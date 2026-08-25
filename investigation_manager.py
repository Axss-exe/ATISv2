#!/usr/bin/env python3
"""
investigation_manager.py — ATIS Investigation / Query String Backend Layer

A thin orchestration and persistence layer that sits ABOVE the existing
ATIS query engine.  It does NOT duplicate query logic.

Persistence:
    output/investigations/{investigation_id}.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from atis_context import PerspectiveContext
from ATIS_Query import run_query_pipeline
from llm_client import get_client

logger = logging.getLogger("ATIS_Investigation")

# =============================================================================
# Configuration
# =============================================================================
INVESTIGATIONS_DIR = Path(os.getenv("INVESTIGATIONS_DIR", "./output/investigations"))

# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    tmp_path = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    try:
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

def _load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        logger.error("Failed to load investigation JSON %s: %s", path, exc)
        return None

def _extract_entities_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for ent in result.get("key_entities", []):
        if isinstance(ent, dict):
            name = ent.get("entity_name") or ent.get("name", "")
            if name and name not in seen:
                seen.add(name)
                entities.append({
                    "name": name, "type": ent.get("entity_type", ""),
                    "country": ent.get("country", ""), "sector": ent.get("sector", ""),
                    "significance_score": ent.get("significance_score", 0),
                    "summary": ent.get("summary", ""),
                    "source_node": ent.get("source_node", ""),
                })
    for intel in result.get("structured_intelligence", []):
        if isinstance(intel, dict):
            name = intel.get("entity", "")
            if name and name not in seen:
                seen.add(name)
                entities.append({
                    "name": name, "type": intel.get("type", ""),
                    "country": intel.get("country", ""), "sector": "",
                    "significance_score": 0, "summary": intel.get("insight", ""),
                    "source_node": intel.get("source_node", ""),
                })
    for node in result.get("source_nodes", []):
        if isinstance(node, dict):
            name = node.get("id", "")
            if name and name not in seen:
                seen.add(name)
                entities.append({
                    "name": name, "type": node.get("type", ""),
                    "country": "", "sector": "", "significance_score": 0,
                    "summary": "", "source_node": name,
                })
    return entities

def _extract_relationships_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    relationships: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for intel in result.get("structured_intelligence", []):
        if isinstance(intel, dict):
            entity = intel.get("entity", "")
            rel = intel.get("relationship", "")
            if entity and rel:
                key = f"{entity}|{rel}"
                if key not in seen:
                    seen.add(key)
                    relationships.append({
                        "from_entity": entity, "relationship_type": rel,
                        "to_entity": intel.get("related_entity", ""),
                        "insight": intel.get("insight", ""),
                        "source_node": intel.get("source_node", ""),
                    })
    for bridge in result.get("cross_border_bridges", []):
        if isinstance(bridge, dict):
            key = f"{bridge.get('from_node', '')}|{bridge.get('to_node', '')}"
            if key and key not in seen:
                seen.add(key)
                relationships.append({
                    "from_entity": bridge.get("from_node", ""),
                    "relationship_type": bridge.get("relationship_type", ""),
                    "to_entity": bridge.get("to_node", ""),
                    "from_country": bridge.get("from_country", ""),
                    "to_country": bridge.get("to_country", ""),
                    "source_node": "",
                })
    return relationships

def _extract_findings_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    raw_findings = result.get("findings_cited") or result.get("findings", [])
    for f in raw_findings:
        if isinstance(f, dict):
            findings.append({"text": f.get("text", str(f)), "source_nodes": f.get("source_nodes", [])})
        elif isinstance(f, str):
            findings.append({"text": f, "source_nodes": []})
    return findings

def _extract_opportunities_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    opportunities: List[Dict[str, Any]] = []
    raw = result.get("opportunities_cited") or result.get("opportunities", [])
    for o in raw:
        if isinstance(o, dict):
            opportunities.append({
                "title": o.get("title", ""), "type": o.get("type", ""),
                "text": o.get("text", ""), "perspective_actor": o.get("perspective_actor", ""),
                "pathway": o.get("pathway", ""), "urgency_score": o.get("urgency_score", 0),
                "feasibility_score": o.get("feasibility_score", 0),
                "source_nodes": o.get("source_nodes", []),
            })
        elif isinstance(o, str):
            opportunities.append({
                "title": o, "type": "", "text": o, "perspective_actor": "",
                "pathway": "", "urgency_score": 0, "feasibility_score": 0, "source_nodes": [],
            })
    return opportunities

def _extract_risks_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    risks: List[Dict[str, Any]] = []
    raw = result.get("risks_cited") or result.get("risks", [])
    for r in raw:
        if isinstance(r, dict):
            risks.append({"text": r.get("text", str(r)), "source_nodes": r.get("source_nodes", [])})
        elif isinstance(r, str):
            risks.append({"text": r, "source_nodes": []})
    return risks

def _extract_sources_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for node in result.get("source_nodes", []):
        if isinstance(node, dict):
            nid = node.get("id", "")
            if nid and nid not in seen:
                seen.add(nid)
                sources.append({"node_id": nid, "node_type": node.get("type", "")})
    return sources

# =============================================================================
# Core Investigation Functions
# =============================================================================

def create_investigation(
    question: str,
    perspective_country: str = "Zimbabwe",
    perspective_country_code: str = "ZW",
    vault_path: str | Path = "./vault",
) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Question is required to start an investigation.")

    investigation_id = str(uuid.uuid4())
    now = _now_iso()
    perspective = PerspectiveContext.from_values(perspective_country, perspective_country_code)

    logger.info("Investigation %s | Executing initial query: %s", investigation_id, question)
    query_result = run_query_pipeline(question, vault_path, perspective)

    query_id = str(uuid.uuid4())
    query_entry = {
        "query_id": query_id, "sequence": 1,
        "question": question.strip(), "parent_query_id": None,
        "result": query_result, "created_at": now,
    }

    investigation = {
        "investigation_id": investigation_id,
        "title": _generate_title(question),
        "status": "active",
        "perspective": perspective.as_dict(),
        "root_question": question.strip(),
        "queries": [query_entry],
        "aggregated_context": _build_empty_aggregated_context(),
        "report": None,
        "created_at": now, "updated_at": now,
    }

    investigation["aggregated_context"] = aggregate_investigation(investigation)
    _persist_investigation(investigation)
    logger.info("Investigation %s created with %d queries.", investigation_id, len(investigation["queries"]))
    return investigation

def add_query_to_investigation(
    investigation_id: str,
    question: str,
    parent_query_id: Optional[str] = None,
    vault_path: str | Path = "./vault",
) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Question is required.")

    investigation = get_investigation(investigation_id)
    if investigation is None:
        raise FileNotFoundError(f"Investigation not found: {investigation_id}")

    p = investigation.get("perspective", {})
    perspective = PerspectiveContext.from_values(p.get("country"), p.get("country_code"))

    if parent_query_id is None and investigation["queries"]:
        parent_query_id = investigation["queries"][-1]["query_id"]

    if parent_query_id is not None:
        parent_exists = any(q["query_id"] == parent_query_id for q in investigation["queries"])
        if not parent_exists:
            raise ValueError(f"Parent query not found: {parent_query_id}")

    logger.info("Investigation %s | Executing query #%d: %s",
                investigation_id, len(investigation["queries"]) + 1, question)
    query_result = run_query_pipeline(question, vault_path, perspective)

    next_sequence = max((q["sequence"] for q in investigation["queries"]), default=0) + 1
    query_id = str(uuid.uuid4())
    now = _now_iso()

    query_entry = {
        "query_id": query_id, "sequence": next_sequence,
        "question": question.strip(), "parent_query_id": parent_query_id,
        "result": query_result, "created_at": now,
    }

    investigation["queries"].append(query_entry)
    investigation["aggregated_context"] = aggregate_investigation(investigation)
    investigation["updated_at"] = now

    _persist_investigation(investigation)
    logger.info("Investigation %s | Added query %s (seq=%d).", investigation_id, query_id, next_sequence)
    return investigation

def get_investigation(investigation_id: str) -> Optional[Dict[str, Any]]:
    path = INVESTIGATIONS_DIR / f"{investigation_id}.json"
    return _load_json_safe(path)

def list_investigations() -> List[Dict[str, Any]]:
    _ensure_dir(INVESTIGATIONS_DIR)
    results: List[Dict[str, Any]] = []
    for path in sorted(INVESTIGATIONS_DIR.glob("*.json")):
        inv = _load_json_safe(path)
        if inv:
            results.append({
                "investigation_id": inv.get("investigation_id", ""),
                "title": inv.get("title", ""),
                "status": inv.get("status", ""),
                "query_count": len(inv.get("queries", [])),
                "created_at": inv.get("created_at", ""),
                "updated_at": inv.get("updated_at", ""),
            })
    return results

def aggregate_investigation(investigation: Dict[str, Any]) -> Dict[str, Any]:
    aggregated = _build_empty_aggregated_context()
    all_entities: Dict[str, Dict[str, Any]] = {}
    all_relationships: Dict[str, Dict[str, Any]] = {}
    all_findings: List[Dict[str, Any]] = []
    all_opportunities: List[Dict[str, Any]] = []
    all_risks: List[Dict[str, Any]] = []
    all_sources: Dict[str, Dict[str, Any]] = {}

    for query in investigation.get("queries", []):
        result = query.get("result", {})
        if not result:
            continue
        for ent in _extract_entities_from_result(result):
            name = ent.get("name", "")
            if name:
                if name not in all_entities:
                    all_entities[name] = ent
                else:
                    existing = all_entities[name]
                    if ent.get("significance_score", 0) > existing.get("significance_score", 0):
                        existing["significance_score"] = ent["significance_score"]
                    if ent.get("summary") and not existing.get("summary"):
                        existing["summary"] = ent["summary"]
        for rel in _extract_relationships_from_result(result):
            key = f"{rel.get('from_entity', '')}|{rel.get('relationship_type', '')}|{rel.get('to_entity', '')}"
            if key and key not in all_relationships:
                all_relationships[key] = rel
        for f in _extract_findings_from_result(result):
            all_findings.append(f)
        for o in _extract_opportunities_from_result(result):
            all_opportunities.append(o)
        for r in _extract_risks_from_result(result):
            all_risks.append(r)
        for src in _extract_sources_from_result(result):
            nid = src.get("node_id", "")
            if nid and nid not in all_sources:
                all_sources[nid] = src

    aggregated["entities"] = list(all_entities.values())
    aggregated["relationships"] = list(all_relationships.values())
    aggregated["findings"] = all_findings
    aggregated["opportunities"] = all_opportunities
    aggregated["risks"] = all_risks
    aggregated["sources"] = list(all_sources.values())
    return aggregated

def generate_investigation_report(investigation_id: str) -> Dict[str, Any]:
    investigation = get_investigation(investigation_id)
    if investigation is None:
        raise FileNotFoundError(f"Investigation not found: {investigation_id}")

    queries = investigation.get("queries", [])
    if not queries:
        raise ValueError("Investigation has no queries to report on.")

    aggregated = investigation.get("aggregated_context", _build_empty_aggregated_context())
    synthesis_context = _build_report_synthesis_context(investigation, aggregated)
    client = get_client()

    logger.info("Investigation %s | Generating knowledge report...", investigation_id)
    try:
        raw_response = client.chat(
            [
                {"role": "system", "content": _REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": synthesis_context},
            ],
            temperature=0.0, seed=42,
        )
    except Exception as exc:
        logger.error("LLM report generation failed for %s: %s", investigation_id, exc)
        raise RuntimeError(f"Report generation failed: {exc}") from exc

    report = _parse_report_response(raw_response, investigation, aggregated)
    investigation["report"] = report
    investigation["updated_at"] = _now_iso()
    _persist_investigation(investigation)
    logger.info("Investigation %s | Report generated.", investigation_id)
    return report

# =============================================================================
# Internal helpers
# =============================================================================

def _generate_title(question: str) -> str:
    title = question.strip()
    if len(title) > 80:
        title = title[:77] + "..."
    return title

def _build_empty_aggregated_context() -> Dict[str, Any]:
    return {
        "entities": [], "relationships": [], "findings": [],
        "risks": [], "opportunities": [], "sources": [],
    }

def _persist_investigation(investigation: Dict[str, Any]) -> None:
    inv_id = investigation["investigation_id"]
    path = INVESTIGATIONS_DIR / f"{inv_id}.json"
    _atomic_write_json(path, investigation)

def _build_report_synthesis_context(
    investigation: Dict[str, Any], aggregated: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# ATIS INVESTIGATION KNOWLEDGE REPORT — SYNTHESIS CONTEXT")
    lines.append("")
    lines.append("## Investigation Metadata")
    lines.append(f"- Title: {investigation.get('title', '')}")
    lines.append(f"- Root Question: {investigation.get('root_question', '')}")
    lines.append(f"- Perspective: {investigation.get('perspective', {}).get('country', '')} ({investigation.get('perspective', {}).get('country_code', '')})")
    lines.append(f"- Total Queries: {len(investigation.get('queries', []))}")
    lines.append("")
    lines.append("## Query Sequence")
    for q in investigation.get("queries", []):
        lines.append(f"### Q{q.get('sequence', '?')}: {q.get('question', '')}")
        result = q.get("result", {})
        if result.get("executive_summary"):
            summary = result["executive_summary"].replace("\n", " ").strip()
            lines.append(f"Summary: {summary[:300]}")
        lines.append("")
    lines.append("## Aggregated Entities")
    for ent in aggregated.get("entities", [])[:50]:
        lines.append(f"- {ent.get('name', '')} ({ent.get('type', '')}, {ent.get('country', '')}) — {ent.get('summary', '')[:100]}")
    lines.append("")
    lines.append("## Aggregated Relationships")
    for rel in aggregated.get("relationships", [])[:30]:
        lines.append(f"- {rel.get('from_entity', '')} → [{rel.get('relationship_type', '')}] → {rel.get('to_entity', '')}")
    lines.append("")
    lines.append("## Key Findings")
    for i, f in enumerate(aggregated.get("findings", [])[:30], 1):
        lines.append(f"{i}. {f.get('text', '')}")
    lines.append("")
    lines.append("## Opportunities")
    for i, o in enumerate(aggregated.get("opportunities", [])[:20], 1):
        lines.append(f"{i}. {o.get('title', o.get('text', ''))} — Actor: {o.get('perspective_actor', 'N/A')}, Pathway: {o.get('pathway', 'N/A')}")
    lines.append("")
    lines.append("## Risks")
    for i, r in enumerate(aggregated.get("risks", [])[:20], 1):
        lines.append(f"{i}. {r.get('text', '')}")
    lines.append("")
    lines.append("## Sources Referenced")
    for src in aggregated.get("sources", [])[:30]:
        lines.append(f"- {src.get('node_id', '')} ({src.get('node_type', '')})")
    lines.append("")
    lines.append("## REPORT GENERATION INSTRUCTIONS")
    lines.append("Generate a comprehensive Knowledge Report using the sections below.")
    lines.append("Ground every claim in the evidence above. Do not invent facts.")
    lines.append("Cite specific query numbers (Q1, Q2, etc.) and source nodes where applicable.")
    lines.append("")
    return "\n".join(lines)

_REPORT_SYSTEM_PROMPT: str = """
You are the ATIS Investigation Report Synthesis Engine.
Your task is to produce a structured Knowledge Report from an accumulated investigation.

CRITICAL RULES:
1. Every claim MUST be grounded in the provided evidence. Do NOT invent facts.
2. Cite specific queries (Q1, Q2, etc.) and source nodes where applicable.
3. If information is missing for a section, state "Not found in investigation evidence."
4. Do NOT use external knowledge beyond what is provided in the synthesis context.
5. Keep the report professional, analytical, and actionable.

OUTPUT SCHEMA (raw JSON only, no markdown fences):
{
  "executive_summary": "2-3 paragraph comprehensive overview...",
  "investigation_objective": "What the investigation set out to answer...",
  "key_findings": [
    {"finding": "...", "evidence_queries": ["Q1"], "source_nodes": ["Node_ID"]}
  ],
  "key_entities": [
    {"name": "...", "type": "...", "significance": "...", "evidence_queries": ["Q1"]}
  ],
  "key_relationships": [
    {"from": "...", "relationship": "...", "to": "...", "evidence_queries": ["Q1"]}
  ],
  "timeline": "Chronological narrative...",
  "evidence_summary": "Overview of evidence base...",
  "risks": [
    {"risk": "...", "severity": "High|Medium|Low", "evidence_queries": ["Q1"]}
  ],
  "opportunities": [
    {"opportunity": "...", "actor": "...", "pathway": "...", "evidence_queries": ["Q1"]}
  ],
  "contradictions": [
    {"description": "...", "conflicting_queries": ["Q1", "Q2"]}
  ],
  "knowledge_gaps": [
    {"gap": "...", "recommended_query": "..."}
  ],
  "unanswered_questions": ["..."],
  "conclusion": "Synthesized conclusion..."
}
"""

def _parse_report_response(
    raw: str, investigation: Dict[str, Any], aggregated: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    return {
        "executive_summary": data.get("executive_summary", ""),
        "investigation_objective": data.get("investigation_objective", ""),
        "key_findings": data.get("key_findings", []),
        "key_entities": data.get("key_entities", []),
        "key_relationships": data.get("key_relationships", []),
        "timeline": data.get("timeline", ""),
        "evidence_summary": data.get("evidence_summary", ""),
        "risks": data.get("risks", []),
        "opportunities": data.get("opportunities", []),
        "contradictions": data.get("contradictions", []),
        "knowledge_gaps": data.get("knowledge_gaps", []),
        "unanswered_questions": data.get("unanswered_questions", []),
        "conclusion": data.get("conclusion", ""),
        "generated_at": _now_iso(),
        "based_on_queries": len(investigation.get("queries", [])),
        "evidence_entities_count": len(aggregated.get("entities", [])),
        "evidence_sources_count": len(aggregated.get("sources", [])),
    }
