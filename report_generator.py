"""ATIS Investigation Report Generator — payload-based, no local persistence.

This module generates structured Knowledge Reports from an Investigation
payload sent by the frontend. It does NOT read from or write to any
database or local file system.

It reuses the existing llm_client.py for all LLM interactions.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from llm_client import get_client, LLMRequestError

logger = logging.getLogger("ATIS_Report")

# =============================================================================
# Public API
# =============================================================================

def generate_investigation_report(investigation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured Knowledge Report from a complete Investigation payload.

    Args:
        investigation: The complete investigation state from the frontend.
            Expected keys (all optional except where noted):
            - investigation_id: str
            - title: str
            - root_question / original_question: str
            - perspective: dict with country, country_code
            - queries: list of query objects with question, answer, findings, etc.
            - entities: list of entity objects
            - relationships: list of relationship objects
            - sources: list of source objects
            - findings: list of finding objects
            - research_required: list or bool
            - accumulated_context: dict

    Returns:
        A structured InvestigationReport dict.

    Raises:
        ValueError: If the investigation payload is missing required fields
            or is too minimal to generate a report.
        RuntimeError: If the LLM call fails or returns unparseable output.
    """
    normalized = _normalize_investigation(investigation)
    if not normalized.get("queries"):
        raise ValueError("Investigation payload must contain at least one query.")

    synthesis_context = _build_synthesis_context(normalized)

    logger.info("Generating report for investigation %s | queries=%d | entities=%d | relationships=%d",
                normalized.get("investigation_id", "unknown"),
                len(normalized.get("queries", [])),
                len(normalized.get("entities", [])),
                len(normalized.get("relationships", [])))

    try:
        client = get_client()
        raw_response = client.chat(
            [
                {"role": "system", "content": _REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": synthesis_context},
            ],
            temperature=0.0,
            max_tokens=8192,
            seed=42,
        )
    except LLMRequestError as exc:
        logger.error("LLM report generation failed: %s", exc)
        raise RuntimeError("Report generation failed: LLM provider error") from exc
    except Exception as exc:
        logger.error("LLM report generation failed unexpectedly: %s", exc)
        raise RuntimeError(f"Report generation failed: {exc}") from exc

    report = _parse_and_validate_report(raw_response, normalized)
    return report


# =============================================================================
# Normalization
# =============================================================================

def _normalize_investigation(inv: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize frontend investigation payload to a stable internal shape."""
    normalized: Dict[str, Any] = {
        "investigation_id": _str_or_empty(inv.get("investigation_id") or inv.get("id")),
        "title": _str_or_empty(inv.get("title") or inv.get("investigation_title")),
        "original_question": _str_or_empty(
            inv.get("original_question")
            or inv.get("root_question")
            or inv.get("question")
        ),
        "perspective_country": _str_or_empty(
            inv.get("perspective", {}).get("country")
            if isinstance(inv.get("perspective"), dict)
            else inv.get("perspective_country")
        ),
        "perspective_country_code": _str_or_empty(
            inv.get("perspective", {}).get("country_code")
            if isinstance(inv.get("perspective"), dict)
            else inv.get("perspective_country_code")
        ),
        "queries": _normalize_queries(inv),
        "entities": _normalize_entities(inv),
        "relationships": _normalize_relationships(inv),
        "sources": _normalize_sources(inv),
        "findings": _normalize_findings(inv),
        "research_required": _normalize_research_required(inv),
        "accumulated_context": inv.get("accumulated_context") or {},
    }

    # Derive title if missing
    if not normalized["title"] and normalized["original_question"]:
        normalized["title"] = _generate_title(normalized["original_question"])

    return normalized


def _str_or_empty(val: Any) -> str:
    return str(val).strip() if val is not None else ""


def _generate_title(question: str) -> str:
    title = question.strip()
    if len(title) > 80:
        title = title[:77] + "..."
    return title


def _normalize_queries(inv: Dict[str, Any]) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    raw_queries = inv.get("queries") or inv.get("query_sequence") or []
    if not isinstance(raw_queries, list):
        return queries

    for i, q in enumerate(raw_queries):
        if not isinstance(q, dict):
            continue
        seq = q.get("sequence") or (i + 1)
        question = _str_or_empty(q.get("question") or q.get("query"))
        answer = _str_or_empty(q.get("answer") or q.get("result", {}).get("executive_summary", ""))
        findings = []
        raw_findings = q.get("findings") or q.get("result", {}).get("findings", []) or []
        for f in raw_findings:
            if isinstance(f, dict):
                findings.append({"text": _str_or_empty(f.get("text")), "source_nodes": f.get("source_nodes", [])})
            elif isinstance(f, str):
                findings.append({"text": f, "source_nodes": []})

        queries.append({
            "sequence": seq,
            "question": question,
            "answer": answer,
            "findings": findings,
            "result": q.get("result") or {},
        })
    return queries


def _normalize_entities(inv: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    # Prefer top-level aggregated entities
    raw_entities = inv.get("entities") or inv.get("aggregated_context", {}).get("entities", [])
    if not isinstance(raw_entities, list):
        raw_entities = []

    for e in raw_entities:
        if not isinstance(e, dict):
            continue
        name = _str_or_empty(e.get("name") or e.get("entity_name") or e.get("entity") or e.get("id"))
        if not name or name in seen:
            continue
        seen.add(name)
        entities.append({
            "name": name,
            "type": _str_or_empty(e.get("type") or e.get("entity_type")),
            "country": _str_or_empty(e.get("country")),
            "sector": _str_or_empty(e.get("sector")),
            "significance": _str_or_empty(e.get("significance") or e.get("significance_score")),
            "summary": _str_or_empty(e.get("summary") or e.get("insight")),
            "evidence_queries": e.get("evidence_queries", []),
        })

    # Also extract from query results if top-level is sparse
    if len(entities) < 3:
        for q in inv.get("queries", []):
            if not isinstance(q, dict):
                continue
            result = q.get("result", {})
            for ent in result.get("key_entities", []) or []:
                if not isinstance(ent, dict):
                    continue
                name = _str_or_empty(ent.get("entity_name") or ent.get("name"))
                if not name or name in seen:
                    continue
                seen.add(name)
                entities.append({
                    "name": name,
                    "type": _str_or_empty(ent.get("entity_type")),
                    "country": _str_or_empty(ent.get("country")),
                    "sector": _str_or_empty(ent.get("sector")),
                    "significance": str(ent.get("significance_score", "")),
                    "summary": _str_or_empty(ent.get("summary")),
                    "evidence_queries": [],
                })

    return entities


def _normalize_relationships(inv: Dict[str, Any]) -> List[Dict[str, Any]]:
    relationships: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    raw_rels = inv.get("relationships") or inv.get("aggregated_context", {}).get("relationships", [])
    if not isinstance(raw_rels, list):
        raw_rels = []

    for r in raw_rels:
        if not isinstance(r, dict):
            continue
        from_ent = _str_or_empty(r.get("from_entity") or r.get("from") or r.get("source"))
        to_ent = _str_or_empty(r.get("to_entity") or r.get("to") or r.get("target"))
        rel_type = _str_or_empty(r.get("relationship_type") or r.get("relationship") or r.get("type"))
        key = f"{from_ent}|{rel_type}|{to_ent}"
        if not from_ent or key in seen:
            continue
        seen.add(key)
        relationships.append({
            "from_entity": from_ent,
            "to_entity": to_ent,
            "relationship_type": rel_type,
            "insight": _str_or_empty(r.get("insight") or r.get("description")),
            "evidence_queries": r.get("evidence_queries", []),
        })

    return relationships


def _normalize_sources(inv: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    raw_sources = inv.get("sources") or inv.get("aggregated_context", {}).get("sources", [])
    if not isinstance(raw_sources, list):
        raw_sources = []

    for s in raw_sources:
        if not isinstance(s, dict):
            continue
        sid = _str_or_empty(s.get("node_id") or s.get("id") or s.get("source_id") or s.get("name"))
        if not sid or sid in seen:
            continue
        seen.add(sid)
        sources.append({
            "id": sid,
            "type": _str_or_empty(s.get("node_type") or s.get("type")),
            "url": _str_or_empty(s.get("url") or s.get("link")),
            "description": _str_or_empty(s.get("description")),
        })

    return sources


def _normalize_findings(inv: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    raw_findings = inv.get("findings") or inv.get("aggregated_context", {}).get("findings", [])
    if not isinstance(raw_findings, list):
        raw_findings = []

    for f in raw_findings:
        if isinstance(f, dict):
            findings.append({
                "text": _str_or_empty(f.get("text")),
                "source_nodes": f.get("source_nodes", []),
            })
        elif isinstance(f, str):
            findings.append({"text": f, "source_nodes": []})

    return findings


def _normalize_research_required(inv: Dict[str, Any]) -> List[str]:
    raw = inv.get("research_required") or inv.get("researchRequired") or []
    if isinstance(raw, bool):
        return ["Further research recommended"] if raw else []
    if isinstance(raw, list):
        return [str(r) for r in raw if r]
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return []


# =============================================================================
# Prompt Construction
# =============================================================================

_REPORT_SYSTEM_PROMPT: str = """
You are the ATIS Investigation Report Synthesis Engine.
Your task is to produce a structured Knowledge Report from an accumulated investigation.

CRITICAL RULES:
1. Use ONLY the supplied investigation information. Do NOT invent facts, sources, entities, or relationships.
2. Distinguish evidence from inference. Mark inferential leaps clearly.
3. Preserve source attribution. Cite specific query numbers (Q1, Q2, etc.) and source IDs where applicable.
4. Identify uncertainty and confidence levels explicitly.
5. Identify research gaps and unanswered questions.
6. Synthesize the sequence of questions into a coherent investigation narrative. Explain how later queries changed or expanded understanding.
7. The report should reveal accumulated intelligence rather than simply concatenating answers.
8. Keep the report professional, analytical, and actionable.
9. Output MUST be valid JSON only. No markdown code fences, no explanatory text outside the JSON.

OUTPUT SCHEMA (valid JSON only):
{
  "title": "string",
  "executive_summary": "string — 2-3 paragraphs comprehensive overview",
  "original_question": "string",
  "investigation_narrative": "string — coherent narrative of how the investigation evolved across queries",
  "key_findings": [
    {"finding": "string", "confidence": "High|Medium|Low", "evidence_queries": ["Q1"], "source_nodes": ["node_id"]}
  ],
  "important_entities": [
    {"name": "string", "type": "string", "significance": "string", "evidence_queries": ["Q1"]}
  ],
  "important_relationships": [
    {"from_entity": "string", "relationship_type": "string", "to_entity": "string", "insight": "string", "evidence_queries": ["Q1"]}
  ],
  "evidence_and_sources": [
    {"source_id": "string", "type": "string", "relevance": "string"}
  ],
  "unresolved_questions": ["string"],
  "research_required": ["string"],
  "implications": "string — implications from the selected perspective",
  "confidence_and_limitations": "string — overall confidence assessment and limitations of the evidence",
  "generated_at": "ISO timestamp (optional, backend will set if missing)"
}
""".strip()


def _build_synthesis_context(investigation: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# ATIS INVESTIGATION KNOWLEDGE REPORT — SYNTHESIS CONTEXT")
    lines.append("")
    lines.append("## Investigation Metadata")
    lines.append(f"- Title: {investigation.get('title', 'Untitled Investigation')}")
    lines.append(f"- Original Question: {investigation.get('original_question', '')}")
    lines.append(f"- Perspective: {investigation.get('perspective_country', '')} ({investigation.get('perspective_country_code', '')})")
    lines.append(f"- Total Queries: {len(investigation.get('queries', []))}")
    lines.append("")

    lines.append("## Query Sequence")
    for q in investigation.get("queries", []):
        seq = q.get("sequence", "?")
        lines.append(f"### Q{seq}: {q.get('question', '')}")
        if q.get("answer"):
            answer = q["answer"].replace("\n", " ").strip()
            lines.append(f"Answer Summary: {answer[:400]}")
        for f in q.get("findings", []):
            lines.append(f"- Finding: {f.get('text', '')}")
        lines.append("")

    lines.append("## Important Entities")
    for ent in investigation.get("entities", [])[:50]:
        lines.append(f"- {ent.get('name', '')} ({ent.get('type', '')}, {ent.get('country', '')}) — {ent.get('summary', '')[:150]}")
    lines.append("")

    lines.append("## Important Relationships")
    for rel in investigation.get("relationships", [])[:30]:
        lines.append(f"- {rel.get('from_entity', '')} → [{rel.get('relationship_type', '')}] → {rel.get('to_entity', '')}")
        if rel.get("insight"):
            lines.append(f"  Insight: {rel['insight'][:150]}")
    lines.append("")

    lines.append("## Key Findings")
    for i, f in enumerate(investigation.get("findings", [])[:30], 1):
        lines.append(f"{i}. {f.get('text', '')}")
    lines.append("")

    lines.append("## Sources Referenced")
    for src in investigation.get("sources", [])[:30]:
        lines.append(f"- {src.get('id', '')} ({src.get('type', '')})")
    lines.append("")

    if investigation.get("research_required"):
        lines.append("## Research Required")
        for r in investigation["research_required"]:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("## REPORT GENERATION INSTRUCTIONS")
    lines.append("Generate a comprehensive Knowledge Report in valid JSON matching the schema above.")
    lines.append("Ground every claim in the evidence provided. Do not invent facts.")
    lines.append("Cite specific query numbers (Q1, Q2, etc.) and source nodes where applicable.")
    lines.append("Explain how the investigation evolved — how did later queries change or expand understanding?")
    lines.append("Output ONLY valid JSON. No markdown fences, no extra text.")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Parsing & Validation
# =============================================================================

def _parse_and_validate_report(raw: str, investigation: Dict[str, Any]) -> Dict[str, Any]:
    """Parse LLM JSON response and validate required fields."""
    cleaned = raw.strip()

    # Strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try direct parse
    data: Dict[str, Any] = {}
    parse_error: Optional[str] = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
        # Try to extract JSON object with regex as fallback extraction
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                parse_error = None
            except json.JSONDecodeError as exc2:
                parse_error = str(exc2)

    if parse_error:
        logger.error("Failed to parse LLM report response as JSON: %s", parse_error)
        raise RuntimeError("Report generation failed: LLM returned invalid JSON. Please retry.")

    if not isinstance(data, dict):
        raise RuntimeError("Report generation failed: LLM returned non-object JSON.")

    # Build final report with fallbacks for missing fields
    report: Dict[str, Any] = {
        "title": _str_or_empty(data.get("title")) or investigation.get("title", "Investigation Report"),
        "executive_summary": _str_or_empty(data.get("executive_summary")),
        "original_question": _str_or_empty(data.get("original_question")) or investigation.get("original_question", ""),
        "investigation_narrative": _str_or_empty(data.get("investigation_narrative")),
        "key_findings": _ensure_list_of_dicts(data.get("key_findings")),
        "important_entities": _ensure_list_of_dicts(data.get("important_entities")),
        "important_relationships": _ensure_list_of_dicts(data.get("important_relationships")),
        "evidence_and_sources": _ensure_list_of_dicts(data.get("evidence_and_sources")),
        "unresolved_questions": _ensure_list_of_strings(data.get("unresolved_questions")),
        "research_required": _ensure_list_of_strings(data.get("research_required")),
        "implications": _str_or_empty(data.get("implications")),
        "confidence_and_limitations": _str_or_empty(data.get("confidence_and_limitations")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "based_on_queries": len(investigation.get("queries", [])),
        "evidence_entities_count": len(investigation.get("entities", [])),
        "evidence_sources_count": len(investigation.get("sources", [])),
    }

    # Validate minimum content
    if not report["executive_summary"] and not report["key_findings"]:
        logger.warning("LLM returned report with empty executive_summary and key_findings")
        # We don't fail here — the frontend can decide how to handle sparse reports.
        # But we log it for observability.

    return report


def _ensure_list_of_dicts(val: Any) -> List[Dict[str, Any]]:
    if isinstance(val, list):
        return [v for v in val if isinstance(v, dict)]
    return []


def _ensure_list_of_strings(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    if isinstance(val, str):
        return [val]
    return []
