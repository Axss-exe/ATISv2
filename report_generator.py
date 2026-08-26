"""ATIS Investigation Report Generator — payload-based, no local persistence.

This module generates structured Knowledge Reports from an Investigation
payload sent by the frontend. It does NOT read from or write to any
database or local file system.

It reuses the existing llm_client.py for all LLM interactions.

TOKEN SAFETY:
- Adaptive max_tokens based on investigation complexity
- Input token estimation with automatic context compression
- Truncation detection with retry escalation
- Model capability awareness (uses provider-specific limits)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator
from llm_client import get_client, LLMRequestError, LLMConfigError, LLMTokenLimitError

logger = logging.getLogger("ATIS_Report")

# =============================================================================
# TOKEN BUDGET CONFIGURATION
# =============================================================================

# Base output tokens for any report
_BASE_OUTPUT_TOKENS = 2048

# Additional tokens per content item
_TOKENS_PER_FINDING = 128
_TOKENS_PER_ENTITY = 96
_TOKENS_PER_RELATIONSHIP = 96
_TOKENS_PER_SOURCE = 64
_TOKENS_PER_QUERY = 256

# Hard ceiling to prevent runaway requests
_MAX_OUTPUT_TOKENS_SAFE = 8192

# Input context compression thresholds
_MAX_INPUT_TOKENS_BEFORE_COMPRESSION = 80_000
_MAX_ENTITIES_IN_CONTEXT = 40
_MAX_RELATIONSHIPS_IN_CONTEXT = 25
_MAX_SOURCES_IN_CONTEXT = 30
_MAX_FINDINGS_IN_CONTEXT = 25
_MAX_QUERY_ANSWER_LENGTH = 350

# =============================================================================
# STRICT REPORT SCHEMA (Backend-defined, LLM output validated against this)
# =============================================================================

class KeyFinding(BaseModel):
    """A single key finding from the investigation."""
    finding: str = Field(..., min_length=1, description="The finding text")
    confidence: str = Field(default="Medium", description="Confidence level: High, Medium, or Low")
    source_nodes: List[str] = Field(default_factory=list, description="Source node IDs referenced")
    evidence_queries: List[str] = Field(default_factory=list, description="Query IDs that support this finding")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        v = str(v).strip().capitalize()
        allowed = {"High", "Medium", "Low"}
        if v not in allowed:
            logger.warning("Invalid confidence value '%s', defaulting to Medium", v)
            return "Medium"
        return v


class ImportantEntity(BaseModel):
    """An important entity identified in the investigation."""
    name: str = Field(..., min_length=1, description="Entity name")
    type: str = Field(default="unknown", description="Entity type")
    significance: str = Field(default="", description="Why this entity is significant")
    evidence_queries: List[str] = Field(default_factory=list, description="Query IDs that support this entity")


class ImportantRelationship(BaseModel):
    """An important relationship between entities."""
    insight: str = Field(default="", description="Insight about the relationship")
    to_entity: str = Field(..., min_length=1, description="Target entity name")
    from_entity: str = Field(..., min_length=1, description="Source entity name")
    evidence_queries: List[str] = Field(default_factory=list, description="Query IDs that support this relationship")
    relationship_type: str = Field(default="", description="Type of relationship")


class EvidenceAndSource(BaseModel):
    """An evidence source referenced in the investigation."""
    type: str = Field(default="", description="Source type")
    relevance: str = Field(default="", description="Why this source is relevant")
    source_id: str = Field(..., min_length=1, description="Source identifier")


class InvestigationReport(BaseModel):
    """
    Canonical Investigation Report structure.

    The backend defines this schema. The LLM generates content that is
    validated against this schema. Deterministic fields (title, original_question,
    generated_at, based_on_queries) are set by the backend, not the LLM.
    """
    title: str = Field(..., min_length=1, description="Investigation title")
    generated_at: str = Field(..., description="ISO 8601 timestamp when the report was generated")
    implications: str = Field(default="", description="Strategic implications")
    key_findings: List[KeyFinding] = Field(default_factory=list, description="Key findings")
    based_on_queries: int = Field(..., ge=0, description="Number of queries this report is based on")
    executive_summary: str = Field(default="", description="Executive summary")
    original_question: str = Field(..., min_length=1, description="Original investigation question")
    research_required: List[str] = Field(default_factory=list, description="Research still required")
    important_entities: List[ImportantEntity] = Field(default_factory=list, description="Important entities")
    evidence_and_sources: List[EvidenceAndSource] = Field(default_factory=list, description="Evidence and sources")
    unresolved_questions: List[str] = Field(default_factory=list, description="Unresolved questions")
    evidence_sources_count: int = Field(default=0, ge=0, description="Total number of evidence sources")
    evidence_entities_count: int = Field(default=0, ge=0, description="Total number of evidence entities")
    important_relationships: List[ImportantRelationship] = Field(default_factory=list, description="Important relationships")
    investigation_narrative: str = Field(default="", description="Narrative of how the investigation evolved")
    confidence_and_limitations: str = Field(default="", description="Confidence assessment and limitations")

    @model_validator(mode="after")
    def validate_report_completeness(self) -> "InvestigationReport":
        """Ensure the report has meaningful content."""
        has_content = (
            bool(self.executive_summary.strip())
            or len(self.key_findings) > 0
            or len(self.important_entities) > 0
        )
        if not has_content:
            raise ValueError("Report must contain at least an executive_summary, key_findings, or important_entities")
        return self


# =============================================================================
# Public API
# =============================================================================

def generate_investigation_report(investigation: Dict[str, Any]) -> InvestigationReport:
    """
    Generate a structured, validated InvestigationReport from a complete Investigation payload.

    Token safety:
    1. Estimates input tokens and compresses context if too large.
    2. Calculates adaptive max_tokens based on investigation complexity.
    3. Detects truncated JSON and retries with increased token budget.
    4. Falls back to a smaller report if the full report cannot fit.

    Args:
        investigation: The complete investigation state from the frontend.

    Returns:
        A validated InvestigationReport Pydantic model.

    Raises:
        ValueError: If the investigation payload is missing required fields.
        RuntimeError: If the LLM call fails or returns unparseable output after retries.
    """
    normalized = _normalize_investigation(investigation)
    if not normalized.get("queries"):
        raise ValueError("Investigation payload must contain at least one query.")

    deterministic_fields = _extract_deterministic_fields(normalized)

    # Token-aware context building
    synthesis_context, is_compressed = _build_synthesis_context(normalized)
    estimated_input_tokens = _estimate_tokens(synthesis_context)

    # Calculate adaptive max_tokens
    adaptive_max_tokens = _calculate_adaptive_max_tokens(normalized)

    logger.info(
        "Generating report for investigation %s | queries=%d | entities=%d | relationships=%d | "
        "input_tokens≈%d | output_tokens=%d | compressed=%s",
        normalized.get("investigation_id", "unknown"),
        len(normalized.get("queries", [])),
        len(normalized.get("entities", [])),
        len(normalized.get("relationships", [])),
        estimated_input_tokens,
        adaptive_max_tokens,
        is_compressed,
    )

    # Attempt generation with truncation recovery
    report = _generate_with_retry(
        synthesis_context=synthesis_context,
        normalized=normalized,
        deterministic_fields=deterministic_fields,
        initial_max_tokens=adaptive_max_tokens,
        estimated_input_tokens=estimated_input_tokens,
    )

    return report


# =============================================================================
# Token Management
# =============================================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def _calculate_adaptive_max_tokens(investigation: Dict[str, Any]) -> int:
    """
    Calculate max_tokens based on investigation complexity.
    More content items = more tokens needed for the structured JSON output.
    """
    queries = investigation.get("queries", [])
    entities = investigation.get("entities", [])
    relationships = investigation.get("relationships", [])
    sources = investigation.get("sources", [])
    findings = investigation.get("findings", [])

    required = (
        _BASE_OUTPUT_TOKENS
        + len(queries) * _TOKENS_PER_QUERY
        + len(entities) * _TOKENS_PER_ENTITY
        + len(relationships) * _TOKENS_PER_RELATIONSHIP
        + len(sources) * _TOKENS_PER_SOURCE
        + len(findings) * _TOKENS_PER_FINDING
    )

    # Cap at safe maximum
    return min(required, _MAX_OUTPUT_TOKENS_SAFE)


def _is_json_truncated(raw: str) -> bool:
    """Detect if JSON response appears to be cut off mid-stream."""
    raw = raw.strip()
    if not raw:
        return True
    # Check for common truncation signatures
    if raw.endswith("...") or raw.endswith("\\") or raw.endswith('"'):
        return True
    # Check if braces are balanced
    open_braces = raw.count("{") - raw.count("}")
    open_brackets = raw.count("[") - raw.count("]")
    if open_braces > 0 or open_brackets > 0:
        return True
    # Check for unterminated strings
    if raw.count('"') % 2 != 0:
        return True
    return False


def _generate_with_retry(
    synthesis_context: str,
    normalized: Dict[str, Any],
    deterministic_fields: Dict[str, Any],
    initial_max_tokens: int,
    estimated_input_tokens: int,
    max_attempts: int = 3,
) -> InvestigationReport:
    """
    Attempt report generation with automatic truncation recovery.
    If JSON is truncated, retry with increased token budget.
    If still failing, fall back to a minimal report prompt.
    """
    client = get_client()
    max_tokens = initial_max_tokens
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        logger.info("Report generation attempt %d/%d | max_tokens=%d", attempt, max_attempts, max_tokens)

        try:
            raw_response = client.chat(
                [
                    {"role": "system", "content": _REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": synthesis_context},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                output_format="json",
            )
        except LLMTokenLimitError as exc:
            # Input too large — compress further and retry
            logger.warning("Token limit exceeded on attempt %d: %s. Compressing context...", attempt, exc)
            synthesis_context = _build_compressed_synthesis_context(normalized)
            max_tokens = _MAX_OUTPUT_TOKENS_SAFE
            continue
        except (LLMRequestError, LLMConfigError) as exc:
            logger.error("LLM report generation failed (attempt %d): %s", attempt, exc)
            last_error = exc
            break
        except Exception as exc:
            logger.error("LLM report generation failed unexpectedly (attempt %d): %s", attempt, exc)
            last_error = exc
            break

        # Check for truncation
        if _is_json_truncated(raw_response):
            logger.warning("Truncated JSON detected on attempt %d. Increasing token budget...", attempt)
            max_tokens = min(max_tokens * 2, _MAX_OUTPUT_TOKENS_SAFE)
            last_error = RuntimeError("LLM returned truncated JSON")
            continue

        # Parse and validate
        try:
            report = _parse_and_validate_report(raw_response, normalized, deterministic_fields)
            logger.info("Report generated successfully on attempt %d", attempt)
            return report
        except RuntimeError as exc:
            logger.warning("Parse/validation failed on attempt %d: %s", attempt, exc)
            last_error = exc
            # If parse failed due to size, try increasing tokens
            if "truncated" in str(exc).lower() or "unexpected end" in str(exc).lower():
                max_tokens = min(max_tokens * 2, _MAX_OUTPUT_TOKENS_SAFE)
                continue
            # Otherwise break — bad JSON structure
            break

    # All attempts exhausted — try emergency minimal report
    logger.error("All %d generation attempts failed. Trying emergency minimal report...", max_attempts)
    return _generate_emergency_report(client, normalized, deterministic_fields, last_error)


def _generate_emergency_report(
    client: Any,
    normalized: Dict[str, Any],
    deterministic_fields: Dict[str, Any],
    original_error: Optional[Exception],
) -> InvestigationReport:
    """
    Last-resort fallback: generate a minimal report with heavily compressed context.
    This ensures we always return a valid InvestigationReport, even if reduced.
    """
    minimal_context = _build_minimal_synthesis_context(normalized)

    try:
        raw_response = client.chat(
            [
                {"role": "system", "content": _MINIMAL_SYSTEM_PROMPT},
                {"role": "user", "content": minimal_context},
            ],
            temperature=0.0,
            max_tokens=4096,
            output_format="json",
        )
        report = _parse_and_validate_report(raw_response, normalized, deterministic_fields)
        logger.warning("Emergency minimal report generated successfully")
        return report
    except Exception as exc:
        logger.error("Emergency report generation also failed: %s", exc)
        # Absolute fallback: construct a minimal valid report from investigation data directly
        return _construct_fallback_report(normalized, deterministic_fields, original_error)


def _construct_fallback_report(
    investigation: Dict[str, Any],
    deterministic_fields: Dict[str, Any],
    error: Optional[Exception],
) -> InvestigationReport:
    """
    Absolute last resort: construct a report directly from investigation data
    without LLM synthesis. Guarantees a valid response.
    """
    queries = investigation.get("queries", [])
    entities = investigation.get("entities", [])
    relationships = investigation.get("relationships", [])
    sources = investigation.get("sources", [])
    findings = investigation.get("findings", [])
    research_required = investigation.get("research_required", [])

    # Build minimal key_findings from raw findings
    key_findings = []
    for i, f in enumerate(findings[:15], 1):
        key_findings.append({
            "finding": f.get("text", str(f))[:300],
            "confidence": "Medium",
            "source_nodes": f.get("source_nodes", [])[:5],
            "evidence_queries": [],
        })

    # Build minimal entities
    important_entities = []
    for e in entities[:10]:
        important_entities.append({
            "name": e.get("name", "Unknown"),
            "type": e.get("type", "unknown"),
            "significance": e.get("significance", e.get("summary", ""))[:200],
            "evidence_queries": [],
        })

    # Build minimal relationships
    important_relationships = []
    for r in relationships[:10]:
        important_relationships.append({
            "from_entity": r.get("from_entity", "Unknown"),
            "to_entity": r.get("to_entity", "Unknown"),
            "relationship_type": r.get("relationship_type", ""),
            "insight": r.get("insight", "")[:200],
            "evidence_queries": [],
        })

    # Build minimal sources
    evidence_and_sources = []
    for s in sources[:10]:
        evidence_and_sources.append({
            "source_id": s.get("id", "unknown"),
            "type": s.get("type", ""),
            "relevance": s.get("description", "")[:200],
        })

    report_data = dict(deterministic_fields)
    # Ensure original_question is never empty in fallback
    if not report_data.get("original_question"):
        report_data["original_question"] = report_data.get("title", "Investigation Report")

    report_data.update({
        "executive_summary": (
            f"This report was generated as a fallback due to LLM generation failure. "
            f"The investigation contains {len(queries)} queries, {len(entities)} entities, "
            f"and {len(relationships)} relationships. "
            f"Error: {error if error else 'Unknown'}"
        ),
        "implications": "Unable to synthesize implications due to generation failure.",
        "investigation_narrative": f"Investigation comprised {len(queries)} queries exploring: {report_data.get('original_question', '')}",
        "confidence_and_limitations": (
            f"Low confidence — this is a fallback report. "
            f"Original error: {error if error else 'None'}. "
            f"Evidence base: {len(sources)} sources, {len(entities)} entities."
        ),
        "key_findings": key_findings,
        "important_entities": important_entities,
        "important_relationships": important_relationships,
        "evidence_and_sources": evidence_and_sources,
        "unresolved_questions": research_required[:5] if research_required else ["Further research required"],
        "research_required": research_required[:5] if research_required else [],
    })

    logger.critical("Returning absolute fallback report — LLM synthesis completely failed")
    return InvestigationReport.model_validate(report_data)


# =============================================================================
# Deterministic Field Extraction
# =============================================================================

def _extract_deterministic_fields(normalized: Dict[str, Any]) -> Dict[str, Any]:
    queries = normalized.get("queries", [])
    entities = normalized.get("entities", [])
    sources = normalized.get("sources", [])

    title = normalized.get("title", "Investigation Report") or "Investigation Report"
    original_question = normalized.get("original_question", "") or title

    return {
        "title": title,
        "original_question": original_question,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "based_on_queries": len(queries),
        "evidence_sources_count": len(sources),
        "evidence_entities_count": len(entities),
    }


# =============================================================================
# Normalization (unchanged from previous version)
# =============================================================================

def _normalize_investigation(inv: Dict[str, Any]) -> Dict[str, Any]:
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
    if not normalized["title"] and normalized["original_question"]:
        normalized["title"] = _generate_title(normalized["original_question"])
    # Derive original_question if missing (frontend may only send title)
    if not normalized["original_question"] and normalized["title"]:
        normalized["original_question"] = normalized["title"]

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
        query_id = _str_or_empty(q.get("query_id") or q.get("id") or f"Q{seq}")
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
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "findings": findings,
            "result": q.get("result") or {},
        })
    return queries


def _normalize_entities(inv: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    seen: Set[str] = set()
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
            findings.append({"text": _str_or_empty(f.get("text")), "source_nodes": f.get("source_nodes", [])})
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
# Prompt Construction (Token-Aware)
# =============================================================================

_REPORT_SYSTEM_PROMPT: str = """
You are the ATIS Investigation Report Synthesis Engine.
Produce a structured Knowledge Report from the investigation below.

RULES:
1. Use ONLY supplied evidence. Do NOT invent facts, sources, or relationships.
2. Cite query IDs (Q1, Q2, etc.) and source nodes where applicable.
3. Identify confidence levels (High/Medium/Low) and research gaps.
4. Synthesize queries into a coherent narrative.
5. Output MUST be valid JSON only. No markdown fences, no extra text.
6. Do NOT include title, original_question, generated_at, based_on_queries, evidence_sources_count, or evidence_entities_count — the backend sets these.

OUTPUT SCHEMA:
{
  "executive_summary": "2-3 paragraphs",
  "implications": "strategic implications",
  "investigation_narrative": "how understanding evolved across queries",
  "key_findings": [
    {"finding": "...", "confidence": "High|Medium|Low", "evidence_queries": ["Q1"], "source_nodes": ["id"]}
  ],
  "important_entities": [
    {"name": "...", "type": "...", "significance": "...", "evidence_queries": ["Q1"]}
  ],
  "important_relationships": [
    {"from_entity": "...", "relationship_type": "...", "to_entity": "...", "insight": "...", "evidence_queries": ["Q1"]}
  ],
  "evidence_and_sources": [
    {"source_id": "...", "type": "...", "relevance": "..."}
  ],
  "unresolved_questions": ["..."],
  "research_required": ["..."],
  "confidence_and_limitations": "overall assessment"
}
""".strip()


_MINIMAL_SYSTEM_PROMPT: str = """
You are the ATIS Investigation Report Synthesis Engine.
The investigation context is large. Produce a CONCISE but complete structured report.
Be brief but cover all sections. Output valid JSON only. No markdown fences.
Do NOT include title, original_question, generated_at, based_on_queries, evidence_sources_count, or evidence_entities_count.

OUTPUT SCHEMA:
{
  "executive_summary": "1-2 paragraphs",
  "implications": "brief",
  "investigation_narrative": "brief evolution",
  "key_findings": [{"finding": "...", "confidence": "High|Medium|Low", "evidence_queries": [], "source_nodes": []}],
  "important_entities": [{"name": "...", "type": "...", "significance": "...", "evidence_queries": []}],
  "important_relationships": [{"from_entity": "...", "relationship_type": "...", "to_entity": "...", "insight": "...", "evidence_queries": []}],
  "evidence_and_sources": [{"source_id": "...", "type": "...", "relevance": "..."}],
  "unresolved_questions": ["..."],
  "research_required": ["..."],
  "confidence_and_limitations": "brief"
}
""".strip()


def _build_synthesis_context(investigation: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Build synthesis context with automatic compression if too large.
    Returns (context_string, was_compressed).
    """
    lines: List[str] = []
    lines.append("# ATIS INVESTIGATION REPORT — SYNTHESIS CONTEXT")
    lines.append("")
    lines.append(f"Title: {investigation.get('title', 'Untitled')}")
    lines.append(f"Question: {investigation.get('original_question', '')}")
    lines.append(f"Perspective: {investigation.get('perspective_country', '')} ({investigation.get('perspective_country_code', '')})")
    lines.append(f"Queries: {len(investigation.get('queries', []))}")
    lines.append("")

    lines.append("## Query Sequence")
    for q in investigation.get("queries", []):
        seq = q.get("sequence", "?")
        qid = q.get("query_id", f"Q{seq}")
        lines.append(f"### {qid}: {q.get('question', '')}")
        if q.get("answer"):
            answer = q["answer"].replace("\n", " ").strip()
            lines.append(f"Answer: {answer[:_MAX_QUERY_ANSWER_LENGTH]}")
        for f in q.get("findings", []):
            lines.append(f"- {f.get('text', '')}")
        lines.append("")

    lines.append("## Entities")
    for ent in investigation.get("entities", [])[:_MAX_ENTITIES_IN_CONTEXT]:
        lines.append(f"- {ent.get('name', '')} ({ent.get('type', '')}) — {ent.get('summary', '')[:120]}")
    lines.append("")

    lines.append("## Relationships")
    for rel in investigation.get("relationships", [])[:_MAX_RELATIONSHIPS_IN_CONTEXT]:
        lines.append(f"- {rel.get('from_entity', '')} → [{rel.get('relationship_type', '')}] → {rel.get('to_entity', '')}: {rel.get('insight', '')[:100]}")
    lines.append("")

    lines.append("## Findings")
    for i, f in enumerate(investigation.get("findings", [])[:_MAX_FINDINGS_IN_CONTEXT], 1):
        lines.append(f"{i}. {f.get('text', '')}")
    lines.append("")

    lines.append("## Sources")
    for src in investigation.get("sources", [])[:_MAX_SOURCES_IN_CONTEXT]:
        lines.append(f"- {src.get('id', '')} ({src.get('type', '')})")
    lines.append("")

    if investigation.get("research_required"):
        lines.append("## Research Required")
        for r in investigation["research_required"]:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("## INSTRUCTIONS")
    lines.append("Generate valid JSON matching the schema. Ground every claim. No invented facts.")
    lines.append("")

    context = "\n".join(lines)
    estimated_tokens = _estimate_tokens(context)
    was_compressed = False

    # If still too large, switch to compressed version
    if estimated_tokens > _MAX_INPUT_TOKENS_BEFORE_COMPRESSION:
        logger.warning("Context too large (%d tokens). Switching to compressed mode.", estimated_tokens)
        context = _build_compressed_synthesis_context(investigation)
        was_compressed = True

    return context, was_compressed


def _build_compressed_synthesis_context(investigation: Dict[str, Any]) -> str:
    """Heavily compressed context for very large investigations."""
    lines: List[str] = []
    lines.append("# ATIS INVESTIGATION — COMPRESSED CONTEXT")
    lines.append("")
    lines.append(f"Title: {investigation.get('title', 'Untitled')}")
    lines.append(f"Question: {investigation.get('original_question', '')}")
    lines.append(f"Queries: {len(investigation.get('queries', []))}")
    lines.append("")

    lines.append("## Queries (summarized)")
    for q in investigation.get("queries", []):
        qid = q.get("query_id", f"Q{q.get('sequence', '?')}")
        lines.append(f"{qid}: {q.get('question', '')}")
        if q.get("answer"):
            lines.append(f"  → {q['answer'].replace(chr(10), ' ')[:200]}")
    lines.append("")

    lines.append("## Top Entities")
    for ent in investigation.get("entities", [])[:20]:
        sig = ent.get('significance', '') or ent.get('summary', '')
        lines.append(f"- {ent.get('name', '')} ({ent.get('type', '')}): {sig[:80]}")
    lines.append("")

    lines.append("## Top Relationships")
    for rel in investigation.get("relationships", [])[:15]:
        lines.append(f"- {rel.get('from_entity', '')} → {rel.get('relationship_type', '')} → {rel.get('to_entity', '')}")
    lines.append("")

    lines.append("## Key Findings")
    for f in investigation.get("findings", [])[:20]:
        lines.append(f"- {f.get('text', '')[:150]}")
    lines.append("")

    lines.append("## Sources")
    for s in investigation.get("sources", [])[:20]:
        lines.append(f"- {s.get('id', '')}")
    lines.append("")

    lines.append("Generate a concise but complete JSON report per the schema.")
    return "\n".join(lines)


def _build_minimal_synthesis_context(investigation: Dict[str, Any]) -> str:
    """Minimal context for emergency fallback generation."""
    queries = investigation.get("queries", [])
    entities = investigation.get("entities", [])
    relationships = investigation.get("relationships", [])
    sources = investigation.get("sources", [])
    findings = investigation.get("findings", [])

    lines: List[str] = []
    lines.append("# ATIS INVESTIGATION — MINIMAL CONTEXT")
    lines.append(f"Q: {investigation.get('original_question', '')}")
    lines.append(f"Queries: {len(queries)} | Entities: {len(entities)} | Relationships: {len(relationships)} | Sources: {len(sources)}")
    lines.append("")

    for q in queries[:5]:
        lines.append(f"Query: {q.get('question', '')}")
    lines.append("")

    for e in entities[:8]:
        lines.append(f"Entity: {e.get('name', '')} ({e.get('type', '')})")
    lines.append("")

    for f in findings[:8]:
        lines.append(f"Finding: {f.get('text', '')[:120]}")
    lines.append("")

    lines.append("Generate the most concise valid JSON report possible.")
    return "\n".join(lines)


# =============================================================================
# Parsing & Validation
# =============================================================================

def _parse_and_validate_report(
    raw: str,
    investigation: Dict[str, Any],
    deterministic_fields: Dict[str, Any],
) -> InvestigationReport:
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
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                parse_error = None
            except json.JSONDecodeError as exc2:
                parse_error = str(exc2)

    if parse_error:
        logger.error("Failed to parse LLM report response as JSON: %s", parse_error)
        raise RuntimeError(f"Report generation failed: LLM returned invalid JSON. Parse error: {parse_error}")

    if not isinstance(data, dict):
        raise RuntimeError("Report generation failed: LLM returned non-object JSON.")

    # Build report with deterministic fields + LLM content
    report_data: Dict[str, Any] = dict(deterministic_fields)
    report_data["executive_summary"] = _str_or_empty(data.get("executive_summary"))
    report_data["implications"] = _str_or_empty(data.get("implications"))
    report_data["investigation_narrative"] = _str_or_empty(data.get("investigation_narrative"))
    report_data["confidence_and_limitations"] = _str_or_empty(data.get("confidence_and_limitations"))
    report_data["key_findings"] = _validate_key_findings(data.get("key_findings"), investigation)
    report_data["important_entities"] = _validate_important_entities(data.get("important_entities"), investigation)
    report_data["important_relationships"] = _validate_important_relationships(data.get("important_relationships"), investigation)
    report_data["evidence_and_sources"] = _validate_evidence_and_sources(data.get("evidence_and_sources"), investigation)
    report_data["unresolved_questions"] = _ensure_list_of_strings(data.get("unresolved_questions"))
    report_data["research_required"] = _ensure_list_of_strings(data.get("research_required"))

    # Validate against Pydantic model
    try:
        report = InvestigationReport.model_validate(report_data)
    except Exception as exc:
        logger.error("Pydantic validation failed for generated report: %s", exc)
        validation_errors = _extract_pydantic_errors(exc)
        raise RuntimeError(
            f"Report validation failed: {validation_errors}. "
            f"Raw data keys: {list(report_data.keys())}. "
            f"key_findings: {len(report_data.get('key_findings', []))}. "
            f"important_entities: {len(report_data.get('important_entities', []))}."
        ) from exc

    # Cross-validation of evidence query IDs
    valid_query_ids = {q.get("query_id", f"Q{q.get('sequence', i+1)}") for i, q in enumerate(investigation.get("queries", []))}
    _cross_validate_evidence_queries(report, valid_query_ids)

    logger.info(
        "Report validated | findings=%d entities=%d relationships=%d sources=%d",
        len(report.key_findings),
        len(report.important_entities),
        len(report.important_relationships),
        len(report.evidence_and_sources),
    )
    return report


def _validate_key_findings(raw: Any, investigation: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = _ensure_list_of_dicts(raw)
    result: List[Dict[str, Any]] = []
    for f in findings:
        finding_text = _str_or_empty(f.get("finding") or f.get("text"))
        if not finding_text:
            continue
        source_nodes = f.get("source_nodes", [])
        source_nodes = [str(s) for s in source_nodes] if isinstance(source_nodes, list) else []
        evidence_queries = f.get("evidence_queries", [])
        evidence_queries = [str(q) for q in evidence_queries] if isinstance(evidence_queries, list) else []
        confidence = str(f.get("confidence", "Medium")).strip().capitalize()
        if confidence not in {"High", "Medium", "Low"}:
            confidence = "Medium"
        result.append({
            "finding": finding_text,
            "confidence": confidence,
            "source_nodes": source_nodes,
            "evidence_queries": evidence_queries,
        })
    return result


def _validate_important_entities(raw: Any, investigation: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = _ensure_list_of_dicts(raw)
    result: List[Dict[str, Any]] = []
    seen_names: Set[str] = set()
    for e in entities:
        name = _str_or_empty(e.get("name") or e.get("entity_name"))
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        evidence_queries = e.get("evidence_queries", [])
        evidence_queries = [str(q) for q in evidence_queries] if isinstance(evidence_queries, list) else []
        result.append({
            "name": name,
            "type": _str_or_empty(e.get("type") or e.get("entity_type")),
            "significance": _str_or_empty(e.get("significance")),
            "evidence_queries": evidence_queries,
        })
    return result


def _validate_important_relationships(raw: Any, investigation: Dict[str, Any]) -> List[Dict[str, Any]]:
    relationships = _ensure_list_of_dicts(raw)
    result: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    for r in relationships:
        from_ent = _str_or_empty(r.get("from_entity") or r.get("from"))
        to_ent = _str_or_empty(r.get("to_entity") or r.get("to"))
        rel_type = _str_or_empty(r.get("relationship_type") or r.get("relationship") or r.get("type"))
        if not from_ent or not to_ent:
            continue
        key = f"{from_ent.lower()}|{rel_type.lower()}|{to_ent.lower()}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        evidence_queries = r.get("evidence_queries", [])
        evidence_queries = [str(q) for q in evidence_queries] if isinstance(evidence_queries, list) else []
        result.append({
            "from_entity": from_ent,
            "to_entity": to_ent,
            "relationship_type": rel_type,
            "insight": _str_or_empty(r.get("insight") or r.get("description")),
            "evidence_queries": evidence_queries,
        })
    return result


def _validate_evidence_and_sources(raw: Any, investigation: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = _ensure_list_of_dicts(raw)
    result: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for s in sources:
        sid = _str_or_empty(s.get("source_id") or s.get("id") or s.get("node_id"))
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)
        result.append({
            "source_id": sid,
            "type": _str_or_empty(s.get("type") or s.get("node_type")),
            "relevance": _str_or_empty(s.get("relevance") or s.get("description")),
        })
    return result


def _cross_validate_evidence_queries(report: InvestigationReport, valid_query_ids: Set[str]) -> None:
    for finding in report.key_findings:
        for qid in finding.evidence_queries:
            if qid not in valid_query_ids:
                logger.warning("Finding references unknown query ID: %s", qid)
    for entity in report.important_entities:
        for qid in entity.evidence_queries:
            if qid not in valid_query_ids:
                logger.warning("Entity '%s' references unknown query ID: %s", entity.name, qid)
    for rel in report.important_relationships:
        for qid in rel.evidence_queries:
            if qid not in valid_query_ids:
                logger.warning("Relationship references unknown query ID: %s", qid)


def _extract_pydantic_errors(exc: Exception) -> str:
    if hasattr(exc, "errors"):
        errors = exc.errors()
        messages = []
        for err in errors:
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "unknown error")
            messages.append(f"{loc}: {msg}")
        return "; ".join(messages)
    return str(exc)


def _ensure_list_of_dicts(val: Any) -> List[Dict[str, Any]]:
    if isinstance(val, list):
        return [v for v in val if isinstance(v, dict)]
    return []


def _ensure_list_of_strings(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    if isinstance(val, str):
        return [val] if val.strip() else []
    return []
