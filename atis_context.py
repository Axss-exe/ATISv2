"""Shared analytical perspective context, determinism engine, and validation for ATIS pipelines."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

logger = logging.getLogger("atis_context")

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_PERSPECTIVE_COUNTRY = "Zimbabwe"
DEFAULT_PERSPECTIVE_COUNTRY_CODE = "ZW"

COUNTRY_CODES = {
    "botswana": "BW", "kenya": "KE", "south africa": "ZA", "zambia": "ZM", "zimbabwe": "ZW",
    "nigeria": "NG", "ghana": "GH", "tanzania": "TZ", "uganda": "UG", "rwanda": "RW",
    "ethiopia": "ET", "egypt": "EG", "morocco": "MA", "algeria": "DZ", "tunisia": "TN",
    "libya": "LY", "sudan": "SD", "south sudan": "SS", "democratic republic of the congo": "CD",
    "republic of the congo": "CG", "angola": "AO", "namibia": "NA", "mozambique": "MZ",
    "malawi": "MW", "eswatini": "SZ", "lesotho": "LS", "madagascar": "MG", "mauritius": "MU",
    "seychelles": "SC", "comoros": "KM", "djibouti": "DJ", "eritrea": "ER", "somalia": "SO",
    "central african republic": "CF", "chad": "TD", "cameroon": "CM", "equatorial guinea": "GQ",
    "gabon": "GA", "sao tome and principe": "ST", "benin": "BJ", "burkina faso": "BF",
    "cape verde": "CV", "cote d'ivoire": "CI", "gambia": "GM", "guinea": "GN",
    "guinea-bissau": "GW", "liberia": "LR", "mali": "ML", "mauritania": "MR", "niger": "NE",
    "senegal": "SN", "sierra leone": "SL", "togo": "TG",
}

CROSS_BORDER_PATHWAYS: Set[str] = {
    "export", "procurement", "supplier relationship", "regional tender", "joint venture",
    "partnership", "investment", "financing", "logistics", "professional services",
    "technology transfer", "regional infrastructure", "power trade", "regulatory arbitrage",
    "market entry", "cross-border energy trade", "regional power pool integration",
    "cross-border supply", "regional distribution", "bilateral trade", "regional value chain",
}

DOMESTIC_PATHWAYS: Set[str] = {
    "domestic procurement", "local supply", "domestic investment", "local financing",
    "domestic logistics", "local services", "domestic market", "local distribution",
}

ANALYSIS_VERSION = "2.1.0-perspective-deterministic"
SCHEMA_VERSION = "2.1.0"


# =============================================================================
# PERSPECTIVE CONTEXT
# =============================================================================

@dataclass(frozen=True)
class PerspectiveContext:
    country: str = DEFAULT_PERSPECTIVE_COUNTRY
    country_code: str = DEFAULT_PERSPECTIVE_COUNTRY_CODE

    @classmethod
    def from_values(cls, country: str | None = None, country_code: str | None = None) -> "PerspectiveContext":
        resolved_country = (country or DEFAULT_PERSPECTIVE_COUNTRY).strip()
        resolved_code = (country_code or COUNTRY_CODES.get(resolved_country.lower(), "")).strip().upper()
        if not resolved_code:
            raise ValueError(f"A country code is required for perspective country '{resolved_country}'.")
        return cls(resolved_country, resolved_code)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "PerspectiveContext":
        payload = payload or {}
        perspective = payload.get("perspective")
        if isinstance(perspective, Mapping):
            return cls.from_values(perspective.get("country"), perspective.get("country_code"))
        return cls.from_values(payload.get("perspective_country"), payload.get("perspective_country_code"))

    def as_dict(self) -> dict[str, str]:
        return {"country": self.country, "country_code": self.country_code}

    def as_fields(self) -> dict[str, str]:
        return {"perspective_country": self.country, "perspective_country_code": self.country_code}


# =============================================================================
# KNOWLEDGE STATE / VAULT VERSIONING
# =============================================================================

@dataclass
class KnowledgeState:
    vault_path: Path
    vault_version: str = field(default="")
    knowledge_state_hash: str = field(default="")
    evidence_set_hash: str = field(default="")
    indexed_at: str = field(default="")
    total_nodes: int = field(default=0)
    total_files: int = field(default=0)

    def compute(self) -> "KnowledgeState":
        if not self.vault_path.exists():
            self.knowledge_state_hash = "empty_vault"
            self.evidence_set_hash = "empty_vault"
            self.indexed_at = datetime.now(timezone.utc).isoformat()
            return self

        md_files = sorted(self.vault_path.rglob("*.md"), key=lambda p: str(p))
        self.total_files = len(md_files)

        hasher = hashlib.sha256()
        for md_path in md_files:
            rel = str(md_path.relative_to(self.vault_path))
            mtime = str(md_path.stat().st_mtime)
            size = str(md_path.stat().st_size)
            hasher.update(f"{rel}|{mtime}|{size}\n".encode("utf-8"))
        self.knowledge_state_hash = hasher.hexdigest()[:32]

        evidence_hasher = hashlib.sha256()
        for md_path in md_files:
            try:
                content = md_path.read_text(encoding="utf-8")
                evidence_hasher.update(content.encode("utf-8"))
            except Exception:
                pass
        self.evidence_set_hash = evidence_hasher.hexdigest()[:32]

        self.vault_version = f"v{self.knowledge_state_hash}"
        self.indexed_at = datetime.now(timezone.utc).isoformat()
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "vault_version": self.vault_version,
            "knowledge_state_hash": self.knowledge_state_hash,
            "evidence_set_hash": self.evidence_set_hash,
            "indexed_at": self.indexed_at,
            "total_nodes": self.total_nodes,
            "total_files": self.total_files,
        }


# =============================================================================
# ANALYSIS FINGERPRINT
# =============================================================================

def compute_analysis_fingerprint(
    story_id: str,
    perspective: PerspectiveContext,
    evidence_ids: List[str],
    entity_ids: List[str],
    relationship_ids: List[str],
    analysis_version: str = ANALYSIS_VERSION,
    schema_version: str = SCHEMA_VERSION,
    knowledge_state_hash: str = "",
) -> str:
    material = {
        "story_id": story_id,
        "perspective_country": perspective.country,
        "perspective_country_code": perspective.country_code,
        "evidence_ids": sorted(set(evidence_ids)),
        "entity_ids": sorted(set(entity_ids)),
        "relationship_ids": sorted(set(relationship_ids)),
        "analysis_version": analysis_version,
        "schema_version": schema_version,
        "knowledge_state_hash": knowledge_state_hash,
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def compute_opportunity_identity(
    title: str,
    perspective_country: str,
    source_country: str,
    event_country: str,
    opportunity_country: str,
    perspective_actor: str,
    perspective_capability: str,
    pathway: str,
    source_nodes: List[str],
) -> str:
    material = {
        "title": title.strip().lower(),
        "perspective_country": perspective_country.strip().lower(),
        "source_country": source_country.strip().lower(),
        "event_country": event_country.strip().lower(),
        "opportunity_country": opportunity_country.strip().lower(),
        "perspective_actor": perspective_actor.strip().lower(),
        "perspective_capability": perspective_capability.strip().lower(),
        "pathway": pathway.strip().lower(),
        "source_nodes": sorted(set(source_nodes)),
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"))
    hash_val = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"OPP-{hash_val.upper()}"


# =============================================================================
# CACHE ISOLATION
# =============================================================================

class AnalysisCache:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path("./.atis_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}

    def _cache_path(self, fingerprint: str) -> Path:
        return self.cache_dir / f"analysis_{fingerprint}.json"

    def get(self, fingerprint: str) -> dict[str, Any] | None:
        if fingerprint in self._memory_cache:
            logger.info("Cache HIT (memory): %s", fingerprint)
            return dict(self._memory_cache[fingerprint])

        cache_path = self._cache_path(fingerprint)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                self._memory_cache[fingerprint] = data
                logger.info("Cache HIT (disk): %s", fingerprint)
                return data
            except Exception as exc:
                logger.warning("Cache read failed for %s: %s", fingerprint, exc)
                return None
        return None

    def set(self, fingerprint: str, data: dict[str, Any]) -> None:
        self._memory_cache[fingerprint] = dict(data)
        cache_path = self._cache_path(fingerprint)
        try:
            cache_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.info("Cache SET: %s", fingerprint)
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", fingerprint, exc)

    def invalidate_by_evidence(self, evidence_id: str) -> int:
        removed = 0
        for cache_file in sorted(self.cache_dir.glob("analysis_*.json")):
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                deps = data.get("evidence_ids", [])
                if evidence_id in deps:
                    cache_file.unlink()
                    removed += 1
                    fp = cache_file.stem.replace("analysis_", "")
                    self._memory_cache.pop(fp, None)
            except Exception:
                continue
        logger.info("Invalidated %d cached analyses dependent on evidence %s", removed, evidence_id)
        return removed

    def invalidate_all(self) -> int:
        removed = 0
        for cache_file in sorted(self.cache_dir.glob("analysis_*.json")):
            try:
                cache_file.unlink()
                removed += 1
            except Exception:
                continue
        self._memory_cache.clear()
        logger.info("Invalidated %d cached analyses (full cache clear)", removed)
        return removed


# =============================================================================
# OPPORTUNITY VALIDATION
# =============================================================================

def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return str(text).strip().lower()


def _pathway_is_cross_border(pathway: str | None) -> bool:
    if not pathway:
        return False
    pw = _normalize(pathway)
    for cbp in CROSS_BORDER_PATHWAYS:
        if cbp in pw:
            return True
    return False


def _pathway_is_domestic(pathway: str | None) -> bool:
    if not pathway:
        return False
    pw = _normalize(pathway)
    for dp in DOMESTIC_PATHWAYS:
        if dp in pw:
            return True
    return False


class NodeValidationResult:
    def __init__(self, node_id: str, exists: bool, node_type: str = ""):
        self.node_id = node_id
        self.exists = exists
        self.node_type = node_type


def validate_node_references(
    node_references: List[str],
    existing_node_ids: Set[str],
) -> Tuple[List[NodeValidationResult], List[NodeValidationResult], List[NodeValidationResult]]:
    existing: List[NodeValidationResult] = []
    research: List[NodeValidationResult] = []
    inferences: List[NodeValidationResult] = []

    for ref in node_references:
        ref_norm = _normalize(ref)
        if not ref_norm:
            continue

        matched = False
        for existing_id in existing_node_ids:
            if _normalize(existing_id) == ref_norm:
                existing.append(NodeValidationResult(existing_id, True, "existing_node"))
                matched = True
                break
            if ref_norm in _normalize(existing_id) or _normalize(existing_id) in ref_norm:
                existing.append(NodeValidationResult(existing_id, True, "existing_node"))
                matched = True
                break

        if not matched:
            if any(word in ref_norm for word in ["verify", "check", "research", "investigate", "confirm", "determine", "find"]):
                research.append(NodeValidationResult(ref, False, "research_requirement"))
            elif len(ref_norm) < 5 or ref_norm.isdigit():
                inferences.append(NodeValidationResult(ref, False, "inference"))
            else:
                research.append(NodeValidationResult(ref, False, "research_requirement"))

    return existing, research, inferences


def validate_opportunity(
    opportunity: dict[str, Any],
    perspective: PerspectiveContext,
    source_node_ids: set[str] | None = None,
    perspective_node_ids: set[str] | None = None,
    cross_border_bridges: list[dict[str, Any]] | None = None,
    existing_node_ids: set[str] | None = None,
) -> dict[str, Any]:
    result = dict(opportunity)
    cross_border_bridges = cross_border_bridges or []
    source_node_ids = source_node_ids or set()
    perspective_node_ids = perspective_node_ids or set()
    existing_node_ids = existing_node_ids or set()

    result.update(perspective.as_fields())

    source_country = result.get("source_country") or result.get("event_country") or ""
    event_country = result.get("event_country") or source_country or ""
    result["source_country"] = source_country
    result["event_country"] = event_country

    opportunity_country = result.get("opportunity_country", "")
    if not opportunity_country:
        result["opportunity_country"] = ""

    # Node validation
    all_referenced_nodes: List[str] = []
    for key in ["source_nodes", "required_nodes", "required_missing_nodes", "entity_nodes"]:
        vals = result.get(key, [])
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, str):
                    all_referenced_nodes.append(v)

    existing_nodes, research_reqs, inferences = validate_node_references(
        all_referenced_nodes, existing_node_ids | source_node_ids | perspective_node_ids
    )

    result["node_validation"] = {
        "existing_nodes": [{"node_id": n.node_id, "type": n.node_type} for n in existing_nodes],
        "research_requirements": [{"node_id": n.node_id, "type": n.node_type} for n in research_reqs],
        "inferences": [{"node_id": n.node_id, "type": n.node_type} for n in inferences],
    }

    # Perspective actor validation
    perspective_actor = result.get("perspective_actor", "")
    actor_is_evidence = False
    actor_node_id = None

    if perspective_actor and perspective_node_ids:
        actor_norm = _normalize(perspective_actor)
        for pid in perspective_node_ids:
            if actor_norm == _normalize(pid):
                actor_is_evidence = True
                actor_node_id = pid
                break
            if actor_norm in _normalize(pid) or _normalize(pid) in actor_norm:
                actor_is_evidence = True
                actor_node_id = pid
                break

    result["perspective_actor_evidence"] = actor_is_evidence
    if actor_node_id:
        result["perspective_actor_node_id"] = actor_node_id

    # Capability validation
    perspective_capability = result.get("perspective_capability", "")
    capability_is_evidence = bool(perspective_capability) and bool(perspective_actor) and actor_is_evidence
    result["perspective_capability_evidence"] = capability_is_evidence

    # Pathway validation
    pathway = result.get("pathway", "")
    pathway_is_evidence = False
    pathway_type = "unknown"

    if pathway:
        pw_norm = _normalize(pathway)
        if _pathway_is_cross_border(pathway):
            pathway_type = "cross_border"
            if cross_border_bridges:
                pathway_is_evidence = True
        elif _pathway_is_domestic(pathway):
            pathway_type = "domestic"
            if _normalize(source_country) == _normalize(perspective.country):
                pathway_is_evidence = True
            else:
                pathway_is_evidence = False
        else:
            pathway_type = "unrecognized"
            pathway_is_evidence = bool(cross_border_bridges)

    result["pathway_type"] = pathway_type
    result["pathway_evidence"] = pathway_is_evidence

    # Cross-border status
    cross_border = False
    if pathway_type == "cross_border" and cross_border_bridges:
        cross_border = True
    result["cross_border"] = cross_border

    if cross_border:
        countries = set()
        if perspective.country:
            countries.add(perspective.country)
        if opportunity_country:
            countries.add(opportunity_country)
        if source_country and _normalize(source_country) != _normalize(perspective.country):
            countries.add(source_country)
        result["cross_border_countries"] = sorted(countries)
    else:
        result["cross_border_countries"] = []

    # Source node validation
    source_nodes = result.get("source_nodes", [])
    if not isinstance(source_nodes, list):
        source_nodes = []
    if source_node_ids:
        source_nodes = [node for node in source_nodes if node in source_node_ids]
    result["source_nodes"] = source_nodes

    # Status assignment
    validation_errors: List[str] = []
    validation_notes: List[str] = []

    if inferences:
        validation_errors.append(f"Invented nodes detected: {[n.node_id for n in inferences]}")
        validation_notes.append("REJECTED: LLM invented node IDs that do not exist in the vault.")

    if research_reqs:
        validation_notes.append(f"Research required for: {[n.node_id for n in research_reqs]}")

    if not perspective_actor:
        validation_errors.append("Missing perspective_actor.")
    elif not actor_is_evidence:
        validation_errors.append(f"Perspective actor '{perspective_actor}' is not evidenced in the vault.")

    if not perspective_capability:
        validation_errors.append("Missing perspective_capability.")
    elif not capability_is_evidence:
        validation_errors.append("Perspective capability is not associated with an evidenced actor.")

    if not pathway:
        validation_errors.append("Missing pathway.")
    elif not pathway_is_evidence:
        if pathway_type == "cross_border":
            validation_errors.append("Cross-border pathway lacks vault evidence (no cross-border bridges found).")
        elif pathway_type == "domestic" and _normalize(source_country) != _normalize(perspective.country):
            validation_errors.append("Domestic pathway is invalid for a foreign-source event without cross-border mechanism.")
        else:
            validation_errors.append("Pathway is not evidenced or recognized.")

    if not source_nodes:
        validation_errors.append("No source nodes cited.")

    if not result.get("opportunity_country"):
        validation_errors.append("Missing opportunity_country (LLM did not determine where the opportunity exists).")

    # Final status
    if inferences:
        result["status"] = "RESEARCH_REQUIRED"
        result["validation_note"] = " | ".join(validation_errors + validation_notes)
    elif validation_errors:
        result["status"] = "RESEARCH_REQUIRED"
        result["validation_note"] = " | ".join(validation_errors + validation_notes)
        result["required_missing_nodes"] = result.get("required_missing_nodes", []) + validation_errors
    elif not existing_nodes and not research_reqs:
        result["status"] = "NO_VALIDATED_OPPORTUNITY"
        result["validation_note"] = "No validated opportunity could be constructed from available evidence."
    else:
        result["status"] = "VALID"
        result["validation_note"] = "All perspective-side evidence verified."

    # Deterministic scoring factors
    result["scoring_factors"] = {
        "evidence_strength": _score_evidence_strength(existing_nodes, source_nodes),
        "perspective_fit": _score_perspective_fit(actor_is_evidence, capability_is_evidence),
        "actor_capability": _score_actor_capability(actor_is_evidence, capability_is_evidence),
        "pathway_strength": _score_pathway_strength(pathway_is_evidence, cross_border_bridges),
        "cross_border_validity": 1.0 if cross_border else (0.5 if pathway_type == "domestic" else 0.0),
    }

    return result


def _score_evidence_strength(existing_nodes: List[NodeValidationResult], source_nodes: List[str]) -> float:
    if not existing_nodes:
        return 0.0
    ratio = len(existing_nodes) / max(len(source_nodes), 1)
    return min(1.0, ratio)


def _score_perspective_fit(actor_evidence: bool, capability_evidence: bool) -> float:
    if actor_evidence and capability_evidence:
        return 1.0
    elif actor_evidence:
        return 0.5
    return 0.0


def _score_actor_capability(actor_evidence: bool, capability_evidence: bool) -> float:
    if actor_evidence and capability_evidence:
        return 1.0
    elif actor_evidence:
        return 0.3
    return 0.0


def _score_pathway_strength(pathway_evidence: bool, bridges: List[dict]) -> float:
    if pathway_evidence and bridges:
        return min(1.0, 0.5 + 0.1 * len(bridges))
    elif pathway_evidence:
        return 0.5
    return 0.0
