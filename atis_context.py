"""Shared analytical perspective context for ATIS pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Set


DEFAULT_PERSPECTIVE_COUNTRY = "Zimbabwe"
DEFAULT_PERSPECTIVE_COUNTRY_CODE = "ZW"

COUNTRY_CODES = {
    "botswana": "BW",
    "kenya": "KE",
    "south africa": "ZA",
    "zambia": "ZM",
    "zimbabwe": "ZW",
}


# Enumerated pathways that constitute actual cross-border mechanisms.
# An opportunity is only cross-border if its pathway is in this set
# AND there is evidence of a cross-border bridge in the vault.
CROSS_BORDER_PATHWAYS: Set[str] = {
    "export",
    "procurement",
    "supplier relationship",
    "regional tender",
    "joint venture",
    "partnership",
    "investment",
    "financing",
    "logistics",
    "professional services",
    "technology transfer",
    "regional infrastructure",
    "power trade",
    "regulatory arbitrage",
    "market entry",
    "cross-border energy trade",
    "regional power pool integration",
}

# Pathways that are domestic to the perspective country.
# If the source event is in a different country but the pathway is domestic,
# the opportunity is NOT valid unless there is a cross-border bridge.
DOMESTIC_PATHWAYS: Set[str] = {
    "domestic procurement",
    "local supply",
    "domestic investment",
    "local financing",
    "domestic logistics",
    "local services",
}


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


def _normalize(text: str | None) -> str:
    """Normalize a string for comparison."""
    if not text:
        return ""
    return str(text).strip().lower()


def _is_perspective_node(node_id: str, node_country: str | None, perspective: PerspectiveContext) -> bool:
    """Determine if a vault node belongs to the perspective country."""
    if not node_country:
        return False
    return _normalize(node_country) == _normalize(perspective.country)


def _pathway_is_cross_border(pathway: str | None) -> bool:
    """Check if a pathway string describes a cross-border mechanism."""
    if not pathway:
        return False
    pw = _normalize(pathway)
    for cbp in CROSS_BORDER_PATHWAYS:
        if cbp in pw:
            return True
    return False


def _pathway_is_domestic(pathway: str | None) -> bool:
    """Check if a pathway string describes a domestic mechanism."""
    if not pathway:
        return False
    pw = _normalize(pathway)
    for dp in DOMESTIC_PATHWAYS:
        if dp in pw:
            return True
    return False


def validate_opportunity(
    opportunity: dict[str, Any],
    perspective: PerspectiveContext,
    source_node_ids: set[str] | None = None,
    perspective_node_ids: set[str] | None = None,
    cross_border_bridges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Deterministically validate an opportunity against perspective-side evidence.

    Parameters
    ----------
    opportunity : dict
        The raw opportunity dict from the LLM.
    perspective : PerspectiveContext
        The user's operating perspective.
    source_node_ids : set[str] | None
        Set of vault node IDs that are source-event nodes (evidence for the event).
    perspective_node_ids : set[str] | None
        Set of vault node IDs that are perspective-country nodes (evidence for actors/capabilities).
    cross_border_bridges : list[dict] | None
        List of cross-border bridge dicts, each with 'from_node', 'to_node', 'relationship_type'.
        Evidence that a perspective-side node connects to a source-country node.

    Returns
    -------
    dict
        The enriched opportunity with status, geography, and validation notes.
    """
    result = dict(opportunity)

    # Attach perspective metadata
    result.update(perspective.as_fields())

    # ------------------------------------------------------------------
    # 1. SOURCE / EVENT GEOGRAPHY (from LLM or inferred)
    # ------------------------------------------------------------------
    source_country = result.get("source_country") or result.get("event_country") or ""
    event_country = result.get("event_country") or source_country or ""
    result["source_country"] = source_country
    result["event_country"] = event_country

    # ------------------------------------------------------------------
    # 2. OPPORTUNITY GEOGRAPHY — DO NOT DEFAULT TO PERSPECTIVE
    # ------------------------------------------------------------------
    opportunity_country = result.get("opportunity_country")
    if not opportunity_country:
        # If the LLM did not set it, we cannot assume perspective country.
        # Leave it empty — the validator will flag this.
        result["opportunity_country"] = ""
    else:
        result["opportunity_country"] = opportunity_country

    # ------------------------------------------------------------------
    # 3. PERSPECTIVE ACTOR VALIDATION
    # ------------------------------------------------------------------
    perspective_actor = result.get("perspective_actor", "")
    actor_is_evidence = False
    actor_node_id = None

    if perspective_actor and perspective_node_ids:
        # Check if the actor name matches any perspective node ID (exact or canonical)
        actor_norm = _normalize(perspective_actor)
        for pid in perspective_node_ids:
            if actor_norm == _normalize(pid):
                actor_is_evidence = True
                actor_node_id = pid
                break
            # Also check if actor name is a substring of node ID or vice versa
            if actor_norm in _normalize(pid) or _normalize(pid) in actor_norm:
                actor_is_evidence = True
                actor_node_id = pid
                break

    result["perspective_actor_evidence"] = actor_is_evidence
    if actor_node_id:
        result["perspective_actor_node_id"] = actor_node_id

    # ------------------------------------------------------------------
    # 4. PERSPECTIVE CAPABILITY VALIDATION
    # ------------------------------------------------------------------
    perspective_capability = result.get("perspective_capability", "")
    capability_is_evidence = bool(perspective_capability) and bool(perspective_actor)

    # Capability must be non-empty and associated with an evidenced actor.
    # We cannot programmatically verify the capability text against node content
    # without the full node content, but we can require that an actor exists.
    result["perspective_capability_evidence"] = capability_is_evidence

    # ------------------------------------------------------------------
    # 5. PATHWAY VALIDATION
    # ------------------------------------------------------------------
    pathway = result.get("pathway", "")
    pathway_is_evidence = False
    pathway_type = "unknown"

    if pathway:
        pw_norm = _normalize(pathway)
        if _pathway_is_cross_border(pathway):
            pathway_type = "cross_border"
            # Cross-border pathway requires cross-border bridge evidence
            if cross_border_bridges:
                pathway_is_evidence = True
        elif _pathway_is_domestic(pathway):
            pathway_type = "domestic"
            # Domestic pathway is only valid if the opportunity is domestic
            # (i.e., source_country == perspective_country)
            if _normalize(source_country) == _normalize(perspective.country):
                pathway_is_evidence = True
            else:
                pathway_is_evidence = False
        else:
            # Unrecognized pathway — allow it but flag for review
            pathway_type = "unrecognized"
            pathway_is_evidence = bool(cross_border_bridges)  # requires some bridge evidence

    result["pathway_type"] = pathway_type
    result["pathway_evidence"] = pathway_is_evidence

    # ------------------------------------------------------------------
    # 6. CROSS-BORDER STATUS — DETERMINISTIC, NOT AUTOMATIC
    # ------------------------------------------------------------------
    # Cross-border is true ONLY when:
    #   a) pathway is cross-border type AND
    #   b) there is at least one cross-border bridge in evidence
    cross_border = False
    if pathway_type == "cross_border" and cross_border_bridges:
        cross_border = True
    result["cross_border"] = cross_border

    if cross_border:
        # Build cross_border_countries from actual geography
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

    # ------------------------------------------------------------------
    # 7. SOURCE NODE VALIDATION
    # ------------------------------------------------------------------
    source_nodes = result.get("source_nodes", [])
    if not isinstance(source_nodes, list):
        source_nodes = []
    if source_node_ids is not None:
        source_nodes = [node for node in source_nodes if node in source_node_ids]
    result["source_nodes"] = source_nodes

    # ------------------------------------------------------------------
    # 8. DETERMINISTIC STATUS ASSIGNMENT
    # ------------------------------------------------------------------
    validation_errors: list[str] = []

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

    if validation_errors:
        result["status"] = "RESEARCH_REQUIRED"
        result["validation_note"] = " | ".join(validation_errors)
        result["required_missing_nodes"] = result.get("required_missing_nodes", []) + validation_errors
    else:
        result["status"] = "VALID"
        result["validation_note"] = "All perspective-side evidence verified."

    return result
