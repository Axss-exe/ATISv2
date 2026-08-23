"""Shared analytical perspective context for ATIS pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_PERSPECTIVE_COUNTRY = "Zimbabwe"
DEFAULT_PERSPECTIVE_COUNTRY_CODE = "ZW"

COUNTRY_CODES = {
    "botswana": "BW",
    "kenya": "KE",
    "south africa": "ZA",
    "zambia": "ZM",
    "zimbabwe": "ZW",
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


def validate_opportunity(opportunity: dict[str, Any], perspective: PerspectiveContext,
                         source_node_ids: set[str] | None = None) -> dict[str, Any]:
    """Add explicit geography and deterministically flag unsupported opportunities."""
    result = dict(opportunity)
    result.update(perspective.as_fields())
    source_nodes = result.get("source_nodes", [])
    if not isinstance(source_nodes, list):
        source_nodes = []
    if source_node_ids is not None:
        source_nodes = [node for node in source_nodes if node in source_node_ids]
    result["source_nodes"] = source_nodes
    result.setdefault("source_country", result.get("event_country", ""))
    result.setdefault("event_country", result.get("source_country", ""))
    result.setdefault("opportunity_country", perspective.country)
    cross_border = bool(result.get("cross_border")) or bool(
        result.get("source_country") and result.get("source_country", "").lower() != perspective.country.lower()
    )
    result["cross_border"] = cross_border
    result.setdefault("cross_border_countries", [perspective.country, result["source_country"]] if result["source_country"] else [perspective.country])
    has_pathway = bool(result.get("perspective_actor") and result.get("perspective_capability") and result.get("pathway"))
    result["status"] = "VALID" if has_pathway and source_nodes else "RESEARCH_REQUIRED"
    result.setdefault("required_missing_nodes", [])
    if result["status"] == "RESEARCH_REQUIRED":
        result.setdefault("validation_note", "Perspective-country actor, capability, pathway, or supporting source node is missing.")
    return result