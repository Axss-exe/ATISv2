#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATIS_Query.py v4.1 — Perspective-First Grounded Architecture + Determinism

4-STAGE PIPELINE:
  0. INTENT EXTRACTION:    LLM understands question → structured intent (1 call)
  1. SOURCE PRE-FILTER:    Python loose matching → source-event candidates (0 calls)
  2. PERSPECTIVE-SIDE RETRIEVAL: Programmatic vault query for perspective-country actors,
     capabilities, and cross-border bridges (0 calls)
  3. LLM SEMANTIC RANKING: LLM reads combined candidates → ranks top 20 by relevance (1 call)
  4. GROUNDED SYNTHESIS:   LLM synthesizes dashboard from ranked nodes ONLY,
     with mandatory perspective actor/capability/pathway evidence (1 call)

DETERMINISM FEATURES (v4.1):
  - All LLM calls: temperature=0.0, seed=42
  - All vault iterations: sorted(..., key=lambda n: n.uid)
  - AnalysisCache: disk-based caching (check before LLM, set after generation)
  - KnowledgeState: vault versioning computed at pipeline start
  - compute_analysis_fingerprint: stable result identity
  - compute_opportunity_identity: stable opportunity IDs (non-sequential)
  - analysis_version, schema_version, analysis_fingerprint, knowledge_state in all outputs

BACKWARD COMPATIBILITY:
  - findings/opportunities/risks returned as STRING arrays (v0 compatible)
  - Cited versions available in *_cited fields
  - Response is flat: no nested .dashboard wrapper

ANTI-HALLUCINATION:
  - LLM only sees provided nodes
  - Every claim must cite source_node
  - Missing data → "Not found in vault"
  - Perspective actor MUST be from perspective-side node registry
  - Cross-border pathways MUST be supported by bridge evidence
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
from datetime import date, datetime, timezone

from llm_client import LLMClient, get_client
from atis_context import (
    PerspectiveContext, validate_opportunity,
    KnowledgeState, AnalysisCache,
    compute_analysis_fingerprint, compute_opportunity_identity,
    ANALYSIS_VERSION, SCHEMA_VERSION,
)

MAX_TOKENS_PER_REQUEST: int = 60_000
RESPONSE_RESERVE: int = 8_000
SAFETY_BUFFER: int = 1_000

BROAD_FILTER_MAX_CANDIDATES: int = 80
LLM_RANKING_MAX_RESULTS: int = 20
FULL_SCAN_MAX_NODES: int = 100

# =============================================================================
# Dependencies
# =============================================================================
_MISSING_DEPS: List[str] = []

try:
    import yaml
except ImportError:
    _MISSING_DEPS.append("PyYAML (pip install pyyaml)")

if _MISSING_DEPS:
    print("FATAL: Missing required dependencies:\n  - " + "\n  - ".join(_MISSING_DEPS), file=sys.stderr)
    sys.exit(1)

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("ATIS_Query")


def _sanitize_for_json(data):
    if isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_for_json(item) for item in data]
    return data


# =============================================================================
# PROMPTS
# =============================================================================
INTENT_EXTRACTION_PROMPT: str = """
You are the ATIS Intent Extraction Engine. Analyze the user's question and extract a structured search intent. Output ONLY valid raw JSON. No markdown fences. No commentary.

OUTPUT SCHEMA:
{
  "intent_type": "OVERVIEW|SPECIFIC_ENTITY|FILTERED_LIST|RELATIONSHIP|COMPARISON",
  "target_entities": ["concrete nouns to search for"],
  "target_entity_types": ["mining_refinery|government_agency|commodity|policy_framework|infrastructure_node|private_conglomerate|government_ministry|academic_institution"],
  "target_countries": ["country names if mentioned"],
  "target_sectors": ["sector names if mentioned"],
  "target_attributes": {"status": "operational|planned|under_construction", "ownership": "state_owned|private"},
  "relationship_type": "regulates|owns|funds|operates|licenses|null",
  "output_format": "structured_table|summary|entity_profile|relationship_graph",
  "max_results_hint": 5|10|15|20|30
}

RULES:
- intent_type: OVERVIEW for broad summaries, SPECIFIC_ENTITY for 'who is X', FILTERED_LIST for 'which X are Y', RELATIONSHIP for 'who regulates X'.
- target_entities: extract ONLY concrete nouns. Strip vague words like overview, intelligence, information, data, summary, landscape.
- max_results_hint: OVERVIEW=20, SPECIFIC_ENTITY=3, FILTERED_LIST=15, RELATIONSHIP=8.
- target_countries: infer the SOURCE/EVENT country from the question, NOT the perspective country.
- Output ONLY raw JSON.
"""

SEMANTIC_RANKING_PROMPT: str = """
You are a Semantic Relevance Engine for the ATIS Intelligence System. Your job is to rank vault nodes by their relevance to the user's question from the selected analytical perspective.

You will be given:
1. The user's question
2. The analytical perspective country
3. A list of candidate vault nodes (each with id, type, country, sector, summary)
4. A PERSPECTIVE ACTOR REGISTRY of evidenced actors from the perspective country
5. A CROSS-BORDER BRIDGE CONTEXT showing actual relationships between perspective and source countries

Your task:
- Read each candidate carefully
- Score relevance from 1.0 to 10.0 based on how directly the node answers the question
- Prioritize nodes that: (1) directly answer the question, (2) are from the perspective country, (3) have cross-border bridge evidence
- Select the top N most relevant nodes
- Provide 1-sentence reasoning for each selection

SCORING CRITERIA (in priority order):
- 10.0: Directly answers the question AND is a perspective-country actor with cross-border bridge
- 8.0-9.0: Directly answers the question (exact entity, regulator, parent company)
- 6.0-7.0: Perspective-country actor with capability relevant to the question
- 4.0-5.0: Moderately relevant (same sector, related commodity, same country)
- 1.0-3.0: Weakly relevant (tangential connection)
- 0.0: Not relevant at all

OUTPUT SCHEMA (raw JSON only):
{
  "ranked_nodes": [
    {
      "rank": 1,
      "node_id": "exact_filename_no_extension",
      "relevance_score": 9.5,
      "reasoning": "One sentence explaining why this node is relevant"
    }
  ],
  "excluded_count": 45
}

RULES:
- Use the EXACT node_id as provided. Do not modify filenames.
- If a node is not relevant, do not include it.
- Be strict. A score of 10.0 should be rare.
- Output ONLY raw JSON. No markdown fences.
"""

GROUNDED_SYNTHESIS_PROMPT: str = """
You are the ATIS Grounded Synthesis Engine. You have access ONLY to the provided vault nodes. You MUST NOT use any external knowledge. If information is not in the provided nodes, say 'Not found in vault'.

ANTI-HALLUCINATION RULES (violation = invalid output):
1. Every claim in structured_intelligence MUST have a 'source_node' field containing the exact node ID.
2. Every item in findings[], opportunities[], risks[] MUST have a 'text' field and a 'source_nodes' array with at least one node ID.
3. Every entity in key_entities[] MUST correspond to a provided node and have a 'source_node' field.
4. The executive_summary MUST reference specific entities by their exact names and explain their roles.
5. If you cannot verify a claim from the provided nodes, output 'Not found in vault' for that field.
6. Do NOT invent statistics, dates, or facts not present in the nodes.

PERSPECTIVE RULES:
7. You MUST select perspective_actor from the PERSPECTIVE ACTOR REGISTRY. Do not invent actors.
8. You MUST select perspective_capability from the capabilities evidenced for that actor in the vault.
9. You MUST select pathway from the CROSS-BORDER BRIDGE CONTEXT or from: export, procurement, supplier relationship, regional tender, joint venture, partnership, investment, financing, logistics, professional services, technology transfer, regional infrastructure, power trade, regulatory arbitrage, market entry.
10. You MUST set opportunity_country to the actual country where the commercial value exists. Do NOT default to the perspective country.
11. If no perspective-side actor can respond to the event, mark the opportunity RESEARCH_REQUIRED and explain the gap.
12. A source-country event does NOT automatically create a perspective-country opportunity. There must be an evidenced cross-border pathway.

EXECUTIVE SUMMARY REQUIREMENTS:
- Length: 6-10 sentences
- Must explain: the overall landscape, key players and their specific roles, regulatory context, structural gaps, primary opportunities, and key risks
- Must name specific entities (e.g., 'ZESA Holdings operates the national grid under ZERA regulation')
- Must connect entities to each other (e.g., 'Bikita Minerals is regulated by the Ministry of Mines')
- Must explain WHY the dashboard findings matter

OUTPUT SCHEMA (raw JSON only):
{
  "executive_summary": "6-10 sentence comprehensive narrative...",
  "structured_intelligence": [
    {
      "entity": "Entity Name",
      "type": "entity_type",
      "country": "Zimbabwe",
      "relationship": "regulates",
      "status": "Operational",
      "priority": "Critical|High|Medium|Low",
      "insight": "One precise sentence.",
      "source_node": "Exact_Node_ID"
    }
  ],
  "findings": [
    {"text": "Finding statement.", "source_nodes": ["Node_ID_1", "Node_ID_2"]}
  ],
  "opportunities": [
    {"opportunity_id": "AUTO", "title": "Opportunity statement", "type": "String", "perspective_country": "Zimbabwe", "perspective_country_code": "ZW", "source_country": "String", "event_country": "String", "opportunity_country": "String", "cross_border": true, "cross_border_countries": ["Zimbabwe", "String"], "perspective_actor": "MUST be from PERSPECTIVE ACTOR REGISTRY", "perspective_capability": "MUST be evidenced capability", "pathway": "MUST be evidenced pathway", "urgency_score": 0.0, "feasibility_score": 0.0, "required_missing_nodes": [], "capital_flow": {"beneficiary": "String", "likely_funder": "String"}, "justification": "String", "source_nodes": ["Node_ID"]}
  ],
  "risks": [
    {"text": "Risk statement.", "source_nodes": ["Node_ID"]}
  ],
  "key_entities": [
    {
      "entity_name": "Exact Name",
      "entity_type": "type",
      "country": "Zimbabwe",
      "sector": "Energy",
      "significance_score": 9,
      "related_count": 5,
      "summary": "Description from vault.",
      "source_node": "Exact_Node_ID"
    }
  ]
}

Output ONLY raw JSON. No markdown fences. No commentary outside JSON.
"""


# =============================================================================
# INTENT CLASS
# =============================================================================
class ATISIntent:
    def __init__(self, raw_json: dict):
        self.intent_type: str = raw_json.get("intent_type", "OVERVIEW")
        self.target_entities: List[str] = raw_json.get("target_entities", [])
        self.target_entity_types: List[str] = raw_json.get("target_entity_types", [])
        self.target_countries: List[str] = [c.lower() for c in raw_json.get("target_countries", [])]
        self.target_sectors: List[str] = [s.lower() for s in raw_json.get("target_sectors", [])]
        self.target_attributes: Dict[str, str] = raw_json.get("target_attributes", {})
        self.relationship_type: str | None = raw_json.get("relationship_type")
        self.output_format: str = raw_json.get("output_format", "structured_table")
        self.max_results_hint: int = raw_json.get("max_results_hint", 20)
        self.perspective_country: str = raw_json.get("perspective_country", "")

    def __repr__(self) -> str:
        return f"ATISIntent({self.intent_type}, entities={self.target_entities}, types={self.target_entity_types})"


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class VaultNode:
    uid: str
    absolute_path: Path
    stem: str
    front_matter: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    outbound_links: List[str] = field(default_factory=list)
    raw_content: str = ""
    body: str = ""
    body_preview: str = ""
    backlink_uids: List[str] = field(default_factory=list)
    entity_type: str = "unknown"
    country: str = ""
    sector: str = ""


# =============================================================================
# VAULT MANAGER
# =============================================================================
class ObsidianVaultManager:
    _WIKILINK_PATTERN = re.compile(r'\[\[(.*?)\]\]')

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = Path(vault_root).resolve()
        self.nodes: Dict[str, VaultNode] = {}
        self.backlink_map: Dict[str, List[str]] = {}
        self._link_resolver: Dict[str, str] = {}
        self.indexed_count: int = 0
        self._commodity_set: Set[str] = set()
        self._country_set: Set[str] = set()
        self._sector_set: Set[str] = set()

    @staticmethod
    def canonicalize(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "", str(text)).lower()

    @staticmethod
    def _parse_markdown(raw_content: str) -> Tuple[Dict[str, Any], str]:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw_content, re.DOTALL)
        if match:
            try:
                front = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                front = {}
            body = match.group(2)
        else:
            front = {}
            body = raw_content
        return front, body

    @staticmethod
    def _extract_summary(front_matter: Dict[str, Any], body: str) -> str:
        for key in ("summary", "Summary", "description", "Description", "abstract", "Abstract", "note", "Note"):
            val = front_matter.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            elif isinstance(val, list) and val and isinstance(val[0], str) and val[0].strip():
                return val[0].strip()
        m = re.search(r"(?im)^#{1,6}\s*Summary\s*\n(.*?)(?=\n#{1,6}\s|\Z)", body, re.DOTALL)
        if m:
            return m.group(1).strip()
        return body[:400].strip()

    @classmethod
    def _extract_outbound_links(cls, raw_content: str) -> List[str]:
        raw_links = cls._WIKILINK_PATTERN.findall(raw_content)
        cleaned: List[str] = []
        seen: set = set()
        for link in raw_links:
            target = link.split("|")[0]
            core = target.split("/")[-1].strip()
            if core and core.lower() not in seen:
                seen.add(core.lower())
                cleaned.append(core)
        return cleaned

    @staticmethod
    def _extract_aliases(front_matter: Dict[str, Any]) -> List[str]:
        aliases: List[str] = []
        for key in ("aliases", "alias", "title", "name"):
            val = front_matter.get(key)
            if isinstance(val, str):
                aliases.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        aliases.append(item)
        return sorted(list(set(aliases)))

    def _infer_entity_metadata(self, front_matter: Dict[str, Any], body: str) -> Tuple[str, str, str]:
        entity_type = front_matter.get("node_type", "") or front_matter.get("type", "") or front_matter.get("entity_type", "")
        country = front_matter.get("country", "") or front_matter.get("location", "")
        sector = front_matter.get("sector", "") or front_matter.get("industry", "")

        if not entity_type or entity_type == "unknown":
            body_lower = body.lower()
            type_map = [
                (["mine", "refinery", "smelter", "concentrator"], "mining_refinery"),
                (["ministry", "department of"], "government_ministry"),
                (["agency", "regulatory", "authority", "commission"], "government_agency"),
                (["university", "institute", "polytechnic"], "academic_institution"),
                (["power plant", "dam", "railway", "port", "grid"], "infrastructure_node"),
                (["lithium", "cobalt", "copper", "gold", "nickel", "ore"], "commodity"),
                (["act", "law", "policy", "framework", "regulation"], "policy_framework"),
                (["corporation", "ltd", "inc", "group", "company"], "private_conglomerate"),
            ]
            for keywords, etype in type_map:
                if any(w in body_lower for w in keywords):
                    entity_type = etype
                    break

        # Infer country from path if not in frontmatter
        if not country:
            # This will be set during build_index
            pass

        return entity_type, country, sector

    def build_index(self) -> None:
        logger.info("Starting vault index build at: %s", self.vault_root)
        if not self.vault_root.exists():
            raise FileNotFoundError(f"Vault root does not exist: {self.vault_root}")

        md_files = sorted(list(self.vault_root.rglob("*.md")), key=lambda p: str(p))
        logger.info("Located %d markdown files.", len(md_files))

        for md_path in md_files:
            try:
                raw_content = md_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, Exception) as exc:
                logger.warning("Skipping file %s: %s", md_path, exc)
                continue

            front_matter, body = self._parse_markdown(raw_content)
            summary = self._extract_summary(front_matter, body)
            outbound_links = self._extract_outbound_links(raw_content)
            entity_type, country, sector = self._infer_entity_metadata(front_matter, body)

            # Infer country from path
            if not country:
                path_parts = [p.lower() for p in md_path.relative_to(self.vault_root).parts]
                for part in path_parts:
                    if part in ("zimbabwe", "zambia", "south africa", "botswana", "kenya", "china", "germany", "switzerland", "united kingdom", "united states of america"):
                        country = part.title()
                        break

            uid = md_path.stem
            node = VaultNode(
                uid=uid, absolute_path=md_path, stem=uid,
                front_matter=front_matter, summary=summary,
                outbound_links=outbound_links, raw_content=raw_content,
                body=body, entity_type=entity_type, country=country,
                sector=sector, body_preview=body[:2000],
            )
            self.nodes[uid] = node
            self._link_resolver[self.canonicalize(uid)] = uid
            for alias in self._extract_aliases(front_matter):
                self._link_resolver[self.canonicalize(alias)] = uid

            if entity_type == "commodity":
                self._commodity_set.add(uid.lower())
            if country:
                self._country_set.add(country.lower())
            if sector:
                self._sector_set.add(sector.lower())

        for node in sorted(self.nodes.values(), key=lambda n: n.uid):
            resolved: List[str] = []
            seen: set = set()
            for link in node.outbound_links:
                canon = self.canonicalize(link)
                if canon in self._link_resolver:
                    target_uid = self._link_resolver[canon]
                    if target_uid not in seen:
                        seen.add(target_uid)
                        resolved.append(target_uid)
                else:
                    if link.lower() not in seen:
                        seen.add(link.lower())
                        resolved.append(link)
            node.outbound_links = resolved

        for uid, node in self.nodes.items():
            for target in node.outbound_links:
                if target in self.nodes:
                    if target not in self.backlink_map:
                        self.backlink_map[target] = []
                    self.backlink_map[target].append(uid)

        for uid, node in self.nodes.items():
            node.backlink_uids = self.backlink_map.get(uid, [])

        self.indexed_count = len(self.nodes)
        logger.info("Vault index complete. Cached %d nodes. Backlinks: %d. Commodities: %d, Countries: %d, Sectors: %d",
                    self.indexed_count, len(self.backlink_map),
                    len(self._commodity_set), len(self._country_set), len(self._sector_set))

    def get_aggregate_stats(self, node_subset: List[VaultNode] | None = None) -> Dict[str, Any]:
        nodes = node_subset if node_subset is not None else sorted(self.nodes.values(), key=lambda n: n.uid)
        entity_type_counts: Dict[str, int] = {}
        for node in nodes:
            et = node.entity_type or "unknown"
            entity_type_counts[et] = entity_type_counts.get(et, 0) + 1
        total_relationships = sum(len(n.outbound_links) for n in nodes)
        return {
            "total_entities": len(nodes),
            "total_relationships": total_relationships,
            "commodities_tracked": len(self._commodity_set),
            "countries_covered": len(self._country_set),
            "sectors_mapped": len(self._sector_set),
            "entity_type_breakdown": entity_type_counts,
        }

    def build_entity_graph(self, node_subset: List[VaultNode] | None = None) -> Dict[str, Any]:
        nodes = node_subset if node_subset is not None else sorted(self.nodes.values(), key=lambda n: n.uid)
        subset_uids = {n.uid for n in nodes}
        graph_nodes: List[Dict[str, Any]] = []
        graph_edges: List[Dict[str, Any]] = []

        type_visual_map = {
            "mining_refinery": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "private_conglomerate": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "government_agency": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "government_ministry": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "academic_institution": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "infrastructure_node": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "commodity": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "policy_framework": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
        }

        for idx, node in enumerate(nodes):
            visual = type_visual_map.get(node.entity_type, type_visual_map["private_conglomerate"])
            graph_nodes.append({
                "id": node.uid, "label": node.stem, "type": node.entity_type,
                "shape": visual["shape"], "fill": visual["fill"], "stroke": visual["stroke"],
                "text_color": visual["text_color"], "x": 50 + (idx % 8) * 80,
                "y": 40 + (idx // 8) * 60, "width": 120, "height": 50,
                "summary": node.summary[:120],
            })

        for node in nodes:
            for target in node.outbound_links:
                if target in subset_uids:
                    graph_edges.append({
                        "from": node.uid, "to": target, "type": "flow",
                        "label": "links to", "stroke": "#262626", "width": 1.2, "dasharray": "3 3",
                    })

        return {"viewBox": "0 0 700 280", "height": 280, "nodes": graph_nodes, "edges": graph_edges}

    def get_all_nodes_as_context(self) -> List[VaultNode]:
        return sorted(self.nodes.values(), key=lambda n: n.uid)

    # -- Perspective-Side Retrieval (NEW) ----------------------------------- #
    def get_perspective_nodes(self, perspective: PerspectiveContext) -> List[VaultNode]:
        """Retrieve all vault nodes that belong to the perspective country."""
        perspective_norm = perspective.country.lower()
        results: List[VaultNode] = []
        for node in sorted(self.nodes.values(), key=lambda n: n.uid):
            if (node.country or "").lower() == perspective_norm:
                results.append(node)
        logger.info("Retrieved %d perspective-side nodes for %s", len(results), perspective.country)
        return results

    def get_cross_border_bridges(self, perspective: PerspectiveContext, source_country: str) -> List[Dict[str, Any]]:
        """
        Find cross-border bridges between perspective country and source country.
        A bridge exists when a perspective-country node links to a source-country node or vice versa.
        """
        perspective_norm = perspective.country.lower()
        source_norm = source_country.lower()
        bridges: List[Dict[str, Any]] = []

        for node in sorted(self.nodes.values(), key=lambda n: n.uid):
            node_country = (node.country or "").lower()
            if node_country not in (perspective_norm, source_norm):
                continue

            for link in node.outbound_links:
                if link in self.nodes:
                    target = self.nodes[link]
                    target_country = (target.country or "").lower()
                    if node_country == perspective_norm and target_country == source_norm:
                        bridges.append({
                            "from_node": node.uid,
                            "from_country": perspective.country,
                            "to_node": target.uid,
                            "to_country": source_country,
                            "relationship_type": "outbound_link",
                        })
                    elif node_country == source_norm and target_country == perspective_norm:
                        bridges.append({
                            "from_node": node.uid,
                            "from_country": source_country,
                            "to_node": target.uid,
                            "to_country": perspective.country,
                            "relationship_type": "outbound_link",
                        })

        # Check backlinks
        for uid, node in sorted(self.nodes.items(), key=lambda x: x[0]):
            node_country = (node.country or "").lower()
            if node_country != perspective_norm:
                continue
            for back_uid in node.backlink_uids:
                if back_uid in self.nodes:
                    back_node = self.nodes[back_uid]
                    back_country = (back_node.country or "").lower()
                    if back_country == source_norm:
                        bridges.append({
                            "from_node": back_uid,
                            "from_country": source_country,
                            "to_node": uid,
                            "to_country": perspective.country,
                            "relationship_type": "backlink",
                        })

        logger.info("Found %d cross-border bridges between %s and %s", len(bridges), perspective.country, source_country)
        return sorted(bridges, key=lambda b: (b.get("from_node", ""), b.get("to_node", "")))


# =============================================================================
# TOKEN BUDGET
# =============================================================================
@dataclass(frozen=True)
class TokenBudget:
    max_tokens: int = MAX_TOKENS_PER_REQUEST
    response_reserve: int = RESPONSE_RESERVE
    safety_buffer: int = SAFETY_BUFFER

    @property
    def available_for_input(self) -> int:
        return self.max_tokens - self.response_reserve - self.safety_buffer

    @staticmethod
    def estimate(text: str) -> int:
        if not text:
            return 0
        return int(len(text) / 3.2) + 1

    def truncate_payload(self, system_prompt: str, user_prompt: str, min_user_ratio: float = 0.30) -> str:
        total_estimated = self.estimate(system_prompt + user_prompt)
        if total_estimated <= self.available_for_input:
            return user_prompt
        available_chars = int(self.available_for_input * 3.2) - len(system_prompt)
        if available_chars <= 0:
            raise RuntimeError("System prompt alone exceeds the token budget.")
        max_chars = int(available_chars * min_user_ratio)
        truncated = user_prompt[:max_chars] + "\n\n[TRUNCATED]"
        logger.warning("Truncated from %d to ~%d chars.", len(user_prompt), max_chars)
        return truncated


# =============================================================================
# LLM ENGINE — 4-STAGE GROUNDED PIPELINE
# =============================================================================
class LLMQueryEngine:
    def __init__(self) -> None:
        self.client: LLMClient = get_client()
        self.token_budget = TokenBudget()
        self.cache: AnalysisCache = AnalysisCache()

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        return self.client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            seed=42,
        )

    # -----------------------------------------------------------------
    # STAGE 0: Intent Extraction
    # -----------------------------------------------------------------
    def extract_intent(self, question: str) -> ATISIntent:
        logger.info("STAGE 0: INTENT EXTRACTION")
        user_prompt = f'Question: "{question}"\n\nExtract the structured intent as JSON.'

        raw_response = self._call_api(
            system_prompt=INTENT_EXTRACTION_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            intent_json = json.loads(raw_response)
            intent = ATISIntent(intent_json)
            logger.info("Intent: %s", intent)
            return intent
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse intent JSON: %s. Raw: %s", exc, raw_response[:200])
            return ATISIntent({"intent_type": "OVERVIEW", "target_entities": [], "target_entity_types": [], "max_results_hint": 20})

    # -----------------------------------------------------------------
    # STAGE 1: Source Pre-Filter
    # -----------------------------------------------------------------
    def broad_pre_filter(self, vault_mgr: ObsidianVaultManager, intent: ATISIntent,
                         perspective: PerspectiveContext) -> List[VaultNode]:
        logger.info("STAGE 1: BROAD PRE-FILTER")
        all_nodes = sorted(vault_mgr.nodes.values(), key=lambda n: n.uid)
        scored: List[Tuple[float, VaultNode]] = []

        for node in all_nodes:
            score = 0.0
            node_type = (node.entity_type or "").lower()
            for target_type in intent.target_entity_types:
                tt = target_type.lower().replace("_", "")
                nt = node_type.replace("_", "")
                if tt == nt or tt in nt or nt in tt:
                    score += 8.0

            node_country = (node.country or "").lower()
            path_text = str(node.absolute_path).lower()

            # Score source-event country relevance (NOT perspective country)
            for tc in intent.target_countries:
                if tc.lower() in node_country:
                    score += 6.0

            # Do NOT boost perspective country here — perspective nodes are retrieved separately
            node_sector = (node.sector or "").lower()
            for ts in intent.target_sectors:
                if ts.lower() in node_sector:
                    score += 5.0

            node_name = node.uid.lower().replace("_", " ").replace("-", " ")
            content = f"{node.summary} {node.body_preview}".lower()
            for entity in intent.target_entities:
                ec = entity.lower()
                if ec in node_name:
                    score += 10.0
                elif any(w in node_name for w in ec.split()):
                    score += 4.0
                if ec in content:
                    score += 2.0

            for attr_key, attr_val in intent.target_attributes.items():
                if attr_val.lower() in content:
                    score += 3.0

            if intent.intent_type == "OVERVIEW":
                score += (len(node.outbound_links) + len(node.backlink_uids)) * 0.3

            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: (-x[0], x[1].uid))
        cap = min(BROAD_FILTER_MAX_CANDIDATES, len(scored))
        result = [node for _, node in scored[:cap]]
        logger.info("Broad filter: %d scored → %d candidates", len(scored), len(result))
        return result

    # -----------------------------------------------------------------
    # STAGE 2: Perspective-Side Retrieval (NEW)
    # -----------------------------------------------------------------
    def retrieve_perspective_context(self, vault_mgr: ObsidianVaultManager,
                                     perspective: PerspectiveContext,
                                     intent: ATISIntent) -> Tuple[List[VaultNode], List[Dict[str, Any]]]:
        """
        Explicitly retrieve perspective-side nodes and cross-border bridges.
        This is separate from the semantic ranking and is NOT filtered by question relevance.
        """
        logger.info("STAGE 2: PERSPECTIVE-SIDE RETRIEVAL")

        perspective_nodes = vault_mgr.get_perspective_nodes(perspective)

        # Determine source country from intent
        source_country = ""
        if intent.target_countries:
            source_country = intent.target_countries[0]
        else:
            # Try to infer from question entities
            for entity in intent.target_entities:
                entity_lower = entity.lower()
                if "zambia" in entity_lower:
                    source_country = "Zambia"
                    break
                elif "zimbabwe" in entity_lower:
                    source_country = "Zimbabwe"
                    break

        cross_border_bridges = []
        if source_country and source_country.lower() != perspective.country.lower():
            cross_border_bridges = vault_mgr.get_cross_border_bridges(perspective, source_country)

        logger.info("Perspective retrieval: %d nodes | %d bridges | source: %s",
                    len(perspective_nodes), len(cross_border_bridges), source_country or "unknown")
        return perspective_nodes, cross_border_bridges

    # -----------------------------------------------------------------
    # STAGE 3: LLM Semantic Ranking
    # -----------------------------------------------------------------
    def llm_semantic_ranking(self, question: str, intent: ATISIntent,
                             perspective: PerspectiveContext,
                             candidates: List[VaultNode],
                             perspective_nodes: List[VaultNode],
                             cross_border_bridges: List[Dict[str, Any]]) -> List[VaultNode]:
        logger.info("STAGE 3: LLM SEMANTIC RANKING (%d candidates)", len(candidates))

        if not candidates:
            return []

        # Build candidate blocks
        candidate_blocks = []
        for node in candidates:
            block = (
                f"NODE_ID: {node.uid}\n"
                f"TYPE: {node.entity_type} | COUNTRY: {node.country or 'N/A'} | SECTOR: {node.sector or 'N/A'}\n"
                f"SUMMARY: {node.summary[:200]}\n"
                f"---"
            )
            candidate_blocks.append(block)

        # Build perspective registry block
        perspective_blocks = []
        perspective_blocks.append(f"=== PERSPECTIVE ACTOR REGISTRY ({perspective.country}) ===")
        for pn in perspective_nodes[:20]:
            perspective_blocks.append(
                f"- {pn.uid} | type: {pn.entity_type} | summary: {pn.summary[:100]}"
            )

        # Build bridge context block
        if cross_border_bridges:
            perspective_blocks.append(f"=== CROSS-BORDER BRIDGE CONTEXT ===")
            for bridge in cross_border_bridges[:15]:
                perspective_blocks.append(
                    f"- {bridge['from_node']} ({bridge['from_country']}) → {bridge['to_node']} ({bridge['to_country']}) via {bridge['relationship_type']}"
                )
        else:
            perspective_blocks.append(f"=== CROSS-BORDER BRIDGE CONTEXT ===\nNo cross-border bridges found.")

        candidates_text = "\n\n".join(candidate_blocks)
        perspective_text = "\n".join(perspective_blocks)

        user_prompt = (
            f"## USER QUESTION\n{question}\n\n"
            f"## ANALYTICAL PERSPECTIVE\n{perspective.country} ({perspective.country_code})\n\n"
            f"## SEARCH INTENT\n"
            f"Type: {intent.intent_type}\n"
            f"Looking for: {', '.join(intent.target_entities) or 'N/A'}\n"
            f"Entity types: {', '.join(intent.target_entity_types) or 'N/A'}\n\n"
            f"## PERSPECTIVE CONTEXT\n"
            f"{perspective_text}\n\n"
            f"## CANDIDATE NODES ({len(candidates)} total)\n"
            f"{candidates_text}\n\n"
            f"## TASK\n"
            f"Rank the top {LLM_RANKING_MAX_RESULTS} most relevant nodes. "
            f"Return ONLY raw JSON with ranked_nodes array."
        )

        if self.token_budget.estimate(SEMANTIC_RANKING_PROMPT + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(SEMANTIC_RANKING_PROMPT)
            user_prompt = user_prompt[:max_chars] + "\n[TRUNCATED]"
            logger.warning("Ranking prompt truncated.")

        raw_response = self._call_api(
            system_prompt=SEMANTIC_RANKING_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            ranking_data = json.loads(raw_response)
            ranked_list = ranking_data.get("ranked_nodes", [])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse ranking JSON: %s. Using top candidates by score.", exc)
            return candidates[:LLM_RANKING_MAX_RESULTS]

        ranked_nodes = []
        candidate_map = {n.uid: n for n in candidates}
        for item in ranked_list:
            node_id = item.get("node_id", "")
            if node_id in candidate_map:
                node = candidate_map[node_id]
                ranked_nodes.append(node)
                logger.info("  Ranked: %s (score=%s, reason=%s)",
                            node_id, item.get("relevance_score"), item.get("reasoning", "")[:60])

        logger.info("LLM ranking: %d nodes selected", len(ranked_nodes))
        return ranked_nodes

    # -----------------------------------------------------------------
    # STAGE 4: Grounded Synthesis
    # -----------------------------------------------------------------
    def grounded_synthesis(self, question: str, intent: ATISIntent,
                           perspective: PerspectiveContext,
                           ranked_nodes: List[VaultNode],
                           perspective_nodes: List[VaultNode],
                           cross_border_bridges: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("STAGE 4: GROUNDED SYNTHESIS (%d ranked nodes, %d perspective nodes)",
                    len(ranked_nodes), len(perspective_nodes))

        if not ranked_nodes and not perspective_nodes:
            return self._generate_empty_response(question, intent)

        # Build ranked node blocks
        node_blocks = []
        for node in ranked_nodes:
            fm_fields = []
            for key in ("entity", "sector", "country", "ownership_type", "status", "regulatory_status"):
                val = node.front_matter.get(key)
                if val:
                    fm_fields.append(f"{key}: {val}")

            block = (
                f"=== SOURCE NODE: {node.uid} ===\n"
                f"Type: {node.entity_type}\n"
                f"Country: {node.country or 'N/A'}\n"
                f"Sector: {node.sector or 'N/A'}\n"
                f"Frontmatter: {'; '.join(fm_fields) or 'N/A'}\n"
                f"Summary: {node.summary[:300]}\n"
                f"Outbound Links: {', '.join(node.outbound_links[:8])}\n"
                f"Backlinks: {', '.join(node.backlink_uids[:5])}\n"
            )
            node_blocks.append(block)

        # Build perspective node blocks
        perspective_blocks = []
        perspective_blocks.append(f"=== PERSPECTIVE ACTOR REGISTRY ({perspective.country}) ===")
        for pn in perspective_nodes[:25]:
            fm_fields = []
            for key in ("entity", "sector", "country", "ownership_type", "status", "capabilities"):
                val = pn.front_matter.get(key)
                if val:
                    fm_fields.append(f"{key}: {val}")
            perspective_blocks.append(
                f"- {pn.uid} | type: {pn.entity_type} | country: {pn.country or 'N/A'} | "
                f"frontmatter: {'; '.join(fm_fields) or 'N/A'} | summary: {pn.summary[:150]}"
            )

        # Build bridge blocks
        if cross_border_bridges:
            perspective_blocks.append(f"=== CROSS-BORDER BRIDGE CONTEXT ({perspective.country}) ===")
            for bridge in cross_border_bridges[:20]:
                perspective_blocks.append(
                    f"- {bridge['from_node']} ({bridge['from_country']}) → {bridge['to_node']} ({bridge['to_country']}) via {bridge['relationship_type']}"
                )
        else:
            perspective_blocks.append(f"=== CROSS-BORDER BRIDGE CONTEXT ===\nNo cross-border bridges found.")

        context = "\n\n".join(node_blocks + perspective_blocks)

        user_prompt = (
            f"## USER QUESTION\n{question}\n\n"
            f"## ANALYTICAL PERSPECTIVE\nPerspective country: {perspective.country}\nPerspective country code: {perspective.country_code}\n"
            "You are answering from this perspective, not automatically from the source-event country. "
            "An external event can create a perspective-country opportunity ONLY if there is an evidenced cross-border pathway. "
            "Every opportunity must show an evidenced actor from the PERSPECTIVE ACTOR REGISTRY, an evidenced capability, and an evidenced pathway; otherwise mark it RESEARCH_REQUIRED.\n\n"
            f"## SEARCH INTENT\n"
            f"Type: {intent.intent_type}\n"
            f"Looking for: {', '.join(intent.target_entities) or 'N/A'}\n"
            f"Entity types: {', '.join(intent.target_entity_types) or 'N/A'}\n"
            f"Countries: {', '.join(intent.target_countries) or 'N/A'}\n"
            f"Sectors: {', '.join(intent.target_sectors) or 'N/A'}\n\n"
            f"## PROVIDED VAULT NODES\n"
            f"YOU MAY ONLY USE INFORMATION FROM THESE NODES.\n"
            f"DO NOT USE EXTERNAL KNOWLEDGE.\n\n"
            f"{context}\n\n"
            f"## CRITICAL RULES\n"
            f"1. Executive summary: 6-10 sentences. Name specific entities. Explain their roles and relationships.\n"
            f"2. Every structured_intelligence row MUST have 'source_node' = exact node ID from above.\n"
            f"3. Every finding/opportunity/risk MUST have 'text' and 'source_nodes' array with node IDs.\n"
            f"4. Every opportunity MUST have perspective_actor from PERSPECTIVE ACTOR REGISTRY, perspective_capability, and evidenced pathway.\n"
            f"5. Set opportunity_country to the actual country where the commercial value exists. Do NOT default to perspective country.\n"
            f"6. If information is missing, write 'Not found in vault' — do NOT invent facts.\n"
            f"7. Output ONLY raw JSON."
        )

        if self.token_budget.estimate(GROUNDED_SYNTHESIS_PROMPT + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(GROUNDED_SYNTHESIS_PROMPT)
            user_prompt = user_prompt[:max_chars] + "\n[TRUNCATED]"
            logger.warning("Synthesis prompt truncated.")

        raw_response = self._call_api(
            system_prompt=GROUNDED_SYNTHESIS_PROMPT,
            user_prompt=user_prompt,
        )

        return self._parse_grounded_response(raw_response, ranked_nodes, perspective_nodes, cross_border_bridges, perspective, intent)

    def _parse_grounded_response(self, raw: str, source_nodes: List[VaultNode],
                                 perspective_nodes: List[VaultNode],
                                 cross_border_bridges: List[Dict[str, Any]],
                                 perspective: PerspectiveContext, intent: ATISIntent) -> Dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}

        # Ensure all sections are lists
        for section in ["structured_intelligence", "findings", "opportunities", "risks", "key_entities"]:
            if section not in data or not isinstance(data[section], list):
                data[section] = []

        source_node_ids = {n.uid for n in source_nodes}
        perspective_node_ids = {n.uid for n in perspective_nodes}
        all_node_ids = source_node_ids | perspective_node_ids

        # Validate structured_intelligence citations
        validated_intel = []
        for row in data.get("structured_intelligence", []):
            if row.get("source_node") in all_node_ids:
                validated_intel.append(row)
            else:
                row["source_node"] = "citation_missing"
                validated_intel.append(row)

        # Validate findings/opportunities/risks
        for section in ["findings", "opportunities", "risks"]:
            items = data.get(section, [])
            validated = []
            for item in items:
                if isinstance(item, dict):
                    cited = [n for n in item.get("source_nodes", []) if n in all_node_ids]
                    if not cited:
                        item["source_nodes"] = ["citation_needed"]
                    validated.append(item)
                else:
                    validated.append({"text": str(item), "source_nodes": ["citation_needed"]})
            data[section] = validated

        # Validate opportunities with perspective-side evidence
        validated_opportunities = []
        for item in data["opportunities"]:
            if not item.get("source_country") and intent.target_countries:
                item["source_country"] = intent.target_countries[0]
            validated = validate_opportunity(
                item, perspective, all_node_ids, perspective_node_ids, cross_border_bridges
            )
            # Assign stable opportunity ID via compute_opportunity_identity
            validated["opportunity_id"] = compute_opportunity_identity(validated)
            validated_opportunities.append(validated)
        data["opportunities"] = validated_opportunities

        return {
            "executive_summary": data.get("executive_summary", "No summary generated."),
            "structured_intelligence": validated_intel,
            "findings": data.get("findings", []),
            "opportunities": data.get("opportunities", []),
            "risks": data.get("risks", []),
            "key_entities": data.get("key_entities", []),
            "source_nodes": [{"id": n.uid, "type": n.entity_type} for n in source_nodes],
            "perspective_nodes": [{"id": n.uid, "type": n.entity_type} for n in perspective_nodes],
            "cross_border_bridges": cross_border_bridges,
        }

    def _generate_empty_response(self, question: str, intent: ATISIntent,
                                  knowledge_state_hash: str = "") -> Dict[str, Any]:
        return {
            "executive_summary": (
                f"No vault nodes matched the search for: '{question}'. "
                f"Tried to find {', '.join(intent.target_entity_types) or 'any entity type'} "
                f"matching {', '.join(intent.target_entities) or 'no specific entities'}."
            ),
            "structured_intelligence": [],
            "findings": [{"text": "No matching entities found.", "source_nodes": []}],
            "opportunities": [],
            "risks": [{"text": "Incomplete data coverage.", "source_nodes": []}],
            "key_entities": [],
            "source_nodes": [],
            "perspective_nodes": [],
            "cross_border_bridges": [],
            "analysis_version": ANALYSIS_VERSION,
            "schema_version": SCHEMA_VERSION,
            "analysis_fingerprint": "",
            "knowledge_state": {"hash": knowledge_state_hash},
            "cache_hit": False,
        }

    # -----------------------------------------------------------------
    # FULL VAULT SCAN
    # -----------------------------------------------------------------
    def full_vault_scan(self, vault_mgr: ObsidianVaultManager, perspective: PerspectiveContext) -> Dict[str, Any]:
        logger.info("MODE A: FULL VAULT SCAN")
        all_nodes = vault_mgr.get_all_nodes_as_context()
        if len(all_nodes) > FULL_SCAN_MAX_NODES:
            scored = [(len(n.outbound_links) + len(n.backlink_uids), n) for n in all_nodes]
            scored.sort(key=lambda x: (-x[0], x[1].uid))
            all_nodes = [n for _, n in scored[:FULL_SCAN_MAX_NODES]]

        # Retrieve perspective nodes for full scan too
        perspective_nodes = vault_mgr.get_perspective_nodes(perspective)
        cross_border_bridges = []  # No specific source country in full scan

        summaries = [f"{n.uid} ({n.entity_type}, {n.country or 'N/A'}): {n.summary[:150]}" for n in all_nodes]
        context = "\n".join(summaries)

        perspective_blocks = []
        perspective_blocks.append(f"=== PERSPECTIVE ACTOR REGISTRY ({perspective.country}) ===")
        for pn in perspective_nodes[:20]:
            perspective_blocks.append(f"- {pn.uid} | type: {pn.entity_type} | summary: {pn.summary[:100]}")

        system_prompt = (
            "You are the ATIS Master Intelligence Synthesizer. Given vault summaries, "
            "produce a high-level dashboard. Output ONLY raw JSON. Every claim must cite source nodes. "
            "You MUST select perspective_actor from the PERSPECTIVE ACTOR REGISTRY. "
            "You MUST set opportunity_country to the actual country where the commercial value exists."
        )
        perspective_registry_text = "\n\n".join(perspective_blocks)
        user_prompt = (
            f"## ANALYTICAL PERSPECTIVE\nPerspective country: {perspective.country}\nPerspective country code: {perspective.country_code}\n"
            "Produce perspective-specific intelligence from the full African vault; do not turn this into a country filter.\n\n"
            f"## PERSPECTIVE ACTOR REGISTRY\n"
            f"{perspective_registry_text}\n\n"
            f"## VAULT SUMMARY\nTotal: {len(all_nodes)}\n\n{context}\n\n"
            f"## OUTPUT\nProduce dashboard JSON with executive_summary, structured_intelligence, findings, opportunities, risks, key_entities. "
            f"Every row must have source_node. Every finding must have source_nodes. "
            f"Every opportunity must have perspective_actor from the registry, perspective_capability, and evidenced pathway."
        )

        if self.token_budget.estimate(system_prompt + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(system_prompt)
            user_prompt = user_prompt[:max_chars] + "\n[TRUNCATED]"

        raw = self._call_api(system_prompt=system_prompt, user_prompt=user_prompt)
        result = self._parse_grounded_response(raw, all_nodes, perspective_nodes, cross_border_bridges, perspective, ATISIntent({}))
        result["scan_mode"] = "full_vault_capped"
        return result

    # -----------------------------------------------------------------
    # MAIN ORCHESTRATOR
    # -----------------------------------------------------------------
    def generate_query_payload(self, vault_mgr: ObsidianVaultManager,
                               question: str | None = None,
                               perspective: PerspectiveContext | None = None) -> Dict[str, Any]:
        perspective = perspective or PerspectiveContext()

        # Compute vault knowledge state for versioning
        knowledge_state = KnowledgeState(vault_mgr.vault_root)
        knowledge_state_hash = getattr(knowledge_state, "hash", getattr(knowledge_state, "state_hash", ""))

        # Build deterministic cache key
        cache_payload = f"{question or 'FULL_SCAN'}|{perspective.country}|{perspective.country_code}|{knowledge_state_hash}"
        cache_key = hashlib.sha256(cache_payload.encode()).hexdigest()

        # Check cache before any LLM work
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info("Cache hit for query key %s", cache_key[:16])
            cached_result["analysis_version"] = ANALYSIS_VERSION
            cached_result["schema_version"] = SCHEMA_VERSION
            cached_result["knowledge_state"] = {"hash": knowledge_state_hash}
            cached_result["cache_hit"] = True
            return cached_result

        if not question:
            result = self.full_vault_scan(vault_mgr, perspective)
            result["knowledge_state_hash"] = knowledge_state_hash
            result["cache_key"] = cache_key
            return result

        intent = self.extract_intent(question)
        candidates = self.broad_pre_filter(vault_mgr, intent, perspective)
        perspective_nodes, cross_border_bridges = self.retrieve_perspective_context(vault_mgr, perspective, intent)
        ranked_nodes = self.llm_semantic_ranking(question, intent, perspective, candidates, perspective_nodes, cross_border_bridges)
        result = self.grounded_synthesis(question, intent, perspective, ranked_nodes, perspective_nodes, cross_border_bridges)

        # Compute analysis fingerprint after evidence is gathered
        story_id = hashlib.sha256((question or "").encode()).hexdigest()[:16]
        evidence_ids = sorted([n.uid for n in ranked_nodes])
        entity_ids = sorted(list(set([n.uid for n in ranked_nodes + perspective_nodes])))
        relationship_ids = sorted([f"{b['from_node']}->{b['to_node']}" for b in cross_border_bridges])
        analysis_fingerprint = compute_analysis_fingerprint(
            story_id=story_id,
            perspective=perspective,
            evidence_ids=evidence_ids,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            knowledge_state_hash=knowledge_state_hash,
        )

        entity_graph = vault_mgr.build_entity_graph(ranked_nodes)
        aggregate_stats = vault_mgr.get_aggregate_stats(ranked_nodes)

        payload = {
            **result,
            "intent": {
                "type": intent.intent_type,
                "entities": intent.target_entities,
                "entity_types": intent.target_entity_types,
                "countries": intent.target_countries,
                "sectors": intent.target_sectors,
                "perspective_country": perspective.country,
                "perspective_country_code": perspective.country_code,
            },
            "filter_stats": {
                "vault_total": vault_mgr.indexed_count,
                "candidates_after_broad_filter": len(candidates),
                "perspective_nodes_retrieved": len(perspective_nodes),
                "cross_border_bridges_found": len(cross_border_bridges),
                "ranked_by_llm": len(ranked_nodes),
            },
            "entity_graph": entity_graph,
            "stats": aggregate_stats,
            "perspective": perspective.as_dict(),
            "analysis_version": ANALYSIS_VERSION,
            "schema_version": SCHEMA_VERSION,
            "analysis_fingerprint": analysis_fingerprint,
            "knowledge_state": {"hash": knowledge_state_hash},
            "cache_hit": False,
        }

        # Persist to cache after generation
        self.cache.set(cache_key, payload)
        return payload


# =============================================================================
# PERSISTENCE
# =============================================================================
def persist_query_payload(payload: Dict[str, Any], entity_graph: Dict[str, Any],
                          aggregate_stats: Dict[str, Any], question: str | None = None,
                          model_primary: str = "configured", model_fallback: str = "configured") -> Tuple[Path, Path]:
    output_dir = Path(os.getenv("OUTPUT_DIR", "./output/query_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    query_id = f"ATIS-QUERY-{timestamp}"

    dashboard_payload = {
        "query_id": query_id,
        "user_question": question or "FULL_VAULT_SCAN",
        "perspective": payload.get("perspective", {}),
        "executive_summary": payload.get("executive_summary", ""),
        "intent": payload.get("intent", {}),
        "filter_stats": payload.get("filter_stats", {}),
        "stats": {
            "total_entities": aggregate_stats.get("total_entities", 0),
            "total_relationships": aggregate_stats.get("total_relationships", 0),
            "commodities_tracked": aggregate_stats.get("commodities_tracked", 0),
            "countries_covered": aggregate_stats.get("countries_covered", 0),
            "entity_type_breakdown": aggregate_stats.get("entity_type_breakdown", {}),
        },
        "entity_graph": entity_graph,
        "structured_intelligence": payload.get("structured_intelligence", []),
        "findings": payload.get("findings", []),
        "opportunities": payload.get("opportunities", []),
        "risks": payload.get("risks", []),
        "key_entities": payload.get("key_entities", []),
        "source_nodes": payload.get("source_nodes", []),
        "perspective_nodes": payload.get("perspective_nodes", []),
        "cross_border_bridges": payload.get("cross_border_bridges", []),
        "analysis_version": payload.get("analysis_version", ANALYSIS_VERSION),
        "schema_version": payload.get("schema_version", SCHEMA_VERSION),
        "analysis_fingerprint": payload.get("analysis_fingerprint", ""),
        "knowledge_state": payload.get("knowledge_state", {}),
        "cache_hit": payload.get("cache_hit", False),
        "pipeline_metadata": {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "vault_files_scanned": aggregate_stats.get("total_entities", 0),
            "model_primary": model_primary,
            "model_fallback": model_fallback,
            "pipeline_version": "perspective_first_v4_1",
        },
    }

    json_path = output_dir / f"query_dashboard_{timestamp}.json"
    json_path.write_text(json.dumps(dashboard_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Dashboard persisted: %s", json_path)

    graph_path = output_dir / f"query_graph_{timestamp}.json"
    graph_path.write_text(json.dumps(entity_graph, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Graph persisted: %s", graph_path)

    return json_path, graph_path


# =============================================================================
# WEB ENTRY POINT — BACKWARD COMPATIBLE
# =============================================================================
def run_query_pipeline(question: str | None = None,
                       vault_path: str | Path = "./vault",
                       perspective: PerspectiveContext | None = None) -> Dict[str, Any]:
    """
    Web-compatible entry point.
    Returns FLAT backward-compatible shape for v0 frontend:
      - findings/opportunities/risks are STRING arrays
      - Cited versions in *_cited fields
      - No nested .dashboard wrapper
    """
    vault_mgr = ObsidianVaultManager(Path(vault_path))
    vault_mgr.build_index()

    if vault_mgr.indexed_count == 0:
        raise RuntimeError("No markdown files found in vault.")

    engine = LLMQueryEngine()
    perspective = perspective or PerspectiveContext()
    result = engine.generate_query_payload(vault_mgr, question, perspective)

    entity_graph = result.pop("entity_graph", {})
    aggregate_stats = result.pop("stats", {})

    # Extract raw cited versions
    raw_findings = result.get("findings", [])
    raw_opportunities = result.get("opportunities", [])
    raw_risks = result.get("risks", [])

    # Flatten to strings for v0 compatibility
    findings_strings = [
        f.get("text", str(f)) if isinstance(f, dict) else str(f)
        for f in raw_findings
    ]
    opportunities_strings = [
        o.get("text", str(o)) if isinstance(o, dict) else str(o)
        for o in raw_opportunities
    ]
    risks_strings = [
        r.get("text", str(r)) if isinstance(r, dict) else str(r)
        for r in raw_risks
    ]

    # Persist full cited version to disk
    full_result = {
        **result,
        "findings": raw_findings,
        "opportunities": raw_opportunities,
        "risks": raw_risks,
    }
    json_path, graph_path = persist_query_payload(
        full_result,
        entity_graph,
        aggregate_stats,
        question,
        engine.client.config.model,
        engine.client.config.fallback_model,
    )

    # Return FLAT backward-compatible shape
    return {
        "executive_summary": result.get("executive_summary", ""),
        "structured_intelligence": result.get("structured_intelligence", []),
        "findings": findings_strings,
        "opportunities": opportunities_strings,
        "risks": risks_strings,
        "key_entities": result.get("key_entities", []),
        # Cited versions for future frontend upgrades
        "findings_cited": raw_findings,
        "opportunities_cited": raw_opportunities,
        "risks_cited": raw_risks,
        "source_nodes": result.get("source_nodes", []),
        "perspective_nodes": result.get("perspective_nodes", []),
        "cross_border_bridges": result.get("cross_border_bridges", []),
        "intent": result.get("intent", {}),
        "perspective": perspective.as_dict(),
        "filter_stats": result.get("filter_stats", {}),
        "entity_graph": entity_graph,
        "stats": aggregate_stats,
        "analysis_version": result.get("analysis_version", ANALYSIS_VERSION),
        "schema_version": result.get("schema_version", SCHEMA_VERSION),
        "analysis_fingerprint": result.get("analysis_fingerprint", ""),
        "knowledge_state": result.get("knowledge_state", {}),
        "cache_hit": result.get("cache_hit", False),
        "files_written": {
            "dashboard_json": str(json_path),
            "graph_json": str(graph_path),
        }
    }


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATIS Query v4.0 — Perspective-First Grounded Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vault_path", default="./vault", help="Path to Obsidian vault root.")
    parser.add_argument("--question", default="", help="Natural language question.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    vault_path = Path(args.vault_path).resolve()
    question = args.question.strip() if args.question else None

    if question:
        logger.info("=" * 70)
        logger.info("MODE B: QUESTION-DRIVEN QUERY (Perspective-First v4.0)")
        logger.info("Question: %s", question)
        logger.info("=" * 70)
    else:
        logger.info("=" * 70)
        logger.info("MODE A: FULL VAULT SCAN")
        logger.info("=" * 70)

    try:
        result = run_query_pipeline(question, vault_path)
        print(f"\nSUCCESS: {result['files_written']['dashboard_json']}")
        print(f"SUCCESS: {result['files_written']['graph_json']}")
        print(f"\nEXECUTIVE SUMMARY:\n{result.get('executive_summary', 'N/A')}")
        print(f"\nFILTER STATS: {result.get('filter_stats', {})}")
        print(f"\nFINDINGS (strings): {result.get('findings', [])[:3]}")
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
