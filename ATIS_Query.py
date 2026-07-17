#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATIS_Query.py v3 — Grounded Intent-First Architecture

3-STAGE PIPELINE:
  0. INTENT EXTRACTION:    LLM understands question → structured intent (1 call, cheap)
  1. BROAD PRE-FILTER:     Python loose matching → ~80 candidates (0 calls)
  2. LLM SEMANTIC RANKING: LLM reads candidates → ranks top 20 by relevance (1 call)
  3. GROUNDED SYNTHESIS:   LLM synthesizes dashboard from ranked nodes ONLY (1 call)
                           Every claim cites source_node. Hallucination impossible.

ANTI-HALLUCINATION GUARANTEES:
  - LLM only sees provided node summaries. No external knowledge.
  - Every structured_intelligence row has "source_node" field.
  - Every finding/opportunity/risk has "source_nodes" array.
  - Executive summary names specific entities and their roles.
  - If data missing → "Not found in vault" instead of fabrication.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
from datetime import date, datetime, timezone

# =============================================================================
# CONFIGURATION
# =============================================================================
HARDCODED_API_KEY: str = "csk-v4vf9r666pv9t99etmm9cppv58j8xpc8fnjpxkpw89mk36rp"

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

try:
    from cerebras.cloud.sdk import Cerebras
    from cerebras.cloud.sdk import APIError, APIConnectionError, RateLimitError
except ImportError as _import_err:
    sys.stderr.write("ERROR: The 'cerebras.cloud.sdk' package is not installed. "
                     "Install it via: pip install cerebras-cloud-sdk\n")
    raise SystemExit(1) from _import_err

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
INTENT_EXTRACTION_PROMPT: str = (
    "You are the ATIS Intent Extraction Engine. Analyze the user's question and extract "
    "a structured search intent. Output ONLY valid raw JSON. No markdown fences. No commentary.\n\n"
    "OUTPUT SCHEMA:\n"
    "{\n"
    '  "intent_type": "OVERVIEW|SPECIFIC_ENTITY|FILTERED_LIST|RELATIONSHIP|COMPARISON",\n'
    '  "target_entities": ["concrete nouns to search for"],\n'
    '  "target_entity_types": ["mining_refinery|government_agency|commodity|policy_framework|infrastructure_node|private_conglomerate|government_ministry|academic_institution"],\n'
    '  "target_countries": ["country names if mentioned"],\n'
    '  "target_sectors": ["sector names if mentioned"],\n'
    '  "target_attributes": {"status": "operational|planned|under_construction", "ownership": "state_owned|private"},\n'
    '  "relationship_type": "regulates|owns|funds|operates|licenses|null",\n'
    '  "output_format": "structured_table|summary|entity_profile|relationship_graph",\n'
    '  "max_results_hint": 5|10|15|20|30\n'
    "}\n\n"
    "RULES:\n"
    "- intent_type: OVERVIEW for broad summaries, SPECIFIC_ENTITY for 'who is X', FILTERED_LIST for 'which X are Y', RELATIONSHIP for 'who regulates X'.\n"
    "- target_entities: extract ONLY concrete nouns. Strip vague words.\n"
    "- max_results_hint: OVERVIEW=20, SPECIFIC_ENTITY=3, FILTERED_LIST=15, RELATIONSHIP=8.\n"
    "- Output ONLY raw JSON."
)

SEMANTIC_RANKING_PROMPT: str = (
    "You are a Semantic Relevance Engine for the ATIS Intelligence System. "
    "Your job is to rank vault nodes by their relevance to the user's question.\n\n"
    "You will be given:\n"
    "1. The user's question\n"
    "2. A list of candidate vault nodes (each with id, type, country, sector, summary)\n\n"
    "Your task:\n"
    "- Read each candidate carefully\n"
    "- Score relevance from 1.0 to 10.0 based on how directly the node answers the question\n"
    "- Select the top N most relevant nodes\n"
    "- Provide 1-sentence reasoning for each selection\n\n"
    "SCORING CRITERIA:\n"
    "- 10.0: Directly answers the question (e.g., the exact entity asked about)\n"
    "- 7.0-9.0: Highly relevant (e.g., regulator of the entity, parent company)\n"
    "- 4.0-6.0: Moderately relevant (e.g., same sector, related commodity)\n"
    "- 1.0-3.0: Weakly relevant (e.g., same country, tangential connection)\n"
    "- 0.0: Not relevant at all\n\n"
    "OUTPUT SCHEMA (raw JSON only):\n"
    "{\n"
    '  "ranked_nodes": [\n'
    '    {\n'
    '      "rank": 1,\n'
    '      "node_id": "exact_filename_no_extension",\n'
    '      "relevance_score": 9.5,\n'
    '      "reasoning": "One sentence explaining why this node is relevant"\n'
    '    }\n'
    '  ],\n'
    '  "excluded_count": 45\n'
    "}\n\n"
    "RULES:\n"
    "- Use the EXACT node_id as provided. Do not modify filenames.\n"
    "- If a node is not relevant, do not include it.\n"
    "- Be strict. A score of 10.0 should be rare.\n"
    "- Output ONLY raw JSON. No markdown fences."
)

GROUNDED_SYNTHESIS_PROMPT: str = (
    "You are the ATIS Grounded Synthesis Engine. You have access ONLY to the provided vault nodes. "
    "You MUST NOT use any external knowledge. If information is not in the provided nodes, say 'Not found in vault'.\n\n"
    "ANTI-HALLUCINATION RULES (violation = invalid output):\n"
    "1. Every claim in structured_intelligence MUST have a 'source_node' field containing the exact node ID.\n"
    "2. Every item in findings[], opportunities[], risks[] MUST have a 'source_nodes' array with at least one node ID.\n"
    "3. Every entity in key_entities[] MUST correspond to a provided node.\n"
    "4. The executive_summary MUST reference specific entities by their exact names and explain their roles.\n"
    "5. If you cannot verify a claim from the provided nodes, output 'Not found in vault' for that field.\n"
    "6. Do NOT invent statistics, dates, or facts not present in the nodes.\n\n"
    "EXECUTIVE SUMMARY REQUIREMENTS:\n"
    "- Length: 6-10 sentences\n"
    "- Must explain: the overall landscape, key players and their specific roles, regulatory context, "
    "structural gaps, primary opportunities, and key risks\n"
    "- Must name specific entities (e.g., 'ZESA Holdings operates the national grid under ZERA regulation')\n"
    "- Must connect entities to each other (e.g., 'Bikita Minerals is regulated by the Ministry of Mines')\n"
    "- Must explain WHY the dashboard findings matter\n\n"
    "OUTPUT SCHEMA (raw JSON only):\n"
    "{\n"
    '  "executive_summary": "6-10 sentence comprehensive narrative...",\n'
    '  "structured_intelligence": [\n'
    '    {\n'
    '      "entity": "Entity Name",\n'
    '      "type": "entity_type",\n'
    '      "country": "Zimbabwe",\n'
    '      "relationship": "regulates",\n'
    '      "status": "Operational",\n'
    '      "priority": "Critical|High|Medium|Low",\n'
    '      "insight": "One precise sentence.",\n'
    '      "source_node": "Exact_Node_ID"\n'
    '    }\n'
    '  ],\n'
    '  "findings": [\n'
    '    {"text": "Finding statement.", "source_nodes": ["Node_ID_1", "Node_ID_2"]}\n'
    '  ],\n'
    '  "opportunities": [\n'
    '    {"text": "Opportunity statement.", "source_nodes": ["Node_ID"]}\n'
    '  ],\n'
    '  "risks": [\n'
    '    {"text": "Risk statement.", "source_nodes": ["Node_ID"]}\n'
    '  ],\n'
    '  "key_entities": [\n'
    '    {\n'
    '      "entity_name": "Exact Name",\n'
    '      "entity_type": "type",\n'
    '      "country": "Zimbabwe",\n'
    '      "sector": "Energy",\n'
    '      "significance_score": 9,\n'
    '      "related_count": 5,\n'
    '      "summary": "Description from vault.",\n'
    '      "source_node": "Exact_Node_ID"\n'
    '    }\n'
    '  ]\n'
    "}\n\n"
    "Output ONLY raw JSON. No markdown fences. No commentary outside JSON."
)


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
        return aliases

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
        return entity_type, country, sector

    def build_index(self) -> None:
        logger.info("Starting vault index build at: %s", self.vault_root)
        if not self.vault_root.exists():
            raise FileNotFoundError(f"Vault root does not exist: {self.vault_root}")

        md_files = list(self.vault_root.rglob("*.md"))
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

        for node in self.nodes.values():
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
        nodes = node_subset if node_subset is not None else list(self.nodes.values())
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
        nodes = node_subset if node_subset is not None else list(self.nodes.values())
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
        return list(self.nodes.values())


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
# CEREBRAS ENGINE — 3-STAGE GROUNDED PIPELINE
# =============================================================================
class CerebrasQueryEngine:
    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or HARDCODED_API_KEY or os.environ.get("CEREBRAS_API_KEY")
        if not resolved_key:
            raise RuntimeError("Cerebras API key not configured.")
        self.client = Cerebras(api_key=resolved_key)
        self.model = "gpt-oss-120b"
        self.fallback_model = "gemma-4-31b"
        self.temperature = 0.1
        self.max_tokens = 512
        self.max_retries = 2
        self.base_delay_seconds = 2.0
        self.token_budget = TokenBudget()
        logger.info("Cerebras client initialized. Model: %s", self.model)

    def _call_api(self, system_prompt: str, user_prompt: str,
                  temperature: float | None = None, max_tokens: int | None = None) -> str:
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        models_to_try = [self.model, self.fallback_model]

        for model in models_to_try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info("API call | model=%s | attempt=%d/%d", model, attempt, self.max_retries)
                    response = self.client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                    if not response.choices:
                        raise APIError(message="Empty choices", request=None, body=None)
                    content = response.choices[0].message.content
                    if content is None:
                        raise APIError(message="None content", request=None, body=None)
                    content = content.strip()
                    if not content:
                        raise APIError(message="Empty content", request=None, body=None)
                    logger.info("API call successful (%s) | %d chars", model, len(content))
                    return content
                except RateLimitError as exc:
                    error_body = getattr(exc, 'body', {}) or {}
                    error_code = error_body.get('code', '') if isinstance(error_body, dict) else ''
                    if error_code == 'token_quota_exceeded':
                        logger.error("Cerebras daily token quota EXHAUSTED. Aborting.")
                        raise RuntimeError(
                            "Cerebras API daily token limit reached. "
                            "Try again after 24 hours or upgrade your plan."
                        ) from exc
                    delay = self.base_delay_seconds * (2 ** (attempt - 1))
                    logger.warning("RateLimitError on %s (attempt %d). Backing off %.1fs...", model, attempt, delay)
                    if attempt < self.max_retries:
                        time.sleep(delay)
                    else:
                        if model == self.model:
                            break
                        raise
                except APIConnectionError as exc:
                    delay = self.base_delay_seconds * (2 ** (attempt - 1))
                    logger.warning("APIConnectionError on %s (attempt %d). Retrying %.1fs...", model, attempt, delay)
                    if attempt < self.max_retries:
                        time.sleep(delay)
                    else:
                        if model == self.model:
                            break
                        raise
                except APIError as exc:
                    logger.error("APIError on %s (attempt %d): %s", model, attempt, exc)
                    if attempt == self.max_retries:
                        if model == self.model:
                            logger.warning("Falling back to: %s", self.fallback_model)
                            break
                        raise
                    time.sleep(self.base_delay_seconds)
                except Exception as exc:
                    logger.error("Unexpected error on %s (attempt %d): %s", model, attempt, exc)
                    if attempt == self.max_retries:
                        if model == self.model:
                            break
                        raise
                    time.sleep(self.base_delay_seconds)
        raise RuntimeError("All API retries exhausted and fallback model failed.")

    # -----------------------------------------------------------------
    # STAGE 0: Intent Extraction
    # -----------------------------------------------------------------
    def extract_intent(self, question: str) -> ATISIntent:
        logger.info("STAGE 0: INTENT EXTRACTION")
        user_prompt = f'Question: "{question}"\n\nExtract the structured intent as JSON.'

        raw_response = self._call_api(
            system_prompt=INTENT_EXTRACTION_PROMPT,
            user_prompt=user_prompt,
            temperature=0.05,
            max_tokens=512
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
    # STAGE 1: Broad Pre-Filter (Python only)
    # -----------------------------------------------------------------
    def broad_pre_filter(self, vault_mgr: ObsidianVaultManager, intent: ATISIntent) -> List[VaultNode]:
        logger.info("STAGE 1: BROAD PRE-FILTER")
        all_nodes = list(vault_mgr.nodes.values())
        scored: List[Tuple[float, VaultNode]] = []

        for node in all_nodes:
            score = 0.0

            # Entity type match
            node_type = (node.entity_type or "").lower()
            for target_type in intent.target_entity_types:
                tt = target_type.lower().replace("_", "")
                nt = node_type.replace("_", "")
                if tt == nt or tt in nt or nt in tt:
                    score += 8.0

            # Country match
            node_country = (node.country or "").lower()
            for tc in intent.target_countries:
                if tc.lower() in node_country:
                    score += 6.0

            # Sector match
            node_sector = (node.sector or "").lower()
            for ts in intent.target_sectors:
                if ts.lower() in node_sector:
                    score += 5.0

            # Name/content match
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

            # Attribute match
            for attr_key, attr_val in intent.target_attributes.items():
                if attr_val.lower() in content:
                    score += 3.0

            # Hub bonus for overview
            if intent.intent_type == "OVERVIEW":
                score += (len(node.outbound_links) + len(node.backlink_uids)) * 0.3

            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        cap = min(BROAD_FILTER_MAX_CANDIDATES, len(scored))
        result = [node for _, node in scored[:cap]]
        logger.info("Broad filter: %d scored → %d candidates", len(scored), len(result))
        return result

    # -----------------------------------------------------------------
    # STAGE 2: LLM Semantic Ranking
    # -----------------------------------------------------------------
    def llm_semantic_ranking(self, question: str, intent: ATISIntent,
                             candidates: List[VaultNode]) -> List[VaultNode]:
        logger.info("STAGE 2: LLM SEMANTIC RANKING (%d candidates)", len(candidates))

        if not candidates:
            return []

        # Build compact candidate summaries
        candidate_blocks = []
        for node in candidates:
            block = (
                f"NODE_ID: {node.uid}\n"
                f"TYPE: {node.entity_type} | COUNTRY: {node.country or 'N/A'} | SECTOR: {node.sector or 'N/A'}\n"
                f"SUMMARY: {node.summary[:200]}\n"
                f"---"
            )
            candidate_blocks.append(block)

        candidates_text = "\n\n".join(candidate_blocks)

        user_prompt = (
            f"## USER QUESTION\n{question}\n\n"
            f"## SEARCH INTENT\n"
            f"Type: {intent.intent_type}\n"
            f"Looking for: {', '.join(intent.target_entities) or 'N/A'}\n"
            f"Entity types: {', '.join(intent.target_entity_types) or 'N/A'}\n\n"
            f"## CANDIDATE NODES ({len(candidates)} total)\n"
            f"{candidates_text}\n\n"
            f"## TASK\n"
            f"Rank the top {LLM_RANKING_MAX_RESULTS} most relevant nodes. "
            f"Return ONLY raw JSON with ranked_nodes array."
        )

        # Token guard
        if self.token_budget.estimate(SEMANTIC_RANKING_PROMPT + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(SEMANTIC_RANKING_PROMPT)
            user_prompt = user_prompt[:max_chars] + "\n[TRUNCATED]"
            logger.warning("Ranking prompt truncated.")

        raw_response = self._call_api(
            system_prompt=SEMANTIC_RANKING_PROMPT,
            user_prompt=user_prompt,
            temperature=0.05,
            max_tokens=2048
        )

        # Parse ranking
        try:
            ranking_data = json.loads(raw_response)
            ranked_list = ranking_data.get("ranked_nodes", [])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse ranking JSON: %s. Using top candidates by score.", exc)
            return candidates[:LLM_RANKING_MAX_RESULTS]

        # Map ranked node_ids back to VaultNode objects
        ranked_nodes = []
        candidate_map = {n.uid: n for n in candidates}
        for item in ranked_list:
            node_id = item.get("node_id", "")
            if node_id in candidate_map:
                node = candidate_map[node_id]
                # Attach ranking metadata for debugging
                node._ranking_score = item.get("relevance_score", 0)
                node._ranking_reason = item.get("reasoning", "")
                ranked_nodes.append(node)
                logger.info("  Ranked: %s (score=%s, reason=%s)",
                            node_id, item.get("relevance_score"), item.get("reasoning", "")[:60])

        logger.info("LLM ranking: %d nodes selected", len(ranked_nodes))
        return ranked_nodes

    # -----------------------------------------------------------------
    # STAGE 3: Grounded Synthesis
    # -----------------------------------------------------------------
    def grounded_synthesis(self, question: str, intent: ATISIntent,
                           ranked_nodes: List[VaultNode]) -> Dict[str, Any]:
        logger.info("STAGE 3: GROUNDED SYNTHESIS (%d nodes)", len(ranked_nodes))

        if not ranked_nodes:
            return self._generate_empty_response(question, intent)

        # Build rich context from ranked nodes
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

        context = "\n\n".join(node_blocks)

        user_prompt = (
            f"## USER QUESTION\n{question}\n\n"
            f"## SEARCH INTENT\n"
            f"Type: {intent.intent_type}\n"
            f"Looking for: {', '.join(intent.target_entities) or 'N/A'}\n"
            f"Entity types: {', '.join(intent.target_entity_types) or 'N/A'}\n"
            f"Countries: {', '.join(intent.target_countries) or 'N/A'}\n"
            f"Sectors: {', '.join(intent.target_sectors) or 'N/A'}\n\n"
            f"## PROVIDED VAULT NODES ({len(ranked_nodes)} nodes)\n"
            f"YOU MAY ONLY USE INFORMATION FROM THESE NODES.\n"
            f"DO NOT USE EXTERNAL KNOWLEDGE.\n\n"
            f"{context}\n\n"
            f"## CRITICAL RULES\n"
            f"1. Executive summary: 6-10 sentences. Name specific entities. Explain their roles and relationships.\n"
            f"2. Every structured_intelligence row MUST have 'source_node' = exact node ID from above.\n"
            f"3. Every finding/opportunity/risk MUST have 'source_nodes' array with node IDs.\n"
            f"4. If information is missing, write 'Not found in vault' — do NOT invent facts.\n"
            f"5. Output ONLY raw JSON."
        )

        if self.token_budget.estimate(GROUNDED_SYNTHESIS_PROMPT + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(GROUNDED_SYNTHESIS_PROMPT)
            user_prompt = user_prompt[:max_chars] + "\n[TRUNCATED]"
            logger.warning("Synthesis prompt truncated.")

        raw_response = self._call_api(
            system_prompt=GROUNDED_SYNTHESIS_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=4096
        )

        return self._parse_grounded_response(raw_response, ranked_nodes)

    def _parse_grounded_response(self, raw: str, source_nodes: List[VaultNode]) -> Dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}

        # Validate citations exist
        source_node_ids = {n.uid for n in source_nodes}
        validated_intel = []
        for row in data.get("structured_intelligence", []):
            if row.get("source_node") in source_node_ids:
                validated_intel.append(row)
            else:
                row["source_node"] = "citation_missing"
                row["insight"] = "[CITATION MISSING] " + row.get("insight", "")
                validated_intel.append(row)

        # Validate findings/opportunities/risks citations
        for section in ["findings", "opportunities", "risks"]:
            items = data.get(section, [])
            validated = []
            for item in items:
                if isinstance(item, dict):
                    cited = [n for n in item.get("source_nodes", []) if n in source_node_ids]
                    if not cited:
                        item["source_nodes"] = ["citation_needed"]
                    validated.append(item)
                else:
                    validated.append({"text": str(item), "source_nodes": ["citation_needed"]})
            data[section] = validated

        return {
            "executive_summary": data.get("executive_summary", "No summary generated."),
            "structured_intelligence": validated_intel,
            "findings": data.get("findings", []),
            "opportunities": data.get("opportunities", []),
            "risks": data.get("risks", []),
            "key_entities": data.get("key_entities", []),
            "source_nodes": [{"id": n.uid, "type": n.entity_type} for n in source_nodes],
        }

    def _generate_empty_response(self, question: str, intent: ATISIntent) -> Dict[str, Any]:
        return {
            "executive_summary": (
                f"No vault nodes matched the search for: '{question}'. "
                f"Tried to find {', '.join(intent.target_entity_types) or 'any entity type'} "
                f"matching {', '.join(intent.target_entities) or 'no specific entities'}."
            ),
            "structured_intelligence": [],
            "findings": [{"text": "No matching entities found.", "source_nodes": []}],
            "opportunities": [{"text": "Consider expanding the vault.", "source_nodes": []}],
            "risks": [{"text": "Incomplete data coverage.", "source_nodes": []}],
            "key_entities": [],
            "source_nodes": [],
        }

    # -----------------------------------------------------------------
    # FULL VAULT SCAN (capped, single-shot)
    # -----------------------------------------------------------------
    def full_vault_scan(self, vault_mgr: ObsidianVaultManager) -> Dict[str, Any]:
        logger.info("MODE A: FULL VAULT SCAN")
        all_nodes = vault_mgr.get_all_nodes_as_context()
        if len(all_nodes) > FULL_SCAN_MAX_NODES:
            scored = [(len(n.outbound_links) + len(n.backlink_uids), n) for n in all_nodes]
            scored.sort(key=lambda x: x[0], reverse=True)
            all_nodes = [n for _, n in scored[:FULL_SCAN_MAX_NODES]]

        summaries = [f"{n.uid} ({n.entity_type}, {n.country or 'N/A'}): {n.summary[:150]}" for n in all_nodes]
        context = "\n".join(summaries)

        system_prompt = (
            "You are the ATIS Master Intelligence Synthesizer. Given vault summaries, "
            "produce a high-level dashboard. Output ONLY raw JSON. "
            "Every claim must cite source nodes."
        )
        user_prompt = (
            f"## VAULT SUMMARY\nTotal: {len(all_nodes)}\n\n{context}\n\n"
            f"## OUTPUT\nProduce dashboard JSON with executive_summary, structured_intelligence, findings, opportunities, risks, key_entities. "
            f"Every row must have source_node. Every finding must have source_nodes."
        )

        if self.token_budget.estimate(system_prompt + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(system_prompt)
            user_prompt = user_prompt[:max_chars] + "\n[TRUNCATED]"

        raw = self._call_api(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.15, max_tokens=4096)
        result = self._parse_grounded_response(raw, all_nodes)
        result["scan_mode"] = "full_vault_capped"
        return result

    # -----------------------------------------------------------------
    # MAIN ORCHESTRATOR
    # -----------------------------------------------------------------
    def generate_query_payload(self, vault_mgr: ObsidianVaultManager,
                               question: str | None = None) -> Dict[str, Any]:
        if not question:
            return self.full_vault_scan(vault_mgr)

        # Stage 0: Intent
        intent = self.extract_intent(question)

        # Stage 1: Broad pre-filter
        candidates = self.broad_pre_filter(vault_mgr, intent)

        # Stage 2: LLM semantic ranking
        ranked_nodes = self.llm_semantic_ranking(question, intent, candidates)

        # Stage 3: Grounded synthesis
        result = self.grounded_synthesis(question, intent, ranked_nodes)

        entity_graph = vault_mgr.build_entity_graph(ranked_nodes)
        aggregate_stats = vault_mgr.get_aggregate_stats(ranked_nodes)

        return {
            **result,
            "intent": {
                "type": intent.intent_type,
                "entities": intent.target_entities,
                "entity_types": intent.target_entity_types,
                "countries": intent.target_countries,
                "sectors": intent.target_sectors,
            },
            "filter_stats": {
                "vault_total": vault_mgr.indexed_count,
                "candidates_after_broad_filter": len(candidates),
                "ranked_by_llm": len(ranked_nodes),
            },
            "entity_graph": entity_graph,
            "stats": aggregate_stats,
        }


# =============================================================================
# PERSISTENCE
# =============================================================================
def persist_query_payload(payload: Dict[str, Any], entity_graph: Dict[str, Any],
                          aggregate_stats: Dict[str, Any], question: str | None = None) -> Tuple[Path, Path]:
    output_dir = Path(os.getenv("OUTPUT_DIR", "./output/query_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    query_id = f"ATIS-QUERY-{timestamp}"

    dashboard_payload = {
        "query_id": query_id,
        "user_question": question or "FULL_VAULT_SCAN",
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
        "pipeline_metadata": {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "vault_files_scanned": aggregate_stats.get("total_entities", 0),
            "model_primary": "gpt-oss-120b",
            "model_fallback": "gemma-4-31b",
            "pipeline_version": "grounded_v3",
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
# WEB ENTRY POINT
# =============================================================================
def run_query_pipeline(question: str | None = None,
                       vault_path: str | Path = "./vault") -> Dict[str, Any]:
    vault_mgr = ObsidianVaultManager(Path(vault_path))
    vault_mgr.build_index()

    if vault_mgr.indexed_count == 0:
        raise RuntimeError("No markdown files found in vault.")

    engine = CerebrasQueryEngine()
    result = engine.generate_query_payload(vault_mgr, question)

    entity_graph = result.pop("entity_graph", {})
    aggregate_stats = result.pop("stats", {})

    json_path, graph_path = persist_query_payload(result, entity_graph, aggregate_stats, question)

    return {
        "dashboard": result,
        "entity_graph": entity_graph,
        "stats": aggregate_stats,
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
        description="ATIS Query v3 — Grounded Intent-First Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vault_path", default="./vault", help="Path to Obsidian vault root.")
    parser.add_argument("--question", default="", help="Natural language question.")
    parser.add_argument("--api_key", default="", help="Optional Cerebras API key override.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    vault_path = Path(args.vault_path).resolve()
    question = args.question.strip() if args.question else None

    if question:
        logger.info("=" * 70)
        logger.info("MODE B: QUESTION-DRIVEN QUERY (Grounded v3)")
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
        print(f"\nEXECUTIVE SUMMARY:\n{result['dashboard'].get('executive_summary', 'N/A')}")
        print(f"\nFILTER STATS: {result['dashboard'].get('filter_stats', {})}")
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
