#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATIS_Query.py
Query Intelligence Layer of the ATIS Intelligence Suite.

MODE A — Full Vault Scan (no --question):
  Scans the entire indexed Obsidian vault and emits a master dashboard.

MODE B — Question-Driven Query (with --question):
  Accepts a natural-language question, searches the vault for relevant
  anchor nodes, crawls the subgraph, and synthesizes a targeted dashboard
  response that answers the question with full structured intelligence.

Populates every pane of the Query Dashboard:
  1. Query Hero Card (executive summary + stats)
  2. Entity Relationship Network Graph
  3. Structured Intelligence Table
  4. Three-Column Info Cards (Findings / Opportunities / Risks)

Usage:
    # Mode A — Full vault scan
    python ATIS_Query.py --vault_path "C:/Users/tmaki/Documents/AKSOS/ATIS/Data"

    # Mode B — Question-driven
    python ATIS_Query.py --vault_path "C:/Users/tmaki/Documents/AKSOS/ATIS/Data" \
        --question "What lithium refineries are operational in Zimbabwe and who regulates them?"
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
CHUNK_SIZE: int = 20
INTER_CHUNK_DELAY_SECONDS: float = 1.5

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
# Data structures
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
# ObsidianVaultManager
# =============================================================================
class ObsidianVaultManager:
    _WIKILINK_PATTERN = re.compile(r'\[\[(.*?)\]\]')

    STOP_WORDS = {
        "and", "the", "of", "for", "in", "with", "institute", "agency",
        "a", "an", "to", "is", "on", "at", "by", "from", "as", "or",
        "it", "its", "this", "that", "these", "those", "be", "are",
        "what", "who", "where", "when", "why", "how", "which", "are", "does",
        "show", "me", "all", "any", "some", "many", "much", "more", "most",
        "have", "has", "had", "do", "did", "can", "could", "will", "would",
        "should", "shall", "may", "might", "must", "about", "into", "through",
        "during", "before", "after", "above", "below", "between", "under",
        "again", "further", "then", "once", "here", "there", "so", "than",
        "too", "very", "just", "now", "only", "own", "same", "such", "each",
        "few", "other", "some", "time", "way", "no", "not", "only", "own",
    }

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
        entity_type = front_matter.get("node_type", "")
        if not entity_type:
            entity_type = front_matter.get("type", "unknown")
        country = front_matter.get("country", "") or front_matter.get("location", "")
        sector = front_matter.get("sector", "") or front_matter.get("industry", "")

        if not entity_type or entity_type == "unknown":
            body_lower = body.lower()
            if any(w in body_lower for w in ["mine", "refinery", "smelter", "concentrator", "processing plant"]):
                entity_type = "mining_refinery"
            elif any(w in body_lower for w in ["ministry", "department of", "bureau of"]):
                entity_type = "government_ministry"
            elif any(w in body_lower for w in ["agency", "regulatory", "authority", "commission"]):
                entity_type = "government_agency"
            elif any(w in body_lower for w in ["university", "institute", "polytechnic", "research lab"]):
                entity_type = "academic_institution"
            elif any(w in body_lower for w in ["power plant", "dam", "railway", "port", "laboratory", "grid"]):
                entity_type = "infrastructure_node"
            elif any(w in body_lower for w in ["lithium", "cobalt", "copper", "gold", "nickel", "ore", "acid", "sulfuric"]):
                entity_type = "commodity"
            elif any(w in body_lower for w in ["act", "law", "ban", "policy", "framework", "regulation", "initiative"]):
                entity_type = "policy_framework"
            elif any(w in body_lower for w in ["corporation", "ltd", "inc", "group", "company", "holdings"]):
                entity_type = "private_conglomerate"
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
            "opportunity": {"shape": "rect", "fill": "#fff", "stroke": "#fff", "text_color": "#000"},
            "hub": {"shape": "circle", "fill": "#fff", "stroke": "#fff", "text_color": "#000"},
            "mining_refinery": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "private_conglomerate": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "government_agency": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "government_ministry": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "academic_institution": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "infrastructure_node": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "commodity": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "policy_framework": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#333", "text_color": "#fff"},
            "risk": {"shape": "rect", "fill": "#1c1c1e", "stroke": "#ff453a", "text_color": "#ff453a"},
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

    # -----------------------------------------------------------------
    # Question-driven search (Anchor + Crawl)
    # -----------------------------------------------------------------
    def _extract_question_tokens(self, question: str) -> Set[str]:
        words = re.findall(r"[a-zA-Z0-9]+", question.lower())
        return {w for w in words if w not in self.STOP_WORDS and len(w) > 2}

    def search_for_question(self, question: str) -> List[VaultNode]:
        """
        Find anchor nodes matching the question tokens, then crawl outbound
        and inbound backlinks to build the relevant subgraph.
        """
        tokens = self._extract_question_tokens(question)
        if not tokens:
            logger.warning("No valid search tokens extracted from question. Returning all nodes.")
            return list(self.nodes.values())

        logger.info("Question tokens: %s", sorted(tokens))

        scored: List[Tuple[int, VaultNode]] = []
        for node in self.nodes.values():
            aliases = self._extract_aliases(node.front_matter)
            alias_text = " ".join(aliases)
            text_block = f"{node.uid} {alias_text} {node.summary} {node.country} {node.sector} {node.entity_type}".lower()
            bonus_text = f"{node.uid} {node.stem} {node.front_matter.get('title', '')} {node.front_matter.get('name', '')}".lower()

            score = 0
            for token in tokens:
                if token in bonus_text:
                    score += 3
                elif token in text_block:
                    score += 1

            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_anchors = [node for _, node in scored[:15]]
        logger.info("Anchor detection: %d anchors from %d candidates.", len(top_anchors), len(scored))

        if not top_anchors:
            logger.warning("No anchors found. Returning all nodes.")
            return list(self.nodes.values())

        # Crawl subgraph
        cluster: Dict[str, VaultNode] = {}
        for anchor in top_anchors:
            cluster[anchor.uid] = anchor
            for target_uid in anchor.outbound_links:
                if target_uid in self.nodes and target_uid not in cluster:
                    cluster[target_uid] = self.nodes[target_uid]
            for source_uid in self.backlink_map.get(anchor.uid, []):
                if source_uid in self.nodes and source_uid not in cluster:
                    cluster[source_uid] = self.nodes[source_uid]

        result = list(cluster.values())
        logger.info("Subgraph crawl returned %d nodes (%d anchors + %d related).",
                    len(result), len(top_anchors), len(result) - len(top_anchors))
        return result


# =============================================================================
# Token Budget Manager
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
        truncated = user_prompt[:max_chars] + "\n\n[USER PROMPT TRUNCATED TO RESPECT TOKEN BUDGET]"
        logger.warning("User prompt truncated from %d to ~%d chars.", len(user_prompt), max_chars)
        return truncated


# =============================================================================
# LLM Integration Layer
# =============================================================================
class CerebrasQueryEngine:
    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or HARDCODED_API_KEY or os.environ.get("CEREBRAS_API_KEY")
        if not resolved_key:
            raise RuntimeError("Cerebras API key not configured.")
        self.client = Cerebras(api_key=resolved_key)
        self.model = "gpt-oss-120b"
        self.fallback_model = "gemma-4-31b"
        self.temperature = 0.15
        self.max_tokens = 4096
        self.max_retries = 3
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
                    delay = self.base_delay_seconds * (2 ** (attempt - 1))
                    logger.warning("RateLimitError on %s (attempt %d): %s. Backing off %.1fs...",
                                   model, attempt, exc, delay)
                    if attempt < self.max_retries:
                        time.sleep(delay)
                    else:
                        if model == self.model:
                            break
                        raise
                except APIConnectionError as exc:
                    delay = self.base_delay_seconds * (2 ** (attempt - 1))
                    logger.warning("APIConnectionError on %s (attempt %d): %s. Retrying %.1fs...",
                                   model, attempt, exc, delay)
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
    # MAP: Chunk extraction
    # -----------------------------------------------------------------
    def _map_chunk(self, chunk_nodes: List[VaultNode], chunk_index: int,
                   total_chunks: int, question: str | None = None) -> str:
        logger.info("MAP phase | Chunk %d/%d (%d nodes)", chunk_index + 1, total_chunks, len(chunk_nodes))

        node_blocks: List[str] = []
        for node in chunk_nodes:
            node_blocks.append(
                f"--- NODE: {node.uid} ---\n"
                f"Type: {node.entity_type}\n"
                f"Country: {node.country or 'N/A'}\n"
                f"Sector: {node.sector or 'N/A'}\n"
                f"Summary: {node.summary}\n"
                f"Frontmatter: {json.dumps(node.front_matter, default=str)}\n"
                f"Outbound Links: {', '.join(node.outbound_links[:10])}\n"
                f"Backlinks: {', '.join(node.backlink_uids[:10])}\n"
                f"Content Preview:\n{node.body_preview[:800]}\n"
            )

        chunk_context = "\n".join(node_blocks)

        if question:
            system_prompt = (
                "You are the ATIS Query Intelligence Engine. A user has asked a specific question. "
                "Your job is to read the provided vault nodes and extract ONLY information relevant to "
                "answering that question. Focus on: entities, relationships, regulatory status, "
                "infrastructure constraints, and commercial opportunities. You MUST use XML tags."
            )
            user_prompt = (
                f"## USER QUESTION\n{question}\n\n"
                f"## VAULT NODES (Chunk {chunk_index + 1} of {total_chunks})\n"
                f"{chunk_context}\n\n"
                "## EXTRACTION INSTRUCTIONS\n"
                "Extract only facts relevant to the user's question. "
                "Your response MUST be split into two XML tags:\n\n"
                "1. <chunk_intelligence>\n"
                "   Dense bullet-point list of relevant facts. Include entities, relationships, "
                "   regulatory gaps, infrastructure constraints, and risk indicators. "
                "   If irrelevant, state 'IRRELEVANT'. No markdown headers inside.\n"
                "   </chunk_intelligence>\n\n"
                "2. <chunk_entities>\n"
                "   Valid JSON array of objects: entity_name, entity_type, country, sector, "
                "   related_entities (array), significance_score (1-10). Empty array [] if none.\n"
                "   </chunk_entities>\n\n"
                "No text outside these two XML tags."
            )
        else:
            system_prompt = (
                "You are a strategic intelligence extraction engine for ATIS. "
                "Extract structured intelligence from vault nodes. Focus on: entities, relationships, "
                "commodities, regulatory gaps, infrastructure bottlenecks, opportunities, and risks. "
                "You MUST use XML tags."
            )
            user_prompt = (
                f"## VAULT NODES (Chunk {chunk_index + 1} of {total_chunks})\n"
                f"{chunk_context}\n\n"
                "## EXTRACTION INSTRUCTIONS\n"
                "Extract structured intelligence. Response MUST be split into two XML tags:\n\n"
                "1. <chunk_intelligence>\n"
                "   Dense bullet-point list: entities, relationships, commodities, regulatory gaps, "
                "   infrastructure constraints, risk indicators. If irrelevant, state 'IRRELEVANT'.\n"
                "   </chunk_intelligence>\n\n"
                "2. <chunk_entities>\n"
                "   Valid JSON array: entity_name, entity_type, country, sector, related_entities, significance_score.\n"
                "   </chunk_entities>\n\n"
                "No text outside these two XML tags."
            )

        if self.token_budget.estimate(system_prompt + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(system_prompt)
            user_prompt = user_prompt[:max_chars] + "\n[CHUNK TRUNCATED]"
            logger.warning("Chunk %d truncated to fit token budget.", chunk_index + 1)

        raw_response = self._call_api(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1, max_tokens=2048)
        if not raw_response:
            raise RuntimeError(f"Empty API response for chunk {chunk_index + 1}")
        logger.info("Chunk %d/%d response received (%d chars).", chunk_index + 1, total_chunks, len(raw_response))
        return raw_response

    # -----------------------------------------------------------------
    # REDUCE: Master synthesis
    # -----------------------------------------------------------------
    def _reduce_synthesize(self, chunk_intelligences: List[str],
                         aggregate_stats: Dict[str, Any], question: str | None = None) -> str:
        logger.info("REDUCE phase | Synthesizing %d chunks.", len(chunk_intelligences))
        consolidated = "\n\n--- CHUNK BOUNDARY ---\n\n".join(chunk_intelligences)
        stats_json = json.dumps(_sanitize_for_json(aggregate_stats), indent=2, default=str)

        if question:
            system_prompt = (
                "You are the ATIS Query Intelligence Synthesis Engine. A user has asked a specific question. "
                "Synthesize the extracted vault intelligence into a comprehensive, question-driven dashboard payload. "
                "Your outputs are deterministic, analytical, and strictly formatted. You MUST use XML tags."
            )
            user_prompt = (
                f"# ATIS QUERY DASHBOARD — QUESTION-DRIVEN SYNTHESIS\n\n"
                f"## USER QUESTION\n{question}\n\n"
                f"## RELEVANT SUBGRAPH STATISTICS\n```json\n{stats_json}\n```\n\n"
                f"## CONSOLIDATED CHUNK INTELLIGENCE\n{consolidated}\n\n"
                "---\n\n"
                "## STRICT OUTPUT INSTRUCTIONS\n"
                "Produce a comprehensive dashboard payload that DIRECTLY ANSWERS the user's question. "
                "Your response MUST be split into FOUR XML tags:\n\n"
                "1. <executive_summary>\n"
                "   A direct answer to the question (3-5 sentences). Include scope, key entities, "
                "   dominant sectors, and highest-priority structural gaps relevant to the question.\n"
                "   </executive_summary>\n\n"
                "2. <structured_intelligence>\n"
                "   Valid JSON array of intelligence rows. Each object: entity, type, country, relationship, "
                "   status, priority (Critical/High/Medium/Low), insight. 8-15 rows relevant to the question.\n"
                "   </structured_intelligence>\n\n"
                "3. <findings_opportunities_risks>\n"
                "   Valid JSON object with keys 'findings', 'opportunities', 'risks'. Each maps to array of 4-6 strings. "
                "   All content must be relevant to the user's question.\n"
                "   </findings_opportunities_risks>\n\n"
                "4. <key_entities>\n"
                "   Valid JSON array of top 10 entities. Each: entity_name, entity_type, country, sector, "
                "   significance_score (1-10), related_count (integer), summary (string).\n"
                "   </key_entities>\n\n"
                "No text outside these four XML tags."
            )
        else:
            system_prompt = (
                "You are the ATIS Query Intelligence Synthesis Engine. Synthesize vault intelligence into a "
                "unified commercial-intelligence dashboard payload. Outputs are deterministic, analytical, "
                "strictly formatted. You MUST use XML tags."
            )
            user_prompt = (
                "# ATIS QUERY DASHBOARD — MASTER INTELLIGENCE SYNTHESIS\n\n"
                f"## VAULT AGGREGATE STATISTICS\n```json\n{stats_json}\n```\n\n"
                f"## CONSOLIDATED CHUNK INTELLIGENCE\n{consolidated}\n\n"
                "---\n\n"
                "## STRICT OUTPUT INSTRUCTIONS\n"
                "Produce a comprehensive Query Dashboard payload. Response MUST be split into FOUR XML tags:\n\n"
                "1. <executive_summary> — Single paragraph (3-5 sentences) on African trade/mining landscape.\n"
                "2. <structured_intelligence> — JSON array (8-15 rows): entity, type, country, relationship, status, priority, insight.\n"
                "3. <findings_opportunities_risks> — JSON object: findings, opportunities, risks (each 4-6 strings).\n"
                "4. <key_entities> — JSON array (top 10): entity_name, entity_type, country, sector, significance_score, related_count, summary.\n\n"
                "No text outside these four XML tags."
            )

        if self.token_budget.estimate(system_prompt + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(system_prompt)
            user_prompt = user_prompt[:max_chars] + "\n[FINAL PROMPT TRUNCATED]"
            logger.warning("Final reduce prompt truncated to fit token budget.")

        return self._call_api(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.15, max_tokens=4096)

    # -----------------------------------------------------------------
    # Orchestrator
    # -----------------------------------------------------------------
    def generate_query_payload(self, vault_nodes: List[VaultNode],
                             aggregate_stats: Dict[str, Any], question: str | None = None) -> Dict[str, Any]:
        chunk_intelligences: List[str] = []
        all_chunk_entities: List[Dict[str, Any]] = []

        if not vault_nodes:
            logger.warning("No vault nodes provided; generating minimal payload.")
            raw_reduce = self._reduce_synthesize(["NO VAULT NODES DISCOVERED."], aggregate_stats, question)
            return self._parse_reduce_output(raw_reduce, aggregate_stats)

        chunks: List[List[VaultNode]] = []
        for i in range(0, len(vault_nodes), CHUNK_SIZE):
            chunks.append(vault_nodes[i:i + CHUNK_SIZE])

        total_chunks = len(chunks)
        logger.info("Map-Reduce initialized: %d nodes -> %d chunks.", len(vault_nodes), total_chunks)

        for idx, chunk in enumerate(chunks):
            raw_response = self._map_chunk(chunk, idx, total_chunks, question)
            intel_match = re.search(r"<chunk_intelligence>\s*(.*?)\s*</chunk_intelligence>", raw_response, re.DOTALL)
            if intel_match:
                chunk_intelligences.append(intel_match.group(1).strip())
            else:
                chunk_intelligences.append(raw_response.strip())
                logger.warning("Chunk %d: <chunk_intelligence> tag not found; using raw response.", idx + 1)

            entities_match = re.search(r"<chunk_entities>\s*(.*?)\s*</chunk_entities>", raw_response, re.DOTALL)
            if entities_match:
                try:
                    entities_array = json.loads(entities_match.group(1).strip())
                    if isinstance(entities_array, list):
                        all_chunk_entities.extend(entities_array)
                        logger.info("Chunk %d: extracted %d entities.", idx + 1, len(entities_array))
                except json.JSONDecodeError as exc:
                    logger.warning("Chunk %d: failed to parse chunk_entities JSON: %s", idx + 1, exc)
            else:
                logger.warning("Chunk %d: <chunk_entities> tag not found.", idx + 1)

            if idx < total_chunks - 1:
                time.sleep(INTER_CHUNK_DELAY_SECONDS)

        raw_reduce = self._reduce_synthesize(chunk_intelligences, aggregate_stats, question)
        return self._parse_reduce_output(raw_reduce, aggregate_stats, all_chunk_entities)

    def _parse_reduce_output(self, raw_reduce: str, aggregate_stats: Dict[str, Any],
                             supplemental_entities: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        summary_match = re.search(r"<executive_summary>\s*(.*?)\s*</executive_summary>", raw_reduce, re.DOTALL)
        executive_summary = summary_match.group(1).strip() if summary_match else "No executive summary generated."

        intel_match = re.search(r"<structured_intelligence>\s*(.*?)\s*</structured_intelligence>", raw_reduce, re.DOTALL)
        structured_intelligence = []
        if intel_match:
            try:
                structured_intelligence = json.loads(intel_match.group(1).strip())
                if not isinstance(structured_intelligence, list):
                    structured_intelligence = []
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse structured_intelligence JSON: %s", exc)

        for_match = re.search(r"<findings_opportunities_risks>\s*(.*?)\s*</findings_opportunities_risks>", raw_reduce, re.DOTALL)
        findings, opportunities, risks = [], [], []
        if for_match:
            try:
                for_data = json.loads(for_match.group(1).strip())
                findings = for_data.get("findings", [])
                opportunities = for_data.get("opportunities", [])
                risks = for_data.get("risks", [])
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse findings_opportunities_risks JSON: %s", exc)

        key_match = re.search(r"<key_entities>\s*(.*?)\s*</key_entities>", raw_reduce, re.DOTALL)
        key_entities = supplemental_entities or []
        if key_match:
            try:
                parsed_entities = json.loads(key_match.group(1).strip())
                if isinstance(parsed_entities, list):
                    key_entities = parsed_entities
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse key_entities JSON: %s", exc)

        return {
            "executive_summary": executive_summary,
            "structured_intelligence": structured_intelligence,
            "findings": findings,
            "opportunities": opportunities,
            "risks": risks,
            "key_entities": key_entities,
        }


# =============================================================================
# Output persistence
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
        "stats": {
            "total_entities": aggregate_stats.get("total_entities", 0),
            "total_relationships": aggregate_stats.get("total_relationships", 0),
            "commodities_tracked": aggregate_stats.get("commodities_tracked", 0),
            "countries_covered": aggregate_stats.get("countries_covered", 0),
            "risk_flags": aggregate_stats.get("sectors_mapped", 0),
            "entity_type_breakdown": aggregate_stats.get("entity_type_breakdown", {}),
        },
        "entity_graph": entity_graph,
        "structured_intelligence": payload.get("structured_intelligence", []),
        "findings": payload.get("findings", []),
        "opportunities": payload.get("opportunities", []),
        "risks": payload.get("risks", []),
        "key_entities": payload.get("key_entities", []),
        "pipeline_metadata": {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "vault_files_scanned": aggregate_stats.get("total_entities", 0),
            "model_primary": "gpt-oss-120b",
            "model_fallback": "gemma-4-31b",
        },
    }

    json_path = output_dir / f"query_dashboard_{timestamp}.json"
    json_path.write_text(json.dumps(dashboard_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Query dashboard payload persisted to: %s", json_path)

    graph_path = output_dir / f"query_graph_{timestamp}.json"
    graph_path.write_text(json.dumps(entity_graph, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Query graph companion persisted to: %s", graph_path)

    return json_path, graph_path


# =============================================================================
# Main orchestration
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATIS Query Layer — Vault Intelligence Dashboard Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vault_path", default=r"C:\Users\tmaki\Documents\AKSOS\ATIS\Vault",
                        help="Absolute path to the Obsidian vault root.")
    parser.add_argument("--question", default="",
                        help="Natural-language question to query the vault against.")
    parser.add_argument("--api_key", default="",
                        help="Optional Cerebras API key override.")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                        help=f"Nodes per LLM chunk. (default: {CHUNK_SIZE})")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    vault_path = Path(args.vault_path).resolve()
    question = args.question.strip() if args.question else None

    if question:
        logger.info("=" * 70)
        logger.info("MODE B: QUESTION-DRIVEN QUERY")
        logger.info("Question: %s", question)
        logger.info("=" * 70)
    else:
        logger.info("=" * 70)
        logger.info("MODE A: FULL VAULT SCAN")
        logger.info("=" * 70)

    # 1. Index vault
    vault_mgr = ObsidianVaultManager(vault_path)
    try:
        vault_mgr.build_index()
    except Exception as exc:
        logger.error("Vault indexing failed: %s", exc)
        sys.exit(1)

    if vault_mgr.indexed_count == 0:
        logger.error("No markdown files found in vault. Aborting.")
        sys.exit(1)

    # 2. Determine node scope (full vault or question-relevant subgraph)
    if question:
        relevant_nodes = vault_mgr.search_for_question(question)
        aggregate_stats = vault_mgr.get_aggregate_stats(relevant_nodes)
        entity_graph = vault_mgr.build_entity_graph(relevant_nodes)
        all_nodes = relevant_nodes
        logger.info("Question mode: using %d relevant nodes.", len(relevant_nodes))
    else:
        aggregate_stats = vault_mgr.get_aggregate_stats()
        entity_graph = vault_mgr.build_entity_graph()
        all_nodes = vault_mgr.get_all_nodes_as_context()

    logger.info("Vault summary — Entities: %d, Relationships: %d, Commodities: %d, Countries: %d",
                aggregate_stats["total_entities"], aggregate_stats["total_relationships"],
                aggregate_stats["commodities_tracked"], aggregate_stats["countries_covered"])

    # 3. LLM generation
    engine = CerebrasQueryEngine(api_key=args.api_key if args.api_key else None)
    try:
        result = engine.generate_query_payload(all_nodes, aggregate_stats, question)
    except Exception as exc:
        logger.error("Query payload generation failed: %s", exc)
        sys.exit(1)

    # 4. Persist
    try:
        json_path, graph_path = persist_query_payload(result, entity_graph, aggregate_stats, question)
    except Exception as exc:
        logger.error("Failed to persist outputs: %s", exc)
        sys.exit(1)

    print(f"\nSUCCESS: Query dashboard payload written to:\n  {json_path}")
    print(f"SUCCESS: Graph companion written to:\n  {graph_path}")
    print(f"\nEXECUTIVE SUMMARY:\n{result.get('executive_summary', 'N/A')}")


# =============================================================================
# Web entry point
# =============================================================================
def run_query_pipeline(question: str | None = None) -> Dict[str, Any]:
    """
    Web-compatible entry point. Accepts optional question string.
    Returns full query dashboard payload.
    """
    vault_path = Path(os.getenv("VAULT_PATH", "./vault"))
    vault_mgr = ObsidianVaultManager(vault_path)
    vault_mgr.build_index()

    if vault_mgr.indexed_count == 0:
        raise RuntimeError("No markdown files found in vault.")

    if question:
        relevant_nodes = vault_mgr.search_for_question(question)
        aggregate_stats = vault_mgr.get_aggregate_stats(relevant_nodes)
        entity_graph = vault_mgr.build_entity_graph(relevant_nodes)
        all_nodes = relevant_nodes
    else:
        aggregate_stats = vault_mgr.get_aggregate_stats()
        entity_graph = vault_mgr.build_entity_graph()
        all_nodes = vault_mgr.get_all_nodes_as_context()

    engine = CerebrasQueryEngine()
    result = engine.generate_query_payload(all_nodes, aggregate_stats, question)

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


if __name__ == "__main__":
    main()
