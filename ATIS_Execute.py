#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATIS_Execute.py
Execution Layer of the ATIS Intelligence Suite.

Scans an indexed Obsidian vault against a target opportunity from an ATIS
dashboard, compiles a tactical transaction roadmap via the Cerebras API,
and persists the result as a markdown file.

Map-Reduce Architecture:
  1. MAP: Split vault nodes into 4-node chunks. Extract legal waivers,
     restrictions, and tactical facts per chunk.
  2. REDUCE: Feed condensed chunk summaries into a final synthesis prompt
     to generate the master roadmap.

Usage:
    python ATIS_Execute.py \
        --dashboard_path "C:/Users/tmaki/Documents/AKSOS/ATIS/dashboard.json" \
        --opportunity_id "OPP-001"
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
from typing import Any, Dict, List, Tuple
from datetime import date, datetime

# =============================================================================
# CONFIGURATION — Hardcode your API key here if desired
# =============================================================================
HARDCODED_API_KEY: str = "csk-v4vf9r666pv9t99etmm9cppv58j8xpc8fnjpxkpw89mk36rp"

# =============================================================================
# Token Budget Configuration
# =============================================================================
MAX_TOKENS_PER_REQUEST: int = 60_000
RESPONSE_RESERVE: int = 8_000
SAFETY_BUFFER: int = 1_000
CHUNK_SIZE: int = 12
INTER_CHUNK_DELAY_SECONDS: float = 0.0

# =============================================================================
# Dependency guards with actionable install hints
# =============================================================================
_MISSING_DEPS: List[str] = []

try:
    import yaml
except ImportError:
    _MISSING_DEPS.append("PyYAML (pip install pyyaml)")

if _MISSING_DEPS:
    print(
        "FATAL: Missing required dependencies:\n  - "
        + "\n  - ".join(_MISSING_DEPS),
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from cerebras.cloud.sdk import Cerebras
    from cerebras.cloud.sdk import APIError, APIConnectionError, RateLimitError
except ImportError as _import_err:
    sys.stderr.write(
        "ERROR: The 'cerebras.cloud.sdk' package is not installed. "
        "Install it via: pip install cerebras-cloud-sdk\n"
    )
    raise SystemExit(1) from _import_err

# =============================================================================
# Logging configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ATIS_Execute")


# =============================================================================
# JSON Serialization Sanitizer
# =============================================================================
def _sanitize_for_json(data):
    """
    Recursively traverses lists and dicts, converting native date/datetime
    objects into ISO-8601 strings so they survive json.dumps().
    """
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
    """
    Represents a fully parsed vault entity with graph-relational metadata.
    """
    uid: str
    absolute_path: Path
    stem: str
    front_matter: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    outbound_links: List[str] = field(default_factory=list)
    raw_content: str = ""
    body: str = ""
    extracted_sections: Dict[str, str] = field(default_factory=dict)
    body_preview: str = ""
    is_anchor: bool = False
    backlink_uids: List[str] = field(default_factory=list)


# =============================================================================
# ObsidianVaultManager
# =============================================================================
class ObsidianVaultManager:
    """
    Two-pass graph indexer for an Obsidian vault.
    Pass 1: Ingest every markdown file into a VaultNode (summary, outbound links, raw content).
    Pass 2: Invert outbound links into a global backlink_map.
    """

    _WIKILINK_PATTERN = re.compile(r'\[\[(.*?)\]\]')
    _SECTION_PATTERNS = [
        re.compile(r"(?im)^#{1,6}\s*Key Contacts\s*\n(.*?)(?=\n#{1,6}\s|\Z)", re.DOTALL),
        re.compile(r"(?im)^#{1,6}\s*Decision Makers\s*\n(.*?)(?=\n#{1,6}\s|\Z)", re.DOTALL),
        re.compile(r"(?im)^#{1,6}\s*Licenses\s*\n(.*?)(?=\n#{1,6}\s|\Z)", re.DOTALL),
        re.compile(r"(?im)^#{1,6}\s*Active Projects\s*\n(.*?)(?=\n#{1,6}\s|\Z)", re.DOTALL),
    ]

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = Path(vault_root).resolve()
        self.nodes: Dict[str, VaultNode] = {}
        self.backlink_map: Dict[str, List[str]] = {}
        self._link_resolver: Dict[str, str] = {}
        self.indexed_count: int = 0

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
        return body[:300].strip()

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

    def _extract_relevant_sections(self, body: str) -> Dict[str, str]:
        extracted: Dict[str, str] = {}
        for pattern in self._SECTION_PATTERNS:
            m = pattern.search(body)
            if m:
                header = pattern.pattern.split(r"\s*")[0].replace("(?im)^#{1,6}\\s*", "")
                header = header.strip().title()
                extracted[header] = m.group(1).strip()
        return extracted

    def build_index(self) -> None:
        logger.info("Starting vault index build at: %s", self.vault_root)
        if not self.vault_root.exists():
            raise FileNotFoundError(f"Vault root does not exist: {self.vault_root}")

        md_files = list(self.vault_root.rglob("*.md"))
        logger.info("Located %d markdown files.", len(md_files))

        for md_path in md_files:
            try:
                raw_content = md_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                logger.warning("Skipping unreadable file %s: %s", md_path, exc)
                continue
            except Exception as exc:
                logger.warning("Failed to read %s: %s", md_path, exc)
                continue

            front_matter, body = self._parse_markdown(raw_content)
            summary = self._extract_summary(front_matter, body)
            outbound_links = self._extract_outbound_links(raw_content)
            extracted_sections = self._extract_relevant_sections(body)

            uid = md_path.stem
            node = VaultNode(
                uid=uid,
                absolute_path=md_path,
                stem=uid,
                front_matter=front_matter,
                summary=summary,
                outbound_links=outbound_links,
                raw_content=raw_content,
                body=body,
                extracted_sections=extracted_sections,
                body_preview=body[:2000],
            )
            self.nodes[uid] = node

            self._link_resolver[self.canonicalize(uid)] = uid
            for alias in self._extract_aliases(front_matter):
                self._link_resolver[self.canonicalize(alias)] = uid

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
        logger.info(
            "Vault index complete. Cached %d nodes. Backlink map has %d entries.",
            self.indexed_count, len(self.backlink_map)
        )

    def _find_anchor_nodes(self, seed_terms: List[str]) -> List[VaultNode]:
        if not seed_terms:
            logger.warning("No seed terms provided; skipping anchor detection.")
            return []

        STOP_WORDS = {
            "and", "the", "of", "for", "in", "with", "institute", "agency",
            "a", "an", "to", "is", "on", "at", "by", "from", "as", "or",
            "it", "its", "this", "that", "these", "those", "be", "are",
        }

        clean_tokens: set = set()
        for phrase in seed_terms:
            if not phrase:
                continue
            words = re.findall(r"[a-zA-Z0-9]+", phrase.lower())
            for word in words:
                if word not in STOP_WORDS and len(word) > 1:
                    clean_tokens.add(word)

        if not clean_tokens:
            logger.warning("No valid search tokens after stop-word filtering.")
            return []

        logger.info("Anchor search tokens (%d): %s", len(clean_tokens), sorted(clean_tokens))

        scored: List[Tuple[int, VaultNode]] = []
        MIN_TOKEN_MATCHES = 1

        for node in self.nodes.values():
            aliases = self._extract_aliases(node.front_matter)
            alias_text = " ".join(aliases)
            text_block = f"{node.uid} {alias_text} {node.summary}".lower()

            bonus_parts = [node.uid, node.stem]
            for key in ("title", "name", "position", "role", "executive",
                        "job_title", "designation", "organization", "company"):
                val = node.front_matter.get(key)
                if isinstance(val, str):
                    bonus_parts.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            bonus_parts.append(item)
            bonus_text = " ".join(bonus_parts).lower()

            overlap_score = 0
            for token in clean_tokens:
                if token in bonus_text:
                    overlap_score += 3
                elif token in text_block:
                    overlap_score += 1

            if overlap_score >= MIN_TOKEN_MATCHES:
                scored.append((overlap_score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(5, len(scored))
        top_anchors = [node for _, node in scored[:k]]

        for score, node in scored[:k]:
            logger.info("Anchor node selected: %s (score %d)", node.uid, score)

        logger.info(
            "Anchor detection complete: %d anchors selected from %d candidates "
            "using %d tokens.",
            len(top_anchors), len(scored), len(clean_tokens)
        )

        return top_anchors

    def _crawl_subgraph(self, anchors: List[VaultNode]) -> Dict[str, VaultNode]:
        cluster: Dict[str, VaultNode] = {}
        for anchor in anchors:
            anchor.is_anchor = True
            cluster[anchor.uid] = anchor

            for target_uid in anchor.outbound_links:
                if target_uid in self.nodes and target_uid not in cluster:
                    cluster[target_uid] = self.nodes[target_uid]

            for source_uid in self.backlink_map.get(anchor.uid, []):
                if source_uid in self.nodes and source_uid not in cluster:
                    cluster[source_uid] = self.nodes[source_uid]

        return cluster

    def search(self, opportunity: Dict[str, Any], seed_terms: List[str]) -> List[VaultNode]:
        anchors = self._find_anchor_nodes(seed_terms)
        if not anchors:
            logger.warning("No anchor nodes detected for this opportunity.")
            return []

        cluster = self._crawl_subgraph(anchors)
        nodes = list(cluster.values())
        logger.info(
            "Vault search returned %d nodes (%d anchors, %d related).",
            len(nodes), len(anchors), len(nodes) - len(anchors)
        )
        return nodes


# =============================================================================
# Opportunity loader
# =============================================================================
def load_opportunity(dashboard_path: Path, opportunity_id: str) -> Dict[str, Any]:
    logger.info("Loading dashboard: %s", dashboard_path)
    if not dashboard_path.exists():
        raise FileNotFoundError(f"Dashboard file not found: {dashboard_path}")

    try:
        with dashboard_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in dashboard file: {exc}") from exc

    opportunities: List[Dict[str, Any]] = []
    if isinstance(data, list):
        opportunities = data
    elif isinstance(data, dict):
        for key in ("opportunities", "data", "results", "items"):
            if key in data and isinstance(data[key], list):
                opportunities = data[key]
                break
        if not opportunities:
            opportunities = [data]
    else:
        raise ValueError("Dashboard JSON structure not recognized.")

    target = None
    for opp in opportunities:
        if opp.get("id") == opportunity_id or opp.get("opportunity_id") == opportunity_id:
            target = opp
            break

    if target is None:
        raise ValueError(f"Opportunity '{opportunity_id}' not found in dashboard.")

    logger.info("Isolated opportunity: %s", opportunity_id)
    return target


# =============================================================================
# Keyword expansion
# =============================================================================
def expand_keywords(opportunity: Dict[str, Any]) -> List[str]:
    keywords: List[str] = []

    title = opportunity.get("title") or opportunity.get("name", "")
    if title:
        keywords.append(str(title))

    missing = opportunity.get("required_missing_nodes", [])
    if isinstance(missing, list):
        for node in missing:
            if isinstance(node, str):
                keywords.append(node)

    capital_flow = opportunity.get("capital_flow", {})
    if isinstance(capital_flow, dict):
        beneficiary = capital_flow.get("beneficiary")
        if beneficiary:
            keywords.append(str(beneficiary))
        funder = capital_flow.get("likely_funder")
        if funder:
            keywords.append(str(funder))

    seen: set = set()
    deduped: List[str] = []
    for kw in keywords:
        kw_clean = kw.strip()
        if kw_clean and kw_clean.lower() not in seen:
            seen.add(kw_clean.lower())
            deduped.append(kw_clean)

    logger.info("Expanded %d unique seed keywords.", len(deduped))
    return deduped


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

    def truncate_payload(
        self,
        system_prompt: str,
        article_text: str,
        graph_context: str,
        min_article_ratio: float = 0.25,
    ) -> Tuple[str, str]:
        total_estimated = self.estimate(system_prompt + article_text + graph_context)
        if total_estimated <= self.available_for_input:
            return article_text, graph_context

        available_chars = int(self.available_for_input * 3.2) - len(system_prompt)
        if available_chars <= 0:
            raise RuntimeError("System prompt alone exceeds the token budget.")

        min_article_chars = int(available_chars * min_article_ratio)
        article_chars = min(len(article_text), min_article_chars)
        graph_chars = available_chars - article_chars

        truncated_article = article_text
        truncated_graph = graph_context

        if len(article_text) > article_chars:
            truncated_article = (
                article_text[:article_chars]
                + "\n\n[ARTICLE TRUNCATED TO RESPECT TOKEN BUDGET]"
            )
            logger.warning(
                "Article truncated from %d to ~%d chars to fit token budget.",
                len(article_text),
                article_chars,
            )

        if len(graph_context) > graph_chars:
            truncated_graph = (
                graph_context[:graph_chars]
                + "\n\n[GRAPH CONTEXT TRUNCATED TO RESPECT TOKEN BUDGET]"
            )
            logger.warning(
                "Graph context truncated from %d to ~%d chars to fit token budget.",
                len(graph_context),
                graph_chars,
            )

        revised_estimate = self.estimate(
            system_prompt + truncated_article + truncated_graph
        )
        logger.info(
            "Revised payload estimate after truncation: %d tokens (budget: %d)",
            revised_estimate,
            self.available_for_input,
        )
        return truncated_article, truncated_graph


# =============================================================================
# LLM Integration Layer — Map-Reduce Chunked Execution
# =============================================================================
class CerebrasExecutionEngine:
    """
    Map-Reduce architecture:
      MAP: Split vault nodes into chunks. Extract legal waivers, restrictions,
           and tactical facts per chunk via Cerebras.
      REDUCE: Synthesize chunk summaries into the final master roadmap.
    """

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or HARDCODED_API_KEY or os.environ.get("CEREBRAS_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Cerebras API key is not configured. Set the HARDCODED_API_KEY constant "
                "at the top of this script, pass --api_key, or export the "
                "CEREBRAS_API_KEY environment variable."
            )

        self.client = Cerebras(api_key=resolved_key)
        self.model = "gpt-oss-120b"
        self.fallback_model = "gemma-4-31b"
        self.temperature = 0.15
        self.max_tokens = 4096
        self.max_retries = 3
        self.base_delay_seconds = 2.0
        self.token_budget = TokenBudget()
        logger.info("Cerebras client initialized. Model: %s", self.model)

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        models_to_try = [self.model, self.fallback_model]

        for model in models_to_try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info(
                        "API call | model=%s | attempt=%d/%d",
                        model,
                        attempt,
                        self.max_retries,
                    )
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    content = response.choices[0].message.content
                    logger.info("API call successful (%s)", model)
                    return content
                except RateLimitError as exc:
                    delay = self.base_delay_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "RateLimitError on %s (attempt %d): %s. "
                        "Backing off for %.1f seconds...",
                        model,
                        attempt,
                        exc,
                        delay,
                    )
                    if attempt < self.max_retries:
                        time.sleep(delay)
                    else:
                        if model == self.model:
                            break
                        raise
                except APIConnectionError as exc:
                    delay = self.base_delay_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "APIConnectionError on %s (attempt %d): %s. "
                        "Retrying in %.1f seconds...",
                        model,
                        attempt,
                        exc,
                        delay,
                    )
                    if attempt < self.max_retries:
                        time.sleep(delay)
                    else:
                        if model == self.model:
                            break
                        raise
                except APIError as exc:
                    logger.error(
                        "APIError on %s (attempt %d): %s", model, attempt, exc
                    )
                    if attempt == self.max_retries:
                        if model == self.model:
                            logger.warning(
                                "Falling back to smaller model: %s",
                                self.fallback_model,
                            )
                            break
                        raise
                    time.sleep(self.base_delay_seconds)
                except Exception as exc:
                    logger.error(
                        "Unexpected error on %s (attempt %d): %s",
                        model,
                        attempt,
                        exc,
                    )
                    if attempt == self.max_retries:
                        if model == self.model:
                            break
                        raise
                    time.sleep(self.base_delay_seconds)

        raise RuntimeError("All API retries exhausted and fallback model failed.")

    # ---------------------------------------------------------------------
    # MAP: Chunk extraction
    # ---------------------------------------------------------------------
    def _map_chunk(
        self,
        opportunity: Dict[str, Any],
        chunk_nodes: List[VaultNode],
        chunk_index: int,
        total_chunks: int,
    ) -> str:
        """
        Sends a small subset of vault nodes to the LLM and asks it to extract
        only the legal waivers, restrictions, and tactical facts relevant to
        the opportunity. Returns the RAW LLM response containing XML tags.
        """
        logger.info(
            "MAP phase | Chunk %d/%d (%d nodes)",
            chunk_index + 1,
            total_chunks,
            len(chunk_nodes),
        )

        # Build dense chunk context
        node_blocks: List[str] = []
        for node in chunk_nodes:
            node_blocks.append(
                f"--- NODE: {node.uid} ---\n"
                f"Summary: {node.summary}\n"
                f"Frontmatter: {json.dumps(node.front_matter, default=str)}\n"
                f"Content Preview:\n{node.raw_content[:1500]}\n"
            )

        chunk_context = "\n".join(node_blocks)

        system_prompt = (
            "You are a legal and tactical extraction engine. "
            "Your job is to read the provided vault nodes and extract ONLY the information "
            "relevant to the target opportunity. Do not summarize general facts. "
            "Focus strictly on: legal waivers, regulatory restrictions, licensing requirements, "
            "decision-maker names, contact coordinates, and channel integrity risks. "
            "You MUST format your entire response using XML tags as specified."
        )

        user_prompt = (
            f"## TARGET OPPORTUNITY\n"
            f"```json\n"
            f"{json.dumps(_sanitize_for_json(opportunity), indent=2, default=str)}\n"
            f"```\n\n"
            f"## VAULT NODES (Chunk {chunk_index + 1} of {total_chunks})\n"
            f"{chunk_context}\n\n"
            f"## EXTRACTION INSTRUCTIONS\n"
            f"Analyze the nodes above. Extract only facts, contacts, legal waivers, "
            f"and restrictions that directly affect the target opportunity.\n\n"
            f"Your response MUST be split into two sections wrapped in XML tags:\n\n"
            f"1. <analysis_summary>\n"
            f"   A dense bullet-point list of extracted facts, contacts, legal waivers, "
            f"   and restrictions. If a node is irrelevant, state 'IRRELEVANT'. "
            f"   Do not include pleasantries or markdown headers inside this tag.\n"
            f"   </analysis_summary>\n\n"
            f"2. <ui_lineage_trace>\n"
            f"   A valid JSON array containing objects with these exact fields: "
            f"   source_node, target_concept, relationship_type, extracted_fact, logic_justification. "
            f"   Each object represents a logical trace from a vault node to an opportunity concept. "
            f"   If no traces exist, output an empty array [].\n"
            f"   </ui_lineage_trace>\n\n"
            f"Do not include any text outside these two XML tags."
        )

        # Token guard for individual chunks
        if self.token_budget.estimate(system_prompt + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(system_prompt)
            user_prompt = user_prompt[:max_chars] + "\n[CHUNK TRUNCATED]"
            logger.warning("Chunk %d truncated to fit token budget.", chunk_index + 1)

        raw_response = self._call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=2048,
        )

        logger.info(
            "Chunk %d/%d raw response received (%d chars).",
            chunk_index + 1,
            total_chunks,
            len(raw_response),
        )
        return raw_response

    # ---------------------------------------------------------------------
    # REDUCE: Master synthesis
    # ---------------------------------------------------------------------
    def _reduce_synthesize(
        self,
        opportunity: Dict[str, Any],
        analysis_summaries: List[str],
    ) -> str:
        """
        Takes the condensed analysis summaries and synthesizes the final master
        tactical transaction roadmap with a companion UI thinking graph.
        """
        logger.info(
            "REDUCE phase | Synthesizing %d analysis summaries into master roadmap.",
            len(analysis_summaries),
        )

        consolidated = "\n\n--- CHUNK BOUNDARY ---\n\n".join(analysis_summaries)

        system_prompt = (
            "You are an elite B2B transaction architect and sovereign deal strategist. "
            "Your outputs are deterministic, highly analytical, and strictly formatted. "
            "You never omit required sections. You write in professional business English. "
            "You MUST format your entire response using XML tags as specified."
        )

        user_prompt = (
            "# ATIS EXECUTION LAYER — TACTICAL TRANSACTION ROADMAP\n\n"
            "## ISOLATED OPPORTUNITY JSON\n"
            "```json\n"
            f"{json.dumps(_sanitize_for_json(opportunity), indent=2, default=str)}\n"
            "```\n\n"
            "## CONSOLIDATED CHUNK INTELLIGENCE\n"
            "The following summaries were extracted from a chunked analysis of the internal vault:\n\n"
            f"{consolidated}\n\n"
            "---\n\n"
            "## STRICT OUTPUT INSTRUCTIONS\n\n"
            "Analyze the opportunity JSON and the consolidated chunk intelligence above. "
            "Produce a comprehensive tactical transaction roadmap. "
            "Your response MUST be split into two sections wrapped in XML tags:\n\n"
            "1. <final_roadmap>\n"
            "   The standard Markdown roadmap containing exactly the following three top-level sections, "
            "   using the exact headers provided below. Do not add introductory fluff or closing pleasantries.\n\n"
            "   ### ## THE TRANSACTION PERIMETER\n"
            "   Detail how the operator must position themselves as a vital intermediary. "
            "   Pinpoint exactly where Pre-MOU agreements, channel integrity protocols, "
            "   and proprietary data-gatekeeping must be initialized to protect the "
            "   transaction channel from being bypassed.\n\n"
            "   ### ## OPERATIONAL ROADMAP\n"
            "   A chronological, step-by-step milestone execution map formatted explicitly "
            "   using an XML-safe procedural timeline format (using clear step descriptions, "
            "   operational timing blocks like 'Days 1-5', and success verification parameters).\n\n"
            "   ### ## DIRECT ACTION MATRIX\n"
            "   A clean Markdown table mapping matching vault files directly to execution tasks. "
            "   Columns must read: `[Target Vault Node | Execution Role | "
            "   Strategic Leverage / Assets | Primary Contact & Coordinates]`.\n\n"
            "   If no vault matches exist, state that explicitly in the matrix and "
            "   recommend immediate network expansion actions.\n"
            "   </final_roadmap>\n\n"
            "2. <ui_thinking_graph>\n"
            "   A valid JSON object representing the pipeline topology. It MUST include:\n"
            "   - A `metrics` dictionary with these exact keys: "
            "     `total_vault_files_scanned` (integer), `nodes_extracted` (integer), "
            "     `map_chunks_processed` (integer), `estimated_manual_hours_saved` (number).\n"
            "   - A `convergence_flow` dictionary with these exact keys: "
            "     `tier_1_anchors` (array of strings), `tier_2_processing_chunks` (array of strings), "
            "     `tier_3_synthesis_logic` (string describing the synthesis approach).\n"
            "   Ensure the JSON is minified or pretty-printed, but valid and parseable.\n"
            "   </ui_thinking_graph>\n\n"
            "Do not include any text outside these two XML tags."
        )

        # Final token guard
        if self.token_budget.estimate(system_prompt + user_prompt) > self.token_budget.available_for_input:
            max_chars = int(self.token_budget.available_for_input * 3.2) - len(system_prompt)
            user_prompt = user_prompt[:max_chars] + "\n[FINAL PROMPT TRUNCATED]"
            logger.warning("Final reduce prompt truncated to fit token budget.")

        return self._call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.15,
            max_tokens=4096,
        )

    # ---------------------------------------------------------------------
    # Orchestrator: Map -> Reduce
    # ---------------------------------------------------------------------
    def generate_roadmap(
        self,
        opportunity: Dict[str, Any],
        vault_nodes: List[VaultNode],
    ) -> Dict[str, Any]:
        """
        Orchestrates the Map-Reduce pipeline:
          1. Split nodes into chunks of CHUNK_SIZE.
          2. MAP: Extract tactical intelligence per chunk (analysis_summary + ui_lineage_trace).
          3. REDUCE: Synthesize master roadmap from chunk summaries.
          
        Returns a dict with keys:
          - 'final_roadmap': str
          - 'ui_thinking_graph': str (raw JSON string)
          - 'compiled_lineage_traces': List[Dict[str, Any]]
        """
        compiled_lineage_traces: List[Dict[str, Any]] = []
        analysis_summaries: List[str] = []

        if not vault_nodes:
            logger.warning("No vault nodes provided; generating roadmap from opportunity only.")
            raw_reduce = self._reduce_synthesize(opportunity, ["NO VAULT NODES DISCOVERED."])
            roadmap_match = re.search(r"<final_roadmap>(.*?)</final_roadmap>", raw_reduce, re.DOTALL)
            graph_match = re.search(r"<ui_thinking_graph>(.*?)</ui_thinking_graph>", raw_reduce, re.DOTALL)
            return {
                "final_roadmap": roadmap_match.group(1).strip() if roadmap_match else raw_reduce,
                "ui_thinking_graph": graph_match.group(1).strip() if graph_match else "{}",
                "compiled_lineage_traces": compiled_lineage_traces,
            }

        # Split into chunks
        chunks: List[List[VaultNode]] = []
        for i in range(0, len(vault_nodes), CHUNK_SIZE):
            chunks.append(vault_nodes[i : i + CHUNK_SIZE])

        total_chunks = len(chunks)
        logger.info("Map-Reduce initialized: %d nodes -> %d chunks.", len(vault_nodes), total_chunks)

        # MAP phase
        for idx, chunk in enumerate(chunks):
            raw_response = self._map_chunk(opportunity, chunk, idx, total_chunks)
            
            # Extract <analysis_summary>
            summary_match = re.search(r"<analysis_summary>(.*?)</analysis_summary>", raw_response, re.DOTALL)
            if summary_match:
                analysis_summaries.append(summary_match.group(1).strip())
            else:
                analysis_summaries.append(raw_response.strip())
                logger.warning("Chunk %d: <analysis_summary> tag not found; using raw response.", idx + 1)
            
            # Extract <ui_lineage_trace>
            trace_match = re.search(r"<ui_lineage_trace>(.*?)</ui_lineage_trace>", raw_response, re.DOTALL)
            if trace_match:
                trace_json_str = trace_match.group(1).strip()
                try:
                    trace_array = json.loads(trace_json_str)
                    if isinstance(trace_array, list):
                        compiled_lineage_traces.extend(trace_array)
                        logger.info("Chunk %d: extracted %d lineage traces.", idx + 1, len(trace_array))
                    else:
                        logger.warning("Chunk %d: lineage trace is not a JSON array.", idx + 1)
                except json.JSONDecodeError as exc:
                    logger.warning("Chunk %d: failed to parse lineage trace JSON: %s", idx + 1, exc)
            else:
                logger.warning("Chunk %d: <ui_lineage_trace> tag not found.", idx + 1)
            
            if idx < total_chunks - 1:
                time.sleep(INTER_CHUNK_DELAY_SECONDS)

        # REDUCE phase
        raw_reduce = self._reduce_synthesize(opportunity, analysis_summaries)
        
        # Extract <final_roadmap>
        roadmap_match = re.search(r"<final_roadmap>(.*?)</final_roadmap>", raw_reduce, re.DOTALL)
        final_roadmap = roadmap_match.group(1).strip() if roadmap_match else raw_reduce
        
        # Extract <ui_thinking_graph>
        graph_match = re.search(r"<ui_thinking_graph>(.*?)</ui_thinking_graph>", raw_reduce, re.DOTALL)
        ui_thinking_graph = graph_match.group(1).strip() if graph_match else "{}"

        logger.info(
            "Master roadmap generated (%d chars). Roadmap: %d chars, Graph: %d chars.",
            len(raw_reduce),
            len(final_roadmap),
            len(ui_thinking_graph),
        )
        return {
            "final_roadmap": final_roadmap,
            "ui_thinking_graph": ui_thinking_graph,
            "compiled_lineage_traces": compiled_lineage_traces,
        }


# =============================================================================
# Output persistence
# =============================================================================
def persist_outputs(
    opportunity_id: str,
    final_roadmap: str,
    ui_thinking_graph_raw: str,
    compiled_lineage_traces: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    """
    Twin-file persistence:
      1. Markdown execution roadmap.
      2. Structured JSON companion for React Flow UI.
    """
    output_dir = Path(os.getenv("OUTPUT_DIR", "./output/roadmaps"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Markdown roadmap
    md_path = output_dir / f"execution_{opportunity_id}.md"
    header = f"""# Execution Roadmap — {opportunity_id}

Generated by ATIS_Execute.py  
Opportunity ID: `{opportunity_id}`

---

"""
    md_path.write_text(header + final_roadmap, encoding="utf-8")
    logger.info("Roadmap persisted to: %s", md_path)

    # 2. JSON reasoning companion
    json_path = output_dir / f"reasoning_{opportunity_id}.json"
    try:
        thinking_graph = json.loads(ui_thinking_graph_raw)
        if not isinstance(thinking_graph, dict):
            raise ValueError("ui_thinking_graph root is not a JSON object.")
        
        # Inject granular lineage traces
        thinking_graph["compiled_lineage_traces"] = compiled_lineage_traces
        
        json_path.write_text(
            json.dumps(thinking_graph, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Reasoning JSON persisted to: %s", json_path)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse ui_thinking_graph JSON: %s", exc)
        
        # Write raw unparsed string to debug log
        debug_path = output_dir / f"reasoning_{opportunity_id}_DEBUG.log"
        debug_path.write_text(
            f"JSON PARSE ERROR: {exc}\n\nRAW UI_THINKING_GRAPH:\n{ui_thinking_graph_raw}",
            encoding="utf-8",
        )
        logger.warning("Raw graph string written to debug log: %s", debug_path)
        
        # Prevent crash by writing a fallback JSON with traces and error metadata
        fallback = {
            "parse_error": str(exc),
            "raw_graph_available_in_debug_log": str(debug_path),
            "compiled_lineage_traces": compiled_lineage_traces,
            "metrics": {},
            "convergence_flow": {},
        }
        json_path.write_text(
            json.dumps(fallback, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Fallback reasoning JSON persisted to: %s", json_path)

    return md_path, json_path


# =============================================================================
# Main orchestration
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATIS Execution Layer — Tactical Transaction Roadmap Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dashboard_path",
        required=True,
        help="Absolute path to the generated ATIS JSON dashboard file.",
    )
    parser.add_argument(
        "--opportunity_id",
        required=True,
        help="Target opportunity to execute (e.g., OPP-001).",
    )
    parser.add_argument(
        "--vault_path",
        default=r"C:\Users\tmaki\Documents\AKSOS\ATIS\Vault",
        help="Absolute path to the Obsidian vault root. (default: ATIS Vault)",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="Optional Cerebras API key. Overrides the hardcoded constant and env var.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dashboard_path = Path(args.dashboard_path).resolve()
    vault_path = Path(args.vault_path).resolve()

    # 1. Load opportunity
    try:
        opportunity = load_opportunity(dashboard_path, args.opportunity_id)
    except Exception as exc:
        logger.error("Failed to load opportunity: %s", exc)
        sys.exit(1)

    # 2. Index vault
    vault_mgr = ObsidianVaultManager(vault_path)
    try:
        vault_mgr.build_index()
    except Exception as exc:
        logger.error("Vault indexing failed: %s", exc)
        sys.exit(1)

    # 3. Expand keywords and search (Anchor + Crawl)
    seed_terms = expand_keywords(opportunity)
    try:
        matches = vault_mgr.search(opportunity, seed_terms)
    except Exception as exc:
        logger.error("Vault search failed: %s", exc)
        sys.exit(1)

    # 4. LLM generation — Map-Reduce
    engine = CerebrasExecutionEngine(api_key=args.api_key if args.api_key else None)
    try:
        result = engine.generate_roadmap(opportunity, matches)
    except Exception as exc:
        logger.error("Roadmap generation failed: %s", exc)
        sys.exit(1)

    # 5. Persist twin files
    try:
        md_path, json_path = persist_outputs(
            args.opportunity_id,
            result["final_roadmap"],
            result["ui_thinking_graph"],
            result["compiled_lineage_traces"],
        )
    except Exception as exc:
        logger.error("Failed to persist outputs: %s", exc)
        sys.exit(1)

    print(f"\nSUCCESS: Execution roadmap written to:\n  {md_path}")
    print(f"SUCCESS: Reasoning JSON written to:\n  {json_path}")


# =============================================================================
# Web entry point
# =============================================================================
def run_execute_pipeline(dashboard_json: Dict[str, Any], opportunity_id: str) -> Dict[str, Any]:
    """
    Web-compatible entry point. Accepts dashboard dict and opportunity ID.
    Returns dict with final_roadmap, ui_thinking_graph, and compiled_lineage_traces.
    """
    vault_path = Path(os.getenv("VAULT_PATH", "./vault"))
    vault_mgr = ObsidianVaultManager(vault_path)
    vault_mgr.build_index()

    opportunity = dashboard_json
    seed_terms = expand_keywords(opportunity)

    matches = vault_mgr.search(opportunity, seed_terms)

    engine = CerebrasExecutionEngine()
    result = engine.generate_roadmap(opportunity, matches)

    # Persist
    md_path, json_path = persist_outputs(
        opportunity_id,
        result["final_roadmap"],
        result["ui_thinking_graph"],
        result["compiled_lineage_traces"],
    )

    return {
        "opportunity_id": opportunity_id,
        "final_roadmap": result["final_roadmap"],
        "ui_thinking_graph": result["ui_thinking_graph"],
        "compiled_lineage_traces": result["compiled_lineage_traces"],
        "files_written": {
            "roadmap_md": str(md_path),
            "reasoning_json": str(json_path),
        }
    }


if __name__ == "__main__":
    main()