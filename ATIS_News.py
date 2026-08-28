#!/usr/bin/env python3
"""
ATIS Constraint-Solving and Market Equilibrium Engine
======================================================

Africa Trade & Intelligence System (ATIS) — Production Orchestration Script.

This module implements a decoupled, state-passing pipeline that:
    1. Extracts economic entities from a news article via the configured LLM.
    2. Reconciles extracted entities against a local Obsidian markdown vault,
       using canonical fuzzy matching and bidirectional backlink crawling.
    3. Retrieves perspective-side actors, capabilities, and cross-border bridges.
    4. Performs macroeconomic constraint-solving analysis via the configured LLM.
    5. Formats the analysis into a structured commercial-intelligence dashboard.
    6. Deterministically validates every opportunity against vault evidence.
    7. Persists the final dashboard JSON to a local `./dashboards/` directory.

Determinism Features:
  - temperature=0.0, seed=42 on all LLM calls
  - sorted vault iterations for stable ordering
  - AnalysisCache for disk-based result caching
  - KnowledgeState for vault versioning
  - compute_analysis_fingerprint for stable identity
  - compute_opportunity_identity for stable opportunity IDs

Constraints:
  - Python 3.10+
  - Strict 60,000-token ceiling per API request (aggressive truncation).
  - Zero placeholders; fully operational.

Environment:
    - LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL must be configured.
  - Article path provided as the first CLI argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_client import LLMClient, get_client
from atis_context import (
    PerspectiveContext,
    validate_opportunity,
    KnowledgeState,
    AnalysisCache,
    compute_analysis_fingerprint,
    compute_opportunity_identity,
    ANALYSIS_VERSION,
    SCHEMA_VERSION,
)


# --------------------------------------------------------------------------- #
# System Prompts (Embedded as Constants)
# --------------------------------------------------------------------------- #
PROMPT_STAGE_1_EXTRACTOR: str = (
    "You are the Entity Extraction Module for an economic intelligence pipeline. "
    "Your sole objective is to extract entities from the provided text and classify them into a strict schema. "
    "Do not analyze or interpret the text.\n"
    "CLASSIFICATION SCHEMA:\n"
    "- [MINING_REFINERY]: Processing plants, smelters, concentrators.\n"
    "- [PRIVATE_CONGLOMERATE]: Mining companies, logistics firms, tech providers.\n"
    "- [GOVERNMENT_AGENCY]: Regulatory bodies, state-owned enterprises, councils.\n"
    "- [GOVERNMENT_MINISTRY]: Sovereign ministries.\n"
    "- [ACADEMIC_INSTITUTION]: Universities, polytechnics, research labs.\n"
    "- [INFRASTRUCTURE_NODE]: Power plants, dams, railways, ports, specific laboratories.\n"
    "- [COMMODITY]: Specific raw or processed materials (e.g., Lithium Ore, Sulfuric Acid).\n"
    "- [POLICY_FRAMEWORK]: Laws, bans, official state initiatives.\n\n"
    "OUTPUT INSTRUCTIONS:\n"
    "Output ONLY valid raw JSON. Do not wrap the response in markdown blocks (```json).\n"
    "JSON SCHEMA:\n"
    "{\n"
    '  "entities": [\n'
    '    {"name": "Exact Name", "class": "[SCHEMA_CLASS]", "context": "Sentence explaining action."}\n'
    "  ],\n"
    '  "core_event": "String summarizing the article main event.",\n'
    '  "source_country": "Country where the event occurred (infer from text)",\n'
    '  "event_country": "Country where the underlying development occurred (infer from text)"\n'
    "}"
)

PROMPT_STAGE_2_SOLVER: str = (
    "You are the ATIS Equilibrium and Constraint Engine. Your objective is to act as a macroeconomic constraint solver. "
    "You will be provided with a [NEW EVENT], a [PERSPECTIVE CONTEXT] containing evidenced actors and capabilities from the perspective country, "
    "and a [CROSS-BORDER BRIDGE CONTEXT] showing actual vault-documented relationships between the perspective country and the source event country. "
    "You must calculate the systemic shifts and unfulfilled requirements caused by the event, BUT you may only propose opportunities that are grounded in the provided perspective-side evidence.\n\n"
    "CRITICAL RULES:\n"
    "1. You MUST select perspective_actor from the [PERSPECTIVE ACTOR REGISTRY] list below. Do not invent actors.\n"
    "2. You MUST select perspective_capability from the capabilities listed for that actor. Do not invent capabilities.\n"
    "3. You MUST select pathway from the [CROSS-BORDER BRIDGE CONTEXT] or from the enumerated list: export, procurement, supplier relationship, regional tender, joint venture, partnership, investment, financing, logistics, professional services, technology transfer, regional infrastructure, power trade, regulatory arbitrage, market entry. The pathway must be supported by evidence.\n"
    "4. You MUST set opportunity_country to the actual country where the commercial opportunity exists — this may be the source country, the perspective country, or a third country. Do NOT default it to the perspective country.\n"
    "5. If no perspective-side actor can respond to the event, state 'NO VALID OPPORTUNITY' and explain the gap.\n"
    "6. Distinguish local source-country opportunities from perspective-country opportunities. A source-country event does NOT automatically create a perspective-country opportunity.\n\n"
    "Follow this exact reasoning sequence in your markdown output:\n"
    "## 1. THE EQUILIBRIUM DELTA: What specific market equilibrium was broken by this event?\n"
    "## 2. CONSTRAINT MATRIX: What new capabilities are now required? What existing capabilities are now insufficient?\n"
    "## 3. PERSPECTIVE-SIDE CAPABILITY AUDIT: Which perspective-country actors have evidenced capabilities that could address the constraints?\n"
    "## 4. CROSS-BORDER BRIDGE AUDIT: Which evidenced pathways connect perspective actors to the source event?\n"
    "## 5. ECONOMIC FLOW: For validated cross-border opportunities, identify who pays, who benefits, and capital flow.\n"
    "## 6. OPPORTUNITY CASCADE: Detail ONLY perspective-validated Primary, Secondary, and Tertiary opportunities. If none, say NONE."
)

PROMPT_STAGE_3_FORMATTER: str = (
    "You are a strict data serialization module. Your objective is to take the provided macroeconomic constraint analysis "
    "and format it into a structured JSON payload for a commercial intelligence dashboard.\n"
    "OUTPUT INSTRUCTIONS:\n"
    "Output ONLY valid raw JSON. Do not wrap the response in markdown blocks (```json). "
    "Calculate urgency_score and feasibility_score on a scale of 1.0 to 10.0.\n"
    "JSON SCHEMA:\n"
    "{\n"
    '  "intelligence_id": "ATIS-INT-GENERIC",\n'
    '  "trigger_event": "String",\n'
    '  "market_equilibrium_shift": "String",\n'
    '  "source_country": "String",\n'
    '  "event_country": "String",\n'
    '  "opportunities": [\n'
    "    {\n"
    '      "opportunity_id": "OPP-001",\n'
    '      "title": "String",\n'
    '      "type": "String",\n'
    '      "perspective_country": "String", "perspective_country_code": "ISO-2",\n'
    '      "source_country": "String", "event_country": "String", "opportunity_country": "String",\n'
    '      "cross_border": true, "cross_border_countries": ["String"],\n'
    '      "perspective_actor": "MUST be from the PERSPECTIVE ACTOR REGISTRY",\n'
    '      "perspective_capability": "MUST be a capability listed for that actor",\n'
    '      "pathway": "MUST be from the enumerated list and supported by evidence",\n'
    '      "source_nodes": ["Exact vault node IDs"],\n'
    '      "urgency_score": Float,\n'
    '      "feasibility_score": Float,\n'
    '      "required_missing_nodes": ["String"],\n'
    '      "capital_flow": {"beneficiary": "String", "likely_funder": "String"},\n'
    '      "justification": "One precise sentence explaining the structural gap and the evidenced pathway."\n'
    "    }\n"
    "  ]\n"
    "}"
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
VAULT_DIR: Path = Path(os.getenv("ATIS_VAULT_DIR", "./vault"))
DASHBOARDS_DIR: Path = Path("./dashboards")
MAX_TOKENS_PER_REQUEST: int = 60_000
RESPONSE_RESERVE: int = 8_000
SAFETY_BUFFER: int = 1_000


# --------------------------------------------------------------------------- #
# Logging Setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger("atis_engine")


# --------------------------------------------------------------------------- #
# Token Budget Manager
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TokenBudget:
    """Aggressive token budget manager enforcing the 60k ceiling."""

    max_tokens: int = MAX_TOKENS_PER_REQUEST
    response_reserve: int = RESPONSE_RESERVE
    safety_buffer: int = SAFETY_BUFFER

    @property
    def available_for_input(self) -> int:
        """Tokens available for system + user prompts after reserves."""
        return self.max_tokens - self.response_reserve - self.safety_buffer

    @staticmethod
    def estimate(text: str) -> int:
        """
        Conservative token estimator.
        Assumes ~3.2 characters per token for mixed English/technical text.
        """
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
        """
        Truncate article and graph context so that the combined input
        (system + article + graph) fits within the token budget.
        Truncation is applied to graph context first, then article text.
        """
        total_estimated = self.estimate(system_prompt + article_text + graph_context)
        if total_estimated <= self.available_for_input:
            return article_text, graph_context

        # Convert available token budget back to a rough character budget
        available_chars = int(self.available_for_input * 3.2) - len(system_prompt)
        if available_chars <= 0:
            raise RuntimeError("System prompt alone exceeds the token budget.")

        # Reserve minimum space for the article
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

        # Verify
        revised_estimate = self.estimate(
            system_prompt + truncated_article + truncated_graph
        )
        logger.info(
            "Revised payload estimate after truncation: %d tokens (budget: %d)",
            revised_estimate,
            self.available_for_input,
        )
        return truncated_article, truncated_graph


# --------------------------------------------------------------------------- #
# Safe JSON Loader — Brace-Balancing Engine
# --------------------------------------------------------------------------- #
def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^~~~(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?~~~\s*$', '', text, flags=re.IGNORECASE)
    return text.strip()


def _extract_balanced_json(text: str) -> str:
    """
    Extract the largest balanced JSON object or array from raw text.
    Uses character-level brace/bracket tracking with full string/escape awareness.
    This avoids the greedy-regex bug that grabs from first '{' to last '}'.
    """
    in_string = False
    escape_next = False
    brace_stack: List[str] = []
    start_idx = -1
    candidates: List[str] = []

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char in '{[':
            if not brace_stack:
                start_idx = i
            brace_stack.append(char)
        elif char in '}]':
            if not brace_stack:
                continue
            opener = brace_stack.pop()
            if (opener == '{' and char != '}') or (opener == '[' and char != ']'):
                brace_stack = []
                start_idx = -1
                continue
            if not brace_stack and start_idx != -1:
                candidates.append(text[start_idx : i + 1])

    if not candidates:
        return ""
    return max(candidates, key=len)


def _fix_common_json_errors(text: str) -> str:
    """Apply safe heuristics to fix common LLM JSON syntax errors."""
    text = re.sub(r',(\s*[\}\]])', r'\1', text)
    text = re.sub(r',+(\s*)', r',', text)
    return text


def safe_json_loads(raw_text: str, stage_name: str) -> Dict[str, Any]:
    """
    Safely parse a JSON string with deterministic, layered fallback strategies.
    Strips markdown fences, extracts balanced structures, fixes trailing commas,
    and raises a clear RuntimeError on absolute failure.
    """
    if not raw_text or not raw_text.strip():
        raise RuntimeError(f"Empty response received from {stage_name}")

    original = raw_text.strip()
    logger.debug("safe_json_loads called for %s (len=%d)", stage_name, len(original))

    cleaned = _strip_markdown_fences(original)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as direct_err:
        logger.warning("Direct JSON parse failed in %s: %s", stage_name, direct_err)

    balanced = _extract_balanced_json(cleaned)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError as bal_err:
            logger.warning("Balanced JSON extraction failed in %s: %s", stage_name, bal_err)

        fixed = _fix_common_json_errors(balanced)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as fix_err:
            logger.warning("Fixed balanced JSON failed in %s: %s", stage_name, fix_err)

    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as regex_err:
            logger.warning("Non-greedy regex JSON recovery failed in %s: %s", stage_name, regex_err)

    heuristic = re.sub(r',(\s*[\}\]])', r'\1', cleaned)
    try:
        return json.loads(heuristic)
    except json.JSONDecodeError as heuristic_err:
        logger.error("All JSON parsing strategies exhausted for %s.", stage_name)
        logger.error("Raw response excerpt (first 1000 chars):\n%s", original[:1000])
        raise RuntimeError(
            f"Failed to parse JSON response from {stage_name} after all recovery strategies."
        ) from heuristic_err


# --------------------------------------------------------------------------- #
# Obsidian Vault Manager (Graph & Inbound Backlink Engine)
# --------------------------------------------------------------------------- #
class ObsidianVaultManager:
    """
    Handles vault indexing, fuzzy filename matching, bidirectional link crawling
    (outbound + inbound backlinks), and perspective-side retrieval for the ATIS
    graph layer.
    """

    def __init__(self, vault_dir: Path = VAULT_DIR) -> None:
        self.vault_dir: Path = vault_dir
        self._ensure_directories()

        # Core Graph Indexing Maps
        self.file_map: Dict[str, str] = {}  # canonical_name -> actual_file_stem
        self.backlink_map: Dict[str, Set[str]] = {}  # canonical_name -> set of actual_file_stems linking to it
        self.node_metadata: Dict[str, Dict[str, Any]] = {}  # canonical_name -> {country, type, summary}

        # Build index immediately on startup
        self._index_vault()

    # -- Directory hygiene --------------------------------------------------- #
    def _ensure_directories(self) -> None:
        """Ensure vault and dashboard directories exist cleanly."""
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        DASHBOARDS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Vault directory verified: %s", self.vault_dir.resolve())

    # -- Canonicalization Engine (Fuzzy Matching) ---------------------------- #
    @staticmethod
    def _canonicalize(name: str) -> str:
        """
        Convert any string/filename to a strict alphanumeric lowercase token.
        Removes spaces, underscores, hyphens, and casing variations to eliminate matching bugs.
        e.g., "Lithium carbonate", "lithium_carbonate", and "LITHIUM-CARBONATE" all become "lithiumcarbonate"
        """
        return re.sub(r"[^a-zA-Z0-9]", "", str(name or "")).lower()

    # -- Institutional Memory Vault Indexer ---------------------------------- #
    def _index_vault(self) -> None:
        """
        Performs a full pass over the vault to catalog existing nodes and map
        bidirectional graph relationships (backlinks) before execution.
        Also extracts country metadata from frontmatter and file paths.
        """
        self.file_map.clear()
        self.backlink_map.clear()
        self.node_metadata.clear()

        # DETERMINISTIC: sort md files by path string
        md_files = sorted(self.vault_dir.rglob("*.md"), key=lambda p: str(p))
        logger.info("Indexing %d existing vault files for graph matching...", len(md_files))

        for file_path in md_files:
            actual_stem = file_path.stem
            canonical_stem = self._canonicalize(actual_stem)

            # Map canonical name to actual file name on disk
            self.file_map[canonical_stem] = actual_stem

            try:
                content = file_path.read_text(encoding="utf-8")
                # Extract frontmatter for country metadata
                country = ""
                node_type = ""
                fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
                if fm_match:
                    try:
                        import yaml
                        front = yaml.safe_load(fm_match.group(1)) or {}
                        country = front.get("country", "") or front.get("location", "")
                        node_type = front.get("node_type", "") or front.get("type", "") or front.get("entity_type", "")
                    except Exception:
                        pass

                # Infer country from path if not in frontmatter
                if not country:
                    path_parts = [p.lower() for p in file_path.relative_to(self.vault_dir).parts]
                    for part in path_parts:
                        if part in ("zimbabwe", "zambia", "south africa", "botswana", "kenya", "china", "germany", "switzerland", "united kingdom", "united states of america"):
                            country = part.title()
                            break

                # Extract summary
                summary = ""
                lines = [l.strip() for l in content.split("\n") if l.strip()]
                for line in lines:
                    if not line.startswith("---") and not line.startswith("#"):
                        summary = line[:200]
                        break

                self.node_metadata[canonical_stem] = {
                    "country": country,
                    "type": node_type,
                    "summary": summary,
                    "path": str(file_path.relative_to(self.vault_dir)),
                }

                # Parse Obsidian style [[WikiLinks]] or [[WikiLinks|Display Name]]
                links = re.findall(r"\[\[(.*?)\]\]", content)
                for link in links:
                    link_target = link.split("|")[0].strip()
                    if not link_target:
                        continue

                    canonical_target = self._canonicalize(link_target)
                    if canonical_target not in self.backlink_map:
                        self.backlink_map[canonical_target] = set()

                    # Register that this file links TO the target
                    self.backlink_map[canonical_target].add(actual_stem)
            except Exception as exc:
                logger.error("Failed to parse backlinks for %s: %s", actual_stem, exc)

    def entity_exists(self, entity_name: str) -> bool:
        return self._canonicalize(entity_name) in self.file_map

    def read_entity(self, entity_name: str) -> str:
        canonical = self._canonicalize(entity_name)
        actual_name = self.file_map.get(canonical)
        if actual_name:
            # Use stored relative path to handle nested subdirectories
            meta = self.node_metadata.get(canonical, {})
            rel_path = meta.get("path", f"{actual_name}.md")
            path = self.vault_dir / rel_path
            if path.exists():
                return path.read_text(encoding="utf-8")
            # Fallback: search by rglob if stored path is stale
            for found in self.vault_dir.rglob(f"{actual_name}.md"):
                if found.exists():
                    return found.read_text(encoding="utf-8")
            logger.warning("Entity file not found: %s (looked at %s and via rglob)", actual_name, path)
        return ""

    def write_entity(self, entity_name: str, content: str) -> None:
        canonical = self._canonicalize(entity_name)
        # Re-use existing name structure if it exists; otherwise use the raw name passed
        actual_name = self.file_map.get(canonical, entity_name.strip())
        path = self.vault_dir / f"{actual_name}.md"
        path.write_text(content, encoding="utf-8")

        # Keep internal indexes hot
        self.file_map[canonical] = actual_name

    # -- Bidirectional Link Crawler ----------------------------------------- #
    def crawl_node_network(self, entity_name: str, visited: Optional[set] = None) -> str:
        """
        Traverses both forward wiki-links and inbound backlinks for an entity,
        bundling structural context so the LLM spots network vulnerabilities.
        """
        if visited is None:
            visited = set()

        canonical = self._canonicalize(entity_name)
        actual_name = self.file_map.get(canonical, entity_name)

        if actual_name in visited:
            return ""
        visited.add(actual_name)

        content = self.read_entity(actual_name)
        if not content:
            return ""

        bundle_parts: List[str] = [
            f"--- CORE NODE: {actual_name} ---\n{content}"
        ]

        # 1. OUTBOUND CONTEXT (What does this note point to?)
        links = re.findall(r"\[\[(.*?)\]\]", content)
        # DETERMINISTIC: sort links before processing
        for link in sorted(links):
            link_clean = link.split("|")[0].strip()
            link_canonical = self._canonicalize(link_clean)
            if link_canonical in self.file_map and link_clean not in visited:
                target_actual = self.file_map[link_canonical]
                # Inline bundle linked file data if it hasn't been crawled yet
                linked_content = self.read_entity(target_actual)
                bundle_parts.append(
                    f"\n--- OUTBOUND LINKED CONTEXT: {target_actual} ---\n{linked_content}"
                )
                visited.add(target_actual)

        # 2. INBOUND CONTEXT / BACKLINKS (What existing assets point to this node?)
        inbound_stems = self.backlink_map.get(canonical, set())
        if inbound_stems:
            bundle_parts.append(
                f"\n--- INBOUND BACKLINKS (EXISTING VAULT RELATIONSHIPS) ---"
            )
            bundle_parts.append(
                f"The following existing nodes in your database are explicitly dependent on or linked to [[{actual_name}]]:"
            )
            # DETERMINISTIC: sort inbound stems
            for inbound in sorted(inbound_stems):
                # Grab a snippet or properties from the backlinked node to maximize context density
                inbound_content = self.read_entity(inbound)
                fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", inbound_content, re.DOTALL)
                meta = f" [{fm_match.group(1).strip().replace(chr(10), ' | ')}]" if fm_match else ""
                bundle_parts.append(f"- [[{inbound}]]{meta}")

        return "\n".join(bundle_parts)

    # -- Perspective-Side Retrieval (NEW) ----------------------------------- #
    def get_perspective_nodes(self, perspective: PerspectiveContext) -> List[Dict[str, Any]]:
        """
        Retrieve all vault nodes that belong to the perspective country.
        Returns a list of dicts with node_id, country, type, summary, content.
        """
        perspective_country_norm = perspective.country.lower()
        results: List[Dict[str, Any]] = []

        # DETERMINISTIC: iterate sorted by canonical stem
        for canonical_stem in sorted(self.file_map.keys()):
            actual_stem = self.file_map[canonical_stem]
            meta = self.node_metadata.get(canonical_stem, {})
            node_country = (meta.get("country") or "").lower()
            if node_country == perspective_country_norm:
                content = self.read_entity(actual_stem)
                results.append({
                    "node_id": actual_stem,
                    "country": meta.get("country", ""),
                    "type": meta.get("type", ""),
                    "summary": meta.get("summary", ""),
                    "content": content[:1500],  # Truncated for token efficiency
                })

        logger.info("Retrieved %d perspective-side nodes for %s", len(results), perspective.country)
        return results

    def get_cross_border_bridges(self, perspective: PerspectiveContext, source_country: str) -> List[Dict[str, Any]]:
        """
        Find nodes in the perspective country that have links to nodes in the source country,
        or nodes in the source country that have links to perspective-country nodes.
        Returns bridge dicts with from_node, to_node, relationship_type.
        """
        perspective_norm = perspective.country.lower()
        source_norm = source_country.lower()
        bridges: List[Dict[str, Any]] = []

        # DETERMINISTIC: iterate sorted by canonical stem
        for canonical_stem in sorted(self.file_map.keys()):
            actual_stem = self.file_map[canonical_stem]
            meta = self.node_metadata.get(canonical_stem, {})
            node_country = (meta.get("country") or "").lower()

            # Only process nodes from either perspective or source country
            if node_country not in (perspective_norm, source_norm):
                continue

            content = self.read_entity(actual_stem)
            links = re.findall(r"\[\[(.*?)\]\]", content)

            for link in links:
                link_clean = link.split("|")[0].strip()
                link_canonical = self._canonicalize(link_clean)
                if link_canonical not in self.file_map:
                    continue

                link_meta = self.node_metadata.get(link_canonical, {})
                link_country = (link_meta.get("country") or "").lower()

                # Check if this link crosses between perspective and source countries
                if node_country == perspective_norm and link_country == source_norm:
                    bridges.append({
                        "from_node": actual_stem,
                        "from_country": perspective.country,
                        "to_node": self.file_map[link_canonical],
                        "to_country": source_country,
                        "relationship_type": "outbound_link",
                    })
                elif node_country == source_norm and link_country == perspective_norm:
                    bridges.append({
                        "from_node": actual_stem,
                        "from_country": source_country,
                        "to_node": self.file_map[link_canonical],
                        "to_country": perspective.country,
                        "relationship_type": "outbound_link",
                    })

        # Also check backlinks — DETERMINISTIC: sort keys
        for canonical_stem in sorted(self.file_map.keys()):
            actual_stem = self.file_map[canonical_stem]
            meta = self.node_metadata.get(canonical_stem, {})
            node_country = (meta.get("country") or "").lower()
            if node_country != perspective_norm:
                continue

            inbound_stems = self.backlink_map.get(canonical_stem, set())
            for inbound in sorted(inbound_stems):
                inbound_canonical = self._canonicalize(inbound)
                inbound_meta = self.node_metadata.get(inbound_canonical, {})
                inbound_country = (inbound_meta.get("country") or "").lower()
                if inbound_country == source_norm:
                    bridges.append({
                        "from_node": inbound,
                        "from_country": source_country,
                        "to_node": actual_stem,
                        "to_country": perspective.country,
                        "relationship_type": "backlink",
                    })

        logger.info("Found %d cross-border bridges between %s and %s", len(bridges), perspective.country, source_country)
        return bridges

    # -- Global Database Indexer -------------------------------------------- #
    def build_global_database_context(self, explicit_names: set[str]) -> str:
        """
        Builds a lightweight structural registry of all remaining nodes in the database
        to ensure background assets are available for pattern matching.
        """
        global_parts: List[str] = []
        canonical_explicits = {self._canonicalize(name) for name in explicit_names}

        # DETERMINISTIC: iterate sorted by canonical stem
        for canonical_stem in sorted(self.file_map.keys()):
            if canonical_stem in canonical_explicits:
                continue

            actual_stem = self.file_map[canonical_stem]
            meta = self.node_metadata.get(canonical_stem, {})
            summary = meta.get("summary", "")

            # Verify file exists before including (handles stale index entries)
            rel_path = meta.get("path", f"{actual_stem}.md")
            full_path = self.vault_dir / rel_path
            if not full_path.exists():
                # Try rglob fallback
                found = list(self.vault_dir.rglob(f"{actual_stem}.md"))
                if not found:
                    logger.debug("Skipping stale index entry: %s", actual_stem)
                    continue
                # Update stored path
                meta["path"] = str(found[0].relative_to(self.vault_dir))

            metadata_str = f"- **{actual_stem}**"
            if meta.get("country"):
                metadata_str += f" [country: {meta['country']}]"
            if meta.get("type"):
                metadata_str += f" [type: {meta['type']}]"
            if summary:
                metadata_str += f" -> Context: {summary}"

            global_parts.append(metadata_str)

        if not global_parts:
            return "No additional background nodes detected."
        return "\n".join(global_parts)

    # -- Graph Context Builder ---------------------------------------------- #
    def build_graph_context(self, entities: List[Dict[str, str]], perspective: PerspectiveContext) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process explicit entities using the new fuzzy-matching and backlink injection engine.
        Also retrieves perspective-side nodes and cross-border bridges.
        Returns: (combined_context, perspective_nodes, cross_border_bridges)
        """
        context_parts: List[str] = []
        handled_canonical_names: set[str] = set()

        for entity in entities:
            name = entity.get("name", "").strip()
            classification = entity.get("class", "").strip()
            if not name:
                continue

            canonical = self._canonicalize(name)
            handled_canonical_names.add(canonical)

            if self.entity_exists(name):
                # Pull forward links AND inbound backlinks
                node_context = self.crawl_node_network(name)
                context_parts.append(
                    f"\n=== MATCHED EXISTING NODE: {self.file_map[canonical]} ===\n{node_context}"
                )
            else:
                # Shadow node provisioning REMOVED per requirements
                context_parts.append(
                    f"\n=== UNMATCHED ENTITY: {name} ===\n"
                    f"Entity not found in vault. No shadow node created."
                )

        # Compile the remaining vault assets into the tracking registry
        logger.info("Compiling global landscape index...")
        global_landscape = self.build_global_database_context(handled_canonical_names)

        context_parts.append(
            f"\n=== GLOBAL SYSTEM GRAPH DATABASE MATRIX ===\n"
            f"Use the following existing database structure to cross-reference constraints. "
            f"Pay special attention to unmentioned downstream assets that are bound by backlinks to the primary constraints:\n"
            f"{global_landscape}"
        )

        source_context = "\n".join(context_parts)

        # Retrieve perspective-side nodes
        perspective_nodes = self.get_perspective_nodes(perspective)

        # Determine source country from entities (best effort)
        source_country = ""
        for entity in entities:
            ctx = entity.get("context", "").lower()
            if "zambia" in ctx:
                source_country = "Zambia"
                break
            elif "zimbabwe" in ctx:
                source_country = "Zimbabwe"
                break

        # If we couldn't infer from entities, try to find any non-perspective country in the source context
        if not source_country:
            source_country = perspective.country  # fallback for domestic analysis

        # Retrieve cross-border bridges
        cross_border_bridges = []
        if source_country.lower() != perspective.country.lower():
            cross_border_bridges = self.get_cross_border_bridges(perspective, source_country)

        # Build perspective context block
        perspective_blocks: List[str] = []
        perspective_blocks.append(f"\n=== PERSPECTIVE ACTOR REGISTRY ({perspective.country}) ===")
        # DETERMINISTIC: sort perspective nodes by node_id before capping
        for pn in sorted(perspective_nodes, key=lambda x: x["node_id"])[:30]:  # Cap for token budget
            perspective_blocks.append(
                f"- {pn['node_id']} | type: {pn['type']} | summary: {pn['summary'][:100]}"
            )

        if cross_border_bridges:
            perspective_blocks.append(f"\n=== CROSS-BORDER BRIDGE CONTEXT ({perspective.country} ↔ {source_country}) ===")
            # DETERMINISTIC: sort bridges before capping
            for bridge in sorted(cross_border_bridges, key=lambda b: (b["from_node"], b["to_node"]))[:20]:  # Cap for token budget
                perspective_blocks.append(
                    f"- {bridge['from_node']} ({bridge['from_country']}) → {bridge['to_node']} ({bridge['to_country']}) via {bridge['relationship_type']}"
                )
        else:
            perspective_blocks.append(f"\n=== CROSS-BORDER BRIDGE CONTEXT ===\nNo evidenced cross-border relationships found between {perspective.country} and {source_country}.")

        combined_context = source_context + "\n" + "\n".join(perspective_blocks)

        return combined_context, perspective_nodes, cross_border_bridges

# --------------------------------------------------------------------------- #
# LLM Pipeline Wrapper
# --------------------------------------------------------------------------- #
class LLMPipeline:
    """
    Encapsulates ATIS stage orchestration around the central LLM client.
    """

    def __init__(self) -> None:
        self.client: LLMClient = get_client()
        self.config = self.client.config
        self.token_budget: TokenBudget = TokenBudget()
        self.cache = AnalysisCache()

    # -- Low-level API call with retries ------------------------------------ #
    def _model_output_cap(self) -> int:
        return self.client.adapter.capabilities.max_output_tokens

    def _is_truncated(self, raw: str) -> bool:
        text = raw.strip()
        if not text:
            return False
        if text.endswith("..."):
            return True
        if text[-1] not in {"}", "]", "\"", ">", "'"}:
            if not (text[-1].isdigit() or text[-1].lower() in {"e", "l"}):
                return True
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        if open_braces > 0 or open_brackets > 0:
            return True
        return False

    def _call_api(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        if max_tokens is None:
            max_tokens = 4096
        cap = self._model_output_cap()
        max_tokens = min(max_tokens, cap)
        return self.client.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.0, seed=42, max_tokens=max_tokens)

    def _call_api_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        stage_name: str = "LLM call",
    ) -> str:
        cap = self._model_output_cap()
        max_tokens = min(max_tokens, cap)
        raw = self._call_api(system_prompt, user_prompt, max_tokens=max_tokens)
        if self._is_truncated(raw):
            retry_tokens = min(max_tokens * 2, cap)
            if retry_tokens > max_tokens:
                logger.warning("%s truncated (len=%d). Retrying with %d tokens.", stage_name, len(raw), retry_tokens)
                raw = self._call_api(system_prompt, user_prompt, max_tokens=retry_tokens)
                if self._is_truncated(raw):
                    logger.error("%s still truncated after retry.", stage_name)
            else:
                logger.error("%s truncated at model cap (%d).", stage_name, cap)
        return raw

    # -- Stage 1: Entity Extraction ----------------------------------------- #
    def stage_1_extract(self, article_text: str) -> Dict[str, Any]:
        """
        Stage 1 — Send the article text to the Entity Extraction Module.
        Returns a Python dict with 'entities', 'core_event', 'source_country', 'event_country'.
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: ENTITY EXTRACTION")
        logger.info("=" * 60)

        # Token guard
        estimated = self.token_budget.estimate(
            PROMPT_STAGE_1_EXTRACTOR + article_text
        )
        logger.info("Estimated Stage-1 input tokens: %d", estimated)

        if estimated > self.token_budget.available_for_input:
            max_chars = int(
                self.token_budget.available_for_input * 3.2
            ) - len(PROMPT_STAGE_1_EXTRACTOR)
            article_text = article_text[:max_chars] + "\n[TRUNCATED]"
            logger.warning(
                "Article aggressively truncated to fit Stage-1 token budget."
            )

        raw_response = self._call_api(PROMPT_STAGE_1_EXTRACTOR, article_text)
        data = safe_json_loads(raw_response, stage_name="Stage 1")

        entity_count = len(data.get("entities", []))
        logger.info(
            "Stage 1 complete. Extracted %d entities. Core event: %s | Source: %s | Event: %s",
            entity_count,
            data.get("core_event", "N/A"),
            data.get("source_country", "N/A"),
            data.get("event_country", "N/A"),
        )
        return data

    # -- Stage 2: Constraint Solving ---------------------------------------- #
    def stage_2_solve(
        self,
        article_text: str,
        graph_context: str,
        perspective: Any,
    ) -> str:
        """
        Stage 2 — Send the [NEW EVENT] + [GRAPH CONTEXT] to the
        Equilibrium and Constraint Engine. Returns raw markdown analysis.
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: CONSTRAINT SOLVING")
        logger.info("=" * 60)

        # Aggressive token management
        article_text, graph_context = self.token_budget.truncate_payload(
            PROMPT_STAGE_2_SOLVER, article_text, graph_context
        )

        user_payload = (
            f"[ANALYTICAL PERSPECTIVE]: {perspective.country} ({perspective.country_code})\n"
            "You MUST use only the actors and capabilities listed in the PERSPECTIVE ACTOR REGISTRY. "
            "You MUST use only the pathways evidenced in the CROSS-BORDER BRIDGE CONTEXT. "
            "Do not invent perspective-side actors, capabilities, or pathways.\n\n"
            f"[NEW EVENT]:\n{article_text}\n\n"
            f"[GRAPH CONTEXT]:\n{graph_context}"
        )

        estimated = self.token_budget.estimate(
            PROMPT_STAGE_2_SOLVER + user_payload
        )
        logger.info("Final Stage-2 input token estimate: %d", estimated)

        raw_analysis = self._call_api(PROMPT_STAGE_2_SOLVER, user_payload)
        logger.info("Stage 2 complete. Analysis length: %d chars", len(raw_analysis))
        return raw_analysis

    # -- Stage 3: Dashboard Formatting -------------------------------------- #
    def stage_3_format(
        self,
        stage_2_analysis: str,
        graph_context: str,
        perspective: Any,
    ) -> Dict[str, Any]:
        """
        Stage 3 — Send the Stage-2 markdown analysis + graph_context to the strict
        data-serialisation module. Returns a structured dashboard JSON dict.
        """
        logger.info("=" * 60)
        logger.info("STAGE 3: DASHBOARD FORMATTING")
        logger.info("=" * 60)

        combined_input = stage_2_analysis + "\n\n" + graph_context
        estimated = self.token_budget.estimate(
            PROMPT_STAGE_3_FORMATTER + combined_input
        )
        logger.info("Estimated Stage-3 input tokens: %d", estimated)

        if estimated > self.token_budget.available_for_input:
            max_chars = int(
                self.token_budget.available_for_input * 3.2
            ) - len(PROMPT_STAGE_3_FORMATTER)
            stage_2_analysis = (
                stage_2_analysis[:max_chars] + "\n[ANALYSIS TRUNCATED]"
            )
            logger.warning(
                "Stage-2 analysis truncated to fit Stage-3 token budget."
            )

        user_prompt = (
            f"## ANALYTICAL PERSPECTIVE\n{perspective.country} ({perspective.country_code})\n"
            "You MUST select perspective_actor from the PERSPECTIVE ACTOR REGISTRY. "
            "You MUST select perspective_capability from the capabilities listed for that actor. "
            "You MUST select pathway from the enumerated list and ensure it is supported by the CROSS-BORDER BRIDGE CONTEXT. "
            "You MUST set opportunity_country to the actual country where the commercial value exists. "
            "Do NOT default opportunity_country to the perspective country. "
            "Mark unsupported opportunities RESEARCH_REQUIRED.\n\n"
            f"## GRAPH CONTEXT\n{graph_context}\n\n"
            + stage_2_analysis
        )

        # IMPORTANT: _call_api_with_retry requires all four arguments.
        # Keep this call explicit so Stage 3 cannot regress to the old 2-argument form.
        raw_response = self._call_api_with_retry(
            system_prompt=PROMPT_STAGE_3_FORMATTER,
            user_prompt=user_prompt,
            max_tokens=4096,
            stage_name="Stage 3",
        )
        dashboard = safe_json_loads(raw_response, stage_name="Stage 3")
        logger.info("Stage 3 complete. Dashboard JSON generated.")
        return dashboard

# --------------------------------------------------------------------------- #
# Main Orchestration Entry Point
# --------------------------------------------------------------------------- #
def process_article_pipeline(
    article_path: str,
    perspective: Any | None = None,
) -> Dict[str, Any]:
    """
    Primary orchestration function for the ATIS pipeline.

    Parameters
    ----------
    article_path : str
        Filesystem path to the plain-text news article.

    Returns
    -------
    Dict[str, Any]
        The final dashboard JSON payload, enriched with pipeline metadata.
    """
    logger.info("=" * 70)
    logger.info("ATIS PIPELINE INITIALISATION")
    logger.info("=" * 70)
    perspective = perspective or PerspectiveContext()
    logger.info("Article path      : %s", article_path)
    logger.info("Perspective       : %s (%s)", perspective.country, perspective.country_code)
    logger.info("Vault directory   : %s", VAULT_DIR.resolve())
    logger.info("Dashboard directory: %s", DASHBOARDS_DIR.resolve())
    logger.info("Model (primary)   : configured")
    logger.info("Model (fallback)  : configured")

    # ------------------------------------------------------------------ #
    # 0. Load article
    # ------------------------------------------------------------------ #
    article_file = Path(article_path)
    if not article_file.exists():
        logger.critical("Article file not found: %s", article_path)
        raise FileNotFoundError(f"Article not found: {article_path}")

    try:
        article_text = article_file.read_text(encoding="utf-8")
    except Exception as exc:
        logger.critical("Failed to read article: %s", exc)
        raise

    logger.info("Article loaded: %d characters", len(article_text))

    # ------------------------------------------------------------------ #
    # 1. Initialise sub-systems
    # ------------------------------------------------------------------ #
    vault_manager = ObsidianVaultManager()
    try:
        pipeline = LLMPipeline()
    except ValueError as exc:
        logger.critical("%s", exc)
        raise

    # Compute knowledge state for determinism
    knowledge_state = KnowledgeState(vault_path=vault_manager.vault_dir)
    knowledge_state.compute()
    knowledge_state_hash = knowledge_state.knowledge_state_hash

    # ------------------------------------------------------------------ #
    # 2. Stage 1 — Entity Extraction
    # ------------------------------------------------------------------ #
    try:
        stage_1_result = pipeline.stage_1_extract(article_text)
    except Exception as exc:
        logger.critical("STAGE 1 FAILED: %s", exc)
        raise

    entities: List[Dict[str, str]] = stage_1_result.get("entities", [])
    core_event: str = stage_1_result.get("core_event", "Unknown event")
    source_country: str = stage_1_result.get("source_country", "")
    event_country: str = stage_1_result.get("event_country", source_country)

    if not entities:
        logger.warning("No entities extracted; graph context will be empty.")

    # ------------------------------------------------------------------ #
    # 3. Graph Processing Layer — WITH PERSPECTIVE RECONCILIATION
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("GRAPH PROCESSING LAYER — PERSPECTIVE RECONCILIATION")
    logger.info("=" * 60)

    graph_context, perspective_nodes, cross_border_bridges = vault_manager.build_graph_context(entities, perspective)
    logger.info(
        "Consolidated graph context built: %d characters | %d perspective nodes | %d cross-border bridges",
        len(graph_context), len(perspective_nodes), len(cross_border_bridges),
    )

    # Build sets for deterministic validation
    perspective_node_ids = {pn["node_id"] for pn in perspective_nodes}

    # Build evidence IDs for fingerprint
    evidence_ids = sorted([e.get("name", "") for e in entities])
    entity_ids = sorted(list(set(
        [perspective.country, source_country, event_country] +
        [pn["node_id"] for pn in perspective_nodes]
    )))
    relationship_ids = sorted(list(set(
        [b["from_node"] for b in cross_border_bridges] +
        [b["to_node"] for b in cross_border_bridges]
    )))

    # Compute analysis fingerprint after stage 1
    analysis_fingerprint = compute_analysis_fingerprint(
        story_id=core_event,
        perspective=perspective,
        evidence_ids=[e for e in evidence_ids if e],
        entity_ids=[e for e in entity_ids if e],
        relationship_ids=[r for r in relationship_ids if r],
        knowledge_state_hash=knowledge_state_hash,
    )
    logger.info("Analysis fingerprint computed: %s", analysis_fingerprint)

    # ------------------------------------------------------------------ #
    # 4. Stage 2 — Constraint Solving (with cache check)
    # ------------------------------------------------------------------ #
    # Check cache before stage 2
    cached_dashboard = pipeline.cache.get(analysis_fingerprint)
    if cached_dashboard is not None:
        logger.info("Cache HIT for fingerprint %s. Returning cached dashboard.", analysis_fingerprint)
        dashboard_payload = cached_dashboard
    else:
        logger.info("Cache MISS for fingerprint %s. Running stages 2 and 3.", analysis_fingerprint)
        try:
            stage_2_analysis = pipeline.stage_2_solve(article_text, graph_context, perspective)
        except Exception as exc:
            logger.critical("STAGE 2 FAILED: %s", exc)
            raise

        # ------------------------------------------------------------------ #
        # 5. Stage 3 — Dashboard Formatting (receives stage 2 output + graph_context)
        # ------------------------------------------------------------------ #
        try:
            dashboard_payload = pipeline.stage_3_format(stage_2_analysis, graph_context, perspective)
        except Exception as exc:
            logger.critical("STAGE 3 FAILED: %s", exc)
            raise

        # Store in cache after dashboard build
        pipeline.cache.set(analysis_fingerprint, dashboard_payload)

    # ------------------------------------------------------------------ #
    # 6. Enrich & Persist — WITH DETERMINISTIC VALIDATION
    # ------------------------------------------------------------------ #
    dashboard_payload["perspective"] = perspective.as_dict()
    dashboard_payload["source_country"] = source_country
    dashboard_payload["event_country"] = event_country

    validated_opportunities = []
    for item in dashboard_payload.get("opportunities", []):
        if isinstance(item, dict):
            validated = validate_opportunity(
                item,
                perspective,
                source_node_ids=None,  # In news pipeline, source nodes are the extracted entities
                perspective_node_ids=perspective_node_ids,
                cross_border_bridges=cross_border_bridges,
            )
            # Replace sequential ID with stable ID
            stable_id = compute_opportunity_identity(
                title=validated.get("title", ""),
                perspective_country=perspective.country,
                source_country=validated.get("source_country", ""),
                event_country=validated.get("event_country", ""),
                opportunity_country=validated.get("opportunity_country", ""),
                pathway=validated.get("pathway", ""),
                perspective_actor=validated.get("perspective_actor", ""),
            )
            validated["opportunity_id"] = stable_id
            validated["stable_opportunity_id"] = stable_id
            validated_opportunities.append(validated)
    dashboard_payload["opportunities"] = validated_opportunities

    dashboard_payload["pipeline_metadata"] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_article": str(article_file.resolve()),
        "extracted_entities_count": len(entities),
        "core_event": core_event,
        "source_country": source_country,
        "event_country": event_country,
        "perspective_country": perspective.country,
        "perspective_country_code": perspective.country_code,
        "perspective_nodes_found": len(perspective_nodes),
        "cross_border_bridges_found": len(cross_border_bridges),
        "model_primary": pipeline.config.model,
        "model_fallback": pipeline.config.fallback_model,
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "analysis_fingerprint": analysis_fingerprint,
        "knowledge_state": knowledge_state.as_dict(),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_filename = f"atis_dashboard_{timestamp}.json"
    output_path_file = DASHBOARDS_DIR / output_filename

    try:
        output_path_file.write_text(
            json.dumps(dashboard_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Dashboard persisted: %s", output_path_file.resolve())
    except Exception as exc:
        logger.error("Failed to write dashboard JSON: %s", exc)
        raise

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    return dashboard_payload

# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATIS Constraint-Solving and Market Equilibrium Engine",
        epilog="Example: python atis_engine.py ./articles/cobalt_news.txt",
    )
    parser.add_argument(
        "article_path",
        metavar="ARTICLE",
        help="Path to the plain-text news article to process.",
    )
    args = parser.parse_args()

    try:
        result = process_article_pipeline(args.article_path)
        # Echo the final JSON to stdout for piping / inspection
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.critical("Pipeline terminated with fatal error: %s", exc)
        sys.exit(1)


# =============================================================================
# Web entry point
# =============================================================================
def run_news_pipeline(
    article_text: str,
    perspective: Any | None = None,
) -> Dict[str, Any]:
    """
    Web-compatible entry point. Accepts raw article text, returns dashboard JSON.

    DETERMINISM FEATURES:
      - Computes KnowledgeState at pipeline start
      - Computes analysis_fingerprint after stage 1
      - Checks AnalysisCache before stage 2
      - Sets AnalysisCache after stage 3
      - Uses compute_opportunity_identity for stable IDs
      - Includes analysis_version, schema_version, analysis_fingerprint, knowledge_state in output
    """
    logger.info("=" * 70)
    perspective = perspective or PerspectiveContext()
    logger.info("ATIS NEWS PIPELINE (WEB) | Perspective: %s (%s)", perspective.country, perspective.country_code)
    logger.info("=" * 70)

    vault_manager = ObsidianVaultManager()
    try:
        pipeline = LLMPipeline()
    except ValueError as exc:
        logger.critical("%s", exc)
        raise

    # Compute knowledge state for determinism
    knowledge_state = KnowledgeState(vault_path=vault_manager.vault_dir)
    knowledge_state.compute()
    knowledge_state_hash = knowledge_state.knowledge_state_hash

    # Stage 1
    try:
        stage_1_result = pipeline.stage_1_extract(article_text)
    except Exception as exc:
        logger.critical("STAGE 1 FAILED: %s", exc)
        raise

    entities = stage_1_result.get("entities", [])
    core_event = stage_1_result.get("core_event", "Unknown event")
    source_country = stage_1_result.get("source_country", "")
    event_country = stage_1_result.get("event_country", source_country)

    # Graph layer — WITH PERSPECTIVE RECONCILIATION
    graph_context, perspective_nodes, cross_border_bridges = vault_manager.build_graph_context(entities, perspective)
    perspective_node_ids = {pn["node_id"] for pn in perspective_nodes}

    # Build evidence IDs for fingerprint
    evidence_ids = sorted([e.get("name", "") for e in entities])
    entity_ids = sorted(list(set(
        [perspective.country, source_country, event_country] +
        [pn["node_id"] for pn in perspective_nodes]
    )))
    relationship_ids = sorted(list(set(
        [b["from_node"] for b in cross_border_bridges] +
        [b["to_node"] for b in cross_border_bridges]
    )))

    # Compute analysis fingerprint after stage 1
    analysis_fingerprint = compute_analysis_fingerprint(
        story_id=core_event,
        perspective=perspective,
        evidence_ids=[e for e in evidence_ids if e],
        entity_ids=[e for e in entity_ids if e],
        relationship_ids=[r for r in relationship_ids if r],
        knowledge_state_hash=knowledge_state_hash,
    )
    logger.info("Analysis fingerprint computed: %s", analysis_fingerprint)

    # Check cache before stage 2
    cached_dashboard = pipeline.cache.get(analysis_fingerprint)
    if cached_dashboard is not None:
        logger.info("Cache HIT for fingerprint %s. Returning cached dashboard.", analysis_fingerprint)
        dashboard_payload = cached_dashboard
    else:
        logger.info("Cache MISS for fingerprint %s. Running stages 2 and 3.", analysis_fingerprint)

        # Stage 2
        try:
            stage_2_analysis = pipeline.stage_2_solve(article_text, graph_context, perspective)
        except Exception as exc:
            logger.critical("STAGE 2 FAILED: %s", exc)
            raise

        # Stage 3 — receives stage 2 output + graph_context
        try:
            dashboard_payload = pipeline.stage_3_format(stage_2_analysis, graph_context, perspective)
        except Exception as exc:
            logger.critical("STAGE 3 FAILED: %s", exc)
            raise

        # Store in cache after dashboard build
        pipeline.cache.set(analysis_fingerprint, dashboard_payload)

    # Enrich with deterministic validation
    dashboard_payload["perspective"] = perspective.as_dict()
    dashboard_payload["source_country"] = source_country
    dashboard_payload["event_country"] = event_country

    validated_opportunities = []
    for item in dashboard_payload.get("opportunities", []):
        if isinstance(item, dict):
            validated = validate_opportunity(
                item,
                perspective,
                source_node_ids=None,
                perspective_node_ids=perspective_node_ids,
                cross_border_bridges=cross_border_bridges,
            )
            # Replace sequential ID with stable ID
            stable_id = compute_opportunity_identity(
                title=validated.get("title", ""),
                perspective_country=perspective.country,
                source_country=validated.get("source_country", ""),
                event_country=validated.get("event_country", ""),
                opportunity_country=validated.get("opportunity_country", ""),
                pathway=validated.get("pathway", ""),
                perspective_actor=validated.get("perspective_actor", ""),
            )
            validated["opportunity_id"] = stable_id
            validated["stable_opportunity_id"] = stable_id
            validated_opportunities.append(validated)
    dashboard_payload["opportunities"] = validated_opportunities

    dashboard_payload["pipeline_metadata"] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_article": "web_upload",
        "extracted_entities_count": len(entities),
        "core_event": core_event,
        "source_country": source_country,
        "event_country": event_country,
        "perspective_country": perspective.country,
        "perspective_country_code": perspective.country_code,
        "perspective_nodes_found": len(perspective_nodes),
        "cross_border_bridges_found": len(cross_border_bridges),
        "model_primary": pipeline.config.model,
        "model_fallback": pipeline.config.fallback_model,
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "analysis_fingerprint": analysis_fingerprint,
        "knowledge_state": knowledge_state.as_dict(),
    }

    # Persist to disk (optional, ephemeral on Render)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_filename = f"atis_dashboard_{timestamp}.json"
    output_path_file = DASHBOARDS_DIR / output_filename
    try:
        output_path_file.write_text(
            json.dumps(dashboard_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Dashboard persisted: %s", output_path_file)
    except Exception as exc:
        logger.error("Failed to write dashboard: %s", exc)

    return dashboard_payload


if __name__ == "__main__":
    main()
