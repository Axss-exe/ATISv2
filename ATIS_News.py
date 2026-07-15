#!/usr/bin/env python3
"""
ATIS Constraint-Solving and Market Equilibrium Engine
======================================================

Africa Trade & Intelligence System (ATIS) — Production Orchestration Script.

This module implements a decoupled, state-passing pipeline that:
  1. Extracts economic entities from a news article via Cerebras LLM.
  2. Reconciles extracted entities against a local Obsidian markdown vault,
     using canonical fuzzy matching and bidirectional backlink crawling.
  3. Scans the entire vault to build a token-efficient global database landscape.
  4. Performs macroeconomic constraint-solving analysis via Cerebras LLM.
  5. Formats the analysis into a structured commercial-intelligence dashboard.
  6. Persists the final dashboard JSON to a local `./dashboards/` directory.

Constraints:
  - Python 3.10+
  - cerebras.cloud.sdk
  - Strict 60,000-token ceiling per API request (aggressive truncation).
  - Zero placeholders; fully operational.

Environment:
  - CEREBRAS_API_KEY must be set in the environment.
  - Article path provided as the first CLI argument.
"""

from __future__ import annotations

import argparse
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

# --------------------------------------------------------------------------- #
# Cerebras SDK
# --------------------------------------------------------------------------- #
try:
    from cerebras.cloud.sdk import Cerebras
    from cerebras.cloud.sdk import APIError, APIConnectionError, RateLimitError
except ImportError as _import_err:
    sys.stderr.write(
        "ERROR: The 'cerebras.cloud.sdk' package is not installed. "
        "Install it via: pip install cerebras-cloud-sdk\n"
    )
    raise SystemExit(1) from _import_err


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
    '  "core_event": "String summarizing the article main event."'
    "\n}"
)

PROMPT_STAGE_2_SOLVER: str = (
    "You are the ATIS Equilibrium and Constraint Engine. Your objective is to act as a macroeconomic constraint solver. "
    "You will be provided with a [NEW EVENT] and a [GRAPH CONTEXT] representing the current known state of the market.\n"
    "Do not summarize the event. You must calculate the systemic shifts and unfulfilled requirements caused by the event.\n"
    "Follow this exact reasoning sequence in your markdown output:\n"
    "## 1. THE EQUILIBRIUM DELTA: What specific market equilibrium was broken by this event?\n"
    "## 2. CONSTRAINT MATRIX: What new capabilities are now required? What existing capabilities are now insufficient?\n"
    "## 3. GRAPH RECONCILIATION: Compare constraints against the [GRAPH CONTEXT] and identify unmet requirements.\n"
    "## 4. ECONOMIC FLOW: For unmet requirements, identify who pays, who benefits, and capital flow.\n"
    "## 5. OPPORTUNITY CASCADE: Detail Primary, Secondary, and Tertiary business/investment opportunities created by this structural gap."
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
    '  "opportunities": [\n'
    "    {\n"
    '      "opportunity_id": "OPP-001",\n'
    '      "title": "String",\n'
    '      "type": "String",\n'
    '      "urgency_score": Float,\n'
    '      "feasibility_score": Float,\n'
    '      "required_missing_nodes": ["String"],\n'
    '      "capital_flow": {"beneficiary": "String", "likely_funder": "String"},\n'
    '      "justification": "One precise sentence explaining the structural gap."'
    "\n    }\n"
    "  ]"
    "\n}"
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
VAULT_DIR: Path = Path(r"C:\Users\tmaki\Documents\AKSOS\ATIS\Data")
DASHBOARDS_DIR: Path = Path("./dashboards")
MAX_TOKENS_PER_REQUEST: int = 60_000
MODEL_NAME: str = "gpt-oss-120b"
FALLBACK_MODEL: str = "gemma-4-31b"
RESPONSE_RESERVE: int = 8_000
SAFETY_BUFFER: int = 1_000
MAX_RETRIES: int = 3
BASE_DELAY_SECONDS: float = 2.0


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
# Obsidian Vault Manager (Graph & Inbound Backlink Engine)
# --------------------------------------------------------------------------- #
class ObsidianVaultManager:
    """
    Handles vault indexing, fuzzy filename matching, bidirectional link crawling
    (outbound + inbound backlinks), and shadow-node provisioning for the ATIS graph layer.
    """

    def __init__(self, vault_dir: Path = VAULT_DIR) -> None:
        self.vault_dir: Path = vault_dir
        self._ensure_directories()

        # Core Graph Indexing Maps
        self.file_map: Dict[str, str] = {}  # canonical_name -> actual_file_stem
        self.backlink_map: Dict[str, Set[str]] = {}  # canonical_name -> set of actual_file_stems linking to it

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
        return re.sub(r"[^a-zA-Z0-9]", "", name).lower()

    # -- Institutional Memory Vault Indexer ---------------------------------- #
    def _index_vault(self) -> None:
        """
        Performs a full pass over the vault to catalog existing nodes and map
        bidirectional graph relationships (backlinks) before execution.
        """
        self.file_map.clear()
        self.backlink_map.clear()

        md_files = list(self.vault_dir.glob("*.md"))
        logger.info("Indexing %d existing vault files for graph matching...", len(md_files))

        for file_path in md_files:
            actual_stem = file_path.stem
            canonical_stem = self._canonicalize(actual_stem)

            # Map canonical name to actual file name on disk
            self.file_map[canonical_stem] = actual_stem

            try:
                content = file_path.read_text(encoding="utf-8")
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
            path = self.vault_dir / f"{actual_name}.md"
            return path.read_text(encoding="utf-8")
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
        for link in links:
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
            for inbound in inbound_stems:
                # Grab a snippet or properties from the backlinked node to maximize context density
                inbound_content = self.read_entity(inbound)
                fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", inbound_content, re.DOTALL)
                meta = f" [{fm_match.group(1).strip().replace(chr(10), ' | ')}]" if fm_match else ""
                bundle_parts.append(f"- [[{inbound}]]{meta}")

        return "\n".join(bundle_parts)

    # -- Global Database Indexer -------------------------------------------- #
    def build_global_database_context(self, explicit_names: set[str]) -> str:
        """
        Builds a lightweight structural registry of all remaining nodes in the database
        to ensure background assets are available for pattern matching.
        """
        global_parts: List[str] = []
        canonical_explicits = {self._canonicalize(name) for name in explicit_names}

        for canonical_stem, actual_stem in self.file_map.items():
            if canonical_stem in canonical_explicits:
                continue

            content = self.read_entity(actual_stem)
            front_matter = ""
            fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if fm_match:
                front_matter = fm_match.group(1).strip().replace("\n", " | ")

            summary = ""
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            for line in lines:
                if not line.startswith("---") and not line.startswith("#"):
                    summary = line[:120]
                    break

            metadata_str = f"- **{actual_stem}**"
            if front_matter:
                metadata_str += f" [{front_matter}]"
            if summary:
                metadata_str += f" -> Context: {summary}"

            global_parts.append(metadata_str)

        if not global_parts:
            return "No additional background nodes detected."
        return "\n".join(global_parts)

    # -- Shadow Node Provisioning ------------------------------------------- #
    def provision_shadow_node(self, entity_name: str, classification: str) -> str:
        if self.entity_exists(entity_name):
            return ""

        content = self._generate_shadow_template(entity_name, classification)
        self.write_entity(entity_name, content)
        logger.info("Provisioned shadow node: %s (%s)", entity_name, classification)
        return content

    def _generate_shadow_template(self, entity_name: str, classification: str) -> str:
        if classification == "[MINING_REFINERY]":
            return (
                f"---\n"
                f"node_type: PRIVATE_CONGLOMERATE\n"
                f"sub_type: MINING_REFINERY\n"
                f"status: shadow\n"
                f"---\n"
                f"# {entity_name}\n\n"
                f"- Requires [[Continuous Baseload Power Grid]]\n"
                f"- Requires [[Industrial Bulk Water Source]]\n"
                f"- Regulated by [[Environmental Management Agency]]\n"
            )
        elif classification == "[GOVERNMENT_AGENCY]":
            return (
                f"---\n"
                f"node_type: GOVERNMENT_AGENCY\n"
                f"status: shadow\n"
                f"---\n"
                f"# {entity_name}\n\n"
                f"- Overseen by [[Overseeing Government Ministry]]\n"
                f"- Established under [[Act of Parliament]]\n"
            )
        else:
            clean_class = classification.strip("[]")
            return (
                f"---\n"
                f"node_type: {clean_class}\n"
                f"status: shadow\n"
                f"---\n"
                f"# {entity_name}\n\n"
                f"- Shadow node for context matching.\n"
            )

    # -- Graph Context Builder ---------------------------------------------- #
    def build_graph_context(self, entities: List[Dict[str, str]]) -> str:
        """
        Process explicit entities using the new fuzzy-matching and backlink injection engine.
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
                shadow_content = self.provision_shadow_node(name, classification)
                context_parts.append(
                    f"\n=== PROVISIONED SHADOW NODE: {name} ===\n{shadow_content}"
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

        return "\n".join(context_parts)


# --------------------------------------------------------------------------- #
# Safe JSON Loader
# --------------------------------------------------------------------------- #
def safe_json_loads(raw_text: str, stage_name: str) -> Dict[str, Any]:
    """
    Safely parse a JSON string with multiple fallback strategies.
    Strips markdown fences, attempts regex recovery, and raises a clear
    RuntimeError on absolute failure.
    """
    cleaned = raw_text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Attempt direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as direct_err:
        logger.warning(
            "Direct JSON parse failed in %s: %s", stage_name, direct_err
        )

    # Fallback: greedy regex search for the outermost JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as regex_err:
            logger.warning(
                "Regex JSON recovery failed in %s: %s", stage_name, regex_err
            )

    # Final fallback: attempt to fix trailing commas or common syntax issues
    heuristic = re.sub(r",(\s*[\}\]])", r"\1", cleaned)
    try:
        return json.loads(heuristic)
    except json.JSONDecodeError as heuristic_err:
        logger.error(
            "All JSON parsing strategies exhausted for %s.", stage_name
        )
        logger.error("Raw response excerpt (first 800 chars):\n%s", raw_text[:800])
        raise RuntimeError(
            f"Failed to parse JSON response from {stage_name}."
        ) from heuristic_err


# --------------------------------------------------------------------------- #
# Cerebras Pipeline Wrapper
# --------------------------------------------------------------------------- #
class CerebrasPipeline:
    """
    Encapsulates all Cerebras API interactions, retry logic,
    token-budget enforcement, and stage orchestration.
    """

    def __init__(
        self,
        api_key: Optional[str] = "csk-v4vf9r666pv9t99etmm9cppv58j8xpc8fnjpxkpw89mk36rp",
        model: str = MODEL_NAME,
        fallback_model: str = FALLBACK_MODEL,
    ) -> None:
        self.api_key: str = api_key or os.environ.get("CEREBRAS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "CEREBRAS_API_KEY is not set in the environment. "
                "Export it before running the pipeline."
            )
        self.client: Cerebras = Cerebras(api_key=self.api_key)
        self.model: str = model
        self.fallback_model: str = fallback_model
        self.token_budget: TokenBudget = TokenBudget()
        self.max_retries: int = MAX_RETRIES
        self.base_delay: float = BASE_DELAY_SECONDS

    # -- Low-level API call with retries ------------------------------------ #
    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 8_000,
    ) -> str:
        """
        Execute a chat completion against the Cerebras API.
        Implements exponential back-off for RateLimitError and
        APIConnectionError. Falls back to the smaller model on persistent
        APIErrors if the primary model is the large one.
        """
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
                    content: str = response.choices[0].message.content
                    logger.info("API call successful (%s)", model)
                    return content

                except RateLimitError as exc:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "RateLimitError on %s (attempt %d): %s. "
                        "Backing off for %.1f seconds...",
                        model,
                        attempt,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

                except APIConnectionError as exc:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "APIConnectionError on %s (attempt %d): %s. "
                        "Retrying in %.1f seconds...",
                        model,
                        attempt,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

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
                            break  # break inner loop, try next model
                        raise
                    time.sleep(self.base_delay)

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
                    time.sleep(self.base_delay)

        raise RuntimeError(
            "All API retries exhausted and fallback model failed."
        )

    # -- Stage 1: Entity Extraction ----------------------------------------- #
    def stage_1_extract(self, article_text: str) -> Dict[str, Any]:
        """
        Stage 1 — Send the article text to the Entity Extraction Module.
        Returns a Python dict with 'entities' and 'core_event'.
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

        raw_response = self._call_api(
            PROMPT_STAGE_1_EXTRACTOR, article_text, temperature=0.1
        )
        data = safe_json_loads(raw_response, stage_name="Stage 1")

        entity_count = len(data.get("entities", []))
        logger.info(
            "Stage 1 complete. Extracted %d entities. Core event: %s",
            entity_count,
            data.get("core_event", "N/A"),
        )
        return data

    # -- Stage 2: Constraint Solving ---------------------------------------- #
    def stage_2_solve(self, article_text: str, graph_context: str) -> str:
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
            f"[NEW EVENT]:\n{article_text}\n\n"
            f"[GRAPH CONTEXT]:\n{graph_context}"
        )

        estimated = self.token_budget.estimate(
            PROMPT_STAGE_2_SOLVER + user_payload
        )
        logger.info("Final Stage-2 input token estimate: %d", estimated)

        raw_analysis = self._call_api(
            PROMPT_STAGE_2_SOLVER, user_payload, temperature=0.2
        )
        logger.info("Stage 2 complete. Analysis length: %d chars", len(raw_analysis))
        return raw_analysis

    # -- Stage 3: Dashboard Formatting -------------------------------------- #
    def stage_3_format(self, stage_2_analysis: str) -> Dict[str, Any]:
        """
        Stage 3 — Send the Stage-2 markdown analysis to the strict
        data-serialisation module. Returns a structured dashboard JSON dict.
        """
        logger.info("=" * 60)
        logger.info("STAGE 3: DASHBOARD FORMATTING")
        logger.info("=" * 60)

        estimated = self.token_budget.estimate(
            PROMPT_STAGE_3_FORMATTER + stage_2_analysis
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

        raw_response = self._call_api(
            PROMPT_STAGE_3_FORMATTER, stage_2_analysis, temperature=0.1
        )
        dashboard = safe_json_loads(raw_response, stage_name="Stage 3")
        logger.info("Stage 3 complete. Dashboard JSON generated.")
        return dashboard


# --------------------------------------------------------------------------- #
# Main Orchestration Entry Point
# --------------------------------------------------------------------------- #
def process_article_pipeline(article_path: str) -> Dict[str, Any]:
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
    logger.info("Article path      : %s", article_path)
    logger.info("Vault directory   : %s", VAULT_DIR.resolve())
    logger.info("Dashboard directory: %s", DASHBOARDS_DIR.resolve())
    logger.info("Model (primary)   : %s", MODEL_NAME)
    logger.info("Model (fallback)  : %s", FALLBACK_MODEL)

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
        pipeline = CerebrasPipeline()
    except ValueError as exc:
        logger.critical("%s", exc)
        raise

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

    if not entities:
        logger.warning("No entities extracted; graph context will be empty.")

    # ------------------------------------------------------------------ #
    # 3. Graph Processing Layer
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("GRAPH PROCESSING LAYER")
    logger.info("=" * 60)

    graph_context = vault_manager.build_graph_context(entities)
    logger.info(
        "Consolidated graph context built: %d characters", len(graph_context)
    )

    # ------------------------------------------------------------------ #
    # 4. Stage 2 — Constraint Solving
    # ------------------------------------------------------------------ #
    try:
        stage_2_analysis = pipeline.stage_2_solve(article_text, graph_context)
    except Exception as exc:
        logger.critical("STAGE 2 FAILED: %s", exc)
        raise

    # ------------------------------------------------------------------ #
    # 5. Stage 3 — Dashboard Formatting
    # ------------------------------------------------------------------ #
    try:
        dashboard_payload = pipeline.stage_3_format(stage_2_analysis)
    except Exception as exc:
        logger.critical("STAGE 3 FAILED: %s", exc)
        raise

    # ------------------------------------------------------------------ #
    # 6. Enrich & Persist
    # ------------------------------------------------------------------ #
    dashboard_payload["pipeline_metadata"] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_article": str(article_file.resolve()),
        "extracted_entities_count": len(entities),
        "core_event": core_event,
        "model_primary": MODEL_NAME,
        "model_fallback": FALLBACK_MODEL,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_filename = f"atis_dashboard_{timestamp}.json"
    output_path = DASHBOARDS_DIR / output_filename

    try:
        output_path.write_text(
            json.dumps(dashboard_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Dashboard persisted: %s", output_path.resolve())
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


if __name__ == "__main__":
    main()