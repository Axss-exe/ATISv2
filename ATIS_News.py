#!/usr/bin/env python3
"""
ATIS News Architecture Rebuild — Intelligent Retrieval, Token Budgeting & Multi-Stage Reasoning
================================================================================================

Rebuilt per ATIS NEWS ARCHITECTURE REBUILD specification:
  1. Backlink tracing with distance-aware scoring
  2. Intelligent knowledge-node selection with inspectable multi-factor scoring
  3. Context/token budgeting against actual model capabilities
  4. LLM call orchestration with automatic workload splitting
  5. Multi-stage reasoning for large knowledge states
  6. Progressive evidence compression (never sends raw massive context to final call)
  7. Evidence provenance preservation through all stages
  8. Diversity-aware selection
  9. Opportunity detection with explicit/derived/potential confidence levels
 10. Observability logging for every analysis

Public API (backward compatible):
  - process_article_pipeline(article_path, perspective=None) -> Dict[str, Any]
  - run_news_pipeline(article_text, perspective=None) -> Dict[str, Any]

Determinism:
  - temperature=0.0 on all LLM calls; provider seed parameters are intentionally omitted.
  - sorted iterations for stable ordering
  - AnalysisCache for disk-based result caching
  - KnowledgeState for vault versioning
  - compute_analysis_fingerprint for stable identity
  - compute_opportunity_identity for stable opportunity IDs
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_client import LLMClient, get_client, LLMTokenLimitError, ModelCapabilities
from atis_context import (
    PerspectiveContext,
    validate_opportunity,
    KnowledgeState,
    AnalysisCache,
    compute_analysis_fingerprint,
    compute_opportunity_identity,
    ANALYSIS_VERSION,
    SCHEMA_VERSION,
    COUNTRY_CODES,
)

# --------------------------------------------------------------------------- #
# Logging Setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger("atis_news")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
VAULT_DIR: Path = Path(os.getenv("ATIS_VAULT_DIR", "./vault"))
DASHBOARDS_DIR: Path = Path("./dashboards")

# Safety margins — never consume 100% of context
OUTPUT_RESERVE_TOKENS: int = 12_000      # Reserve for model response
SAFETY_MARGIN_TOKENS: int = 2_000       # Buffer for token estimation error
SYSTEM_PROMPT_OVERHEAD: int = 500        # Overhead per message block

# Diversity / selection parameters
MAX_DIRECT_EVIDENCE_NODES: int = 30
MAX_FIRST_ORDER_NODES: int = 25
MAX_SECOND_ORDER_NODES: int = 15
MAX_PERIPHERAL_NODES: int = 5
MAX_PERSPECTIVE_NODES: int = 20
MAX_BRIDGE_NODES: int = 15
MAX_GLOBAL_REGISTRY_NODES: int = 50

# Multi-stage partitioning
MAX_NODES_PER_PARTITION: int = 25       # Nodes per evidence-analysis call
MAX_PARTITIONS: int = 10

# Deduplication
MIN_SIMILARITY_THRESHOLD: float = 0.75   # For near-duplicate detection

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class EvidenceCategory(Enum):
    DIRECT = "direct"               # Information directly describing the event
    FIRST_ORDER = "first_order"   # Nodes directly connected to event entities
    SECOND_ORDER = "second_order" # Nodes connected to first-order nodes
    PERIPHERAL = "peripheral"     # Technically connected but low analytical value
    PERSPECTIVE = "perspective"   # Perspective-country actors
    BRIDGE = "bridge"             # Cross-border bridge nodes
    GLOBAL = "global"             # Background registry nodes

class OpportunityType(Enum):
    EXPLICIT = "explicit"         # Source directly indicates opportunity
    DERIVED = "derived"           # Evidence supports reasonable strategic opportunity
    POTENTIAL = "potential"       # Plausible but insufficient evidence

class ReasoningMode(Enum):
    SINGLE = "single"
    MULTI_STAGE = "multi_stage"

# --------------------------------------------------------------------------- #
# Data Structures
# --------------------------------------------------------------------------- #
@dataclass
class ScoredNode:
    """
    Inspectable node selection with multi-factor relevance scoring.
    Every candidate node receives explicit scores so the reasoning is transparent.
    """
    node_id: str
    canonical_id: str
    content: str
    category: EvidenceCategory

    # Core relevance dimensions
    relevance_score: float = 0.0
    direct_match_score: float = 0.0
    relationship_score: float = 0.0
    temporal_score: float = 0.0
    geographic_score: float = 0.0
    sector_score: float = 0.0
    backlink_distance: int = 0
    evidence_strength: float = 0.0
    diversity_score: float = 0.0

    # Metadata
    country: str = ""
    node_type: str = ""
    sector: str = ""
    summary: str = ""
    source_entities: List[str] = field(default_factory=list)
    selection_reason: str = ""

    @property
    def composite_score(self) -> float:
        """Weighted composite for ranking."""
        return (
            self.relevance_score * 0.30 +
            self.direct_match_score * 0.25 +
            self.relationship_score * 0.15 +
            self.geographic_score * 0.10 +
            self.sector_score * 0.08 +
            self.temporal_score * 0.05 +
            self.evidence_strength * 0.05 +
            self.diversity_score * 0.02
        )

@dataclass
class IntermediateFinding:
    """Preserves evidence provenance through intermediate reasoning stages."""
    finding: str
    supporting_nodes: List[str]
    supporting_entities: List[str]
    confidence: float
    reasoning_stage: str
    category: str = ""
    contradictions: List[str] = field(default_factory=list)

@dataclass
class EvidencePartition:
    """A partition of evidence for parallel/sequential analysis."""
    partition_id: int
    nodes: List[ScoredNode]
    theme: str
    estimated_tokens: int

@dataclass
class ReasoningLog:
    """Observability record for a single News analysis."""
    entities_extracted: int = 0
    candidate_nodes: int = 0
    backlink_candidates: int = 0
    relevant_nodes: int = 0
    selected_evidence: int = 0
    estimated_tokens: int = 0
    safe_budget: int = 0
    reasoning_mode: str = ""
    partitions: int = 0
    evidence_calls: int = 0
    synthesis_calls: int = 0
    final_call: int = 0
    total_llm_calls: int = 0
    deduplicated_nodes: int = 0
    weak_backlinks_filtered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities_extracted": self.entities_extracted,
            "candidate_nodes": self.candidate_nodes,
            "backlink_candidates": self.backlink_candidates,
            "relevant_nodes": self.relevant_nodes,
            "selected_evidence": self.selected_evidence,
            "estimated_tokens": self.estimated_tokens,
            "safe_budget": self.safe_budget,
            "reasoning_mode": self.reasoning_mode,
            "partitions": self.partitions,
            "evidence_calls": self.evidence_calls,
            "synthesis_calls": self.synthesis_calls,
            "final_call": self.final_call,
            "total_llm_calls": self.total_llm_calls,
            "deduplicated_nodes": self.deduplicated_nodes,
            "weak_backlinks_filtered": self.weak_backlinks_filtered,
        }

    def log_tree(self) -> str:
        return (
            f"NEWS REASONING\n"
            f"├── entities extracted: {self.entities_extracted}\n"
            f"├── candidate nodes: {self.candidate_nodes}\n"
            f"├── backlink candidates: {self.backlink_candidates}\n"
            f"├── relevant nodes: {self.relevant_nodes}\n"
            f"├── selected evidence: {self.selected_evidence}\n"
            f"├── estimated tokens: {self.estimated_tokens:,}\n"
            f"├── safe budget: {self.safe_budget:,}\n"
            f"├── reasoning mode: {self.reasoning_mode}\n"
            f"├── partitions: {self.partitions}\n"
            f"├── evidence calls: {self.evidence_calls}\n"
            f"├── synthesis calls: {self.synthesis_calls}\n"
            f"├── final call: {self.final_call}\n"
            f"├── total LLM calls: {self.total_llm_calls}\n"
            f"├── deduplicated nodes: {self.deduplicated_nodes}\n"
            f"└── weak backlinks filtered: {self.weak_backlinks_filtered}"
        )


# --------------------------------------------------------------------------- #
# Token Budget Manager — Dynamic against actual model capabilities
# --------------------------------------------------------------------------- #
@dataclass
class TokenBudgetManager:
    """
    Dedicated token-budgeting layer that queries actual model capabilities.
    Never assumes the model can consume the provider's maximum context.
    """
    capabilities: ModelCapabilities
    output_reserve: int = OUTPUT_RESERVE_TOKENS
    safety_margin: int = SAFETY_MARGIN_TOKENS
    system_prompt_overhead: int = SYSTEM_PROMPT_OVERHEAD

    @property
    def provider_context_limit(self) -> int:
        return self.capabilities.max_context_tokens

    @property
    def max_output_tokens(self) -> int:
        return self.capabilities.max_output_tokens

    @property
    def usable_context_budget(self) -> int:
        """Safe input budget after reserving output + safety margin."""
        return self.provider_context_limit - self.output_reserve - self.safety_margin

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Conservative estimate: ~3.2 chars per token for mixed English/technical text."""
        if not text:
            return 0
        return int(len(text) / 3.2) + 1

    @staticmethod
    def estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
        """Estimate tokens for a list of messages including overhead."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += TokenBudgetManager.estimate_tokens(content)
            total += SYSTEM_PROMPT_OVERHEAD  # per-message overhead
        return total

    def fits_in_budget(self, input_tokens: int, requested_output: int) -> bool:
        """Check if a call fits within the safe budget."""
        total = input_tokens + requested_output + self.safety_margin
        return total <= self.provider_context_limit

    def compute_safe_output_tokens(self, input_tokens: int) -> int:
        """Given input size, compute safe max output tokens."""
        available = self.provider_context_limit - input_tokens - self.safety_margin
        return min(available, self.max_output_tokens)

    def require_budget(self, input_tokens: int, requested_output: int, stage_name: str) -> None:
        """
        Guarantee: input + output + safety < provider_context_limit.
        If exceeded, raise LLMTokenLimitError BEFORE sending.
        """
        total = input_tokens + requested_output + self.safety_margin
        if total > self.provider_context_limit:
            raise LLMTokenLimitError(
                f"[{stage_name}] Estimated total tokens ({total:,}) exceeds provider "
                f"context limit ({self.provider_context_limit:,}). "
                f"Input: ~{input_tokens:,}, Requested output: {requested_output:,}, "
                f"Safety margin: {self.safety_margin:,}. "
                f"SPLIT the workload instead of sending."
            )


# --------------------------------------------------------------------------- #
# Safe JSON Loader — Brace-Balancing Engine (preserved from original)
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
    """Extract the largest balanced JSON object or array from raw text."""
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
    """Safely parse JSON with deterministic, layered fallback strategies."""
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
# Obsidian Vault Manager — Enhanced with Intelligent Scoring
# --------------------------------------------------------------------------- #
class ObsidianVaultManager:
    """
    Handles vault indexing, fuzzy filename matching, bidirectional link crawling,
    and intelligent node selection with multi-factor relevance scoring.
    """

    def __init__(self, vault_dir: Path = VAULT_DIR) -> None:
        self.vault_dir: Path = vault_dir
        self._ensure_directories()

        # Core Graph Indexing Maps
        self.file_map: Dict[str, str] = {}          # canonical_name -> actual_file_stem
        self.backlink_map: Dict[str, Set[str]] = {}   # canonical_name -> set of stems linking TO it
        self.node_metadata: Dict[str, Dict[str, Any]] = {}  # canonical_name -> metadata
        self.node_content: Dict[str, str] = {}      # canonical_name -> full content (lazy)
        self.outbound_links: Dict[str, List[str]] = {}  # canonical_name -> outbound link targets

        # Build index immediately on startup
        self._index_vault()

    def _ensure_directories(self) -> None:
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        DASHBOARDS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Vault directory verified: %s", self.vault_dir.resolve())

    @staticmethod
    def _canonicalize(name: str) -> str:
        """Strict alphanumeric lowercase token for deterministic matching."""
        return re.sub(r"[^a-zA-Z0-9]", "", str(name or "")).lower()

    def _index_vault(self) -> None:
        """Full pass: catalog nodes, map bidirectional relationships, extract metadata."""
        self.file_map.clear()
        self.backlink_map.clear()
        self.node_metadata.clear()
        self.node_content.clear()
        self.outbound_links.clear()

        md_files = sorted(self.vault_dir.rglob("*.md"), key=lambda p: str(p))
        logger.info("Indexing %d vault files...", len(md_files))

        for file_path in md_files:
            actual_stem = file_path.stem
            canonical_stem = self._canonicalize(actual_stem)
            self.file_map[canonical_stem] = actual_stem

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.error("Failed to read %s: %s", actual_stem, exc)
                continue

            # Extract frontmatter
            country = ""
            node_type = ""
            sector = ""
            fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if fm_match:
                try:
                    import yaml
                    front = yaml.safe_load(fm_match.group(1)) or {}
                    country = front.get("country", "") or front.get("location", "")
                    node_type = front.get("node_type", "") or front.get("type", "") or front.get("entity_type", "")
                    sector = front.get("sector", "") or front.get("industry", "")
                except Exception:
                    pass

            # Infer country from path if not in frontmatter
            if not country:
                path_parts = [p.lower() for p in file_path.relative_to(self.vault_dir).parts]
                for part in path_parts:
                    if part in COUNTRY_CODES:
                        country = part.title()
                        break

            # Extract summary (first non-empty, non-frontmatter, non-header line)
            summary = ""
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            for line in lines:
                if not line.startswith("---") and not line.startswith("#"):
                    summary = line[:300]
                    break

            rel_path = str(file_path.relative_to(self.vault_dir))
            self.node_metadata[canonical_stem] = {
                "country": country,
                "type": node_type,
                "sector": sector,
                "summary": summary,
                "path": rel_path,
            }
            self.node_content[canonical_stem] = content

            # Parse outbound wiki-links
            links = re.findall(r"\[\[(.*?)\]\]", content)
            cleaned_links: List[str] = []
            seen_links: Set[str] = set()
            for link in links:
                link_target = link.split("|")[0].strip()
                if not link_target:
                    continue
                canon_target = self._canonicalize(link_target)
                if canon_target not in seen_links:
                    seen_links.add(canon_target)
                    cleaned_links.append(link_target)
            self.outbound_links[canonical_stem] = cleaned_links

            # Register backlinks
            for link in cleaned_links:
                canon_target = self._canonicalize(link)
                if canon_target not in self.backlink_map:
                    self.backlink_map[canon_target] = set()
                self.backlink_map[canon_target].add(actual_stem)

        logger.info("Vault index complete: %d nodes, %d backlink targets",
                    len(self.file_map), len(self.backlink_map))

    def entity_exists(self, entity_name: str) -> bool:
        return self._canonicalize(entity_name) in self.file_map

    def read_entity(self, entity_name: str) -> str:
        canonical = self._canonicalize(entity_name)
        if canonical in self.node_content:
            return self.node_content[canonical]
        actual_name = self.file_map.get(canonical)
        if actual_name:
            meta = self.node_metadata.get(canonical, {})
            rel_path = meta.get("path", f"{actual_name}.md")
            path = self.vault_dir / rel_path
            if path.exists():
                return path.read_text(encoding="utf-8")
            for found in self.vault_dir.rglob(f"{actual_name}.md"):
                if found.exists():
                    return found.read_text(encoding="utf-8")
        return ""

    def get_node_meta(self, entity_name: str) -> Dict[str, Any]:
        canonical = self._canonicalize(entity_name)
        return self.node_metadata.get(canonical, {})

    # ------------------------------------------------------------------ #
    # Intelligent Node Selection with Multi-Factor Scoring
    # ------------------------------------------------------------------ #
    def select_evidence_nodes(
        self,
        entities: List[Dict[str, str]],
        perspective: PerspectiveContext,
        reasoning_log: ReasoningLog,
    ) -> List[ScoredNode]:
        """
        Main entry: starting from extracted entities, perform intelligent
        backlink tracing, relevance scoring, diversity selection, and deduplication.
        Returns a ranked, deduplicated list of ScoredNodes.
        """
        entity_names = [e.get("name", "").strip() for e in entities if e.get("name", "").strip()]
        entity_classes = {self._canonicalize(e.get("name", "")): e.get("class", "") for e in entities}

        # Extract countries, sectors, actors from entities
        source_countries: Set[str] = set()
        source_sectors: Set[str] = set()
        for e in entities:
            ctx = e.get("context", "").lower()
            for country in COUNTRY_CODES:
                if country in ctx:
                    source_countries.add(country.title())

        perspective_norm = perspective.country.lower()

        # --- PHASE 1: Direct Evidence ---
        direct_nodes: List[ScoredNode] = []
        for name in entity_names:
            canonical = self._canonicalize(name)
            if canonical not in self.file_map:
                continue
            actual = self.file_map[canonical]
            meta = self.node_metadata.get(canonical, {})
            content = self.read_entity(actual)

            score = ScoredNode(
                node_id=actual,
                canonical_id=canonical,
                content=content,
                category=EvidenceCategory.DIRECT,
                direct_match_score=1.0,
                relationship_score=1.0,
                evidence_strength=1.0,
                backlink_distance=0,
                country=meta.get("country", ""),
                node_type=meta.get("type", ""),
                sector=meta.get("sector", ""),
                summary=meta.get("summary", ""),
                source_entities=[name],
                selection_reason=f"Direct entity match: '{name}'",
            )
            # Geographic score
            if meta.get("country", "").lower() in {c.lower() for c in source_countries}:
                score.geographic_score = 1.0
            direct_nodes.append(score)

        # --- PHASE 2: Backlink Tracing (First & Second Order) ---
        first_order_nodes: Dict[str, ScoredNode] = {}
        second_order_nodes: Dict[str, ScoredNode] = {}
        peripheral_nodes: Dict[str, ScoredNode] = {}

        visited: Set[str] = set(self._canonicalize(n) for n in entity_names)

        # First-order: outbound links from direct nodes + inbound backlinks to direct nodes
        for name in entity_names:
            canonical = self._canonicalize(name)
            if canonical not in self.file_map:
                continue
            actual = self.file_map[canonical]

            # Outbound links
            for link_target in self.outbound_links.get(canonical, []):
                link_canon = self._canonicalize(link_target)
                if link_canon in visited:
                    continue
                visited.add(link_canon)
                if link_canon not in self.file_map:
                    continue
                actual_target = self.file_map[link_canon]
                meta = self.node_metadata.get(link_canon, {})
                content = self.read_entity(actual_target)

                rel_score = self._compute_relationship_score(
                    link_canon, entity_names, perspective_norm, source_countries
                )

                node = ScoredNode(
                    node_id=actual_target,
                    canonical_id=link_canon,
                    content=content,
                    category=EvidenceCategory.FIRST_ORDER,
                    direct_match_score=0.3,
                    relationship_score=rel_score,
                    evidence_strength=0.7,
                    backlink_distance=1,
                    country=meta.get("country", ""),
                    node_type=meta.get("type", ""),
                    sector=meta.get("sector", ""),
                    summary=meta.get("summary", ""),
                    source_entities=[name],
                    selection_reason=f"First-order link from '{name}'",
                )
                first_order_nodes[link_canon] = node

            # Inbound backlinks
            for inbound_stem in self.backlink_map.get(canonical, set()):
                inbound_canon = self._canonicalize(inbound_stem)
                if inbound_canon in visited:
                    continue
                visited.add(inbound_canon)
                if inbound_canon not in self.file_map:
                    continue
                meta = self.node_metadata.get(inbound_canon, {})
                content = self.read_entity(inbound_stem)

                rel_score = self._compute_relationship_score(
                    inbound_canon, entity_names, perspective_norm, source_countries
                )

                node = ScoredNode(
                    node_id=inbound_stem,
                    canonical_id=inbound_canon,
                    content=content,
                    category=EvidenceCategory.FIRST_ORDER,
                    direct_match_score=0.2,
                    relationship_score=rel_score,
                    evidence_strength=0.6,
                    backlink_distance=1,
                    country=meta.get("country", ""),
                    node_type=meta.get("type", ""),
                    sector=meta.get("sector", ""),
                    summary=meta.get("summary", ""),
                    source_entities=[name],
                    selection_reason=f"Inbound backlink to '{name}'",
                )
                first_order_nodes[inbound_canon] = node

        # Second-order: links from first-order nodes
        for canon, first_node in list(first_order_nodes.items()):
            for link_target in self.outbound_links.get(canon, []):
                link_canon = self._canonicalize(link_target)
                if link_canon in visited:
                    continue
                visited.add(link_canon)
                if link_canon not in self.file_map:
                    continue
                actual_target = self.file_map[link_canon]
                meta = self.node_metadata.get(link_canon, {})
                content = self.read_entity(actual_target)

                rel_score = self._compute_relationship_score(
                    link_canon, entity_names, perspective_norm, source_countries
                ) * 0.5  #衰减 for second order

                node = ScoredNode(
                    node_id=actual_target,
                    canonical_id=link_canon,
                    content=content,
                    category=EvidenceCategory.SECOND_ORDER,
                    direct_match_score=0.1,
                    relationship_score=rel_score,
                    evidence_strength=0.4,
                    backlink_distance=2,
                    country=meta.get("country", ""),
                    node_type=meta.get("type", ""),
                    sector=meta.get("sector", ""),
                    summary=meta.get("summary", ""),
                    source_entities=first_node.source_entities,
                    selection_reason=f"Second-order link from '{first_node.node_id}'",
                )
                second_order_nodes[link_canon] = node

            # Second-order via backlinks to first-order nodes
            for inbound_stem in self.backlink_map.get(canon, set()):
                inbound_canon = self._canonicalize(inbound_stem)
                if inbound_canon in visited:
                    continue
                visited.add(inbound_canon)
                if inbound_canon not in self.file_map:
                    continue
                meta = self.node_metadata.get(inbound_canon, {})
                content = self.read_entity(inbound_stem)

                rel_score = self._compute_relationship_score(
                    inbound_canon, entity_names, perspective_norm, source_countries
                ) * 0.4

                node = ScoredNode(
                    node_id=inbound_stem,
                    canonical_id=inbound_canon,
                    content=content,
                    category=EvidenceCategory.SECOND_ORDER,
                    direct_match_score=0.05,
                    relationship_score=rel_score,
                    evidence_strength=0.3,
                    backlink_distance=2,
                    country=meta.get("country", ""),
                    node_type=meta.get("type", ""),
                    sector=meta.get("sector", ""),
                    summary=meta.get("summary", ""),
                    source_entities=first_node.source_entities,
                    selection_reason=f"Second-order backlink to '{first_node.node_id}'",
                )
                second_order_nodes[inbound_canon] = node

        # --- PHASE 3: Perspective-Side Nodes ---
        perspective_nodes: Dict[str, ScoredNode] = {}
        for canon, meta in sorted(self.node_metadata.items()):
            if meta.get("country", "").lower() != perspective_norm:
                continue
            if canon in visited:
                # Already captured — upgrade its perspective relevance
                continue
            actual = self.file_map.get(canon, canon)
            content = self.read_entity(actual)

            rel_score = self._compute_relationship_score(
                canon, entity_names, perspective_norm, source_countries
            )

            node = ScoredNode(
                node_id=actual,
                canonical_id=canon,
                content=content,
                category=EvidenceCategory.PERSPECTIVE,
                direct_match_score=0.0,
                relationship_score=rel_score,
                evidence_strength=0.5,
                backlink_distance=1,
                country=meta.get("country", ""),
                node_type=meta.get("type", ""),
                sector=meta.get("sector", ""),
                summary=meta.get("summary", ""),
                source_entities=[],
                selection_reason=f"Perspective-country actor: {perspective.country}",
            )
            perspective_nodes[canon] = node

        # --- PHASE 4: Cross-Border Bridges ---
        bridge_nodes: Dict[str, ScoredNode] = {}
        source_country = ""
        if source_countries:
            source_country = list(source_countries)[0]
        else:
            # Infer from entity context
            for e in entities:
                ctx = e.get("context", "").lower()
                for country in COUNTRY_CODES:
                    if country in ctx:
                        source_country = country.title()
                        break
                if source_country:
                    break

        if source_country and source_country.lower() != perspective_norm:
            source_norm = source_country.lower()
            for canon, meta in sorted(self.node_metadata.items()):
                node_country = meta.get("country", "").lower()
                if node_country not in (perspective_norm, source_norm):
                    continue

                # Check if this node has cross-border links
                has_bridge = False
                for link_target in self.outbound_links.get(canon, []):
                    link_canon = self._canonicalize(link_target)
                    link_meta = self.node_metadata.get(link_canon, {})
                    link_country = link_meta.get("country", "").lower()
                    if node_country == perspective_norm and link_country == source_norm:
                        has_bridge = True
                        break
                    if node_country == source_norm and link_country == perspective_norm:
                        has_bridge = True
                        break

                if not has_bridge:
                    # Check backlinks
                    for inbound in self.backlink_map.get(canon, set()):
                        inbound_canon = self._canonicalize(inbound)
                        inbound_meta = self.node_metadata.get(inbound_canon, {})
                        inbound_country = inbound_meta.get("country", "").lower()
                        if node_country == perspective_norm and inbound_country == source_norm:
                            has_bridge = True
                            break
                        if node_country == source_norm and inbound_country == perspective_norm:
                            has_bridge = True
                            break

                if has_bridge and canon not in visited:
                    actual = self.file_map.get(canon, canon)
                    content = self.read_entity(actual)
                    node = ScoredNode(
                        node_id=actual,
                        canonical_id=canon,
                        content=content,
                        category=EvidenceCategory.BRIDGE,
                        direct_match_score=0.2,
                        relationship_score=0.8,
                        evidence_strength=0.7,
                        backlink_distance=1,
                        country=meta.get("country", ""),
                        node_type=meta.get("type", ""),
                        sector=meta.get("sector", ""),
                        summary=meta.get("summary", ""),
                        source_entities=[],
                        selection_reason=f"Cross-border bridge: {node_country} ↔ {source_country}",
                    )
                    bridge_nodes[canon] = node

        # --- PHASE 5: Global Registry (diversity fill) ---
        global_nodes: Dict[str, ScoredNode] = {}
        # Only if we need more diversity
        all_selected = set()
        for d in [direct_nodes, list(first_order_nodes.values()), list(second_order_nodes.values()),
                   list(perspective_nodes.values()), list(bridge_nodes.values())]:
            for n in (d if isinstance(d, list) else d):
                all_selected.add(n.canonical_id)

        # Collect sectors and countries already covered
        covered_sectors: Set[str] = set()
        covered_countries: Set[str] = set()
        for n in all_selected:
            meta = self.node_metadata.get(n, {})
            if meta.get("sector"):
                covered_sectors.add(meta["sector"].lower())
            if meta.get("country"):
                covered_countries.add(meta["country"].lower())

        # Fill gaps with diverse global nodes
        for canon, meta in sorted(self.node_metadata.items()):
            if canon in all_selected:
                continue
            sector = meta.get("sector", "").lower()
            country = meta.get("country", "").lower()
            if sector and sector not in covered_sectors:
                actual = self.file_map.get(canon, canon)
                content = self.read_entity(actual)
                node = ScoredNode(
                    node_id=actual,
                    canonical_id=canon,
                    content=content,
                    category=EvidenceCategory.GLOBAL,
                    direct_match_score=0.0,
                    relationship_score=0.2,
                    evidence_strength=0.2,
                    backlink_distance=3,
                    country=meta.get("country", ""),
                    node_type=meta.get("type", ""),
                    sector=meta.get("sector", ""),
                    summary=meta.get("summary", ""),
                    source_entities=[],
                    selection_reason=f"Diversity fill: sector '{meta.get('sector', '')}'",
                )
                global_nodes[canon] = node
                covered_sectors.add(sector)
            elif country and country not in covered_countries and len(global_nodes) < MAX_GLOBAL_REGISTRY_NODES // 2:
                actual = self.file_map.get(canon, canon)
                content = self.read_entity(actual)
                node = ScoredNode(
                    node_id=actual,
                    canonical_id=canon,
                    content=content,
                    category=EvidenceCategory.GLOBAL,
                    direct_match_score=0.0,
                    relationship_score=0.15,
                    evidence_strength=0.15,
                    backlink_distance=3,
                    country=meta.get("country", ""),
                    node_type=meta.get("type", ""),
                    sector=meta.get("sector", ""),
                    summary=meta.get("summary", ""),
                    source_entities=[],
                    selection_reason=f"Diversity fill: country '{meta.get('country', '')}'",
                )
                global_nodes[canon] = node
                covered_countries.add(country)

        # --- PHASE 6: Deduplication ---
        all_nodes: Dict[str, ScoredNode] = {}
        for node in direct_nodes:
            all_nodes[node.canonical_id] = node
        for d in [first_order_nodes, second_order_nodes, perspective_nodes, bridge_nodes, global_nodes]:
            for canon, node in d.items():
                if canon in all_nodes:
                    # Merge: keep higher scores
                    existing = all_nodes[canon]
                    existing.direct_match_score = max(existing.direct_match_score, node.direct_match_score)
                    existing.relationship_score = max(existing.relationship_score, node.relationship_score)
                    existing.evidence_strength = max(existing.evidence_strength, node.evidence_strength)
                    existing.source_entities = list(set(existing.source_entities + node.source_entities))
                    # Upgrade category if better
                    if node.category.value < existing.category.value:
                        existing.category = node.category
                else:
                    all_nodes[canon] = node

        # Near-duplicate detection (by content similarity on summaries)
        deduped: Dict[str, ScoredNode] = {}
        duplicates_removed = 0
        for canon in sorted(all_nodes.keys()):
            node = all_nodes[canon]
            is_duplicate = False
            for existing_canon, existing in deduped.items():
                sim = self._summary_similarity(node.summary, existing.summary)
                if sim > MIN_SIMILARITY_THRESHOLD and node.node_type == existing.node_type:
                    # Merge into existing
                    existing.direct_match_score = max(existing.direct_match_score, node.direct_match_score)
                    existing.relationship_score = max(existing.relationship_score, node.relationship_score)
                    existing.source_entities = list(set(existing.source_entities + node.source_entities))
                    is_duplicate = True
                    duplicates_removed += 1
                    break
            if not is_duplicate:
                deduped[canon] = node

        # --- PHASE 7: Category caps with diversity enforcement ---
        final_selection: List[ScoredNode] = []

        # Direct: take all (they're the most relevant)
        direct = [n for n in deduped.values() if n.category == EvidenceCategory.DIRECT]
        direct.sort(key=lambda n: n.composite_score, reverse=True)
        final_selection.extend(direct[:MAX_DIRECT_EVIDENCE_NODES])

        # First-order
        first = [n for n in deduped.values() if n.category == EvidenceCategory.FIRST_ORDER]
        first.sort(key=lambda n: n.composite_score, reverse=True)
        final_selection.extend(first[:MAX_FIRST_ORDER_NODES])

        # Second-order
        second = [n for n in deduped.values() if n.category == EvidenceCategory.SECOND_ORDER]
        second.sort(key=lambda n: n.composite_score, reverse=True)
        final_selection.extend(second[:MAX_SECOND_ORDER_NODES])

        # Peripheral (low-value second-order)
        peripheral = [n for n in deduped.values() if n.category == EvidenceCategory.PERIPHERAL]
        peripheral.sort(key=lambda n: n.composite_score, reverse=True)
        final_selection.extend(peripheral[:MAX_PERIPHERAL_NODES])

        # Perspective
        persp = [n for n in deduped.values() if n.category == EvidenceCategory.PERSPECTIVE]
        persp.sort(key=lambda n: n.composite_score, reverse=True)
        final_selection.extend(persp[:MAX_PERSPECTIVE_NODES])

        # Bridge
        bridge = [n for n in deduped.values() if n.category == EvidenceCategory.BRIDGE]
        bridge.sort(key=lambda n: n.composite_score, reverse=True)
        final_selection.extend(bridge[:MAX_BRIDGE_NODES])

        # Global (diversity)
        glob = [n for n in deduped.values() if n.category == EvidenceCategory.GLOBAL]
        glob.sort(key=lambda n: n.composite_score, reverse=True)
        final_selection.extend(glob[:MAX_GLOBAL_REGISTRY_NODES])

        # Compute diversity scores
        sector_counts: Dict[str, int] = {}
        country_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for n in final_selection:
            sector_counts[n.sector.lower()] = sector_counts.get(n.sector.lower(), 0) + 1
            country_counts[n.country.lower()] = country_counts.get(n.country.lower(), 0) + 1
            type_counts[n.node_type.lower()] = type_counts.get(n.node_type.lower(), 0) + 1

        for n in final_selection:
            # Higher diversity score if in underrepresented categories
            sector_rarity = 1.0 / max(sector_counts.get(n.sector.lower(), 1), 1)
            country_rarity = 1.0 / max(country_counts.get(n.country.lower(), 1), 1)
            type_rarity = 1.0 / max(type_counts.get(n.node_type.lower(), 1), 1)
            n.diversity_score = (sector_rarity + country_rarity + type_rarity) / 3.0

        # Re-sort by composite score (which now includes diversity)
        final_selection.sort(key=lambda n: n.composite_score, reverse=True)

        # Update reasoning log
        reasoning_log.candidate_nodes = len(self.file_map)
        reasoning_log.backlink_candidates = len(first_order_nodes) + len(second_order_nodes)
        reasoning_log.relevant_nodes = len(deduped)
        reasoning_log.selected_evidence = len(final_selection)
        reasoning_log.deduplicated_nodes = duplicates_removed
        reasoning_log.weak_backlinks_filtered = len(second_order_nodes) - len([n for n in final_selection if n.category == EvidenceCategory.SECOND_ORDER])

        logger.info("Evidence selection: %d direct, %d first-order, %d second-order, "
                    "%d perspective, %d bridge, %d global | %d deduplicated | %d final",
                    len(direct), len(first), len(second), len(persp), len(bridge), len(glob),
                    duplicates_removed, len(final_selection))

        return final_selection

    def _compute_relationship_score(
        self,
        node_canon: str,
        entity_names: List[str],
        perspective_norm: str,
        source_countries: Set[str],
    ) -> float:
        """Score a node's relationship relevance to the event."""
        meta = self.node_metadata.get(node_canon, {})
        content = self.node_content.get(node_canon, "").lower()
        score = 0.0

        # Entity name matches in content
        for name in entity_names:
            name_lower = name.lower()
            if name_lower in content:
                score += 0.3
            canon_name = self._canonicalize(name)
            if canon_name in self._canonicalize(meta.get("summary", "")):
                score += 0.2

        # Country match
        node_country = meta.get("country", "").lower()
        if node_country in source_countries:
            score += 0.25
        if node_country == perspective_norm:
            score += 0.15

        # Type relevance (actors, infrastructure, policy are more relevant)
        node_type = meta.get("type", "").lower()
        if node_type in ("government_agency", "government_ministry", "private_conglomerate"):
            score += 0.1

        # Content richness (more content = more evidence)
        if len(content) > 1000:
            score += 0.05

        return min(score, 1.0)

    @staticmethod
    def _summary_similarity(a: str, b: str) -> float:
        """Simple Jaccard similarity on word sets for near-duplicate detection."""
        if not a or not b:
            return 0.0
        words_a = set(re.findall(r'\b\w+\b', a.lower()))
        words_b = set(re.findall(r'\b\w+\b', b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def build_node_context_block(self, node: ScoredNode, max_chars: int = 2000) -> str:
        """Build a structured context block for a single node."""
        content = node.content[:max_chars]
        if len(node.content) > max_chars:
            content += "\n[CONTENT TRUNCATED FOR TOKEN BUDGET]"

        lines = [
            f"=== NODE: {node.node_id} ===",
            f"Type: {node.node_type or 'unknown'} | Country: {node.country or 'N/A'} | Sector: {node.sector or 'N/A'}",
            f"Relevance: {node.composite_score:.2f} | Category: {node.category.value} | Distance: {node.backlink_distance}",
            f"Selection reason: {node.selection_reason}",
            f"Source entities: {', '.join(node.source_entities) or 'N/A'}",
            "---",
            content,
            "=== END NODE ===",
        ]
        return "\n".join(lines)

    def build_cross_border_bridge_context(
        self,
        perspective: PerspectiveContext,
        source_country: str,
    ) -> List[Dict[str, Any]]:
        """Find cross-border bridges between perspective and source country."""
        perspective_norm = perspective.country.lower()
        source_norm = source_country.lower()
        bridges: List[Dict[str, Any]] = []

        for canon, meta in sorted(self.node_metadata.items()):
            node_country = meta.get("country", "").lower()
            if node_country not in (perspective_norm, source_norm):
                continue

            actual = self.file_map.get(canon, canon)
            for link_target in self.outbound_links.get(canon, []):
                link_canon = self._canonicalize(link_target)
                link_meta = self.node_metadata.get(link_canon, {})
                link_country = link_meta.get("country", "").lower()

                if node_country == perspective_norm and link_country == source_norm:
                    bridges.append({
                        "from_node": actual,
                        "from_country": perspective.country,
                        "to_node": self.file_map.get(link_canon, link_target),
                        "to_country": source_country,
                        "relationship_type": "outbound_link",
                    })
                elif node_country == source_norm and link_country == perspective_norm:
                    bridges.append({
                        "from_node": actual,
                        "from_country": source_country,
                        "to_node": self.file_map.get(link_canon, link_target),
                        "to_country": perspective.country,
                        "relationship_type": "outbound_link",
                    })

            # Backlinks
            for inbound in self.backlink_map.get(canon, set()):
                inbound_canon = self._canonicalize(inbound)
                inbound_meta = self.node_metadata.get(inbound_canon, {})
                inbound_country = inbound_meta.get("country", "").lower()

                if node_country == perspective_norm and inbound_country == source_norm:
                    bridges.append({
                        "from_node": inbound,
                        "from_country": source_country,
                        "to_node": actual,
                        "to_country": perspective.country,
                        "relationship_type": "backlink",
                    })
                elif node_country == source_norm and inbound_country == perspective_norm:
                    bridges.append({
                        "from_node": inbound,
                        "from_country": perspective.country,
                        "to_node": actual,
                        "to_country": source_country,
                        "relationship_type": "backlink",
                    })

        return sorted(bridges, key=lambda b: (b.get("from_node", ""), b.get("to_node", "")))


# --------------------------------------------------------------------------- #
# System Prompts
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

PROMPT_EVIDENCE_ANALYSIS: str = (
    "You are the ATIS Evidence Analysis Module. You will analyze a subset of evidence nodes "
    "related to a news event. Your task is to extract structured findings, identify relationships, "
    "and flag potential opportunities and risks.\n\n"
    "OUTPUT SCHEMA (raw JSON only):\n"
    "{\n"
    '  "findings": [\n'
    '    {"finding": "Precise statement", "confidence": 0.85, "supporting_nodes": ["NodeID"], "category": "economic|political|infrastructure|regulatory"}\n'
    "  ],\n"
    '  "relationships": [\n'
    '    {"from": "NodeA", "to": "NodeB", "relationship": "regulates|funds|operates|supplies", "evidence": "quote or reasoning"}\n'
    "  ],\n"
    '  "opportunity_signals": [\n'
    '    {"signal": "Description", "confidence": 0.72, "type": "explicit|derived|potential", "supporting_nodes": ["NodeID"], "rationale": "Why this is an opportunity"}\n'
    "  ],\n"
    '  "risk_signals": [\n'
    '    {"signal": "Description", "confidence": 0.65, "severity": "high|medium|low", "supporting_nodes": ["NodeID"]}\n'
    "  ],\n"
    '  "key_entities": ["NodeID1", "NodeID2"],\n'
    '  "gaps": ["What information is missing"]\n'
    "}\n\n"
    "RULES:\n"
    "1. Every finding MUST cite supporting_nodes from the provided evidence.\n"
    "2. Do NOT invent facts not present in the evidence.\n"
    "3. Confidence scores must be justified by evidence quality.\n"
    "4. Distinguish: EXPLICIT (source states it), DERIVED (reasonable inference), POTENTIAL (plausible but weak).\n"
    "5. Output ONLY raw JSON."
)

PROMPT_RELATIONSHIP_SYNTHESIS: str = (
    "You are the ATIS Relationship Synthesis Module. You will receive intermediate findings "
    "from multiple evidence partitions. Your task is to synthesize cross-partition relationships, "
    "resolve contradictions, and identify the most important patterns.\n\n"
    "OUTPUT SCHEMA (raw JSON only):\n"
    "{\n"
    '  "synthesized_findings": [\n'
    '    {"finding": "...", "confidence": 0.88, "supporting_partitions": [1, 3], "supporting_nodes": ["NodeID"]}\n'
    "  ],\n"
    '  "cross_relationships": [\n'
    '    {"from_partition": 1, "to_partition": 2, "relationship": "...", "nodes": ["NodeA", "NodeB"]}\n'
    "  ],\n"
    '  "contradictions": [\n'
    '    {"partitions": [1, 2], "description": "...", "resolution": "..."}\n'
    "  ],\n"
    '  "consolidated_opportunities": [\n'
    '    {"title": "...", "type": "explicit|derived|potential", "confidence": 0.75, "supporting_nodes": ["NodeID"], "rationale": "..."}\n'
    "  ],\n"
    '  "consolidated_risks": [\n'
    '    {"description": "...", "severity": "high|medium|low", "supporting_nodes": ["NodeID"]}\n'
    "  ],\n"
    '  "key_themes": ["theme1", "theme2"]\n'
    "}\n\n"
    "RULES:\n"
    "1. Preserve all source node references.\n"
    "2. Resolve contradictions explicitly — do not ignore them.\n"
    "3. Consolidate duplicate opportunity signals into the strongest single entry.\n"
    "4. Output ONLY raw JSON."
)

PROMPT_FINAL_SYNTHESIS: str = (
    "You are the ATIS Final Synthesis Engine. You will receive distilled findings, "
    "relationship discoveries, and opportunity/risk signals from prior reasoning stages. "
    "Your task is to produce the final ATIS News analytical dashboard.\n\n"
    "CRITICAL RULES:\n"
    "1. You MUST select perspective_actor from the PERSPECTIVE ACTOR REGISTRY. Do not invent actors.\n"
    "2. You MUST select perspective_capability from the capabilities listed for that actor.\n"
    "3. You MUST select pathway from the CROSS-BORDER BRIDGE CONTEXT or from: "
    "export, procurement, supplier relationship, regional tender, joint venture, partnership, "
    "investment, financing, logistics, professional services, technology transfer, "
    "regional infrastructure, power trade, regulatory arbitrage, market entry.\n"
    "4. You MUST set opportunity_country to the actual country where the commercial opportunity exists.\n"
    "5. Distinguish local source-country opportunities from perspective-country opportunities.\n"
    "6. Opportunity detection: EXPLICIT (source states it), DERIVED (evidence supports it), POTENTIAL (plausible but weak).\n"
    "7. Do NOT force opportunities. If no defensible opportunity exists, return an empty opportunities array.\n"
    "8. Every opportunity MUST have supporting_nodes from the provided evidence.\n"
    "9. Output ONLY valid raw JSON.\n\n"
    "JSON SCHEMA:\n"
    "{\n"
    '  "intelligence_id": "ATIS-INT-GENERIC",\n'
    '  "trigger_event": "String",\n'
    '  "market_equilibrium_shift": "String",\n'
    '  "source_country": "String",\n'
    '  "event_country": "String",\n'
    '  "executive_summary": "6-10 sentence comprehensive narrative...",\n'
    '  "structured_intelligence": [\n'
    "    {\n"
    '      "entity": "Entity Name", "type": "entity_type", "country": "...", "relationship": "...",\n'
    '      "status": "...", "priority": "Critical|High|Medium|Low", "insight": "...", "source_node": "NodeID"\n'
    "    }\n"
    "  ],\n"
    '  "findings": [\n'
    '    {"text": "Finding.", "source_nodes": ["NodeID"]}\n'
    "  ],\n"
    '  "opportunities": [\n'
    "    {\n"
    '      "opportunity_id": "OPP-...", "title": "...", "type": "String",\n'
    '      "perspective_country": "...", "perspective_country_code": "...",\n'
    '      "source_country": "...", "event_country": "...", "opportunity_country": "...",\n'
    '      "cross_border": true, "cross_border_countries": ["..."],\n'
    '      "perspective_actor": "MUST be from registry", "perspective_capability": "MUST be evidenced",\n'
    '      "pathway": "MUST be evidenced", "urgency_score": 0.0, "feasibility_score": 0.0,\n'
    '      "required_missing_nodes": [], "capital_flow": {"beneficiary": "...", "likely_funder": "..."},\n'
    '      "justification": "...", "source_nodes": ["NodeID"],\n'
    '      "opportunity_type": "explicit|derived|potential", "opportunity_confidence": 0.0\n'
    "    }\n"
    "  ],\n"
    '  "risks": [\n'
    '    {"text": "Risk.", "source_nodes": ["NodeID"], "severity": "high|medium|low"}\n'
    "  ],\n"
    '  "key_entities": [\n'
    "    {\n"
    '      "entity_name": "...", "entity_type": "...", "country": "...", "sector": "...",\n'
    '      "significance_score": 9, "summary": "...", "source_node": "NodeID"\n'
    "    }\n"
    "  ]\n"
    "}"
)

PROMPT_SINGLE_STAGE_ANALYSIS: str = (
    "You are the ATIS Equilibrium and Constraint Engine. Analyze the provided news event "
    "and evidence to produce a complete intelligence dashboard.\n\n"
    "CRITICAL RULES:\n"
    "1. You MUST select perspective_actor from the PERSPECTIVE ACTOR REGISTRY.\n"
    "2. You MUST select perspective_capability from the capabilities listed for that actor.\n"
    "3. You MUST select pathway from the CROSS-BORDER BRIDGE CONTEXT or enumerated list.\n"
    "4. You MUST set opportunity_country to the actual country where the commercial opportunity exists.\n"
    "5. Distinguish: EXPLICIT, DERIVED, and POTENTIAL opportunities with different confidence levels.\n"
    "6. Do NOT force opportunities — return empty array if none are defensible.\n"
    "7. Every claim MUST cite source_node IDs from the provided evidence.\n"
    "8. Output ONLY valid raw JSON.\n\n"
    "Use the same JSON schema as the Final Synthesis prompt above."
)


# --------------------------------------------------------------------------- #
# News LLM Orchestrator
# --------------------------------------------------------------------------- #
class NewsLLMOrchestrator:
    """
    Orchestrates LLM calls for the News pipeline with:
      - Automatic token budget calculation against actual model capabilities
      - Dynamic workload splitting (single-call vs multi-stage)
      - Progressive evidence compression
      - Evidence provenance preservation
      - Failure safety (never exceeds provider context limit)
    """

    def __init__(self) -> None:
        self.client: LLMClient = get_client()
        self.config = self.client.config
        self.budget = TokenBudgetManager(self.client.adapter.capabilities)
        self.cache = AnalysisCache()
        self._call_count = 0

    def _model_output_cap(self) -> int:
        return self.budget.max_output_tokens

    def _is_truncated(self, raw: str) -> bool:
        text = raw.strip()
        if not text:
            return False
        if text.endswith("..."):
            return True
        if text[-1] not in {"}", "]", "\"", ">", "'"}:
            if not (text[-1].isdigit() or text[-1].lower() in {"e", "l"}):
                return True
        # Structural check
        stack: List[str] = []
        in_string = False
        escape = False
        for ch in text:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]" and stack:
                expected = "}" if stack[-1] == "{" else "]"
                if ch == expected:
                    stack.pop()
        return in_string or bool(stack)

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        stage_name: str = "LLM call",
    ) -> str:
        if max_tokens is None:
            max_tokens = 4096
        cap = self._model_output_cap()
        max_tokens = min(max_tokens, cap)

        # Pre-flight token safety check
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_tokens = self.budget.estimate_messages_tokens(messages)
        self.budget.require_budget(input_tokens, max_tokens, stage_name)

        self._call_count += 1
        return self.client.chat(messages, temperature=0.0, max_tokens=max_tokens)

    def _call_api_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        stage_name: str = "LLM call",
    ) -> str:
        cap = self._model_output_cap()
        max_tokens = min(max_tokens, cap)
        raw = self._call_api(system_prompt, user_prompt, max_tokens=max_tokens, stage_name=stage_name)
        if self._is_truncated(raw):
            retry_tokens = min(max_tokens * 2, cap)
            if retry_tokens > max_tokens:
                logger.warning("%s truncated (len=%d). Retrying with %d tokens.", stage_name, len(raw), retry_tokens)
                raw = self._call_api(system_prompt, user_prompt, max_tokens=retry_tokens, stage_name=stage_name)
                if self._is_truncated(raw):
                    logger.error("%s still truncated after retry.", stage_name)
            else:
                logger.error("%s truncated at model cap (%d).", stage_name, cap)
        return raw

    # ------------------------------------------------------------------ #
    # Stage 1: Entity Extraction (always single call — article is input)
    # ------------------------------------------------------------------ #
    def stage_1_extract(self, article_text: str, reasoning_log: ReasoningLog) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("STAGE 1: ENTITY EXTRACTION")
        logger.info("=" * 60)

        estimated = self.budget.estimate_tokens(PROMPT_STAGE_1_EXTRACTOR + article_text)
        logger.info("Estimated Stage-1 input tokens: %d (budget: %d)", estimated, self.budget.usable_context_budget)

        if not self.budget.fits_in_budget(estimated, 2048):
            max_chars = int((self.budget.usable_context_budget - 2048) * 3.2) - len(PROMPT_STAGE_1_EXTRACTOR)
            article_text = article_text[:max(max_chars, 500)] + "\n[ARTICLE TRUNCATED TO RESPECT TOKEN BUDGET]"
            logger.warning("Article truncated to fit Stage-1 token budget.")

        raw_response = self._call_api(PROMPT_STAGE_1_EXTRACTOR, article_text, max_tokens=2048, stage_name="Stage 1")
        data = safe_json_loads(raw_response, stage_name="Stage 1")

        entity_count = len(data.get("entities", []))
        reasoning_log.entities_extracted = entity_count
        logger.info(
            "Stage 1 complete. Extracted %d entities. Core: %s | Source: %s | Event: %s",
            entity_count,
            data.get("core_event", "N/A"),
            data.get("source_country", "N/A"),
            data.get("event_country", "N/A"),
        )
        return data

    # ------------------------------------------------------------------ #
    # Stage 2: Evidence Analysis (partitioned if needed)
    # ------------------------------------------------------------------ #
    def stage_2_analyze_evidence(
        self,
        article_text: str,
        evidence_partitions: List[EvidencePartition],
        perspective: PerspectiveContext,
        bridge_context: str,
        perspective_registry: str,
        reasoning_log: ReasoningLog,
    ) -> List[Dict[str, Any]]:
        """
        Analyze evidence partitions. Each partition gets its own LLM call.
        Returns list of intermediate findings (one per partition).
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: EVIDENCE ANALYSIS (%d partitions)", len(evidence_partitions))
        logger.info("=" * 60)

        partition_results: List[Dict[str, Any]] = []

        for partition in evidence_partitions:
            logger.info("Analyzing partition %d: %d nodes, ~%d tokens | theme: %s",
                        partition.partition_id, len(partition.nodes), partition.estimated_tokens, partition.theme)

            # Build evidence block
            evidence_blocks = []
            for node in partition.nodes:
                block = self._build_evidence_block(node)
                evidence_blocks.append(block)

            evidence_text = "\n\n".join(evidence_blocks)

            user_prompt = (
                f"## NEWS EVENT\n{article_text[:2000]}\n\n"
                f"## ANALYTICAL PERSPECTIVE\n{perspective.country} ({perspective.country_code})\n\n"
                f"## PERSPECTIVE ACTOR REGISTRY\n{perspective_registry}\n\n"
                f"## CROSS-BORDER BRIDGE CONTEXT\n{bridge_context}\n\n"
                f"## EVIDENCE PARTITION {partition.partition_id}: {partition.theme}\n"
                f"{evidence_text}\n\n"
                f"Analyze this evidence partition and return structured findings."
            )

            estimated_input = self.budget.estimate_tokens(PROMPT_EVIDENCE_ANALYSIS + user_prompt)
            logger.info("Partition %d estimated input: %d tokens", partition.partition_id, estimated_input)

            raw = self._call_api_with_retry(
                PROMPT_EVIDENCE_ANALYSIS,
                user_prompt,
                max_tokens=4096,
                stage_name=f"Evidence Analysis P{partition.partition_id}",
            )

            try:
                result = safe_json_loads(raw, stage_name=f"Evidence Analysis P{partition.partition_id}")
            except RuntimeError as exc:
                logger.error("Partition %d analysis failed: %s", partition.partition_id, exc)
                result = {"findings": [], "relationships": [], "opportunity_signals": [],
                          "risk_signals": [], "key_entities": [], "gaps": [str(exc)]}

            result["_partition_id"] = partition.partition_id
            result["_theme"] = partition.theme
            partition_results.append(result)
            reasoning_log.evidence_calls += 1

        return partition_results

    def _build_evidence_block(self, node: ScoredNode, max_chars: int = 1500) -> str:
        content = node.content[:max_chars]
        if len(node.content) > max_chars:
            content += "\n[TRUNCATED]"
        return (
            f"--- NODE: {node.node_id} ---\n"
            f"Type: {node.node_type or 'unknown'} | Country: {node.country or 'N/A'} | Sector: {node.sector or 'N/A'}\n"
            f"Relevance: {node.composite_score:.2f} | Category: {node.category.value}\n"
            f"Reason: {node.selection_reason}\n"
            f"{content}\n"
            f"--- END ---"
        )

    # ------------------------------------------------------------------ #
    # Stage 3: Relationship Synthesis (if multi-partition)
    # ------------------------------------------------------------------ #
    def stage_3_synthesize_relationships(
        self,
        partition_results: List[Dict[str, Any]],
        perspective: PerspectiveContext,
        reasoning_log: ReasoningLog,
    ) -> Dict[str, Any]:
        """Synthesize findings across partitions."""
        logger.info("=" * 60)
        logger.info("STAGE 3: RELATIONSHIP SYNTHESIS (%d partitions)", len(partition_results))
        logger.info("=" * 60)

        # Compress partition results into a synthesis prompt
        compressed_partitions = []
        for pr in partition_results:
            pid = pr.get("_partition_id", 0)
            findings = pr.get("findings", [])
            relationships = pr.get("relationships", [])
            opp_signals = pr.get("opportunity_signals", [])
            risk_signals = pr.get("risk_signals", [])

            compressed = {
                "partition_id": pid,
                "theme": pr.get("_theme", ""),
                "finding_count": len(findings),
                "top_findings": [f["finding"] for f in findings[:5]],
                "relationship_count": len(relationships),
                "top_relationships": relationships[:5],
                "opportunity_signals": [o["signal"] for o in opp_signals[:5]],
                "risk_signals": [r["signal"] for r in risk_signals[:5]],
                "key_entities": pr.get("key_entities", []),
                "gaps": pr.get("gaps", []),
            }
            compressed_partitions.append(compressed)

        partitions_json = json.dumps(compressed_partitions, indent=2, ensure_ascii=False)

        user_prompt = (
            f"## ANALYTICAL PERSPECTIVE\n{perspective.country} ({perspective.country_code})\n\n"
            f"## INTERMEDIATE FINDINGS FROM {len(partition_results)} PARTITIONS\n"
            f"{partitions_json}\n\n"
            f"Synthesize cross-partition relationships, resolve contradictions, and consolidate opportunities."
        )

        estimated = self.budget.estimate_tokens(PROMPT_RELATIONSHIP_SYNTHESIS + user_prompt)

        # If still too large, compress further
        if not self.budget.fits_in_budget(estimated, 4096):
            logger.warning("Synthesis prompt too large (%d tokens). Compressing further.", estimated)
            # Ultra-compressed: just counts and top items
            ultra = []
            for pr in partition_results:
                ultra.append({
                    "partition_id": pr.get("_partition_id", 0),
                    "theme": pr.get("_theme", ""),
                    "findings": [f["finding"] for f in pr.get("findings", [])[:3]],
                    "opportunities": [o["signal"] for o in pr.get("opportunity_signals", [])[:3]],
                    "risks": [r["signal"] for r in pr.get("risk_signals", [])[:3]],
                })
            user_prompt = (
                f"## PERSPECTIVE: {perspective.country}\n"
                f"## PARTITIONS: {json.dumps(ultra, ensure_ascii=False)}\n"
                f"Synthesize and return JSON."
            )
            estimated = self.budget.estimate_tokens(PROMPT_RELATIONSHIP_SYNTHESIS + user_prompt)

        raw = self._call_api_with_retry(
            PROMPT_RELATIONSHIP_SYNTHESIS,
            user_prompt,
            max_tokens=4096,
            stage_name="Relationship Synthesis",
        )

        try:
            result = safe_json_loads(raw, stage_name="Relationship Synthesis")
        except RuntimeError as exc:
            logger.error("Relationship synthesis failed: %s", exc)
            result = {
                "synthesized_findings": [],
                "cross_relationships": [],
                "contradictions": [],
                "consolidated_opportunities": [],
                "consolidated_risks": [],
                "key_themes": [],
            }

        reasoning_log.synthesis_calls += 1
        return result

    # ------------------------------------------------------------------ #
    # Stage 4: Final Synthesis (always receives distilled intelligence)
    # ------------------------------------------------------------------ #
    def stage_4_final_synthesis(
        self,
        article_text: str,
        distilled_findings: Dict[str, Any],
        perspective: PerspectiveContext,
        perspective_registry: str,
        bridge_context: str,
        reasoning_log: ReasoningLog,
    ) -> Dict[str, Any]:
        """Produce final dashboard from distilled findings."""
        logger.info("=" * 60)
        logger.info("STAGE 4: FINAL SYNTHESIS")
        logger.info("=" * 60)

        # Build distilled context (never the raw massive knowledge base)
        distilled_text = json.dumps(distilled_findings, indent=2, ensure_ascii=False)

        user_prompt = (
            f"## NEWS EVENT\n{article_text[:1500]}\n\n"
            f"## ANALYTICAL PERSPECTIVE\n{perspective.country} ({perspective.country_code})\n\n"
            f"## PERSPECTIVE ACTOR REGISTRY\n{perspective_registry}\n\n"
            f"## CROSS-BORDER BRIDGE CONTEXT\n{bridge_context}\n\n"
            f"## DISTILLED INTELLIGENCE\n"
            f"{distilled_text}\n\n"
            f"Produce the final ATIS News analytical dashboard."
        )

        estimated = self.budget.estimate_tokens(PROMPT_FINAL_SYNTHESIS + user_prompt)
        logger.info("Final synthesis estimated input: %d tokens", estimated)

        if not self.budget.fits_in_budget(estimated, 8192):
            # Compress distilled findings further
            logger.warning("Final synthesis too large. Compressing distilled findings.")
            compressed = {
                "synthesized_findings": distilled_findings.get("synthesized_findings", [])[:10],
                "consolidated_opportunities": distilled_findings.get("consolidated_opportunities", [])[:10],
                "consolidated_risks": distilled_findings.get("consolidated_risks", [])[:10],
                "key_themes": distilled_findings.get("key_themes", []),
                "contradictions": distilled_findings.get("contradictions", [])[:5],
            }
            user_prompt = (
                f"## NEWS EVENT\n{article_text[:1000]}\n"
                f"## PERSPECTIVE: {perspective.country}\n"
                f"## BRIDGES: {bridge_context[:500]}\n"
                f"## DISTILLED: {json.dumps(compressed, ensure_ascii=False)}\n"
                f"Produce final dashboard JSON."
            )
            estimated = self.budget.estimate_tokens(PROMPT_FINAL_SYNTHESIS + user_prompt)

        raw = self._call_api_with_retry(
            PROMPT_FINAL_SYNTHESIS,
            user_prompt,
            max_tokens=8192,
            stage_name="Final Synthesis",
        )

        dashboard = safe_json_loads(raw, stage_name="Final Synthesis")
        reasoning_log.final_call = 1
        logger.info("Final synthesis complete.")
        return dashboard

    # ------------------------------------------------------------------ #
    # Single-Stage Analysis (for small workloads)
    # ------------------------------------------------------------------ #
    def single_stage_analysis(
        self,
        article_text: str,
        selected_nodes: List[ScoredNode],
        perspective: PerspectiveContext,
        perspective_registry: str,
        bridge_context: str,
        reasoning_log: ReasoningLog,
    ) -> Dict[str, Any]:
        """Single LLM call for small evidence sets."""
        logger.info("=" * 60)
        logger.info("SINGLE-STAGE ANALYSIS (%d nodes)", len(selected_nodes))
        logger.info("=" * 60)

        evidence_blocks = []
        for node in selected_nodes:
            evidence_blocks.append(self._build_evidence_block(node))

        evidence_text = "\n\n".join(evidence_blocks)

        user_prompt = (
            f"## NEWS EVENT\n{article_text}\n\n"
            f"## ANALYTICAL PERSPECTIVE\n{perspective.country} ({perspective.country_code})\n\n"
            f"## PERSPECTIVE ACTOR REGISTRY\n{perspective_registry}\n\n"
            f"## CROSS-BORDER BRIDGE CONTEXT\n{bridge_context}\n\n"
            f"## EVIDENCE\n"
            f"{evidence_text}\n\n"
            f"Analyze the event and evidence. Produce the final ATIS News dashboard."
        )

        estimated = self.budget.estimate_tokens(PROMPT_SINGLE_STAGE_ANALYSIS + user_prompt)
        logger.info("Single-stage estimated input: %d tokens (budget: %d)",
                    estimated, self.budget.usable_context_budget)

        if not self.budget.fits_in_budget(estimated, 8192):
            raise LLMTokenLimitError(
                f"Single-stage analysis exceeds budget ({estimated} > {self.budget.usable_context_budget}). "
                f"This should have been caught earlier and switched to multi-stage."
            )

        raw = self._call_api_with_retry(
            PROMPT_SINGLE_STAGE_ANALYSIS,
            user_prompt,
            max_tokens=8192,
            stage_name="Single-Stage Analysis",
        )

        dashboard = safe_json_loads(raw, stage_name="Single-Stage Analysis")
        reasoning_log.reasoning_mode = ReasoningMode.SINGLE.value
        reasoning_log.final_call = 1
        reasoning_log.total_llm_calls = reasoning_log.evidence_calls + reasoning_log.synthesis_calls + reasoning_log.final_call + 1  # +1 for stage 1
        logger.info("Single-stage analysis complete.")
        return dashboard


# --------------------------------------------------------------------------- #
# Evidence Partitioning
# --------------------------------------------------------------------------- #
def partition_evidence(
    nodes: List[ScoredNode],
    budget: TokenBudgetManager,
    system_prompt: str,
    article_text: str,
    perspective_registry: str,
    bridge_context: str,
    max_nodes_per_partition: int = MAX_NODES_PER_PARTITION,
) -> List[EvidencePartition]:
    """
    Partition selected evidence into thematically coherent groups that each
    fit within the token budget. Returns partitions sorted by importance.
    """
    if not nodes:
        return []

    # Group nodes by sector for thematic coherence
    sector_groups: Dict[str, List[ScoredNode]] = {}
    for node in nodes:
        sector = node.sector or "general"
        if sector not in sector_groups:
            sector_groups[sector] = []
        sector_groups[sector].append(node)

    # Sort groups by total composite score
    sorted_sectors = sorted(
        sector_groups.items(),
        key=lambda item: sum(n.composite_score for n in item[1]),
        reverse=True,
    )

    partitions: List[EvidencePartition] = []
    current_partition_nodes: List[ScoredNode] = []
    current_theme = ""
    current_tokens = 0
    partition_id = 0

    # Base overhead: system prompt + article snippet + perspective + bridges
    base_text = system_prompt + article_text[:1000] + perspective_registry + bridge_context
    base_tokens = budget.estimate_tokens(base_text)
    available_per_partition = budget.usable_context_budget - base_tokens - 4096  # reserve output

    for sector, sector_nodes in sorted_sectors:
        for node in sorted(sector_nodes, key=lambda n: n.composite_score, reverse=True):
            node_block = _estimate_node_block_tokens(node)

            if not current_partition_nodes:
                current_theme = sector
                current_partition_nodes.append(node)
                current_tokens = node_block
            elif (len(current_partition_nodes) < max_nodes_per_partition and
                  current_tokens + node_block < available_per_partition):
                current_partition_nodes.append(node)
                current_tokens += node_block
            else:
                # Finalize current partition
                partitions.append(EvidencePartition(
                    partition_id=partition_id,
                    nodes=list(current_partition_nodes),
                    theme=current_theme,
                    estimated_tokens=base_tokens + current_tokens,
                ))
                partition_id += 1

                # Start new partition
                current_partition_nodes = [node]
                current_theme = sector
                current_tokens = node_block

    # Don't forget the last partition
    if current_partition_nodes:
        partitions.append(EvidencePartition(
            partition_id=partition_id,
            nodes=list(current_partition_nodes),
            theme=current_theme,
            estimated_tokens=base_tokens + current_tokens,
        ))

    # Cap partitions
    if len(partitions) > MAX_PARTITIONS:
        logger.warning("Too many partitions (%d). Merging smallest into largest.", len(partitions))
        # Sort by importance (total score)
        partitions.sort(key=lambda p: sum(n.composite_score for n in p.nodes), reverse=True)
        merged = partitions[:MAX_PARTITIONS - 1]
        remainder = []
        for p in partitions[MAX_PARTITIONS - 1:]:
            remainder.extend(p.nodes)
        merged.append(EvidencePartition(
            partition_id=MAX_PARTITIONS - 1,
            nodes=remainder,
            theme="mixed",
            estimated_tokens=budget.estimate_tokens(system_prompt + "\n".join(n.summary for n in remainder)),
        ))
        partitions = merged

    logger.info("Evidence partitioned into %d groups", len(partitions))
    for p in partitions:
        logger.info("  Partition %d: %d nodes, theme='%s', ~%d tokens",
                    p.partition_id, len(p.nodes), p.theme, p.estimated_tokens)

    return partitions


def _estimate_node_block_tokens(node: ScoredNode) -> int:
    """Estimate tokens for a single node's evidence block."""
    content_len = min(len(node.content), 1500)
    overhead = len(node.node_id) + len(node.node_type) + len(node.country) + len(node.sector) + 100
    return TokenBudgetManager.estimate_tokens(node.content[:content_len]) + TokenBudgetManager.estimate_tokens(" " * overhead)


# --------------------------------------------------------------------------- #
# Context Builders
# --------------------------------------------------------------------------- #
def build_perspective_registry(
    vault_manager: ObsidianVaultManager,
    perspective: PerspectiveContext,
    selected_nodes: List[ScoredNode],
    max_nodes: int = MAX_PERSPECTIVE_NODES,
) -> str:
    """Build perspective actor registry from selected perspective nodes."""
    perspective_nodes = [n for n in selected_nodes if n.category == EvidenceCategory.PERSPECTIVE]
    perspective_nodes.sort(key=lambda n: n.composite_score, reverse=True)

    lines = [f"=== PERSPECTIVE ACTOR REGISTRY ({perspective.country}) ==="]
    for node in perspective_nodes[:max_nodes]:
        lines.append(
            f"- {node.node_id} | type: {node.node_type or 'unknown'} | "
            f"sector: {node.sector or 'N/A'} | summary: {node.summary[:120]}"
        )

    if not perspective_nodes:
        lines.append(f"No perspective-country actors found in vault.")

    return "\n".join(lines)


def build_bridge_context(bridges: List[Dict[str, Any]], max_bridges: int = MAX_BRIDGE_NODES) -> str:
    """Build cross-border bridge context string."""
    if not bridges:
        return "=== CROSS-BORDER BRIDGE CONTEXT ===\nNo evidenced cross-border relationships found."

    lines = ["=== CROSS-BORDER BRIDGE CONTEXT ==="]
    for bridge in bridges[:max_bridges]:
        lines.append(
            f"- {bridge['from_node']} ({bridge['from_country']}) → "
            f"{bridge['to_node']} ({bridge['to_country']}) via {bridge['relationship_type']}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Opportunity Post-Processing
# --------------------------------------------------------------------------- #
def post_process_opportunities(
    dashboard: Dict[str, Any],
    perspective: PerspectiveContext,
    perspective_node_ids: Set[str],
    cross_border_bridges: List[Dict[str, Any]],
    source_country: str,
    event_country: str,
) -> List[Dict[str, Any]]:
    """
    Validate opportunities and assign explicit/derived/potential types with confidence.
    """
    raw_opportunities = dashboard.get("opportunities", [])
    if not raw_opportunities:
        return []

    validated = []
    for item in raw_opportunities:
        if not isinstance(item, dict):
            continue

        # Run through atis_context validation
        validated_item = validate_opportunity(
            item,
            perspective,
            source_node_ids=None,
            perspective_node_ids=perspective_node_ids,
            cross_border_bridges=cross_border_bridges,
        )

        # Determine opportunity type and confidence
        opp_type = item.get("opportunity_type", "")
        opp_confidence = item.get("opportunity_confidence", 0.0)

        if not opp_type:
            # Infer from evidence
            justification = item.get("justification", "").lower()
            if any(w in justification for w in ["directly", "announced", "signed", "agreed", "awarded"]):
                opp_type = OpportunityType.EXPLICIT.value
                opp_confidence = max(opp_confidence, 0.85)
            elif any(w in justification for w in ["could", "potential", "might", "may", "possible"]):
                opp_type = OpportunityType.POTENTIAL.value
                opp_confidence = min(opp_confidence, 0.55) if opp_confidence else 0.45
            else:
                opp_type = OpportunityType.DERIVED.value
                opp_confidence = max(opp_confidence, 0.60) if opp_confidence else 0.65

        validated_item["opportunity_type"] = opp_type
        validated_item["opportunity_confidence"] = opp_confidence

        # Stable ID
        stable_id = compute_opportunity_identity(
            title=validated_item.get("title", ""),
            perspective_country=perspective.country,
            source_country=validated_item.get("source_country", source_country),
            event_country=validated_item.get("event_country", event_country),
            opportunity_country=validated_item.get("opportunity_country", ""),
            perspective_actor=validated_item.get("perspective_actor", ""),
            perspective_capability=validated_item.get("perspective_capability", ""),
            pathway=validated_item.get("pathway", ""),
            source_nodes=validated_item.get("source_nodes", []),
        )
        validated_item["opportunity_id"] = stable_id
        validated_item["stable_opportunity_id"] = stable_id

        validated.append(validated_item)

    return validated


# --------------------------------------------------------------------------- #
# Main Orchestration Entry Point
# --------------------------------------------------------------------------- #
def _legacy_process_article_pipeline(
    article_path: str,
    perspective: Any | None = None,
) -> Dict[str, Any]:
    """
    Primary orchestration function for the ATIS News pipeline.

    Architecture:
      1. Entity Extraction (single LLM call)
      2. Intelligent Evidence Selection (backlink tracing + scoring + diversity)
      3. Token Budget Calculation
      4. Decision: Single-Call or Multi-Stage
      5. [Single] Direct analysis → Final dashboard
      6. [Multi] Evidence Partitioning → Parallel Analysis → 
                Relationship Synthesis → Final Synthesis
      7. Opportunity Validation + Output

    The caller does NOT need to know how large the knowledge base is.
    """
    logger.info("=" * 70)
    logger.info("ATIS NEWS PIPELINE INITIALISATION")
    logger.info("=" * 70)

    perspective = perspective or PerspectiveContext()
    reasoning_log = ReasoningLog()

    logger.info("Article path      : %s", article_path)
    logger.info("Perspective       : %s (%s)", perspective.country, perspective.country_code)
    logger.info("Vault directory   : %s", VAULT_DIR.resolve())
    logger.info("Model context     : %d tokens", 
                get_client().adapter.capabilities.max_context_tokens)

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
        orchestrator = NewsLLMOrchestrator()
    except ValueError as exc:
        logger.critical("%s", exc)
        raise

    # Knowledge state for determinism
    knowledge_state = KnowledgeState(vault_path=vault_manager.vault_dir)
    knowledge_state.compute()
    knowledge_state_hash = knowledge_state.knowledge_state_hash

    # ------------------------------------------------------------------ #
    # 2. Stage 1 — Entity Extraction
    # ------------------------------------------------------------------ #
    try:
        stage_1_result = orchestrator.stage_1_extract(article_text, reasoning_log)
    except Exception as exc:
        logger.critical("STAGE 1 FAILED: %s", exc)
        raise

    entities: List[Dict[str, str]] = stage_1_result.get("entities", [])
    core_event: str = stage_1_result.get("core_event", "Unknown event")
    source_country: str = stage_1_result.get("source_country", "")
    event_country: str = stage_1_result.get("event_country", source_country)

    if not entities:
        logger.warning("No entities extracted; analysis will rely on article text only.")

    # ------------------------------------------------------------------ #
    # 3. Intelligent Evidence Selection
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("INTELLIGENT EVIDENCE SELECTION")
    logger.info("=" * 60)

    selected_nodes = vault_manager.select_evidence_nodes(entities, perspective, reasoning_log)

    # Build perspective registry and bridge context
    perspective_registry = build_perspective_registry(vault_manager, perspective, selected_nodes)

    # Determine source country for bridges
    inferred_source = source_country or event_country or perspective.country
    if not inferred_source:
        for e in entities:
            ctx = e.get("context", "").lower()
            for country in COUNTRY_CODES:
                if country in ctx:
                    inferred_source = country.title()
                    break
            if inferred_source:
                break

    bridges = vault_manager.build_cross_border_bridge_context(perspective, inferred_source)
    bridge_context = build_bridge_context(bridges)

    perspective_node_ids = {n.node_id for n in selected_nodes if n.category == EvidenceCategory.PERSPECTIVE}

    # ------------------------------------------------------------------ #
    # 4. Token Budget Calculation & Mode Decision
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("TOKEN BUDGET CALCULATION")
    logger.info("=" * 60)

    # Estimate single-stage input
    single_stage_estimate = _estimate_single_stage_input(
        article_text, selected_nodes, perspective_registry, bridge_context, orchestrator.budget
    )
    reasoning_log.estimated_tokens = single_stage_estimate
    reasoning_log.safe_budget = orchestrator.budget.usable_context_budget

    logger.info("Single-stage estimate: %d tokens", single_stage_estimate)
    logger.info("Safe budget: %d tokens", orchestrator.budget.usable_context_budget)

    # Decision: single or multi-stage?
    use_multi_stage = single_stage_estimate > orchestrator.budget.usable_context_budget

    # ------------------------------------------------------------------ #
    # 5. Execute Analysis
    # ------------------------------------------------------------------ #
    try:
        if not use_multi_stage:
            # Single-stage path
            logger.info("MODE: SINGLE-STAGE REASONING")
            reasoning_log.reasoning_mode = ReasoningMode.SINGLE.value
            dashboard = orchestrator.single_stage_analysis(
                article_text, selected_nodes, perspective, perspective_registry, bridge_context, reasoning_log
            )
        else:
            # Multi-stage path
            logger.info("MODE: MULTI-STAGE REASONING")
            reasoning_log.reasoning_mode = ReasoningMode.MULTI_STAGE.value

            # Partition evidence
            partitions = partition_evidence(
                selected_nodes,
                orchestrator.budget,
                PROMPT_EVIDENCE_ANALYSIS,
                article_text,
                perspective_registry,
                bridge_context,
            )
            reasoning_log.partitions = len(partitions)

            # Stage 2: Analyze each partition
            partition_results = orchestrator.stage_2_analyze_evidence(
                article_text, partitions, perspective, bridge_context, perspective_registry, reasoning_log
            )

            # Stage 3: Synthesize relationships
            if len(partition_results) > 1:
                synthesis = orchestrator.stage_3_synthesize_relationships(
                    partition_results, perspective, reasoning_log
                )
            else:
                # Only one partition — pass through
                synthesis = {
                    "synthesized_findings": partition_results[0].get("findings", []) if partition_results else [],
                    "consolidated_opportunities": partition_results[0].get("opportunity_signals", []) if partition_results else [],
                    "consolidated_risks": partition_results[0].get("risk_signals", []) if partition_results else [],
                    "key_themes": [],
                    "contradictions": [],
                    "cross_relationships": [],
                }

            # Stage 4: Final synthesis
            dashboard = orchestrator.stage_4_final_synthesis(
                article_text, synthesis, perspective, perspective_registry, bridge_context, reasoning_log
            )
    except Exception as exc:
        logger.critical("ANALYSIS FAILED: %s", exc)
        raise

    # ------------------------------------------------------------------ #
    # 6. Enrich & Persist
    # ------------------------------------------------------------------ #
    # Post-process opportunities
    validated_opportunities = post_process_opportunities(
        dashboard, perspective, perspective_node_ids, bridges, source_country, event_country
    )
    dashboard["opportunities"] = validated_opportunities

    # Compute fingerprint
    evidence_ids = sorted([e.get("name", "") for e in entities])
    entity_ids = sorted(list(set(
        [perspective.country, source_country, event_country] +
        [n.node_id for n in selected_nodes]
    )))
    relationship_ids = sorted(list(set(
        [b["from_node"] for b in bridges] + [b["to_node"] for b in bridges]
    )))

    analysis_fingerprint = compute_analysis_fingerprint(
        story_id=core_event,
        perspective=perspective,
        evidence_ids=[e for e in evidence_ids if e],
        entity_ids=[e for e in entity_ids if e],
        relationship_ids=[r for r in relationship_ids if r],
        knowledge_state_hash=knowledge_state_hash,
    )

    # Build metadata
    dashboard["perspective"] = perspective.as_dict()
    dashboard["source_country"] = source_country
    dashboard["event_country"] = event_country
    dashboard["pipeline_metadata"] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_article": str(article_file.resolve()),
        "extracted_entities_count": len(entities),
        "core_event": core_event,
        "source_country": source_country,
        "event_country": event_country,
        "perspective_country": perspective.country,
        "perspective_country_code": perspective.country_code,
        "selected_evidence_nodes": len(selected_nodes),
        "cross_border_bridges_found": len(bridges),
        "model_primary": orchestrator.config.model,
        "model_fallback": orchestrator.config.fallback_model,
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "analysis_fingerprint": analysis_fingerprint,
        "knowledge_state": knowledge_state.as_dict(),
        "reasoning_log": reasoning_log.to_dict(),
    }

    # Persist
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_filename = f"atis_dashboard_{timestamp}.json"
    output_path_file = DASHBOARDS_DIR / output_filename

    try:
        output_path_file.write_text(
            json.dumps(dashboard, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Dashboard persisted: %s", output_path_file.resolve())
    except Exception as exc:
        logger.error("Failed to write dashboard JSON: %s", exc)
        raise

    # Log reasoning tree
    logger.info("\n%s", reasoning_log.log_tree())

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    return dashboard


def _estimate_single_stage_input(
    article_text: str,
    selected_nodes: List[ScoredNode],
    perspective_registry: str,
    bridge_context: str,
    budget: TokenBudgetManager,
) -> int:
    """Estimate tokens for a single-stage analysis call."""
    article_tokens = budget.estimate_tokens(article_text)
    evidence_tokens = sum(budget.estimate_tokens(n.content[:1500]) for n in selected_nodes)
    registry_tokens = budget.estimate_tokens(perspective_registry)
    bridge_tokens = budget.estimate_tokens(bridge_context)
    overhead = budget.estimate_tokens(PROMPT_SINGLE_STAGE_ANALYSIS) + SYSTEM_PROMPT_OVERHEAD * 4
    return article_tokens + evidence_tokens + registry_tokens + bridge_tokens + overhead


# --------------------------------------------------------------------------- #
# Web Entry Point
# --------------------------------------------------------------------------- #
def _legacy_run_news_pipeline(
    article_text: str,
    perspective: Any | None = None,
) -> Dict[str, Any]:
    """
    Web-compatible entry point. Accepts raw article text, returns dashboard JSON.

    This is the same pipeline as process_article_pipeline but accepts text directly.
    """
    logger.info("=" * 70)
    perspective = perspective or PerspectiveContext()
    reasoning_log = ReasoningLog()

    logger.info("ATIS NEWS PIPELINE (WEB) | Perspective: %s (%s)", perspective.country, perspective.country_code)
    logger.info("=" * 70)

    # Initialise
    vault_manager = ObsidianVaultManager()
    try:
        orchestrator = NewsLLMOrchestrator()
    except ValueError as exc:
        logger.critical("%s", exc)
        raise

    knowledge_state = KnowledgeState(vault_path=vault_manager.vault_dir)
    knowledge_state.compute()
    knowledge_state_hash = knowledge_state.knowledge_state_hash

    # Stage 1
    try:
        stage_1_result = orchestrator.stage_1_extract(article_text, reasoning_log)
    except Exception as exc:
        logger.critical("STAGE 1 FAILED: %s", exc)
        raise

    entities = stage_1_result.get("entities", [])
    core_event = stage_1_result.get("core_event", "Unknown event")
    source_country = stage_1_result.get("source_country", "")
    event_country = stage_1_result.get("event_country", source_country)

    # Evidence selection
    selected_nodes = vault_manager.select_evidence_nodes(entities, perspective, reasoning_log)
    perspective_registry = build_perspective_registry(vault_manager, perspective, selected_nodes)

    inferred_source = source_country or event_country or perspective.country
    if not inferred_source:
        for e in entities:
            ctx = e.get("context", "").lower()
            for country in COUNTRY_CODES:
                if country in ctx:
                    inferred_source = country.title()
                    break
            if inferred_source:
                break

    bridges = vault_manager.build_cross_border_bridge_context(perspective, inferred_source)
    bridge_context = build_bridge_context(bridges)
    perspective_node_ids = {n.node_id for n in selected_nodes if n.category == EvidenceCategory.PERSPECTIVE}

    # Budget & decision
    single_stage_estimate = _estimate_single_stage_input(
        article_text, selected_nodes, perspective_registry, bridge_context, orchestrator.budget
    )
    reasoning_log.estimated_tokens = single_stage_estimate
    reasoning_log.safe_budget = orchestrator.budget.usable_context_budget
    use_multi_stage = single_stage_estimate > orchestrator.budget.usable_context_budget

    # Execute
    try:
        if not use_multi_stage:
            reasoning_log.reasoning_mode = ReasoningMode.SINGLE.value
            dashboard = orchestrator.single_stage_analysis(
                article_text, selected_nodes, perspective, perspective_registry, bridge_context, reasoning_log
            )
        else:
            reasoning_log.reasoning_mode = ReasoningMode.MULTI_STAGE.value
            partitions = partition_evidence(
                selected_nodes, orchestrator.budget, PROMPT_EVIDENCE_ANALYSIS,
                article_text, perspective_registry, bridge_context,
            )
            reasoning_log.partitions = len(partitions)

            partition_results = orchestrator.stage_2_analyze_evidence(
                article_text, partitions, perspective, bridge_context, perspective_registry, reasoning_log
            )

            if len(partition_results) > 1:
                synthesis = orchestrator.stage_3_synthesize_relationships(
                    partition_results, perspective, reasoning_log
                )
            else:
                synthesis = {
                    "synthesized_findings": partition_results[0].get("findings", []) if partition_results else [],
                    "consolidated_opportunities": partition_results[0].get("opportunity_signals", []) if partition_results else [],
                    "consolidated_risks": partition_results[0].get("risk_signals", []) if partition_results else [],
                    "key_themes": [],
                    "contradictions": [],
                    "cross_relationships": [],
                }

            dashboard = orchestrator.stage_4_final_synthesis(
                article_text, synthesis, perspective, perspective_registry, bridge_context, reasoning_log
            )
    except Exception as exc:
        logger.critical("ANALYSIS FAILED: %s", exc)
        raise

    # Post-process
    validated_opportunities = post_process_opportunities(
        dashboard, perspective, perspective_node_ids, bridges, source_country, event_country
    )
    dashboard["opportunities"] = validated_opportunities

    # Fingerprint
    evidence_ids = sorted([e.get("name", "") for e in entities])
    entity_ids = sorted(list(set(
        [perspective.country, source_country, event_country] +
        [n.node_id for n in selected_nodes]
    )))
    relationship_ids = sorted(list(set(
        [b["from_node"] for b in bridges] + [b["to_node"] for b in bridges]
    )))

    analysis_fingerprint = compute_analysis_fingerprint(
        story_id=core_event,
        perspective=perspective,
        evidence_ids=[e for e in evidence_ids if e],
        entity_ids=[e for e in entity_ids if e],
        relationship_ids=[r for r in relationship_ids if r],
        knowledge_state_hash=knowledge_state_hash,
    )

    dashboard["perspective"] = perspective.as_dict()
    dashboard["source_country"] = source_country
    dashboard["event_country"] = event_country
    dashboard["pipeline_metadata"] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_article": "web_upload",
        "extracted_entities_count": len(entities),
        "core_event": core_event,
        "source_country": source_country,
        "event_country": event_country,
        "perspective_country": perspective.country,
        "perspective_country_code": perspective.country_code,
        "selected_evidence_nodes": len(selected_nodes),
        "cross_border_bridges_found": len(bridges),
        "model_primary": orchestrator.config.model,
        "model_fallback": orchestrator.config.fallback_model,
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "analysis_fingerprint": analysis_fingerprint,
        "knowledge_state": knowledge_state.as_dict(),
        "reasoning_log": reasoning_log.to_dict(),
    }

    # Persist
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_filename = f"atis_dashboard_{timestamp}.json"
    output_path_file = DASHBOARDS_DIR / output_filename
    try:
        output_path_file.write_text(
            json.dumps(dashboard, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Dashboard persisted: %s", output_path_file)
    except Exception as exc:
        logger.error("Failed to write dashboard: %s", exc)

    logger.info("\n%s", reasoning_log.log_tree())
    return dashboard



# =============================================================================
# ATIS NEWS v3 — PERSPECTIVE-FIRST INTELLIGENCE LAYER
# =============================================================================
#
# The legacy NewsLLMOrchestrator and evidence-selection code above are retained
# for backward compatibility with older callers. The production entry points at
# the bottom of this file use the PerspectiveFirstNewsEngine below.
#
# Core rule:
#   ARTICLE -> MEANING -> PERSPECTIVE ECOSYSTEM -> TARGETED DB RETRIEVAL
#   -> ACTUAL GRAPH TRAVERSAL -> CONSEQUENCES -> GAPS -> OPPORTUNITIES/RISKS
#
# The article is NOT used as a keyword query against the vault. The selected
# perspective defines the analytical universe. Database nodes and wiki-links are
# the only authority for what entities and relationships exist in ATIS.
# =============================================================================

PERSPECTIVE_FIRST_VERSION = "3.0.0"
MAX_ARTICLE_CHARS = int(os.getenv("ATIS_NEWS_MAX_ARTICLE_CHARS", "30000"))
MAX_PERSPECTIVE_ECOSYSTEM_NODES = int(os.getenv("ATIS_NEWS_MAX_PERSPECTIVE_NODES", "500"))
MAX_IMPACT_DOMAINS = int(os.getenv("ATIS_NEWS_MAX_IMPACT_DOMAINS", "8"))
MAX_RETRIEVAL_TARGETS = int(os.getenv("ATIS_NEWS_MAX_RETRIEVAL_TARGETS", "30"))
MAX_GRAPH_NODES = int(os.getenv("ATIS_NEWS_MAX_GRAPH_NODES", "80"))
MAX_GRAPH_PATHS = int(os.getenv("ATIS_NEWS_MAX_GRAPH_PATHS", "60"))
MAX_GRAPH_DEPTH = int(os.getenv("ATIS_NEWS_MAX_GRAPH_DEPTH", "2"))
MAX_GRAPH_NODES_PER_LLM_PARTITION = int(os.getenv("ATIS_NEWS_MAX_GRAPH_NODES_PER_PARTITION", "12"))
MAX_FINAL_OUTPUT_TOKENS = int(os.getenv("ATIS_NEWS_MAX_FINAL_OUTPUT_TOKENS", "4096"))
MAX_STAGE_OUTPUT_TOKENS = int(os.getenv("ATIS_NEWS_MAX_STAGE_OUTPUT_TOKENS", "3072"))
MIN_NODE_RESOLUTION_SCORE = float(os.getenv("ATIS_NEWS_MIN_NODE_RESOLUTION_SCORE", "0.72"))
MIN_OPPORTUNITY_GRAPH_SCORE = float(os.getenv("ATIS_NEWS_MIN_OPPORTUNITY_GRAPH_SCORE", "0.55"))


def _pf_norm(value: Any) -> str:
    """Normalize names for deterministic comparison without treating similarity as proof."""
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _pf_tokens(value: Any) -> Set[str]:
    return {x for x in re.findall(r"[a-z0-9]+", _pf_norm(value)) if len(x) > 2}


def _pf_similarity(a: Any, b: Any) -> float:
    aa, bb = _pf_tokens(a), _pf_tokens(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def _pf_country_matches(value: str, country: str) -> bool:
    a = _pf_norm(value)
    b = _pf_norm(country)
    if not b:
        return False
    return a == b or b in a.split() or _pf_norm(COUNTRY_CODES.get(b, b)) == a


def _pf_safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _pf_text(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + " [TRUNCATED]"


class PerspectiveFirstNewsEngine:
    """
    Production News engine implementing the perspective-first ATIS architecture.

    The engine deliberately separates four responsibilities:
      * LLM: article understanding and interpretation.
      * Vault: authoritative perspective ecosystem and node contents.
      * Graph: authoritative relationships/backlinks.
      * LLM: interpretation of already-verified graph evidence.

    No LLM call is permitted to manufacture a graph edge.
    """

    def __init__(self, vault: ObsidianVaultManager, perspective: PerspectiveContext) -> None:
        self.vault = vault
        self.perspective = perspective
        self.client: LLMClient = get_client()
        self.config = self.client.config
        self.budget = TokenBudgetManager(self.client.adapter.capabilities)
        self.cache = AnalysisCache()
        self.calls = 0
        self.truncated_retries = 0

    # ------------------------------------------------------------------ #
    # LLM boundary / token safety
    # ------------------------------------------------------------------ #
    def _model_output_cap(self) -> int:
        return int(self.budget.max_output_tokens)

    @staticmethod
    def _is_truncated(raw: str) -> bool:
        if not raw or not raw.strip():
            return True
        text = raw.strip()
        if text.endswith("..."):
            return True
        stack: List[str] = []
        in_string = False
        escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "]}" and stack:
                expected = "]" if stack[-1] == "[" else "}"
                if ch == expected:
                    stack.pop()
                else:
                    return True
        return in_string or bool(stack)

    def _call_json(self, system_prompt: str, user_prompt: str, requested_output: int, stage: str) -> Dict[str, Any]:
        """Bound every call before transmission; retry once in a more compact form."""
        cap = self._model_output_cap()
        requested = min(max(512, int(requested_output)), cap)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_tokens = self.budget.estimate_messages_tokens(messages)
        if not self.budget.fits_in_budget(input_tokens, requested):
            requested = self.budget.compute_safe_output_tokens(input_tokens)
            if not self.budget.fits_in_budget(input_tokens, requested):
                raise LLMTokenLimitError(
                    f"[{stage}] call rejected before transmission: input~{input_tokens:,}, "
                    f"output={requested:,}, context={self.budget.provider_context_limit:,}"
                )
        self.calls += 1
        logger.info("[LLM] %s | input~%d | output<=%d | context=%d", stage, input_tokens, requested, self.budget.provider_context_limit)
        # IMPORTANT: no seed/random_seed. Leanstral-compatible provider boundary.
        raw = self.client.chat(messages, temperature=0.0, max_tokens=requested)
        if self._is_truncated(raw):
            self.truncated_retries += 1
            logger.warning("[LLM] %s returned truncated output; retrying compactly", stage)
            compact_user = user_prompt[: max(2000, int(len(user_prompt) * 0.70))]
            compact_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": compact_user},
            ]
            compact_input = self.budget.estimate_messages_tokens(compact_messages)
            retry_output = self.budget.compute_safe_output_tokens(compact_input)
            retry_output = min(max(512, retry_output), cap)
            if not self.budget.fits_in_budget(compact_input, retry_output):
                raise LLMTokenLimitError(f"[{stage}] compact retry cannot fit provider context")
            self.calls += 1
            raw = self.client.chat(compact_messages, temperature=0.0, max_tokens=retry_output)
        return safe_json_loads(raw, stage_name=stage)

    # ------------------------------------------------------------------ #
    # Perspective ecosystem — database first, never a generic dump
    # ------------------------------------------------------------------ #
    def load_perspective_ecosystem(self) -> List[Dict[str, Any]]:
        country = _pf_norm(self.perspective.country)
        candidates: List[Dict[str, Any]] = []
        for canon in sorted(self.vault.node_metadata):
            meta = self.vault.node_metadata[canon]
            node_country = _pf_norm(meta.get("country", ""))
            if not node_country:
                path = str(meta.get("path", "")).lower()
                node_country = country if country and country in path else ""
            if node_country != country:
                continue
            candidates.append({
                "node_id": self.vault.file_map.get(canon, canon),
                "canonical_id": canon,
                "type": meta.get("type", ""),
                "sector": meta.get("sector", ""),
                "summary": meta.get("summary", ""),
                "path": meta.get("path", ""),
            })
        candidates.sort(key=lambda x: (x["sector"], x["type"], x["node_id"]))
        logger.info("[PERSPECTIVE] %s ecosystem nodes available: %d", self.perspective.country, len(candidates))
        return candidates[:MAX_PERSPECTIVE_ECOSYSTEM_NODES]

    def _ecosystem_context(self, ecosystem: List[Dict[str, Any]]) -> str:
        lines = [
            f"PERSPECTIVE COUNTRY: {self.perspective.country} ({self.perspective.country_code})",
            "This registry defines the perspective ecosystem. It is retrieval guidance, not proof of impact.",
        ]
        for n in ecosystem:
            lines.append(
                f"- NODE={n['node_id']} | type={n['type'] or 'unknown'} | sector={n['sector'] or 'unknown'} | summary={_pf_text(n['summary'], 240)}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Stage 1 — article facts and meaning
    # ------------------------------------------------------------------ #
    def understand_article(self, article_text: str) -> Dict[str, Any]:
        system = """You are ATIS News Stage 1: Article Understanding.
Your job is to understand ONLY what is contained in the supplied article.
Do not search for external facts. Do not invent entities or relationships.
Separate direct article facts from interpretation.
Return JSON only.

Required shape:
{
  "event": {"title":"", "summary":"", "event_country":"", "source_country":""},
  "facts": [{"fact":"", "evidence":"", "importance":"high|medium|low"}],
  "actors": [{"name":"", "role":"", "evidence":""}],
  "mechanisms": [{"mechanism":"", "evidence":""}],
  "meaning": [{"interpretation":"", "based_on_fact_indexes":[0]}],
  "uncertainties": ["..."]
}
"""
        user = f"ARTICLE:\n{article_text[:MAX_ARTICLE_CHARS]}"
        result = self._call_json(system, user, min(MAX_STAGE_OUTPUT_TOKENS, 3072), "Article Understanding")
        result.setdefault("event", {})
        result.setdefault("facts", [])
        result.setdefault("actors", [])
        result.setdefault("mechanisms", [])
        result.setdefault("meaning", [])
        result.setdefault("uncertainties", [])
        logger.info("[FACTS] %d facts | %d actors | %d mechanisms | %d meanings", len(_pf_safe_list(result["facts"])), len(_pf_safe_list(result["actors"])), len(_pf_safe_list(result["mechanisms"])), len(_pf_safe_list(result["meaning"])))
        return result

    # ------------------------------------------------------------------ #
    # Stage 2 — map meaning against the perspective ecosystem
    # ------------------------------------------------------------------ #
    def map_impact_domains(self, article: Dict[str, Any], ecosystem: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map the event to the perspective ecosystem without sending the whole registry in one call."""
        system = """You are ATIS News Stage 2: Perspective Impact Mapper.
The analytical perspective is the selected country. Determine which areas of THAT
country's supplied ecosystem should be investigated because of the event.

IMPORTANT:
- Do not claim that an impact exists merely because it is plausible.
- Identify investigation targets/domains, not final opportunities.
- Use only the supplied article understanding and supplied perspective ecosystem.
- Do not create database relationships.
- Do not use article entity names as direct vault queries.
- The ecosystem registry contains node IDs that may be used as retrieval hints.
Return JSON only.

Required shape:
{
  "impact_domains":[
    {"domain":"", "why_relevant":"", "mechanism":"", "priority":"high|medium|low", "ecosystem_node_hints":["exact supplied NODE IDs"]}
  ],
  "excluded_domains":[{"domain":"","reason":""}]
}
"""
        # Use the model's actual context capacity to batch the perspective registry.
        article_json = json.dumps(article, ensure_ascii=False)
        safe_input_budget = max(1500, self.budget.usable_context_budget - self.budget.estimate_tokens(system) - 3072)
        per_node_estimate = max(20, self.budget.estimate_tokens("NODE=xxxxxxxx | type= | sector= | summary=" + "x" * 180))
        batch_size = max(8, min(80, safe_input_budget // per_node_estimate))
        batches = [ecosystem[i:i + batch_size] for i in range(0, len(ecosystem), batch_size)] or [[]]
        batches = batches[:MAX_PARTITIONS]
        aggregate_domains: Dict[str, Dict[str, Any]] = {}
        excluded: List[Dict[str, Any]] = []
        for idx, batch in enumerate(batches, 1):
            registry = self._ecosystem_context(batch)
            user = f"ARTICLE UNDERSTANDING:\n{article_json[:9000]}\n\nPERSPECTIVE ECOSYSTEM BATCH {idx}/{len(batches)}:\n{registry}"
            try:
                result = self._call_json(system, user, min(MAX_STAGE_OUTPUT_TOKENS, 3072), f"Perspective Impact Mapping B{idx}")
            except LLMTokenLimitError:
                # Reduce the batch once more instead of failing the entire news analysis.
                smaller = batch[:max(4, len(batch) // 2)]
                registry = self._ecosystem_context(smaller)
                user = f"ARTICLE UNDERSTANDING:\n{article_json[:6000]}\n\nPERSPECTIVE ECOSYSTEM:\n{registry}"
                result = self._call_json(system, user, 2048, f"Perspective Impact Mapping B{idx} Compact")
            for domain in _pf_safe_list(result.get("impact_domains")):
                if not isinstance(domain, dict):
                    continue
                key = _pf_norm(domain.get("domain", ""))
                if not key:
                    continue
                existing = aggregate_domains.get(key)
                if existing is None:
                    aggregate_domains[key] = dict(domain)
                else:
                    existing["ecosystem_node_hints"] = sorted(set(_pf_safe_list(existing.get("ecosystem_node_hints")) + _pf_safe_list(domain.get("ecosystem_node_hints"))))[:12]
                    if str(domain.get("priority", "medium")) == "high":
                        existing["priority"] = "high"
                    if not existing.get("why_relevant") and domain.get("why_relevant"):
                        existing["why_relevant"] = domain.get("why_relevant")
                    if not existing.get("mechanism") and domain.get("mechanism"):
                        existing["mechanism"] = domain.get("mechanism")
            excluded.extend(_pf_safe_list(result.get("excluded_domains")))
        domains = sorted(aggregate_domains.values(), key=lambda d: (0 if d.get("priority") == "high" else 1 if d.get("priority") == "medium" else 2, _pf_norm(d.get("domain", ""))))[:MAX_IMPACT_DOMAINS]
        return {"impact_domains": domains, "excluded_domains": excluded[:MAX_IMPACT_DOMAINS]}

    # ------------------------------------------------------------------ #
    # Targeted DB retrieval
    # ------------------------------------------------------------------ #
    def _node_record(self, canon: str) -> Dict[str, Any]:
        meta = self.vault.node_metadata.get(canon, {})
        return {
            "node_id": self.vault.file_map.get(canon, canon),
            "canonical_id": canon,
            "country": meta.get("country", ""),
            "type": meta.get("type", ""),
            "sector": meta.get("sector", ""),
            "summary": meta.get("summary", ""),
            "path": meta.get("path", ""),
            "content": self.vault.node_content.get(canon, ""),
        }

    def _resolve_supplied_node(self, hint: str, ecosystem: List[Dict[str, Any]]) -> Optional[str]:
        """Resolve only against supplied perspective ecosystem; similarity is never relationship evidence."""
        target = _pf_norm(hint)
        if not target:
            return None
        exact = []
        for n in ecosystem:
            if _pf_norm(n["node_id"]) == target or _pf_norm(n["canonical_id"]) == target:
                exact.append(n["canonical_id"])
        if exact:
            return exact[0]
        # Alias/normalized match against node name only.
        candidates: List[Tuple[float, str]] = []
        for n in ecosystem:
            score = max(_pf_similarity(hint, n["node_id"]), _pf_similarity(hint, n.get("summary", "")))
            if score >= MIN_NODE_RESOLUTION_SCORE:
                candidates.append((score, n["canonical_id"]))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        if candidates:
            # Require a clear winner; do not force ambiguous matches.
            if len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= 0.12:
                return candidates[0][1]
        return None

    def retrieve_targets(self, impact: Dict[str, Any], ecosystem: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        resolved: Dict[str, Dict[str, Any]] = {}
        unresolved: List[Dict[str, Any]] = []
        for domain in _pf_safe_list(impact.get("impact_domains")):
            if not isinstance(domain, dict):
                continue
            hints = domain.get("ecosystem_node_hints", [])
            if not hints:
                # Domain-level targeting: sector/type matching is retrieval, not graph evidence.
                sector = _pf_norm(domain.get("domain", ""))
                hints = [n["node_id"] for n in ecosystem if sector and sector in _pf_norm(n.get("sector", ""))][:8]
            found_for_domain = False
            for hint in hints:
                canon = self._resolve_supplied_node(str(hint), ecosystem)
                if canon:
                    found_for_domain = True
                    rec = self._node_record(canon)
                    rec["target_domains"] = sorted(set(rec.get("target_domains", []) + [domain.get("domain", "")]))
                    resolved[canon] = rec
                else:
                    unresolved.append({"hint": str(hint), "domain": domain.get("domain", ""), "status": "UNRESOLVED"})
            if not found_for_domain:
                unresolved.append({"hint": domain.get("domain", ""), "domain": domain.get("domain", ""), "status": "NO_TARGET_NODE_RESOLVED"})
        nodes = sorted(resolved.values(), key=lambda n: (n.get("sector", ""), n["node_id"]))[:MAX_RETRIEVAL_TARGETS]
        logger.info("[RETRIEVAL] targeted nodes=%d unresolved targets=%d", len(nodes), len(unresolved))
        return nodes, unresolved

    # ------------------------------------------------------------------ #
    # Actual graph traversal
    # ------------------------------------------------------------------ #
    def _canonical_from_actual(self, node_id: str) -> Optional[str]:
        canon = self.vault._canonicalize(node_id)
        if canon in self.vault.file_map:
            return canon
        return None

    def traverse_graph(self, target_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        seeds = sorted({n["canonical_id"] for n in target_nodes if n.get("canonical_id") in self.vault.file_map})
        visited: Dict[str, int] = {s: 0 for s in seeds}
        queue: List[str] = list(seeds)
        paths: List[Dict[str, Any]] = []
        direct_edges: List[Dict[str, Any]] = []
        first_nodes: Set[str] = set()
        second_nodes: Set[str] = set()
        backlink_candidates = 0

        while queue and len(visited) <= MAX_GRAPH_NODES:
            current = queue.pop(0)
            depth = visited[current]
            if depth >= MAX_GRAPH_DEPTH:
                continue
            current_actual = self.vault.file_map.get(current, current)
            outgoing = sorted(self.vault.outbound_links.get(current, []), key=lambda x: self.vault._canonicalize(x))
            incoming = sorted(self.vault.backlink_map.get(current, set()), key=lambda x: self.vault._canonicalize(x))
            edges: List[Tuple[str, str]] = []
            for target in outgoing:
                target_canon = self.vault._canonicalize(target)
                if target_canon in self.vault.file_map:
                    edges.append((target_canon, "outbound_link"))
            for source in incoming:
                source_canon = self.vault._canonicalize(source)
                if source_canon in self.vault.file_map:
                    edges.append((source_canon, "inbound_backlink"))
                    if source_canon not in visited:
                        backlink_candidates += 1
            seen_edge: Set[Tuple[str, str]] = set()
            for neighbour, relation_type in sorted(edges, key=lambda x: (x[0], x[1])):
                if (neighbour, relation_type) in seen_edge:
                    continue
                seen_edge.add((neighbour, relation_type))
                next_depth = depth + 1
                # A node already reached at an equal or lower depth is not a new
                # first/second-order node. This prevents reverse backlinks from
                # turning a seed node into a false second-order discovery.
                if neighbour in visited:
                    continue
                if next_depth == 1:
                    first_nodes.add(neighbour)
                else:
                    second_nodes.add(neighbour)
                edge = {
                    "from_node": current_actual,
                    "to_node": self.vault.file_map.get(neighbour, neighbour),
                    "relationship_type": relation_type,
                    "depth": next_depth,
                    "source": "database",
                }
                direct_edges.append(edge)
                paths.append({
                    "nodes": [self.vault.file_map.get(current, current), self.vault.file_map.get(neighbour, neighbour)],
                    "edges": [edge],
                    "depth": next_depth,
                    "source": "database",
                })
                if len(visited) < MAX_GRAPH_NODES:
                    visited[neighbour] = next_depth
                    queue.append(neighbour)
        # Convert reachable nodes to records.
        direct_set = set(seeds)
        graph_nodes: Dict[str, Dict[str, Any]] = {}
        for canon, depth in sorted(visited.items(), key=lambda x: (x[1], x[0])):
            rec = self._node_record(canon)
            if canon in direct_set:
                category = "target"
            elif depth == 1:
                category = "first_order"
            else:
                category = "second_order"
            rec["graph_depth"] = depth
            rec["graph_category"] = category
            graph_nodes[canon] = rec
        paths = paths[:MAX_GRAPH_PATHS]
        logger.info(
            "[GRAPH] targets=%d | direct=%d | first-order=%d | second-order=%d | backlink candidates=%d | paths=%d",
            len(seeds), len(direct_set), len(first_nodes), len(second_nodes), backlink_candidates, len(paths)
        )
        return {
            "target_nodes": sorted(graph_nodes[c] for c in direct_set if c in graph_nodes),
            "nodes": list(graph_nodes.values()),
            "edges": direct_edges[:MAX_GRAPH_PATHS],
            "paths": paths,
            "direct_nodes": len(direct_set),
            "first_order_nodes": len(first_nodes),
            "second_order_nodes": len(second_nodes),
            "backlink_candidates": backlink_candidates,
        }

    # ------------------------------------------------------------------ #
    # Graph interpretation — LLM sees verified graph, not raw vault universe
    # ------------------------------------------------------------------ #
    def _graph_context(self, graph: Dict[str, Any]) -> str:
        lines = ["VERIFIED DATABASE GRAPH — relationships are authoritative and may not be invented."]
        for edge in graph.get("edges", [])[:MAX_GRAPH_PATHS]:
            lines.append(
                f"EDGE: {edge['from_node']} --{edge['relationship_type']}--> {edge['to_node']} | depth={edge['depth']} | source=database"
            )
        lines.append("NODE EVIDENCE:")
        for n in graph.get("nodes", [])[:MAX_GRAPH_NODES]:
            lines.append(
                f"NODE: {n['node_id']} | country={n.get('country','')} | type={n.get('type','')} | sector={n.get('sector','')} | depth={n.get('graph_depth')} | category={n.get('graph_category')} | summary={_pf_text(n.get('summary',''), 220)}"
            )
        return "\n".join(lines)

    def analyze_graph_partition(self, article: Dict[str, Any], impact: Dict[str, Any], graph_nodes: List[Dict[str, Any]], graph_edges: List[Dict[str, Any]], partition_no: int) -> Dict[str, Any]:
        system = """You are ATIS News Graph Consequence Analyst.
You are given a news event, its meaning, perspective impact domains, and a subset
of VERIFIED database graph nodes/edges.

The database graph is authoritative. You may interpret it, but you may NOT create
new graph edges or factual entities. If a relationship is absent, it is absent.
Distinguish supported consequence from hypothesis and research-required gaps.
Do not turn mere node co-occurrence into a relationship.
Return JSON only.

Shape:
{
 "consequences":[{"statement":"","order":"direct|first_order|second_order","supporting_nodes":[""],"supporting_edges":["FROM->TO"],"confidence":0.0}],
 "gaps":[{"gap":"","missing_relationship":"","status":"RESEARCH_REQUIRED","related_nodes":[""]}],
 "opportunity_signals":[{"signal":"","status":"SUPPORTED|RESEARCH_REQUIRED","supporting_nodes":[""],"supporting_edges":[""]}],
 "risk_signals":[{"signal":"","status":"SUPPORTED|RESEARCH_REQUIRED","supporting_nodes":[""],"supporting_edges":[""]}]
}
"""
        compact_nodes = []
        for n in graph_nodes[:MAX_GRAPH_NODES_PER_LLM_PARTITION]:
            compact_nodes.append({k: n.get(k, "") for k in ("node_id", "country", "type", "sector", "summary", "graph_depth", "graph_category")})
        compact_edges = [
            {"from_node": e["from_node"], "to_node": e["to_node"], "relationship_type": e["relationship_type"], "depth": e["depth"]}
            for e in graph_edges[:MAX_GRAPH_NODES_PER_LLM_PARTITION * 2]
        ]
        user = (
            f"PERSPECTIVE={self.perspective.country} ({self.perspective.country_code})\n"
            f"ARTICLE MEANING={json.dumps(article.get('meaning', []), ensure_ascii=False)}\n"
            f"IMPACT DOMAINS={json.dumps(impact.get('impact_domains', []), ensure_ascii=False)}\n"
            f"VERIFIED NODES={json.dumps(compact_nodes, ensure_ascii=False)}\n"
            f"VERIFIED EDGES={json.dumps(compact_edges, ensure_ascii=False)}"
        )
        return self._call_json(system, user, MAX_STAGE_OUTPUT_TOKENS, f"Graph Consequence Partition {partition_no}")

    def analyze_graph(self, article: Dict[str, Any], impact: Dict[str, Any], graph: Dict[str, Any], reasoning_log: ReasoningLog) -> Dict[str, Any]:
        nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
        edges = [e for e in graph.get("edges", []) if isinstance(e, dict)]
        if not nodes:
            return {"consequences": [], "gaps": [{"gap": "No targeted graph nodes were resolved", "status": "RESEARCH_REQUIRED", "related_nodes": []}], "opportunity_signals": [], "risk_signals": []}
        partitions: List[List[Dict[str, Any]]] = []
        for i in range(0, len(nodes), MAX_GRAPH_NODES_PER_LLM_PARTITION):
            if len(partitions) >= MAX_PARTITIONS:
                break
            partitions.append(nodes[i:i + MAX_GRAPH_NODES_PER_LLM_PARTITION])
        reasoning_log.partitions = len(partitions)
        combined = {"consequences": [], "gaps": [], "opportunity_signals": [], "risk_signals": []}
        for idx, part in enumerate(partitions, 1):
            part_ids = {n["node_id"] for n in part}
            part_edges = [e for e in edges if e.get("from_node") in part_ids or e.get("to_node") in part_ids]
            result = self.analyze_graph_partition(article, impact, part, part_edges, idx)
            for key in combined:
                combined[key].extend(_pf_safe_list(result.get(key)))
            reasoning_log.evidence_calls += 1
        return combined

    # ------------------------------------------------------------------ #
    # Impact-chain construction from verified stages
    # ------------------------------------------------------------------ #
    def build_impact_chain(self, article: Dict[str, Any], impact: Dict[str, Any], graph: Dict[str, Any], consequences: Dict[str, Any]) -> List[Dict[str, Any]]:
        chain: List[Dict[str, Any]] = []
        event = article.get("event", {}) if isinstance(article.get("event"), dict) else {}
        chain.append({
            "stage": "external_event",
            "label": event.get("title") or event.get("summary") or "Article event",
            "status": "SUPPORTED",
            "source": "article",
            "evidence": [f"article_fact_{i}" for i, _ in enumerate(_pf_safe_list(article.get("facts")))],
        })
        meanings = _pf_safe_list(article.get("meaning"))
        if meanings:
            chain.append({
                "stage": "meaning",
                "label": _pf_text(meanings[0].get("interpretation", ""), 500) if isinstance(meanings[0], dict) else _pf_text(meanings[0], 500),
                "status": "INTERPRETATION",
                "source": "llm_article_interpretation",
                "evidence": meanings[0].get("based_on_fact_indexes", []) if isinstance(meanings[0], dict) else [],
            })
        for domain in _pf_safe_list(impact.get("impact_domains"))[:MAX_IMPACT_DOMAINS]:
            if isinstance(domain, dict):
                chain.append({
                    "stage": "zimbabwe_ecosystem" if _pf_norm(self.perspective.country) == "zimbabwe" else "perspective_ecosystem",
                    "label": domain.get("domain", ""),
                    "status": "INVESTIGATION_TARGET",
                    "source": "perspective_ecosystem",
                    "mechanism": domain.get("mechanism", ""),
                    "priority": domain.get("priority", "medium"),
                })
        for edge in graph.get("edges", [])[:MAX_GRAPH_PATHS]:
            chain.append({
                "stage": "graph_relationship",
                "label": f"{edge.get('from_node')} → {edge.get('to_node')}",
                "status": "SUPPORTED",
                "source": "database",
                "relationship": edge.get("relationship_type"),
                "depth": edge.get("depth"),
                "evidence": [f"{edge.get('from_node')}->{edge.get('to_node')}"]
            })
        for item in _pf_safe_list(consequences.get("consequences"))[:20]:
            if isinstance(item, dict):
                chain.append({
                    "stage": "consequence" if item.get("order") != "second_order" else "second_order_effect",
                    "label": item.get("statement", ""),
                    "status": "SUPPORTED" if item.get("supporting_edges") or item.get("supporting_nodes") else "RESEARCH_REQUIRED",
                    "source": "verified_graph_interpretation",
                    "evidence": item.get("supporting_nodes", []),
                    "graph_edges": item.get("supporting_edges", []),
                    "confidence": item.get("confidence", 0.0),
                })
        for gap in _pf_safe_list(consequences.get("gaps"))[:20]:
            if isinstance(gap, dict):
                chain.append({
                    "stage": "gap",
                    "label": gap.get("gap", ""),
                    "status": "RESEARCH_REQUIRED",
                    "source": "database_graph_gap",
                    "evidence": gap.get("related_nodes", []),
                    "missing_relationship": gap.get("missing_relationship", ""),
                })
        return chain[:MAX_GRAPH_PATHS + MAX_IMPACT_DOMAINS + 10]

    # ------------------------------------------------------------------ #
    # Final synthesis from distilled evidence
    # ------------------------------------------------------------------ #
    def final_synthesis(self, article: Dict[str, Any], impact: Dict[str, Any], graph: Dict[str, Any], consequences: Dict[str, Any], impact_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        system = """You are the ATIS News Final Synthesis Engine.
Produce the final dashboard from the supplied structured analysis.

HARD RULES:
1. The selected perspective country is the lens. Never switch to a local source-country perspective.
2. Database nodes and edges are authoritative. Do not invent missing edges.
3. Opportunities must be supported by at least one verified database node AND one verified graph edge/path, unless explicitly labelled RESEARCH_REQUIRED.
4. If evidence is missing, say RESEARCH_REQUIRED rather than filling the gap with world knowledge.
5. Never infer a relationship merely because two nodes appear in the same context.
6. Preserve the impact chain exactly as an evidence trail.
7. Return concise JSON. Do not write an essay.

Return:
{
 "trigger_event":"",
 "market_equilibrium_shift":"",
 "executive_summary":"",
 "analytical_perspective":{"country":"","country_code":"","description":""},
 "facts":[],
 "meaning":[],
 "impact_domains":[],
 "impact_chain":[],
 "findings":[],
 "opportunities":[{"opportunity_id":"","title":"","status":"SUPPORTED|RESEARCH_REQUIRED","perspective_country":"","opportunity_country":"","perspective_actor":"","perspective_capability":"","pathway":"","justification":"","urgency_score":0.0,"feasibility_score":0.0,"source_nodes":[],"graph_paths":[],"required_missing_nodes":[]}],
 "risks":[{"text":"","status":"SUPPORTED|RESEARCH_REQUIRED","severity":"high|medium|low","source_nodes":[],"graph_paths":[]}],
 "gaps":[{"gap":"","status":"RESEARCH_REQUIRED","related_nodes":[],"missing_relationship":""}],
 "entities":[],
 "graph_analysis":{"direct":0,"first_order":0,"second_order":0,"backlink_candidates":0,"paths":[]}
}
"""
        # Final prompt uses distilled graph and consequences, not the entire vault.
        graph_summary = {
            "nodes": [
                {k: n.get(k, "") for k in ("node_id", "country", "type", "sector", "summary", "graph_depth", "graph_category")}
                for n in graph.get("nodes", [])[:MAX_GRAPH_NODES]
            ],
            "edges": graph.get("edges", [])[:MAX_GRAPH_PATHS],
        }
        user = (
            f"PERSPECTIVE={self.perspective.country} ({self.perspective.country_code})\n"
            f"ARTICLE={json.dumps(article, ensure_ascii=False)[:6000]}\n"
            f"IMPACT={json.dumps(impact, ensure_ascii=False)[:5000]}\n"
            f"GRAPH={json.dumps(graph_summary, ensure_ascii=False)[:12000]}\n"
            f"CONSEQUENCES={json.dumps(consequences, ensure_ascii=False)[:10000]}\n"
            f"IMPACT_CHAIN={json.dumps(impact_chain, ensure_ascii=False)[:10000]}"
        )
        result = self._call_json(system, user, MAX_FINAL_OUTPUT_TOKENS, "Final News Synthesis")
        result["impact_chain"] = impact_chain
        return result

    # ------------------------------------------------------------------ #
    # Deterministic validation / post-processing
    # ------------------------------------------------------------------ #
    def validate_and_ground(self, dashboard: Dict[str, Any], graph: Dict[str, Any], ecosystem: List[Dict[str, Any]], article: Dict[str, Any], impact: Dict[str, Any], consequences: Dict[str, Any]) -> Dict[str, Any]:
        valid_nodes = {n["node_id"] for n in graph.get("nodes", [])}
        valid_edges = {f"{e.get('from_node')}->{e.get('to_node')}" for e in graph.get("edges", [])}
        perspective_nodes = {n["node_id"] for n in ecosystem}
        opportunities: List[Dict[str, Any]] = []
        for opp in _pf_safe_list(dashboard.get("opportunities")):
            if not isinstance(opp, dict):
                continue
            source_nodes = [str(x) for x in _pf_safe_list(opp.get("source_nodes")) if str(x) in valid_nodes]
            graph_paths = [str(x) for x in _pf_safe_list(opp.get("graph_paths")) if str(x) in valid_edges]
            actor = str(opp.get("perspective_actor", ""))
            actor_supported = not actor or actor in perspective_nodes
            supported = bool(source_nodes and graph_paths and actor_supported)
            item = dict(opp)
            item["source_nodes"] = source_nodes
            item["graph_paths"] = graph_paths
            if not supported:
                item["status"] = "RESEARCH_REQUIRED"
                item["opportunity_confidence"] = min(float(item.get("opportunity_confidence", 0.0) or 0.0), 0.49)
                if not item.get("required_missing_nodes"):
                    item["required_missing_nodes"] = ["verified graph pathway and/or perspective actor evidence"]
            else:
                item["status"] = "SUPPORTED"
                item["opportunity_confidence"] = max(float(item.get("opportunity_confidence", 0.0) or 0.0), MIN_OPPORTUNITY_GRAPH_SCORE)
            item.setdefault("perspective_country", self.perspective.country)
            item.setdefault("perspective_country_code", self.perspective.country_code)
            opportunities.append(item)
        dashboard["opportunities"] = opportunities

        risks: List[Dict[str, Any]] = []
        for risk in _pf_safe_list(dashboard.get("risks")):
            if not isinstance(risk, dict):
                continue
            item = dict(risk)
            item["source_nodes"] = [str(x) for x in _pf_safe_list(item.get("source_nodes")) if str(x) in valid_nodes]
            item["graph_paths"] = [str(x) for x in _pf_safe_list(item.get("graph_paths")) if str(x) in valid_edges]
            if not item["source_nodes"]:
                item["status"] = "RESEARCH_REQUIRED"
            else:
                item.setdefault("status", "SUPPORTED")
            risks.append(item)
        dashboard["risks"] = risks

        dashboard.setdefault("gaps", consequences.get("gaps", []))
        dashboard.setdefault("findings", [])
        dashboard.setdefault("entities", article.get("actors", []))
        dashboard["analytical_perspective"] = {
            "country": self.perspective.country,
            "country_code": self.perspective.country_code,
            "description": "Country through which this event is interpreted.",
        }
        dashboard["graph_analysis"] = {
            "direct": graph.get("direct_nodes", 0),
            "first_order": graph.get("first_order_nodes", 0),
            "second_order": graph.get("second_order_nodes", 0),
            "backlink_candidates": graph.get("backlink_candidates", 0),
            "paths": graph.get("paths", [])[:MAX_GRAPH_PATHS],
        }
        dashboard["source_country"] = article.get("event", {}).get("source_country", "") if isinstance(article.get("event"), dict) else ""
        dashboard["event_country"] = article.get("event", {}).get("event_country", "") if isinstance(article.get("event"), dict) else ""
        return dashboard

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def run(self, article_text: str, reasoning_log: ReasoningLog) -> Dict[str, Any]:
        if not article_text or not article_text.strip():
            raise ValueError("News article text is empty")
        article_text = article_text.strip()
        if len(article_text) > MAX_ARTICLE_CHARS:
            article_text = article_text[:MAX_ARTICLE_CHARS] + "\n[ARTICLE TRUNCATED BEFORE ANALYSIS]"

        ecosystem = self.load_perspective_ecosystem()
        reasoning_log.perspective_nodes = len(ecosystem)
        article = self.understand_article(article_text)
        reasoning_log.entities_extracted = len(_pf_safe_list(article.get("actors")))
        impact = self.map_impact_domains(article, ecosystem)
        reasoning_log.impact_domains = len(_pf_safe_list(impact.get("impact_domains")))
        targets, unresolved = self.retrieve_targets(impact, ecosystem)
        reasoning_log.retrieval_targets = len(targets)
        reasoning_log.candidate_nodes = len(self.vault.file_map)
        graph = self.traverse_graph(targets)
        reasoning_log.direct_graph_nodes = graph.get("direct_nodes", 0)
        reasoning_log.first_order_graph_nodes = graph.get("first_order_nodes", 0)
        reasoning_log.second_order_graph_nodes = graph.get("second_order_nodes", 0)
        reasoning_log.backlink_candidates = graph.get("backlink_candidates", 0)
        reasoning_log.graph_paths = len(graph.get("paths", []))
        reasoning_log.relevant_nodes = len(graph.get("nodes", []))
        reasoning_log.selected_evidence = len(graph.get("nodes", []))

        if graph.get("nodes"):
            reasoning_log.reasoning_mode = ReasoningMode.MULTI_STAGE.value
            consequences = self.analyze_graph(article, impact, graph, reasoning_log)
        else:
            reasoning_log.reasoning_mode = "perspective_only_no_graph_evidence"
            consequences = {
                "consequences": [],
                "gaps": [{
                    "gap": "No evidence-backed relationship was found between the identified perspective targets and the current graph.",
                    "status": "RESEARCH_REQUIRED",
                    "related_nodes": [n["node_id"] for n in targets],
                }],
                "opportunity_signals": [],
                "risk_signals": [],
            }
        impact_chain = self.build_impact_chain(article, impact, graph, consequences)
        dashboard = self.final_synthesis(article, impact, graph, consequences, impact_chain)
        dashboard = self.validate_and_ground(dashboard, graph, ecosystem, article, impact, consequences)
        dashboard["facts"] = article.get("facts", [])
        dashboard["meaning"] = article.get("meaning", [])
        dashboard["impact_domains"] = impact.get("impact_domains", [])
        dashboard["impact_chain"] = impact_chain
        dashboard["research_required"] = unresolved + _pf_safe_list(consequences.get("gaps"))
        reasoning_log.gaps = len(_pf_safe_list(dashboard.get("gaps")))
        reasoning_log.opportunities = len(_pf_safe_list(dashboard.get("opportunities")))
        reasoning_log.total_llm_calls = self.calls
        return dashboard


# --------------------------------------------------------------------------- #
# Backward-compatible helper builders now operate on the perspective-first data
# --------------------------------------------------------------------------- #
def _pf_build_reasoning_metadata(
    reasoning_log: ReasoningLog,
    engine: PerspectiveFirstNewsEngine,
    vault: ObsidianVaultManager,
    perspective: PerspectiveContext,
    article: Dict[str, Any],
    dashboard: Dict[str, Any],
    knowledge_state: KnowledgeState,
) -> Dict[str, Any]:
    event = article.get("event", {}) if isinstance(article.get("event"), dict) else {}
    core_event = event.get("title") or event.get("summary") or dashboard.get("trigger_event") or "Unknown event"
    try:
        knowledge_hash = knowledge_state.knowledge_state_hash
    except Exception:
        knowledge_hash = getattr(knowledge_state, "hash", "")
    evidence_ids = sorted(str(x) for x in dashboard.get("graph_analysis", {}).get("paths", []))
    entity_ids = sorted(str(x.get("node_id", x)) if isinstance(x, dict) else str(x) for x in dashboard.get("entities", []))
    try:
        fingerprint = compute_analysis_fingerprint(
            story_id=core_event,
            perspective=perspective,
            evidence_ids=evidence_ids,
            entity_ids=entity_ids,
            relationship_ids=sorted(f"{e.get('from_node')}->{e.get('to_node')}" for e in dashboard.get("graph_analysis", {}).get("paths", []) if isinstance(e, dict)),
            knowledge_state_hash=knowledge_hash,
        )
    except Exception:
        fingerprint = hashlib.sha256(json.dumps(dashboard, sort_keys=True, default=str).encode()).hexdigest()[:24]
    reasoning_data = reasoning_log.to_dict()
    # The original ReasoningLog predates the perspective-first counters. Include
    # them without breaking callers that depend on its legacy to_dict shape.
    for attr in (
        "perspective_nodes", "impact_domains", "retrieval_targets",
        "direct_graph_nodes", "first_order_graph_nodes", "second_order_graph_nodes",
        "graph_paths", "gaps", "opportunities",
    ):
        reasoning_data[attr] = getattr(reasoning_log, attr, 0)

    return {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "perspective_first_graph_grounded",
        "architecture_version": PERSPECTIVE_FIRST_VERSION,
        "source_article": "web_upload",
        "core_event": core_event,
        "source_country": dashboard.get("source_country", ""),
        "event_country": dashboard.get("event_country", ""),
        "perspective_country": perspective.country,
        "perspective_country_code": perspective.country_code,
        "model_primary": getattr(engine.config, "model", ""),
        "model_fallback": getattr(engine.config, "fallback_model", ""),
        "seed_sent_to_provider": False,
        "selected_evidence_nodes": dashboard.get("graph_analysis", {}).get("direct", 0) + dashboard.get("graph_analysis", {}).get("first_order", 0) + dashboard.get("graph_analysis", {}).get("second_order", 0),
        "cross_border_bridges_found": len(vault.build_cross_border_bridge_context(perspective, dashboard.get("source_country", "") or perspective.country)),
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "analysis_fingerprint": fingerprint,
        "knowledge_state": knowledge_state.as_dict() if hasattr(knowledge_state, "as_dict") else {"hash": knowledge_hash},
        "reasoning_log": reasoning_data,
        "llm_calls": engine.calls,
        "truncated_retries": engine.truncated_retries,
    }


def _pf_finalize_dashboard(
    dashboard: Dict[str, Any],
    perspective: PerspectiveContext,
    engine: PerspectiveFirstNewsEngine,
    reasoning_log: ReasoningLog,
    vault: ObsidianVaultManager,
) -> Dict[str, Any]:
    """Canonical response shaping while retaining legacy frontend fields."""
    dashboard = dict(dashboard or {})
    dashboard.setdefault("intelligence_id", f"ATIS-INT-{hashlib.sha256(str(dashboard.get('trigger_event','')).encode()).hexdigest()[:12].upper()}")
    dashboard.setdefault("trigger_event", "Unknown event")
    dashboard.setdefault("executive_summary", "No executive summary was generated.")
    dashboard["perspective"] = perspective.as_dict()
    dashboard["analytical_perspective"] = dashboard.get("analytical_perspective") or {
        "country": perspective.country,
        "country_code": perspective.country_code,
        "description": "Country through which this event is interpreted.",
    }
    dashboard.setdefault("cross_border_analysis", {
        "event_country": dashboard.get("event_country", ""),
        "source_country": dashboard.get("source_country", ""),
        "cross_border_bridges": [],
        "cross_border_bridges_count": 0,
    })
    dashboard.setdefault("opportunities", [])
    dashboard.setdefault("risks", [])
    dashboard.setdefault("gaps", [])
    dashboard.setdefault("entities", [])
    dashboard.setdefault("findings", [])
    dashboard.setdefault("facts", [])
    dashboard.setdefault("meaning", [])
    dashboard.setdefault("impact_domains", [])
    dashboard.setdefault("impact_chain", [])
    return dashboard


# --------------------------------------------------------------------------- #
# Production entry points — these override the legacy orchestration functions
# without removing their implementation above.
# --------------------------------------------------------------------------- #
def _run_perspective_first_news(article_text: str, perspective: Any | None = None, source_label: str = "web_upload") -> Dict[str, Any]:
    perspective = perspective or PerspectiveContext()
    reasoning_log = ReasoningLog()
    logger.info("=" * 78)
    logger.info("ATIS NEWS v%s | PERSPECTIVE-FIRST | %s (%s)", PERSPECTIVE_FIRST_VERSION, perspective.country, perspective.country_code)
    logger.info("=" * 78)
    vault = ObsidianVaultManager()
    engine = PerspectiveFirstNewsEngine(vault, perspective)
    knowledge_state = KnowledgeState(vault_path=vault.vault_dir)
    knowledge_state.compute()
    dashboard = engine.run(article_text, reasoning_log)
    dashboard = _pf_finalize_dashboard(dashboard, perspective, engine, reasoning_log, vault)
    dashboard["pipeline_metadata"] = _pf_build_reasoning_metadata(reasoning_log, engine, vault, perspective, {
        "event": {"title": dashboard.get("trigger_event", ""), "source_country": dashboard.get("source_country", ""), "event_country": dashboard.get("event_country", "")},
        "facts": dashboard.get("facts", []),
    }, dashboard, knowledge_state)
    dashboard["pipeline_metadata"]["source_article"] = source_label
    dashboard["pipeline_metadata"]["reasoning_log"] = dict(dashboard["pipeline_metadata"].get("reasoning_log", {}))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = DASHBOARDS_DIR / f"atis_dashboard_{timestamp}.json"
    try:
        output_path.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False), encoding="utf-8")
        dashboard["pipeline_metadata"]["dashboard_path"] = str(output_path)
        logger.info("[FINAL] Dashboard persisted: %s", output_path.resolve())
    except Exception as exc:
        logger.error("[FINAL] Failed to persist dashboard: %s", exc)
    logger.info("\n%s", reasoning_log.log_tree())
    logger.info("=" * 78)
    logger.info("ATIS NEWS PIPELINE COMPLETE")
    logger.info("=" * 78)
    return dashboard


def process_article_pipeline(article_path: str, perspective: Any | None = None) -> Dict[str, Any]:
    article_file = Path(article_path)
    if not article_file.exists():
        raise FileNotFoundError(f"Article not found: {article_path}")
    article_text = article_file.read_text(encoding="utf-8")
    return _run_perspective_first_news(article_text, perspective, source_label="file_upload")


def run_news_pipeline(article_text: str, perspective: Any | None = None) -> Dict[str, Any]:
    return _run_perspective_first_news(article_text, perspective, source_label="web_upload")


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATIS News — Perspective-First, Graph-Grounded Intelligence Engine",
        epilog="Example: python ATIS_News.py ./articles/news.txt",
    )
    parser.add_argument("article_path", metavar="ARTICLE", help="Path to the plain-text news article to process.")
    args = parser.parse_args()
    try:
        result = process_article_pipeline(args.article_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.critical("Pipeline terminated with fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
