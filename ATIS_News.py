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
import math
import os
import re
import sys
import time
import tempfile
import sqlite3
import socket
import traceback
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_client import LLMClient, get_client, LLMTokenLimitError, ModelCapabilities

# --------------------------------------------------------------------------- #
# Durable pipeline state & zero-loss partitioning
# --------------------------------------------------------------------------- #
class PipelineStage(str, Enum):
    ARTICLE_UNDERSTANDING = "ARTICLE_UNDERSTANDING"
    PERSPECTIVE_MAPPING = "PERSPECTIVE_MAPPING"
    DATABASE_RETRIEVAL = "DATABASE_RETRIEVAL"
    GRAPH_TRAVERSAL = "GRAPH_TRAVERSAL"
    IMPACT_ANALYSIS = "IMPACT_ANALYSIS"
    FINAL_SYNTHESIS = "FINAL_SYNTHESIS"
    COMPLETE = "COMPLETE"


@dataclass
class JobState:
    intelligence_id: str
    status: str = "IN_PROGRESS"
    current_stage: str = PipelineStage.ARTICLE_UNDERSTANDING.value
    completed_stages: List[str] = field(default_factory=list)
    stage_data: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def mark_stage_complete(self, stage: PipelineStage, data: Any) -> None:
        self.stage_data[stage.value] = data
        if stage.value not in self.completed_stages:
            self.completed_stages.append(stage.value)
        self.updated_at = time.time()

    def is_completed(self, stage: PipelineStage) -> bool:
        return stage.value in self.completed_stages and stage.value in self.stage_data


class StatePersistenceManager:
    """Durable, atomic checkpoint storage for resumable News jobs."""

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        configured = storage_dir or os.getenv("ATIS_NEWS_JOB_STORE", "./job_store")
        self.storage_dir = Path(configured)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, job_id: str) -> Path:
        safe_id = "".join(c for c in str(job_id) if c.isalnum() or c in ("-", "_")) or "unknown"
        return self.storage_dir / f"{safe_id}.json"

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(k): StatePersistenceManager._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [StatePersistenceManager._json_safe(v) for v in value]
        if hasattr(value, "value"):
            return StatePersistenceManager._json_safe(value.value)
        return str(value)

    def save_state(self, state: JobState) -> None:
        state.updated_at = time.time()
        path = self._get_path(state.intelligence_id)
        payload = self._json_safe(asdict(state))
        # Atomic replace prevents a timeout/process crash from leaving a half JSON file.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(self.storage_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def load_state(self, job_id: str) -> Optional[JobState]:
        path = self._get_path(job_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            return JobState(**raw)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("[STATE] Ignoring unreadable checkpoint %s: %s", path, exc)
            return None


class DataPartitioner:
    """Partitions records/text without discarding any source material."""

    @staticmethod
    def partition_records(records: List[Any], max_per_batch: int = 40) -> List[List[Any]]:
        if not records:
            return []
        if max_per_batch <= 0:
            raise ValueError("max_per_batch must be positive")
        return [records[i:i + max_per_batch] for i in range(0, len(records), max_per_batch)]

    @staticmethod
    def partition_paragraphs(text: str, max_chars: int = 4000) -> List[str]:
        if not text:
            return []
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for paragraph in paragraphs:
            # A single oversized paragraph is further split rather than truncated.
            if len(paragraph) > max_chars:
                if current:
                    chunks.append("\n\n".join(current))
                    current, current_len = [], 0
                chunks.extend(paragraph[i:i + max_chars] for i in range(0, len(paragraph), max_chars))
                continue
            projected = current_len + (2 if current else 0) + len(paragraph)
            if current and projected > max_chars:
                chunks.append("\n\n".join(current))
                current, current_len = [paragraph], len(paragraph)
            else:
                current.append(paragraph)
                current_len = projected
        if current:
            chunks.append("\n\n".join(current))
        return chunks

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

# Explicit analytical priority. Never rely on lexical ordering of enum values.
EVIDENCE_CATEGORY_PRIORITY = {
    EvidenceCategory.DIRECT: 0,
    EvidenceCategory.FIRST_ORDER: 1,
    EvidenceCategory.SECOND_ORDER: 2,
    EvidenceCategory.PERIPHERAL: 3,
    EvidenceCategory.PERSPECTIVE: 4,
    EvidenceCategory.BRIDGE: 5,
    EvidenceCategory.GLOBAL: 6,
}


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
        logger.error("Raw response excerpt (first 1000 chars):\n%s", original)
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
                    summary = line
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
                    if EVIDENCE_CATEGORY_PRIORITY.get(node.category, 999) < EVIDENCE_CATEGORY_PRIORITY.get(existing.category, 999):
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
        direct.sort(key=lambda n: (-n.composite_score, n.canonical_id))
        final_selection.extend(direct[:MAX_DIRECT_EVIDENCE_NODES])

        # First-order
        first = [n for n in deduped.values() if n.category == EvidenceCategory.FIRST_ORDER]
        first.sort(key=lambda n: (-n.composite_score, n.canonical_id))
        final_selection.extend(first[:MAX_FIRST_ORDER_NODES])

        # Second-order
        second = [n for n in deduped.values() if n.category == EvidenceCategory.SECOND_ORDER]
        second.sort(key=lambda n: (-n.composite_score, n.canonical_id))
        final_selection.extend(second[:MAX_SECOND_ORDER_NODES])

        # Peripheral (low-value second-order)
        peripheral = [n for n in deduped.values() if n.category == EvidenceCategory.PERIPHERAL]
        peripheral.sort(key=lambda n: (-n.composite_score, n.canonical_id))
        final_selection.extend(peripheral[:MAX_PERIPHERAL_NODES])

        # Perspective
        persp = [n for n in deduped.values() if n.category == EvidenceCategory.PERSPECTIVE]
        persp.sort(key=lambda n: (-n.composite_score, n.canonical_id))
        final_selection.extend(persp[:MAX_PERSPECTIVE_NODES])

        # Bridge
        bridge = [n for n in deduped.values() if n.category == EvidenceCategory.BRIDGE]
        bridge.sort(key=lambda n: (-n.composite_score, n.canonical_id))
        final_selection.extend(bridge[:MAX_BRIDGE_NODES])

        # Global (diversity)
        glob = [n for n in deduped.values() if n.category == EvidenceCategory.GLOBAL]
        glob.sort(key=lambda n: (-n.composite_score, n.canonical_id))
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
        final_selection.sort(key=lambda n: (-n.composite_score, n.canonical_id))

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
            raise LLMTokenLimitError(
                "Stage 1 source article exceeds the provider context budget; "
                "ATIS refuses to truncate the article."
            )

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
                f"## NEWS EVENT\n{article_text}\n\n"
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
            f"## NEWS EVENT\n{article_text}\n\n"
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
                f"## NEWS EVENT\n{article_text}\n"
                f"## PERSPECTIVE: {perspective.country}\n"
                f"## BRIDGES: {bridge_context}\n"
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
    base_text = system_prompt + article_text + perspective_registry + bridge_context
    base_tokens = budget.estimate_tokens(base_text)
    available_per_partition = budget.usable_context_budget - base_tokens - 4096  # reserve output

    for sector, sector_nodes in sorted_sectors:
        for node in sorted(sector_nodes, key=lambda n: (-n.composite_score, n.canonical_id)):
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
        partitions.sort(key=lambda p: (-sum(n.composite_score for n in p.nodes), p.partition_id))
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
    perspective_nodes.sort(key=lambda n: (-n.composite_score, n.canonical_id))

    lines = [f"=== PERSPECTIVE ACTOR REGISTRY ({perspective.country}) ==="]
    for node in perspective_nodes[:max_nodes]:
        lines.append(
            f"- {node.node_id} | type: {node.node_type or 'unknown'} | "
            f"sector: {node.sector or 'N/A'} | summary: {node.summary}"
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
            json.dumps(StatePersistenceManager._json_safe(dashboard), indent=2, ensure_ascii=False, allow_nan=False),
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
    evidence_tokens = sum(budget.estimate_tokens(n.content) for n in selected_nodes)
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
            json.dumps(StatePersistenceManager._json_safe(dashboard), indent=2, ensure_ascii=False, allow_nan=False),
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

PERSPECTIVE_FIRST_VERSION = "3.7.0-production-durable"

# ---------------------------------------------------------------------------
# Production controls
# ---------------------------------------------------------------------------
# These controls govern source-size safety and operator-configured graph limits.
# HTTP transport lifetime is intentionally not coupled to News intelligence execution.
MAX_ARTICLE_CHARS = int(os.getenv("ATIS_NEWS_MAX_ARTICLE_CHARS", "500000"))
MAX_PERSPECTIVE_ECOSYSTEM_NODES = int(os.getenv("ATIS_NEWS_MAX_PERSPECTIVE_NODES", "0"))  # 0 = unlimited; retained for backward-compatible env naming
MAX_IMPACT_DOMAINS = int(os.getenv("ATIS_NEWS_MAX_IMPACT_DOMAINS", "8"))
MAX_RETRIEVAL_TARGETS = int(os.getenv("ATIS_NEWS_MAX_RETRIEVAL_TARGETS", "0"))  # 0 = unlimited
MAX_GRAPH_NODES = int(os.getenv("ATIS_NEWS_MAX_GRAPH_NODES", "0"))  # 0 = unlimited
MAX_GRAPH_PATHS = int(os.getenv("ATIS_NEWS_MAX_GRAPH_PATHS", "0"))  # 0 = unlimited
MAX_GRAPH_DEPTH = int(os.getenv("ATIS_NEWS_MAX_GRAPH_DEPTH", "2"))
MAX_FINAL_OUTPUT_TOKENS = int(os.getenv("ATIS_NEWS_MAX_FINAL_OUTPUT_TOKENS", "3072"))
MAX_STAGE_OUTPUT_TOKENS = int(os.getenv("ATIS_NEWS_MAX_STAGE_OUTPUT_TOKENS", "3072"))
MIN_NODE_RESOLUTION_SCORE = float(os.getenv("ATIS_NEWS_MIN_NODE_RESOLUTION_SCORE", "0.72"))
MIN_OPPORTUNITY_GRAPH_SCORE = float(os.getenv("ATIS_NEWS_MIN_OPPORTUNITY_GRAPH_SCORE", "0.55"))

# Provider transport protection is deliberately separate from job lifetime.
# A News job is durable and may outlive the HTTP request that submitted it.
# These values bound individual synchronous provider calls; they are NOT an
# end-to-end intelligence deadline and MUST NOT cause completed work to be
# discarded merely because an HTTP request is approaching its upstream limit.
ARTICLE_LLM_TIMEOUT_SECONDS = float(os.getenv("ATIS_NEWS_ARTICLE_TIMEOUT_SECONDS", "45"))
IMPACT_LLM_TIMEOUT_SECONDS = float(os.getenv("ATIS_NEWS_IMPACT_TIMEOUT_SECONDS", "45"))
GRAPH_LLM_TIMEOUT_SECONDS = float(os.getenv("ATIS_NEWS_GRAPH_TIMEOUT_SECONDS", "0"))
FINAL_LLM_TIMEOUT_SECONDS = float(os.getenv("ATIS_NEWS_FINAL_TIMEOUT_SECONDS", "45"))
MIN_LLM_CALL_SECONDS = float(os.getenv("ATIS_NEWS_MIN_LLM_CALL_SECONDS", "2.0"))
PROVIDER_RETRY_COUNT = max(0, int(os.getenv("ATIS_NEWS_PROVIDER_RETRY_COUNT", "1")))
MAX_IMPACT_LLM_CALLS = 1
MAX_GRAPH_LLM_CALLS = 0
LLM_CONCURRENCY = 1

# Retained as a compatibility/read-only telemetry setting. It is no longer an
# intelligence deadline. HTTP callers must use their own transport timeout.
PIPELINE_DEADLINE_SECONDS = 0.0

# Durable queue controls. SQLite is used intentionally so no Redis/Celery
# dependency is required and multiple worker processes can coordinate through
# an atomic lease transaction.
NEWS_QUEUE_DB = Path(os.getenv(
    "ATIS_NEWS_QUEUE_DB",
    os.getenv("ATIS_NEWS_JOB_STORE", "./job_store") + "/news_jobs.sqlite3",
))
NEWS_QUEUE_LEASE_SECONDS = max(10.0, float(os.getenv("ATIS_NEWS_QUEUE_LEASE_SECONDS", "180")))
NEWS_QUEUE_POLL_SECONDS = max(0.2, float(os.getenv("ATIS_NEWS_QUEUE_POLL_SECONDS", "1.0")))
NEWS_QUEUE_MAX_ATTEMPTS = max(1, int(os.getenv("ATIS_NEWS_QUEUE_MAX_ATTEMPTS", "5")))

# The production graph is deterministic. No LLM call is used to manufacture
# graph relationships.



# ---------------------------------------------------------------------------
# Durable News Job Queue — SQLite-backed atomic leases
# ---------------------------------------------------------------------------
class DurableNewsJobQueue:
    """Cross-process durable queue for News jobs.

    The queue is intentionally independent from the HTTP request lifecycle.
    SQLite transactions provide the atomic claim/lease primitive without
    requiring Redis or Celery. A worker crash leaves an expired lease that a
    later worker can reclaim. Payloads are immutable after submission so a
    retried job cannot silently change its article or perspective.
    """

    def __init__(self, db_path: str | Path = NEWS_QUEUE_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_jobs (
                    job_id TEXT PRIMARY KEY,
                    article_text TEXT NOT NULL,
                    perspective_json TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL,
                    lease_until REAL,
                    worker_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_jobs_claim
                ON news_jobs(status, available_at, lease_until)
            """)

    @staticmethod
    def make_job_id(article_text: str, perspective: PerspectiveContext) -> str:
        canonical = json.dumps(
            {
                "article_sha256": hashlib.sha256(article_text.strip().encode("utf-8")).hexdigest(),
                "perspective": perspective.as_dict(),
                "pipeline_version": PERSPECTIVE_FIRST_VERSION,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return "ATIS-NEWS-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    def submit(
        self,
        article_text: str,
        perspective: PerspectiveContext,
        source_label: str = "web_upload",
        job_id: str | None = None,
    ) -> Dict[str, Any]:
        article_text = article_text.strip()
        if not article_text:
            raise ValueError("News article text is empty")
        if len(article_text) > MAX_ARTICLE_CHARS:
            raise ValueError(
                f"Article contains {len(article_text):,} characters, exceeding "
                f"{MAX_ARTICLE_CHARS:,}. ATIS refuses to truncate the source article."
            )
        job_id = job_id or self.make_job_id(article_text, perspective)
        now = time.time()
        payload = json.dumps(perspective.as_dict(), sort_keys=True, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO news_jobs
                (job_id, article_text, perspective_json, source_label, status,
                 attempts, available_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'QUEUED', 0, ?, ?, ?)
                ON CONFLICT(job_id) DO NOTHING
                """,
                (job_id, article_text, payload, source_label, now, now, now),
            )
            row = conn.execute(
                "SELECT * FROM news_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to persist News job {job_id}")
        # A caller-supplied job ID is an idempotency key, not permission to
        # mutate an existing job. Reject accidental ID collisions.
        if row["article_text"] != article_text or row["perspective_json"] != payload:
            raise ValueError(
                f"News job ID {job_id} already belongs to different immutable input"
            )
        return self._row_to_status(row)

    def claim(self, worker_id: str | None = None) -> Optional[Dict[str, Any]]:
        worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        now = time.time()
        lease_until = now + NEWS_QUEUE_LEASE_SECONDS
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM news_jobs
                WHERE attempts < ?
                  AND available_at <= ?
                  AND (
                      status = 'QUEUED'
                      OR (status = 'RUNNING' AND lease_until IS NOT NULL AND lease_until < ?)
                  )
                ORDER BY created_at ASC, job_id ASC
                LIMIT 1
                """,
                (NEWS_QUEUE_MAX_ATTEMPTS, now, now),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            next_attempt = int(row["attempts"]) + 1
            conn.execute(
                """
                UPDATE news_jobs
                SET status='RUNNING', attempts=?, lease_until=?, worker_id=?,
                    updated_at=?, error=NULL
                WHERE job_id=?
                """,
                (next_attempt, lease_until, worker_id, now, row["job_id"]),
            )
            claimed = conn.execute(
                "SELECT * FROM news_jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            conn.commit()
        return self._row_to_job(claimed)

    def renew(self, job_id: str, worker_id: str) -> bool:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE news_jobs
                SET lease_until=?, updated_at=?
                WHERE job_id=? AND status='RUNNING' AND worker_id=?
                """,
                (now + NEWS_QUEUE_LEASE_SECONDS, now, job_id, worker_id),
            )
        return cur.rowcount == 1

    def complete(self, job_id: str, worker_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE news_jobs
                SET status='COMPLETED', lease_until=NULL, worker_id=NULL,
                    completed_at=?, updated_at=?, error=NULL
                WHERE job_id=? AND status='RUNNING' AND worker_id=?
                """,
                (now, now, job_id, worker_id),
            )
        if cur.rowcount != 1:
            raise RuntimeError(f"Lost News job lease while completing {job_id}")

    def fail(self, job_id: str, worker_id: str, error: str) -> None:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT attempts FROM news_jobs WHERE job_id=? AND worker_id=?",
                (job_id, worker_id),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"News job {job_id} is not owned by worker {worker_id}")
            attempts = int(row["attempts"])
            terminal = attempts >= NEWS_QUEUE_MAX_ATTEMPTS
            status = "FAILED" if terminal else "QUEUED"
            available = now if terminal else now + min(60.0, 2.0 ** min(attempts, 5))
            conn.execute(
                """
                UPDATE news_jobs
                SET status=?, lease_until=NULL, worker_id=NULL,
                    available_at=?, updated_at=?, error=?
                WHERE job_id=? AND worker_id=?
                """,
                (status, available, now, str(error)[:4000], job_id, worker_id),
            )

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM news_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return self._row_to_status(row) if row else None

    @staticmethod
    def _row_to_status(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "attempts": int(row["attempts"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "error": row["error"],
            "worker_id": row["worker_id"],
            "lease_until": row["lease_until"],
        }

    @classmethod
    def _row_to_job(cls, row: sqlite3.Row) -> Dict[str, Any]:
        result = cls._row_to_status(row)
        result.update({
            "article_text": row["article_text"],
            "perspective": json.loads(row["perspective_json"]),
            "source_label": row["source_label"],
        })
        return result


def submit_news_job(
    article_text: str,
    perspective: Any | None = None,
    source_label: str = "web_upload",
    job_id: str | None = None,
) -> Dict[str, Any]:
    """Persist a News job and return immediately; no LLM work is done here."""
    perspective_ctx = perspective or PerspectiveContext()
    queue = DurableNewsJobQueue()
    return queue.submit(article_text, perspective_ctx, source_label, job_id)


def get_news_job_status(job_id: str) -> Dict[str, Any]:
    """Return durable queue status plus resumable pipeline checkpoint metadata."""
    result = DurableNewsJobQueue().get(job_id)
    if result is None:
        raise KeyError(f"News job not found: {job_id}")
    state = StatePersistenceManager().load_state(job_id)
    if state is not None:
        result["checkpoint"] = {
            "status": state.status,
            "current_stage": state.current_stage,
            "completed_stages": list(state.completed_stages),
            "updated_at": state.updated_at,
            "error_count": len(state.error_log),
            "resume_available": state.status != "COMPLETED",
        }
        if state.status == "COMPLETED":
            result["result"] = state.stage_data.get(PipelineStage.FINAL_SYNTHESIS.value)
    return result


def run_news_worker_once(worker_id: str | None = None) -> Optional[Dict[str, Any]]:
    """Claim and execute one durable News job in a worker process."""
    queue = DurableNewsJobQueue()
    claimed = queue.claim(worker_id)
    if claimed is None:
        return None

    owner = claimed["worker_id"]
    stop_heartbeat = threading.Event()
    lease_lost = threading.Event()

    def heartbeat() -> None:
        interval = max(2.0, NEWS_QUEUE_LEASE_SECONDS / 3.0)
        while not stop_heartbeat.wait(interval):
            try:
                if not queue.renew(claimed["job_id"], owner):
                    lease_lost.set()
                    logger.error("[WORKER] Lease lost for News job %s", claimed["job_id"])
                    return
            except Exception:
                logger.exception("[WORKER] Lease renewal failed for %s", claimed["job_id"])
                # A transient DB error is not proof of lease loss. The next
                # heartbeat gets another chance unless the lease actually expires.
                continue

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"atis-news-lease-{claimed['job_id'][-12:]}",
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        perspective_payload = claimed["perspective"] if isinstance(claimed["perspective"], dict) else {}
        perspective = PerspectiveContext(
            country=str(perspective_payload.get("country") or "Zimbabwe"),
            country_code=str(perspective_payload.get("country_code") or "ZW"),
        )
        dashboard = _run_perspective_first_news(
            claimed["article_text"],
            perspective,
            source_label=claimed["source_label"],
            job_id=claimed["job_id"],
        )
        if lease_lost.is_set():
            raise RuntimeError(f"Lost durable lease for News job {claimed['job_id']}")

        transport_failures = int(
            (dashboard.get("pipeline_execution") or {}).get("transport_failures", 0)
        )
        if dashboard.get("partial") and transport_failures > 0:
            queue.fail(
                claimed["job_id"],
                owner,
                str((dashboard.get("detail") or "provider transport failure"))[:4000],
            )
        else:
            queue.complete(claimed["job_id"], owner)
        return dashboard
    except Exception as exc:
        logger.exception("[WORKER] News job %s failed: %s", claimed["job_id"], exc)
        try:
            queue.fail(claimed["job_id"], owner, traceback.format_exc())
        except Exception:
            logger.exception("[WORKER] Could not release lease for %s", claimed["job_id"])
        return None
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2.0)


def run_news_worker_forever(worker_id: str | None = None) -> None:
    """Run the durable worker loop as a separate process/service."""
    resolved_worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
    logger.info("[WORKER] ATIS News durable worker started: %s", resolved_worker_id)
    while True:
        job = run_news_worker_once(resolved_worker_id)
        if job is None:
            time.sleep(NEWS_QUEUE_POLL_SECONDS)

# ---------------------------------------------------------------------------
# Shared in-process vault cache
# ---------------------------------------------------------------------------
# ATIS_News used to rebuild all ~716 markdown files for every /api/news call.
# That consumed ~6 seconds before the first LLM request.  This cache lives in
# this file only, preserving the requirement that no other repository file is
# changed.  It invalidates when the vault file set or mtimes change.
_VAULT_CACHE: Dict[str, Tuple[Tuple[Tuple[str, int, int], ...], "ObsidianVaultManager"]] = {}


def _vault_signature(vault_dir: Path) -> Tuple[Tuple[str, int, int], ...]:
    rows: List[Tuple[str, int, int]] = []
    try:
        for path in sorted(vault_dir.rglob("*.md"), key=lambda p: str(p)):
            try:
                stat = path.stat()
                rows.append((str(path.relative_to(vault_dir)), int(stat.st_mtime_ns), int(stat.st_size)))
            except OSError:
                continue
    except OSError:
        return tuple()
    return tuple(rows)


def _get_cached_vault(vault_dir: Path) -> ObsidianVaultManager:
    key = str(vault_dir.resolve())
    signature = _vault_signature(vault_dir)
    cached = _VAULT_CACHE.get(key)
    if cached and cached[0] == signature:
        logger.info("[VAULT] Reusing cached index: %d nodes", len(cached[1].file_map))
        return cached[1]
    vault = ObsidianVaultManager(vault_dir)
    _VAULT_CACHE[key] = (signature, vault)
    return vault


def _pf_norm(value: Any) -> str:
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
    """Normalize text without character truncation.

    The legacy ``limit`` argument remains for compatibility, but is ignored.
    Production context control is record-level, never character-level.
    """
    return str(value or "").strip()


def _pf_safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not (-1e9 < number < 1e9):
        return default
    return number


class NewsPipelineDeadline(RuntimeError):
    """ATIS must stop before the upstream HTTP request deadline."""


class NewsPipelineContextOverflow(RuntimeError):
    """A complete record set cannot fit without silently truncating evidence."""


# ---------------------------------------------------------------------------
# Lossless structured-context helpers
# ---------------------------------------------------------------------------
def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _messages_tokens(budget: TokenBudgetManager, system: str, user: str) -> int:
    return budget.estimate_messages_tokens([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


def _record_cost(budget: TokenBudgetManager, record: Any) -> int:
    return budget.estimate_tokens(_json(record))


def _pack_complete_records(
    records: List[Any],
    budget: TokenBudgetManager,
    fixed_tokens: int,
    output_tokens: int,
    max_records: Optional[int] = None,
) -> List[List[Any]]:
    """Pack complete records into context-safe batches.

    No record is sliced. If a single record itself cannot fit, it is rejected
    explicitly rather than truncated.
    """
    capacity = budget.provider_context_limit - budget.safety_margin - output_tokens - fixed_tokens
    if capacity < 256:
        raise NewsPipelineContextOverflow("No safe context capacity remains for complete records")
    batches: List[List[Any]] = []
    current: List[Any] = []
    used = 0
    for record in records:
        cost = _record_cost(budget, record)
        if cost > capacity:
            raise NewsPipelineContextOverflow(
                f"One complete evidence record requires ~{cost} tokens but only {capacity} are available"
            )
        if current and (used + cost > capacity or (max_records and len(current) >= max_records)):
            batches.append(current)
            current = []
            used = 0
        current.append(record)
        used += cost
    if current:
        batches.append(current)
    return batches


def _dedupe_dict_records(records: List[Any], key_fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            continue
        key = "|".join(_pf_norm(item.get(k, "")) for k in key_fields)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


# ---------------------------------------------------------------------------
# Production engine
# ---------------------------------------------------------------------------
class PerspectiveFirstNewsEngine:
    """Production perspective-first News engine.

    Data authority:
      * Article -> LLM may understand only article facts/meaning.
      * Perspective ecosystem -> vault only.
      * Relationships -> vault outbound links/backlinks only.
      * Consequences -> LLM interprets verified graph evidence; it cannot add edges.
      * Opportunities -> must be grounded in verified nodes/paths or explicitly
        marked RESEARCH_REQUIRED.

    Normal LLM budget is deliberately small:
      1. Article Understanding
      2. Perspective Impact Mapping (one call, or two concurrent complete batches)
      3. Final News Synthesis

    Graph traversal is deterministic and therefore does not require a separate
    graph LLM partition stage.
    """

    def __init__(
        self,
        vault: ObsidianVaultManager,
        perspective: PerspectiveContext,
        started_at: Optional[float] = None,
    ) -> None:
        self.vault = vault
        self.perspective = perspective
        self.client: LLMClient = get_client()
        self.config = self.client.config
        self.budget = TokenBudgetManager(self.client.adapter.capabilities)
        self.cache = AnalysisCache()
        self.calls = 0
        self.truncated_retries = 0
        self.timeouts = 0
        self.deadline_exhausted = False
        self.started_at = started_at if started_at is not None else time.monotonic()
        # Job lifetime is durable and independent of HTTP transport lifetime.
        # There is intentionally no intelligence deadline here.
        self._deadline = float("inf")
        self._stage_durations: Dict[str, float] = {}

    def _remaining_seconds(self) -> float:
        """Return job execution budget; durable workers have no artificial deadline."""
        return float("inf")

    def _remaining_seconds_display(self) -> str:
        """Human/log representation that never leaks Infinity into JSON-facing telemetry."""
        remaining = self._remaining_seconds()
        return "unbounded" if not math.isfinite(remaining) else f"{remaining:.1f}s"

    def _stage_timeout(self, stage: str) -> float:
        name = stage.lower()
        if "article understanding" in name:
            return ARTICLE_LLM_TIMEOUT_SECONDS
        if "perspective impact" in name:
            return IMPACT_LLM_TIMEOUT_SECONDS
        if "final news synthesis" in name:
            return FINAL_LLM_TIMEOUT_SECONDS
        return min(10.0, PIPELINE_DEADLINE_SECONDS)

    def _call_provider(self, messages: List[Dict[str, str]], max_tokens: int, stage: str) -> str:
        """Run a synchronous provider call behind a transport-only timeout.

        This timeout protects the worker from a permanently hung socket. It is
        deliberately NOT a pipeline deadline: a completed stage remains valid
        regardless of how long the job has been running.
        """
        timeout = max(0.1, self._stage_timeout(stage))
        import threading

        attempts = PROVIDER_RETRY_COUNT + 1
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            result: Dict[str, Any] = {}
            error: Dict[str, BaseException] = {}

            def worker() -> None:
                try:
                    result["value"] = self.client.chat(
                        messages, temperature=0.0, max_tokens=max_tokens
                    )
                except BaseException as exc:
                    error["value"] = exc

            started = time.monotonic()
            thread = threading.Thread(
                target=worker,
                name=f"atis-news-{stage[:24]}-{attempt}",
                daemon=True,
            )
            thread.start()
            thread.join(timeout=timeout)
            elapsed = time.monotonic() - started
            self._stage_durations[f"{stage}#attempt{attempt}"] = round(elapsed, 3)

            if thread.is_alive():
                self.timeouts += 1
                last_error = NewsPipelineDeadline(
                    f"[{stage}] provider transport timeout after {timeout:.1f}s"
                )
                logger.warning(
                    "[TRANSPORT] [%s] provider call timed out after %.1fs (attempt %d/%d)",
                    stage, timeout, attempt, attempts,
                )
            elif "value" in error:
                last_error = error["value"]
                logger.warning(
                    "[PROVIDER] [%s] attempt %d/%d failed: %s",
                    stage, attempt, attempts, last_error,
                )
            else:
                return str(result.get("value") or "")

            if attempt < attempts:
                self._increment_reasoning_counter("retry_calls")
                # The provider call itself is retried; the complete input and
                # requested output remain unchanged.
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"[{stage}] provider call returned no result")

    def _increment_reasoning_counter(self, name: str) -> None:
        """Increment optional ReasoningLog counters without coupling engine state to its dataclass."""
        # The run() method exposes the active log only after stage calls, so
        # these counters are also mirrored onto the engine and copied later.
        setattr(self, f"_{name}", getattr(self, f"_{name}", 0) + 1)

    def _call_json(self, system_prompt: str, user_prompt: str, requested_output: int, stage: str) -> Dict[str, Any]:
        """Lossless JSON call with bounded retry.

        A retry NEVER shortens the input. If output is malformed/incomplete,
        the exact same complete input is retried with a larger output allowance
        only when both context and deadline allow it.
        """
        cap = max(512, self.budget.max_output_tokens)
        requested = min(max(512, int(requested_output)), cap)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_tokens = self.budget.estimate_messages_tokens(messages)
        if not self.budget.fits_in_budget(input_tokens, requested):
            safe_output = self.budget.compute_safe_output_tokens(input_tokens)
            if safe_output < 512:
                raise LLMTokenLimitError(
                    f"[{stage}] complete input cannot fit provider context without truncation: "
                    f"input~{input_tokens}, context={self.budget.provider_context_limit}"
                )
            requested = min(requested, safe_output)

        logger.info(
            "[LLM] %s | input~%d | output<=%d | remaining=%s",
            stage, input_tokens, requested, self._remaining_seconds_display()
        )
        self.calls += 1
        stage_l = stage.lower()
        if "article understanding" in stage_l:
            self._increment_reasoning_counter("evidence_calls")
        elif "perspective impact" in stage_l:
            self._increment_reasoning_counter("evidence_calls")
        elif "final news synthesis" in stage_l:
            self._increment_reasoning_counter("final_call")
        raw = self._call_provider(messages, requested, stage)
        try:
            parsed = safe_json_loads(raw, stage_name=stage)
            if not isinstance(parsed, dict):
                raise RuntimeError(f"{stage} returned JSON that is not an object")
            return parsed
        except RuntimeError as first_error:
            retry_output = min(cap, max(requested + 1024, requested * 2))
            if retry_output <= requested:
                raise first_error
            if not self.budget.fits_in_budget(input_tokens, retry_output):
                retry_output = self.budget.compute_safe_output_tokens(input_tokens)
            if retry_output <= requested or retry_output < 512:
                raise first_error
            self.truncated_retries += 1
            self._increment_reasoning_counter("retry_calls")
            logger.warning(
                "[LLM] %s returned unusable JSON; retrying SAME complete input with output<=%d",
                stage, retry_output
            )
            self.calls += 1
            raw_retry = self._call_provider(messages, retry_output, f"{stage} retry")
            parsed = safe_json_loads(raw_retry, stage_name=f"{stage} retry")
            if not isinstance(parsed, dict):
                raise RuntimeError(f"{stage} retry returned JSON that is not an object")
            return parsed

    # ------------------------------------------------------------------
    # Perspective ecosystem
    # ------------------------------------------------------------------
    def load_perspective_ecosystem(self) -> List[Dict[str, Any]]:
        country = _pf_norm(self.perspective.country)
        candidates: List[Dict[str, Any]] = []
        for canon in sorted(self.vault.node_metadata):
            meta = self.vault.node_metadata[canon]
            node_country = _pf_norm(meta.get("country", ""))
            if not node_country:
                path = str(meta.get("path", "")).lower()
                if country and country in path:
                    node_country = country
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
        candidates.sort(key=lambda x: (str(x["sector"]), str(x["type"]), str(x["node_id"])))
        # Never silently discard perspective-country nodes. If the complete registry
        # does not fit one provider request, map_impact_domains partitions it.
        result = candidates
        logger.info("[PERSPECTIVE] %s ecosystem nodes available: %d (complete registry)", self.perspective.country, len(result))
        return result

    @staticmethod
    def _ecosystem_context(ecosystem: List[Dict[str, Any]], country: str, code: str) -> str:
        lines = [
            f"PERSPECTIVE COUNTRY: {country} ({code})",
            "Every NODE ID below is an existing database node. This registry is retrieval guidance, not proof of impact.",
        ]
        for n in ecosystem:
            lines.append(
                f"NODE={n['node_id']} | type={n.get('type') or 'unknown'} | "
                f"sector={n.get('sector') or 'unknown'} | summary={n.get('summary') or ''}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 1: article understanding
    # ------------------------------------------------------------------
    def understand_article(self, article_text: str) -> Dict[str, Any]:
        """Understand the complete article without character truncation."""
        system = """You are ATIS News Stage 1: Article Understanding.
Understand ONLY what is contained in the supplied article segment.
Do not use external knowledge. Do not invent entities, relationships, causes, or facts.
Preserve every material fact, actor, mechanism, meaning and uncertainty present in the supplied segment.
Return ONLY valid JSON.

Schema:
{
  "event":{"title":"","summary":"","event_country":"","source_country":""},
  "facts":[{"fact":"","evidence":"","importance":"high|medium|low"}],
  "actors":[{"name":"","role":"","evidence":""}],
  "mechanisms":[{"mechanism":"","evidence":""}],
  "meaning":[{"interpretation":"","based_on_fact_indexes":[0]}],
  "uncertainties":["..."]
}
"""

        requested = min(MAX_STAGE_OUTPUT_TOKENS, 3072)
        full_user = "ARTICLE SEGMENT (COMPLETE):\n" + article_text
        full_tokens = _messages_tokens(self.budget, system, full_user)
        if self.budget.fits_in_budget(full_tokens, requested):
            result = self._call_json(system, full_user, requested, "Article Understanding")
            result.setdefault("event", {})
            for key in ("facts", "actors", "mechanisms", "meaning", "uncertainties"):
                result.setdefault(key, [])
            logger.info("[FACTS] %d facts | %d actors | %d mechanisms | %d meanings", len(_pf_safe_list(result["facts"])), len(_pf_safe_list(result["actors"])), len(_pf_safe_list(result["mechanisms"])), len(_pf_safe_list(result["meaning"])))
            return result

        batch_chars = int(os.getenv("ATIS_NEWS_ARTICLE_BATCH_CHARS", "12000"))
        batches = DataPartitioner.partition_paragraphs(article_text, max_chars=batch_chars)
        logger.warning("[ARTICLE] complete article does not fit; processing %d lossless article batches", len(batches))
        results: List[Dict[str, Any]] = []
        for index, batch in enumerate(batches, start=1):
            user = f"ARTICLE SEGMENT {index}/{len(batches)} (COMPLETE; DO NOT INFER BEYOND THIS SEGMENT):\n{batch}"
            input_tokens = _messages_tokens(self.budget, system, user)
            if not self.budget.fits_in_budget(input_tokens, requested):
                raise NewsPipelineContextOverflow(f"Article segment {index} cannot fit provider context without truncation")
            results.append(self._call_json(system, user, requested, f"Article Understanding [{index}/{len(batches)}]"))

        merged: Dict[str, Any] = {"event": {}, "facts": [], "actors": [], "mechanisms": [], "meaning": [], "uncertainties": []}
        for result in results:
            event = result.get("event") if isinstance(result.get("event"), dict) else {}
            for key in ("title", "summary", "event_country", "source_country"):
                if not merged["event"].get(key) and event.get(key):
                    merged["event"][key] = event[key]
            for key in ("facts", "actors", "mechanisms", "meaning", "uncertainties"):
                for item in _pf_safe_list(result.get(key)):
                    identity = _json(item)
                    if not any(_json(existing) == identity for existing in merged[key]):
                        merged[key].append(item)
        # Batch-local fact indexes cannot safely be reused after merging. Clear them
        # rather than creating false provenance links. The meaning text itself is preserved.
        for item in merged["meaning"]:
            if isinstance(item, dict):
                item["based_on_fact_indexes"] = []
        logger.info("[FACTS] merged %d facts | %d actors | %d mechanisms | %d meanings from %d batches", len(merged["facts"]), len(merged["actors"]), len(merged["mechanisms"]), len(merged["meaning"]), len(results))
        return merged

    # ------------------------------------------------------------------
    # Stage 2: perspective impact mapping
    # ------------------------------------------------------------------
    def map_impact_domains(self, article: Dict[str, Any], ecosystem: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map the article onto the complete perspective ecosystem.

        The complete registry is sent when it fits. When it does not, the registry
        is partitioned into complete record batches. No node is character-truncated
        or silently discarded. Batch results are merged deterministically.
        """
        system = """You are ATIS News Stage 2: Perspective Impact Mapper.
The selected country is the analytical perspective. Determine which areas of THAT country's supplied ecosystem should be investigated because of the event.

Rules:
- Use only the supplied article understanding and supplied perspective ecosystem.
- Do not create database relationships.
- Do not claim an impact merely because it is plausible.
- Return investigation targets/domains, not final opportunities.
- ecosystem_node_hints MUST be exact NODE IDs supplied in the current registry batch.
- Preserve distinctions between direct article meaning and investigation hypotheses.
- Return ONLY valid JSON.

Schema:
{
  "impact_domains":[{"domain":"","why_relevant":"","mechanism":"","priority":"high|medium|low","ecosystem_node_hints":["exact NODE ID"]}],
  "excluded_domains":[{"domain":"","reason":""}]
}
"""
        article_json = _json(article)

        def normalize(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            domains: List[Dict[str, Any]] = []
            for item in _pf_safe_list(result.get("impact_domains")):
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                item["ecosystem_node_hints"] = sorted({
                    str(x).strip() for x in _pf_safe_list(item.get("ecosystem_node_hints"))
                    if not isinstance(x, (dict, list)) and str(x).strip()
                })
                domains.append(item)
            excluded = [x for x in _pf_safe_list(result.get("excluded_domains")) if isinstance(x, dict)]
            return domains, excluded

        registry = self._ecosystem_context(ecosystem, self.perspective.country, self.perspective.country_code)
        requested = min(MAX_STAGE_OUTPUT_TOKENS, 3072)
        full_user = f"ARTICLE UNDERSTANDING:\n{article_json}\n\nPERSPECTIVE ECOSYSTEM (COMPLETE):\n{registry}"
        full_tokens = _messages_tokens(self.budget, system, full_user)

        batches: List[List[Dict[str, Any]]]
        if self.budget.fits_in_budget(full_tokens, requested):
            batches = [ecosystem]
        else:
            # Start with a conservative record batch and shrink adaptively when a
            # batch still exceeds the actual provider capability.
            batches = DataPartitioner.partition_records(ecosystem, max_per_batch=40)
            logger.warning(
                "[PERSPECTIVE] complete registry is ~%d tokens; partitioning into %d lossless batches",
                full_tokens, len(batches)
            )

        all_domains: List[Dict[str, Any]] = []
        all_excluded: List[Dict[str, Any]] = []
        for batch_index, batch in enumerate(batches, start=1):
            if not batch:
                continue
            batch_registry = self._ecosystem_context(batch, self.perspective.country, self.perspective.country_code)
            user = f"ARTICLE UNDERSTANDING:\n{article_json}\n\nPERSPECTIVE ECOSYSTEM BATCH {batch_index}/{len(batches)}:\n{batch_registry}"
            input_tokens = _messages_tokens(self.budget, system, user)
            if not self.budget.fits_in_budget(input_tokens, requested):
                # Shrink only the batch boundaries, never the records themselves.
                sub_batches = DataPartitioner.partition_records(batch, max_per_batch=max(1, len(batch) // 2))
                if len(sub_batches) > 1:
                    for sub_index, sub_batch in enumerate(sub_batches, start=1):
                        sub_registry = self._ecosystem_context(sub_batch, self.perspective.country, self.perspective.country_code)
                        sub_user = f"ARTICLE UNDERSTANDING:\n{article_json}\n\nPERSPECTIVE ECOSYSTEM SUB-BATCH {batch_index}.{sub_index}:\n{sub_registry}"
                        sub_tokens = _messages_tokens(self.budget, system, sub_user)
                        if not self.budget.fits_in_budget(sub_tokens, requested):
                            raise NewsPipelineContextOverflow(
                                f"Perspective ecosystem batch {batch_index}.{sub_index} cannot fit provider context without truncation"
                            )
                        result = self._call_json(system, sub_user, requested, f"Perspective Impact Mapping [{batch_index}.{sub_index}]")
                        domains, excluded = normalize(result)
                        all_domains.extend(domains)
                        all_excluded.extend(excluded)
                    continue
                raise NewsPipelineContextOverflow(
                    f"Perspective ecosystem node batch {batch_index} cannot fit provider context without truncation"
                )
            result = self._call_json(system, user, requested, f"Perspective Impact Mapping [{batch_index}/{len(batches)}]")
            domains, excluded = normalize(result)
            all_domains.extend(domains)
            all_excluded.extend(excluded)

        # Merge equivalent domains without losing any node hints.
        merged: Dict[str, Dict[str, Any]] = {}
        for domain in all_domains:
            key = _pf_norm(domain.get("domain", "")) or f"domain-{len(merged)}"
            if key not in merged:
                merged[key] = dict(domain)
                merged[key]["ecosystem_node_hints"] = sorted(set(domain.get("ecosystem_node_hints", [])))
                continue
            current = merged[key]
            current["ecosystem_node_hints"] = sorted(set(current.get("ecosystem_node_hints", [])) | set(domain.get("ecosystem_node_hints", [])))
            for field_name in ("why_relevant", "mechanism"):
                if domain.get(field_name) and domain[field_name] not in str(current.get(field_name, "")):
                    current[field_name] = f"{current.get(field_name, '')}; {domain[field_name]}".strip("; ")
            priority_rank = {"high": 0, "medium": 1, "low": 2}
            if priority_rank.get(str(domain.get("priority", "medium")).lower(), 1) < priority_rank.get(str(current.get("priority", "medium")).lower(), 1):
                current["priority"] = domain.get("priority")

        domains = sorted(
            merged.values(),
            key=lambda d: (
                0 if str(d.get("priority", "medium")).lower() == "high" else 1 if str(d.get("priority", "medium")).lower() == "medium" else 2,
                _pf_norm(d.get("domain", "")),
            ),
        )
        return {"impact_domains": domains, "excluded_domains": all_excluded}

    # ------------------------------------------------------------------
    # DB target resolution
    # ------------------------------------------------------------------
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
        target = _pf_norm(hint)
        if not target:
            return None
        for n in ecosystem:
            if _pf_norm(n.get("node_id", "")) == target or _pf_norm(n.get("canonical_id", "")) == target:
                return n["canonical_id"]
        candidates: List[Tuple[float, str]] = []
        for n in ecosystem:
            score = max(_pf_similarity(hint, n.get("node_id", "")), _pf_similarity(hint, n.get("summary", "")))
            if score >= MIN_NODE_RESOLUTION_SCORE:
                candidates.append((score, n["canonical_id"]))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        if candidates and (len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= 0.12):
            return candidates[0][1]
        return None

    def retrieve_targets(self, impact: Dict[str, Any], ecosystem: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        resolved: Dict[str, Dict[str, Any]] = {}
        unresolved: List[Dict[str, Any]] = []
        for domain in _pf_safe_list(impact.get("impact_domains")):
            if not isinstance(domain, dict):
                continue
            hints = _pf_safe_list(domain.get("ecosystem_node_hints"))
            if not hints:
                sector = _pf_norm(domain.get("domain", ""))
                hints = [n["node_id"] for n in ecosystem if sector and sector in _pf_norm(n.get("sector", ""))]
            found = False
            for hint in hints:
                canon = self._resolve_supplied_node(str(hint), ecosystem)
                if canon:
                    found = True
                    rec = self._node_record(canon)
                    domains = set(rec.get("target_domains", []))
                    domains.add(str(domain.get("domain", "")))
                    rec["target_domains"] = sorted(x for x in domains if x)
                    resolved[canon] = rec
                else:
                    unresolved.append({"hint": str(hint), "domain": domain.get("domain", ""), "status": "UNRESOLVED"})
            if not found:
                unresolved.append({"hint": domain.get("domain", ""), "domain": domain.get("domain", ""), "status": "NO_TARGET_NODE_RESOLVED"})
        nodes = sorted(resolved.values(), key=lambda n: (str(n.get("sector", "")), str(n.get("node_id", ""))))
        if MAX_RETRIEVAL_TARGETS > 0 and len(nodes) > MAX_RETRIEVAL_TARGETS:
            # This is an explicit operator-configured cap, never an implicit truncation.
            overflow = nodes[MAX_RETRIEVAL_TARGETS:]
            nodes = nodes[:MAX_RETRIEVAL_TARGETS]
            unresolved.extend({"hint": n.get("node_id", ""), "domain": "retrieval_cap", "status": "RESEARCH_REQUIRED", "reason": "operator_configured_target_cap"} for n in overflow)
        logger.info("[RETRIEVAL] targeted nodes=%d unresolved targets=%d", len(nodes), len(unresolved))
        return nodes, unresolved

    # ------------------------------------------------------------------
    # Deterministic graph traversal
    # ------------------------------------------------------------------
    def traverse_graph(self, target_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Traverse authoritative database links without dropping discovered edges.

        Node/path caps are explicit operator controls. They never masquerade as
        complete graph truth: overflow is reported as research-required metadata.
        """
        seeds = sorted({n["canonical_id"] for n in target_nodes if n.get("canonical_id") in self.vault.file_map})
        visited: Dict[str, int] = {s: 0 for s in seeds}
        queue: List[str] = list(seeds)
        all_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        backlink_candidates = 0

        while queue:
            current = queue.pop(0)
            depth = visited[current]
            if depth >= MAX_GRAPH_DEPTH:
                continue
            outgoing = sorted(self.vault.outbound_links.get(current, []), key=lambda x: self.vault._canonicalize(x))
            incoming = sorted(self.vault.backlink_map.get(current, set()), key=lambda x: self.vault._canonicalize(x))
            neighbours: List[Tuple[str, str]] = []
            for target in outgoing:
                canon = self.vault._canonicalize(target)
                if canon in self.vault.file_map:
                    neighbours.append((canon, "outbound_link"))
            for source in incoming:
                canon = self.vault._canonicalize(source)
                if canon in self.vault.file_map:
                    neighbours.append((canon, "inbound_backlink"))
                    if canon not in visited:
                        backlink_candidates += 1

            for neighbour, relation in sorted(set(neighbours), key=lambda x: (x[0], x[1])):
                next_depth = depth + 1
                if neighbour not in visited:
                    visited[neighbour] = next_depth
                    queue.append(neighbour)
                elif visited[neighbour] < next_depth:
                    next_depth = visited[neighbour]
                edge = {
                    "from_node": self.vault.file_map.get(current, current),
                    "to_node": self.vault.file_map.get(neighbour, neighbour),
                    "relationship_type": relation,
                    "depth": next_depth,
                    "source": "database",
                }
                all_edges[(edge["from_node"], edge["to_node"], relation)] = edge

        ordered_edges = sorted(
            all_edges.values(),
            key=lambda e: (int(e.get("depth") or 0), str(e.get("from_node")), str(e.get("to_node")), str(e.get("relationship_type"))),
        )
        graph_nodes: List[Dict[str, Any]] = []
        overflow_nodes: List[str] = []
        ordered_nodes = sorted(visited.items(), key=lambda x: (x[1], x[0]))
        for index, (canon, depth) in enumerate(ordered_nodes):
            if MAX_GRAPH_NODES > 0 and index >= MAX_GRAPH_NODES:
                overflow_nodes.append(self.vault.file_map.get(canon, canon))
                continue
            rec = self._node_record(canon)
            rec["graph_depth"] = depth
            rec["graph_category"] = "target" if canon in seeds else "first_order" if depth == 1 else "second_order"
            graph_nodes.append(rec)

        allowed_ids = {str(n.get("node_id")) for n in graph_nodes}
        filtered_edges = [
            e for e in ordered_edges
            if str(e.get("from_node")) in allowed_ids and str(e.get("to_node")) in allowed_ids
        ]
        overflow_edges = max(0, len(ordered_edges) - len(filtered_edges))
        if MAX_GRAPH_PATHS > 0 and len(filtered_edges) > MAX_GRAPH_PATHS:
            overflow_edges += len(filtered_edges) - MAX_GRAPH_PATHS
            filtered_edges = filtered_edges[:MAX_GRAPH_PATHS]

        paths = [
            {
                "nodes": [e["from_node"], e["to_node"]],
                "edges": [e],
                "depth": e["depth"],
                "source": "database",
            }
            for e in filtered_edges
        ]
        logger.info(
            "[GRAPH] targets=%d | direct=%d | first-order=%d | second-order=%d | "
            "backlink candidates=%d | edges=%d | paths=%d | overflow_nodes=%d | overflow_edges=%d",
            len(seeds),
            len(seeds),
            sum(1 for n in graph_nodes if n["graph_category"] == "first_order"),
            sum(1 for n in graph_nodes if n["graph_category"] == "second_order"),
            backlink_candidates,
            len(filtered_edges),
            len(paths),
            len(overflow_nodes),
            overflow_edges,
        )
        return {
            "target_nodes": [n for n in graph_nodes if n["graph_category"] == "target"],
            "nodes": graph_nodes,
            "edges": filtered_edges,
            "paths": paths,
            "direct_nodes": len(seeds),
            "first_order_nodes": sum(1 for n in graph_nodes if n["graph_category"] == "first_order"),
            "second_order_nodes": sum(1 for n in graph_nodes if n["graph_category"] == "second_order"),
            "backlink_candidates": backlink_candidates,
            "overflow_nodes": overflow_nodes,
            "overflow_edge_count": overflow_edges,
            "graph_complete": not overflow_nodes and overflow_edges == 0,
            "research_required": [
                {
                    "status": "RESEARCH_REQUIRED",
                    "reason": "operator_configured_graph_cap",
                    "node_id": node_id,
                }
                for node_id in overflow_nodes
            ],
        }

    # ------------------------------------------------------------------
    # Graph consequence interpretation — exactly one call
    # ------------------------------------------------------------------
    def analyze_graph(self, article: Dict[str, Any], impact: Dict[str, Any], graph: Dict[str, Any], reasoning_log: ReasoningLog) -> Dict[str, Any]:
        """Convert the verified graph into machine-grounded evidence.

        There is intentionally NO graph LLM call here.  The graph is a database
        fact set, so relationships are reported deterministically.  The single
        final synthesis call receives these verified nodes/edges and performs
        interpretation. This removes the graph-partition latency seen in the
        production logs.
        """
        nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
        edges = [e for e in graph.get("edges", []) if isinstance(e, dict)]
        consequences: List[Dict[str, Any]] = []
        for edge in edges:
            consequences.append({
                "statement": f"Verified database relationship: {edge.get('from_node')} -> {edge.get('to_node')}.",
                "order": "direct" if int(edge.get("depth") or 1) == 1 else "second_order",
                "supporting_nodes": [str(edge.get("from_node")), str(edge.get("to_node"))],
                "supporting_edges": [f"{edge.get('from_node')}->{edge.get('to_node')}"],
                "confidence": 1.0,
                "source": "database",
            })
        gaps = []
        if not nodes:
            gaps.append({"gap": "No targeted database graph nodes were resolved", "status": "RESEARCH_REQUIRED", "related_nodes": []})
        return {
            "consequences": consequences,
            "gaps": gaps,
            "opportunity_signals": [],
            "risk_signals": [],
            "source": "database_deterministic",
        }

    # ------------------------------------------------------------------
    # Impact chain
    # ------------------------------------------------------------------
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
        for idx, meaning in enumerate(_pf_safe_list(article.get("meaning"))):
            if not isinstance(meaning, dict):
                continue
            chain.append({
                "stage": "meaning",
                "label": str(meaning.get("interpretation", "")),
                "status": "INTERPRETATION",
                "source": "llm_article_interpretation",
                "evidence": meaning.get("based_on_fact_indexes", []),
                "sequence": idx,
            })
        for domain in _pf_safe_list(impact.get("impact_domains")):
            if not isinstance(domain, dict):
                continue
            chain.append({
                "stage": "perspective_ecosystem",
                "label": str(domain.get("domain", "")),
                "status": "INVESTIGATION_TARGET",
                "source": "perspective_ecosystem",
                "mechanism": str(domain.get("mechanism", "")),
                "priority": str(domain.get("priority", "medium")),
                "node_hints": _pf_safe_list(domain.get("ecosystem_node_hints")),
            })
        for edge in graph.get("edges", []):
            chain.append({
                "stage": "graph_relationship",
                "label": f"{edge.get('from_node')} → {edge.get('to_node')}",
                "status": "SUPPORTED",
                "source": "database",
                "relationship": edge.get("relationship_type"),
                "depth": edge.get("depth"),
                "evidence": [f"{edge.get('from_node')}->{edge.get('to_node')}"],
            })
        for item in _pf_safe_list(consequences.get("consequences")):
            if not isinstance(item, dict):
                continue
            chain.append({
                "stage": "second_order_effect" if item.get("order") == "second_order" else "consequence",
                "label": str(item.get("statement", "")),
                "status": "SUPPORTED" if item.get("supporting_edges") or item.get("supporting_nodes") else "RESEARCH_REQUIRED",
                "source": "verified_graph_interpretation",
                "evidence": _pf_safe_list(item.get("supporting_nodes")),
                "graph_edges": _pf_safe_list(item.get("supporting_edges")),
                "confidence": _pf_safe_float(item.get("confidence", 0.0)),
            })
        for gap in _pf_safe_list(consequences.get("gaps")):
            if not isinstance(gap, dict):
                continue
            chain.append({
                "stage": "gap",
                "label": str(gap.get("gap", "")),
                "status": "RESEARCH_REQUIRED",
                "source": "database_graph_gap",
                "evidence": _pf_safe_list(gap.get("related_nodes")),
                "missing_relationship": str(gap.get("missing_relationship", "")),
            })
        return chain

    # ------------------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------------------
    def final_synthesis(
        self,
        article: Dict[str, Any],
        impact: Dict[str, Any],
        graph: Dict[str, Any],
        consequences: Dict[str, Any],
        impact_chain: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Produce final synthesis with genuinely lossless, budget-aware batches.

        The previous implementation partitioned graph *nodes* and then rebuilt
        edges/paths by node membership. A highly connected node could therefore
        pull hundreds of edges and paths into one nominal 40-node batch, making
        the first batch impossible to fit. This implementation packs complete
        records by their actual serialized cost and partitions nodes, edges,
        paths, consequences, and narrative-chain records independently.

        No record is character-truncated. Records are only moved between
        batches. A record that cannot fit even by itself raises an explicit
        context-overflow error.
        """
        system = """You are the ATIS News Final Synthesis Engine.
Produce the final dashboard from the supplied structured analysis.

HARD RULES:
1. The selected perspective country is the lens. Never switch to a source-country/local perspective.
2. Article facts come only from the supplied article understanding.
3. Database nodes and edges are authoritative. Never invent an edge, actor, capability, or relationship.
4. Opportunities must be supported by verified database nodes AND a verified graph path/edge, unless status=RESEARCH_REQUIRED.
5. A plausible commercial idea without database support is RESEARCH_REQUIRED, not SUPPORTED.
6. Preserve supplied impact-chain evidence; do not rewrite its evidence trail.
7. Key entities must correspond to supplied database nodes or article actors and must carry provenance.
8. Return concise but complete JSON. Do not omit fields simply because an analysis is partial.
9. Return ONLY valid JSON.

Schema:
{
 "trigger_event":"",
 "market_equilibrium_shift":"",
 "executive_summary":"",
 "analytical_perspective":{"country":"","country_code":"","description":""},
 "facts":[], "meaning":[], "impact_domains":[], "impact_chain":[],
 "findings":[{"text":"","source_nodes":[],"graph_paths":[],"status":"SUPPORTED|RESEARCH_REQUIRED"}],
 "opportunities":[{"opportunity_id":"","title":"","status":"SUPPORTED|RESEARCH_REQUIRED","perspective_country":"","opportunity_country":"","source_country":"","event_country":"","cross_border":false,"cross_border_countries":[],"perspective_actor":"","perspective_capability":"","pathway":"","justification":"","urgency_score":0.0,"feasibility_score":0.0,"source_nodes":[],"graph_paths":[],"required_missing_nodes":[]}],
 "risks":[{"text":"","status":"SUPPORTED|RESEARCH_REQUIRED","severity":"high|medium|low","source_nodes":[],"graph_paths":[]},
 ],
 "gaps":[{"gap":"","status":"RESEARCH_REQUIRED","related_nodes":[],"missing_relationship":""}],
 "key_entities":[{"entity_name":"","entity_type":"","country":"","sector":"","significance_score":0,"summary":"","source_node":""}]
}
"""

        def _records(value: Any) -> List[Dict[str, Any]]:
            return [dict(x) for x in _pf_safe_list(value) if isinstance(x, dict)]

        def _record_cost(record: Dict[str, Any]) -> int:
            # Estimate the actual serialized record, not its character count.
            return max(1, self.budget.estimate_tokens(_json(record)))

        # Keep the immutable article/perspective context in every call. Large
        # evidence collections are packed independently below.
        base = (
            f"PERSPECTIVE={self.perspective.country} ({self.perspective.country_code})\n"
            f"ARTICLE UNDERSTANDING={_json(article)}\n"
            f"PERSPECTIVE IMPACT={_json(impact)}\n"
        )

        # Consequence lists can themselves be large. Treat each complete
        # dictionary as an atomic record; no slicing of strings or dictionaries.
        consequence_records: List[Dict[str, Any]] = []
        for key in ("consequences", "opportunity_signals", "risk_signals", "gaps"):
            for item in _records(consequences.get(key)):
                consequence_records.append({"collection": key, "record": item})

        narrative_records = [
            {"collection": "impact_chain", "record": item}
            for item in _records(impact_chain)
            if item.get("stage") != "graph_relationship"
        ]

        node_records = [
            {"collection": "graph_nodes", "record": dict(n)}
            for n in _records(graph.get("nodes"))
        ]
        edge_records = [
            {"collection": "graph_edges", "record": dict(e)}
            for e in _records(graph.get("edges"))
        ]
        path_records = [
            {"collection": "graph_paths", "record": dict(path)}
            for path in _records(graph.get("paths"))
        ]

        atomic = consequence_records + narrative_records + node_records + edge_records + path_records

        # We need enough room for the final output as well. Use the same
        # provider-context budget the engine uses for normal calls.
        requested = MAX_FINAL_OUTPUT_TOKENS
        base_tokens = _messages_tokens(self.budget, system, base)
        usable = self.budget.usable_context_budget - requested - base_tokens
        if usable <= 0:
            raise NewsPipelineContextOverflow(
                f"Final synthesis immutable context alone consumes {base_tokens} tokens; "
                f"no room remains for complete evidence records."
            )

        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if current:
                batches.append(current)
                current = []
                current_tokens = 0

        for record in atomic:
            cost = _record_cost(record)
            if cost > usable:
                raise NewsPipelineContextOverflow(
                    "A complete final-synthesis evidence record cannot fit within the "
                    f"provider context budget ({cost} > {usable} tokens). "
                    "ATIS refuses to truncate the record."
                )
            if current and current_tokens + cost > usable:
                flush()
            current.append(record)
            current_tokens += cost
        flush()

        # A final call is still required when there is no graph/evidence: send
        # the immutable context once so the model can produce a truthful
        # research-required dashboard.
        if not batches:
            batches = [[]]

        logger.info(
            "[FINAL] lossless final synthesis: %d complete batches | "
            "base_tokens=%d | evidence_budget=%d | records=%d",
            len(batches), base_tokens, usable, len(atomic),
        )

        results: List[Dict[str, Any]] = []
        for index, batch in enumerate(batches, start=1):
            grouped: Dict[str, List[Any]] = {}
            for item in batch:
                grouped.setdefault(str(item["collection"]), []).append(item["record"])

            payload = {
                "consequences": grouped.get("consequences", []),
                "opportunity_signals": grouped.get("opportunity_signals", []),
                "risk_signals": grouped.get("risk_signals", []),
                "gaps": grouped.get("gaps", []),
                "impact_chain": grouped.get("impact_chain", []),
                "graph": {
                    "nodes": grouped.get("graph_nodes", []),
                    "edges": grouped.get("graph_edges", []),
                    "paths": grouped.get("graph_paths", []),
                },
                "batch_contract": {
                    "batch_index": index,
                    "batch_count": len(batches),
                    "lossless_record_partition": True,
                    "record_count": len(batch),
                },
            }
            user = (
                base
                + f"VERIFIED EVIDENCE BATCH {index}/{len(batches)}="
                + _json(payload)
                + "\n"
                "Analyze ONLY the supplied evidence. Preserve provenance and mark "
                "anything not supported by this batch as RESEARCH_REQUIRED."
            )
            input_tokens = _messages_tokens(self.budget, system, user)
            if not self.budget.fits_in_budget(input_tokens, requested):
                # This should be unreachable because packing used the same
                # estimator. Keep the guard to prevent accidental truncation if
                # estimator behavior changes.
                raise NewsPipelineContextOverflow(
                    f"Final synthesis batch {index} exceeded the provider budget "
                    f"after serialization ({input_tokens} tokens)."
                )
            results.append(
                self._call_json(
                    system,
                    user,
                    requested,
                    f"Final News Synthesis [{index}/{len(batches)}]",
                )
            )

        if not results:
            raise RuntimeError("Final synthesis produced no batch results")

        # Deterministic, loss-preserving merge. Article-level fields come from
        # the first result; evidence-bearing collections are unioned.
        merged = dict(results[0])
        list_fields = ("findings", "opportunities", "risks", "gaps", "key_entities")
        for field_name in list_fields:
            combined: List[Any] = []
            seen: Set[str] = set()
            for result in results:
                for item in _pf_safe_list(result.get(field_name)):
                    key = _json(item)
                    if key not in seen:
                        seen.add(key)
                        combined.append(item)
            merged[field_name] = combined

        merged["facts"] = _pf_safe_list(article.get("facts"))
        merged["meaning"] = _pf_safe_list(article.get("meaning"))
        merged["impact_domains"] = _pf_safe_list(impact.get("impact_domains"))
        merged["impact_chain"] = impact_chain
        return merged

    # ------------------------------------------------------------------
    # Grounding
    # ------------------------------------------------------------------
    def validate_and_ground(
        self,
        dashboard: Dict[str, Any],
        graph: Dict[str, Any],
        ecosystem: List[Dict[str, Any]],
        article: Dict[str, Any],
        impact: Dict[str, Any],
        consequences: Dict[str, Any],
    ) -> Dict[str, Any]:
        valid_nodes = {str(n.get("node_id")) for n in graph.get("nodes", []) if n.get("node_id")}
        valid_edges = {
            f"{e.get('from_node')}->{e.get('to_node')}"
            for e in graph.get("edges", [])
            if e.get("from_node") and e.get("to_node")
        }
        perspective_nodes = {str(n.get("node_id")) for n in ecosystem if n.get("node_id")}

        grounded_opportunities: List[Dict[str, Any]] = []
        for raw in _pf_safe_list(dashboard.get("opportunities")):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            source_nodes = [str(x) for x in _pf_safe_list(item.get("source_nodes")) if str(x) in valid_nodes]
            graph_paths = [str(x) for x in _pf_safe_list(item.get("graph_paths")) if str(x) in valid_edges]
            actor = str(item.get("perspective_actor", ""))
            actor_supported = bool(actor) and actor in perspective_nodes
            perspective_country = str(item.get("perspective_country") or self.perspective.country)
            path_endpoints = {
                part for path in graph_paths
                for part in str(path).split("->", 1)
            }
            supported = bool(
                source_nodes
                and graph_paths
                and actor_supported
                and _pf_norm(perspective_country) == _pf_norm(self.perspective.country)
                and _pf_norm(str(item.get("opportunity_country") or self.perspective.country)) == _pf_norm(self.perspective.country)
                and any(node in path_endpoints for node in source_nodes)
            )
            item["source_nodes"] = source_nodes
            item["graph_paths"] = graph_paths
            item["perspective_country"] = self.perspective.country
            item["perspective_country_code"] = self.perspective.country_code
            if supported:
                item["status"] = "SUPPORTED"
                item["opportunity_confidence"] = max(_pf_safe_float(item.get("opportunity_confidence")), MIN_OPPORTUNITY_GRAPH_SCORE)
            else:
                item["status"] = "RESEARCH_REQUIRED"
                item["opportunity_confidence"] = min(_pf_safe_float(item.get("opportunity_confidence")), 0.49)
                missing = list(_pf_safe_list(item.get("required_missing_nodes")))
                if not source_nodes:
                    missing.append("verified source node")
                if not graph_paths:
                    missing.append("verified database graph path")
                if not actor_supported:
                    missing.append("verified perspective-country actor")
                item["required_missing_nodes"] = sorted(set(str(x) for x in missing if x))
            grounded_opportunities.append(item)
        dashboard["opportunities"] = grounded_opportunities

        grounded_risks: List[Dict[str, Any]] = []
        for raw in _pf_safe_list(dashboard.get("risks")):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item["source_nodes"] = [str(x) for x in _pf_safe_list(item.get("source_nodes")) if str(x) in valid_nodes]
            item["graph_paths"] = [str(x) for x in _pf_safe_list(item.get("graph_paths")) if str(x) in valid_edges]
            item.setdefault("status", "SUPPORTED" if item["source_nodes"] else "RESEARCH_REQUIRED")
            grounded_risks.append(item)
        dashboard["risks"] = grounded_risks

        dashboard["findings"] = [x for x in _pf_safe_list(dashboard.get("findings")) if isinstance(x, dict)]
        dashboard["gaps"] = [x for x in _pf_safe_list(dashboard.get("gaps")) if isinstance(x, dict)]
        dashboard["entities"] = _pf_safe_list(dashboard.get("entities"))
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
            "paths": graph.get("paths", []),
        }
        event = article.get("event", {}) if isinstance(article.get("event"), dict) else {}
        dashboard["source_country"] = event.get("source_country", "") or ""
        dashboard["event_country"] = event.get("event_country", "") or ""
        return dashboard

    # ------------------------------------------------------------------
    # Truthful deterministic fallback
    # ------------------------------------------------------------------
    def _fallback_dashboard(
        self,
        article: Dict[str, Any],
        impact: Dict[str, Any],
        graph: Dict[str, Any],
        consequences: Dict[str, Any],
        impact_chain: List[Dict[str, Any]],
        reason: str,
    ) -> Dict[str, Any]:
        event = article.get("event", {}) if isinstance(article.get("event"), dict) else {}
        return {
            "status": "partial",
            "partial": True,
            "detail": reason,
            "trigger_event": str(event.get("title") or "News event"),
            "market_equilibrium_shift": "",
            "executive_summary": str(event.get("summary") or ""),
            "analytical_perspective": {
                "country": self.perspective.country,
                "country_code": self.perspective.country_code,
                "description": "Country through which this event is interpreted.",
            },
            "facts": _pf_safe_list(article.get("facts")),
            "meaning": _pf_safe_list(article.get("meaning")),
            "impact_domains": _pf_safe_list(impact.get("impact_domains")),
            "impact_chain": impact_chain,
            "findings": _pf_safe_list(consequences.get("consequences")),
            "opportunities": _pf_safe_list(consequences.get("opportunity_signals")),
            "risks": _pf_safe_list(consequences.get("risk_signals")),
            "gaps": _pf_safe_list(consequences.get("gaps")),
            "entities": _pf_safe_list(article.get("actors")),
            "graph_analysis": {
                "direct": graph.get("direct_nodes", 0),
                "first_order": graph.get("first_order_nodes", 0),
                "second_order": graph.get("second_order_nodes", 0),
                "backlink_candidates": graph.get("backlink_candidates", 0),
                "paths": graph.get("paths", []),
            },
            "source_country": event.get("source_country", ""),
            "event_country": event.get("event_country", ""),
        }

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
    def run(
        self,
        article_text: str,
        reasoning_log: ReasoningLog,
        state: Optional[JobState] = None,
        persistence: Optional[StatePersistenceManager] = None,
    ) -> Dict[str, Any]:
        """Run the News pipeline through durable, resumable stage checkpoints."""
        if not article_text or not article_text.strip():
            raise ValueError("News article text is empty")
        article_text = article_text.strip()
        if len(article_text) > MAX_ARTICLE_CHARS:
            raise ValueError(
                f"Article contains {len(article_text):,} characters, exceeding the configured safety limit "
                f"{MAX_ARTICLE_CHARS:,}. ATIS refuses to truncate the source article."
            )

        persistence = persistence or StatePersistenceManager()
        if state is None:
            state = JobState(intelligence_id="transient-news-job")
        state.stage_data.setdefault("raw_article", article_text)
        state.stage_data.setdefault("perspective", self.perspective.as_dict())

        def checkpoint(stage: PipelineStage, data: Any) -> None:
            state.current_stage = stage.value
            state.mark_stage_complete(stage, data)
            state.status = "IN_PROGRESS"
            persistence.save_state(state)
            logger.info("[CHECKPOINT] %s complete | job=%s", stage.value, state.intelligence_id)

        def set_stage(stage: PipelineStage) -> None:
            state.current_stage = stage.value
            state.status = "IN_PROGRESS"
            persistence.save_state(state)

        # Stage 1 — Article Understanding
        if state.is_completed(PipelineStage.ARTICLE_UNDERSTANDING):
            article = state.stage_data[PipelineStage.ARTICLE_UNDERSTANDING.value]
        else:
            set_stage(PipelineStage.ARTICLE_UNDERSTANDING)
            article = self.understand_article(article_text)
            checkpoint(PipelineStage.ARTICLE_UNDERSTANDING, article)

        reasoning_log.entities_extracted = len(_pf_safe_list(article.get("actors")))
        reasoning_log.estimated_tokens = self.budget.estimate_tokens(article_text)
        reasoning_log.safe_budget = self.budget.provider_context_limit - self.budget.safety_margin

        # Stage 2 — Perspective Mapping. The ecosystem is database-derived only.
        if state.is_completed(PipelineStage.PERSPECTIVE_MAPPING):
            stage2 = state.stage_data[PipelineStage.PERSPECTIVE_MAPPING.value]
            ecosystem = _pf_safe_list(stage2.get("ecosystem"))
            impact = stage2.get("impact", {}) if isinstance(stage2, dict) else {}
        else:
            set_stage(PipelineStage.PERSPECTIVE_MAPPING)
            ecosystem = self.load_perspective_ecosystem()
            reasoning_log.perspective_nodes = len(ecosystem)
            impact = self.map_impact_domains(article, ecosystem)
            checkpoint(PipelineStage.PERSPECTIVE_MAPPING, {"ecosystem": ecosystem, "impact": impact})
        reasoning_log.perspective_nodes = len(ecosystem)
        reasoning_log.impact_domains = len(_pf_safe_list(impact.get("impact_domains")))

        # Stage 3 — Database Retrieval
        if state.is_completed(PipelineStage.DATABASE_RETRIEVAL):
            stage3 = state.stage_data[PipelineStage.DATABASE_RETRIEVAL.value]
            targets = _pf_safe_list(stage3.get("targets"))
            unresolved = _pf_safe_list(stage3.get("unresolved"))
        else:
            set_stage(PipelineStage.DATABASE_RETRIEVAL)
            targets, unresolved = self.retrieve_targets(impact, ecosystem)
            checkpoint(PipelineStage.DATABASE_RETRIEVAL, {"targets": targets, "unresolved": unresolved})
        reasoning_log.retrieval_targets = len(targets)
        reasoning_log.candidate_nodes = len(self.vault.file_map)

        # Stage 4 — Deterministic Graph Traversal
        if state.is_completed(PipelineStage.GRAPH_TRAVERSAL):
            graph = state.stage_data[PipelineStage.GRAPH_TRAVERSAL.value]
        else:
            set_stage(PipelineStage.GRAPH_TRAVERSAL)
            graph = self.traverse_graph(targets)
            checkpoint(PipelineStage.GRAPH_TRAVERSAL, graph)
        reasoning_log.direct_graph_nodes = graph.get("direct_nodes", 0)
        reasoning_log.first_order_graph_nodes = graph.get("first_order_nodes", 0)
        reasoning_log.second_order_graph_nodes = graph.get("second_order_nodes", 0)
        reasoning_log.backlink_candidates = graph.get("backlink_candidates", 0)
        reasoning_log.graph_paths = len(graph.get("paths", []))
        reasoning_log.relevant_nodes = len(graph.get("nodes", []))
        reasoning_log.selected_evidence = len(graph.get("nodes", []))
        reasoning_log.reasoning_mode = "perspective_first_graph_grounded" if graph.get("nodes") else "perspective_only_no_graph_evidence"

        # Stage 5 — Impact Analysis. Consequences are interpreted from verified graph data.
        if state.is_completed(PipelineStage.IMPACT_ANALYSIS):
            stage5 = state.stage_data[PipelineStage.IMPACT_ANALYSIS.value]
            consequences = stage5.get("consequences", {})
            impact_chain = _pf_safe_list(stage5.get("impact_chain"))
        else:
            set_stage(PipelineStage.IMPACT_ANALYSIS)
            consequences = self.analyze_graph(article, impact, graph, reasoning_log)
            if not graph.get("nodes"):
                consequences.setdefault("gaps", []).append({
                    "gap": "No evidence-backed relationship was found between the selected perspective targets and the current database graph.",
                    "status": "RESEARCH_REQUIRED",
                    "related_nodes": [n.get("node_id", "") for n in targets],
                })
            impact_chain = self.build_impact_chain(article, impact, graph, consequences)
            checkpoint(PipelineStage.IMPACT_ANALYSIS, {"consequences": consequences, "impact_chain": impact_chain})

        # Stage 6 — Final Synthesis. If the deadline is too small, return a truthful
        # partial result without marking the stage complete; the next invocation resumes here.
        if state.is_completed(PipelineStage.FINAL_SYNTHESIS):
            dashboard = state.stage_data[PipelineStage.FINAL_SYNTHESIS.value]
        else:
            set_stage(PipelineStage.FINAL_SYNTHESIS)
            try:
                dashboard = self.final_synthesis(article, impact, graph, consequences, impact_chain)
                dashboard = self.validate_and_ground(dashboard, graph, ecosystem, article, impact, consequences)
            except (NewsPipelineDeadline, LLMTokenLimitError, NewsPipelineContextOverflow, RuntimeError) as exc:
                logger.warning("[FINAL] %s", exc)
                # A transport failure is a job failure/partial result, not an
                # HTTP deadline. The durable worker may retry the whole job
                # while already-completed checkpoints remain intact.
                dashboard = self._fallback_dashboard(article, impact, graph, consequences, impact_chain, str(exc))
            if not self.deadline_exhausted and dashboard.get("status") != "partial":
                checkpoint(PipelineStage.FINAL_SYNTHESIS, dashboard)

        # Authoritative fields are rebuilt from deterministic/article sources after
        # every resume so a malformed or stale LLM result cannot erase evidence.
        dashboard = dict(dashboard or {})
        dashboard["facts"] = _pf_safe_list(article.get("facts"))
        dashboard["meaning"] = _pf_safe_list(article.get("meaning"))
        dashboard["impact_domains"] = _pf_safe_list(impact.get("impact_domains"))
        dashboard["impact_chain"] = impact_chain
        dashboard["source_nodes"] = [
            {k: n.get(k, "") for k in ("node_id", "canonical_id", "country", "type", "sector", "summary", "path", "graph_depth", "graph_category")}
            for n in graph.get("nodes", []) if isinstance(n, dict)
        ]
        dashboard["perspective_nodes"] = [
            {k: n.get(k, "") for k in ("node_id", "canonical_id", "country", "type", "sector", "summary", "path")}
            for n in ecosystem if isinstance(n, dict)
        ]
        dashboard["cross_border_bridges"] = dashboard.get("cross_border_analysis", {}).get("cross_border_bridges", []) if isinstance(dashboard.get("cross_border_analysis"), dict) else []
        dashboard["structured_intelligence"] = {
            "facts": dashboard["facts"],
            "meaning": dashboard["meaning"],
            "impact_domains": dashboard["impact_domains"],
            "graph": {"nodes": dashboard["source_nodes"], "edges": graph.get("edges", []), "paths": graph.get("paths", [])},
            "consequences": consequences.get("consequences", []),
            "opportunities": dashboard.get("opportunities", []),
            "risks": dashboard.get("risks", []),
            "gaps": dashboard.get("gaps", []),
        }
        dashboard["research_required"] = (
            unresolved
            + _pf_safe_list(consequences.get("gaps"))
            + _pf_safe_list(graph.get("research_required"))
        )
        reasoning_log.gaps = len(_pf_safe_list(dashboard.get("gaps")))
        reasoning_log.opportunities = len(_pf_safe_list(dashboard.get("opportunities")))
        reasoning_log.total_llm_calls = self.calls
        reasoning_log.evidence_calls = getattr(self, "_evidence_calls", 0)
        reasoning_log.final_call = getattr(self, "_final_call", 0)
        reasoning_log.synthesis_calls = 0
        reasoning_log.partitions = max(1, reasoning_log.partitions, 1 if reasoning_log.impact_domains else 0)
        dashboard["status"] = dashboard.get("status", "complete")
        dashboard["partial"] = bool(dashboard.get("partial", False))
        dashboard["pipeline_execution"] = {
            "deadline_seconds": None,
            "execution_model": "durable_worker",
            "transport_timeout_only": True,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
            # None means there is no artificial intelligence deadline. Do not
            # serialize float("inf") because Starlette's JSON encoder rejects it.
            "remaining_seconds": None if not math.isfinite(self._remaining_seconds()) else round(self._remaining_seconds(), 3),
            "llm_calls": self.calls,
            "llm_timeouts": self.timeouts,
            "truncated_retries": self.truncated_retries,
            "retry_calls": getattr(self, "_retry_calls", 0),
            "deadline_exhausted": False,
            "transport_failures": self.timeouts,
            "stage_durations": dict(self._stage_durations),
            "graph_llm_calls": 0,
            "planned_llm_stages": ["Article Understanding", "Perspective Impact Mapping", "Final News Synthesis"],
            "max_normal_llm_calls": None,
            "llm_call_policy": "stage calls are bounded by provider transport timeouts; partitioning may increase call count",
            "character_truncation": False,
            "structured_record_truncation": False,
            "durable_checkpointing": True,
            "checkpoint_job_id": state.intelligence_id,
            "completed_stages": list(state.completed_stages),
            "current_stage": state.current_stage,
        }
        if not self.deadline_exhausted and dashboard.get("status") != "partial":
            state.status = "COMPLETED"
            state.current_stage = PipelineStage.COMPLETE.value
            if PipelineStage.COMPLETE.value not in state.completed_stages:
                state.completed_stages.append(PipelineStage.COMPLETE.value)
            state.stage_data[PipelineStage.FINAL_SYNTHESIS.value] = dashboard
            persistence.save_state(state)
        else:
            state.status = "PARTIAL"
            persistence.save_state(state)
        return dashboard


# ---------------------------------------------------------------------------
# Reasoning metadata / response shaping
# ---------------------------------------------------------------------------
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
    knowledge_hash = getattr(knowledge_state, "knowledge_state_hash", "")
    graph_paths = dashboard.get("graph_analysis", {}).get("paths", []) if isinstance(dashboard.get("graph_analysis"), dict) else []
    evidence_ids = sorted(
        f"{p.get('from_node','')}->{p.get('to_node','')}:{p.get('depth','')}:{p.get('relationship_type','')}"
        for p in graph_paths if isinstance(p, dict)
    )
    entity_ids = sorted(
        str(x.get("node_id", x)) if isinstance(x, dict) else str(x)
        for x in _pf_safe_list(dashboard.get("entities"))
    )
    try:
        fingerprint = compute_analysis_fingerprint(
            story_id=core_event,
            perspective=perspective,
            evidence_ids=evidence_ids,
            entity_ids=entity_ids,
            relationship_ids=sorted(
                f"{p.get('from_node')}->{p.get('to_node')}"
                for p in graph_paths if isinstance(p, dict)
            ),
            knowledge_state_hash=knowledge_hash,
        )
    except Exception:
        fingerprint = hashlib.sha256(_json(dashboard).encode()).hexdigest()[:24]

    reasoning_data = reasoning_log.to_dict()
    for attr in (
        "perspective_nodes", "impact_domains", "retrieval_targets",
        "direct_graph_nodes", "first_order_graph_nodes", "second_order_graph_nodes",
        "graph_paths", "gaps", "opportunities", "graph_llm_calls",
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
        "cross_border_bridges_found": dashboard.get("cross_border_analysis", {}).get("cross_border_bridges_count", 0) if isinstance(dashboard.get("cross_border_analysis"), dict) else 0,
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "analysis_fingerprint": fingerprint,
        "knowledge_state": knowledge_state.as_dict() if hasattr(knowledge_state, "as_dict") else {"hash": knowledge_hash},
        "reasoning_log": reasoning_data,
        "llm_calls": engine.calls,
        "truncated_retries": engine.truncated_retries,
        "pipeline_execution": dashboard.get("pipeline_execution", {}),
    }


def _pf_finalize_dashboard(
    dashboard: Dict[str, Any],
    perspective: PerspectiveContext,
    engine: PerspectiveFirstNewsEngine,
    reasoning_log: ReasoningLog,
    vault: ObsidianVaultManager,
) -> Dict[str, Any]:
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
    for key in ("opportunities", "risks", "gaps", "entities", "findings", "facts", "meaning", "impact_domains", "impact_chain", "structured_intelligence"):
        dashboard.setdefault(key, [])
    return dashboard


# ---------------------------------------------------------------------------
# Production entry points
# ---------------------------------------------------------------------------
def _run_perspective_first_news(article_text: str, perspective: Any | None = None, source_label: str = "web_upload", job_id: str | None = None) -> Dict[str, Any]:
    started_at = time.monotonic()
    perspective = perspective or PerspectiveContext()
    reasoning_log = ReasoningLog()
    logger.info("=" * 78)
    logger.info("ATIS NEWS v%s | PERSPECTIVE-FIRST | %s (%s)", PERSPECTIVE_FIRST_VERSION, perspective.country, perspective.country_code)
    logger.info("=" * 78)

    # Job execution is intentionally independent of HTTP request lifetime.
    # The caller may disconnect while this durable worker continues processing.
    persistence = StatePersistenceManager()
    if not job_id:
        job_id = DurableNewsJobQueue.make_job_id(article_text, perspective)
    state = persistence.load_state(job_id)
    if state is not None and state.stage_data.get("raw_article") != article_text.strip():
        logger.warning("[STATE] Checkpoint %s belongs to different article input; starting a fresh job", job_id)
        state = None
    if state is not None and state.stage_data.get("_pipeline_version") != PERSPECTIVE_FIRST_VERSION:
        logger.info("[STATE] Checkpoint %s was created by an older pipeline version; starting a fresh job", job_id)
        state = None
    if state is not None:
        saved_perspective = state.stage_data.get("perspective", {})
        if isinstance(saved_perspective, dict):
            saved_country = _pf_norm(saved_perspective.get("country", ""))
            saved_code = _pf_norm(saved_perspective.get("country_code", ""))
            if saved_country != _pf_norm(perspective.country) or saved_code != _pf_norm(perspective.country_code):
                logger.warning(
                    "[STATE] Checkpoint %s belongs to a different analytical perspective; starting a fresh job",
                    job_id,
                )
                state = None
    if state is None:
        state = JobState(intelligence_id=job_id)
        state.stage_data["raw_article"] = article_text.strip()
        state.stage_data["perspective"] = perspective.as_dict()
        state.stage_data["_pipeline_version"] = PERSPECTIVE_FIRST_VERSION
        persistence.save_state(state)
    vault = _get_cached_vault(VAULT_DIR)
    engine = PerspectiveFirstNewsEngine(vault, perspective, started_at=started_at)
    knowledge_state = KnowledgeState(vault_path=vault.vault_dir)
    knowledge_state.compute()
    # KnowledgeState in older builds can report zero cached nodes even when the
    # authoritative vault index contains them. News telemetry must describe the
    # actual graph used for this request.
    try:
        knowledge_state.total_nodes = len(vault.file_map)
        knowledge_state.total_files = len(list(vault.vault_dir.rglob("*.md")))
    except Exception:
        pass

    try:
        dashboard = engine.run(article_text, reasoning_log, state=state, persistence=persistence)
    except Exception as exc:
        logger.exception("[NEWS] Production pipeline failed safely: %s", exc)
        state.status = "PARTIAL"
        state.current_stage = state.current_stage or PipelineStage.ARTICLE_UNDERSTANDING.value
        state.error_log.append(str(exc))
        persistence.save_state(state)
        article_state = state.stage_data.get(PipelineStage.ARTICLE_UNDERSTANDING.value, {})
        stage2 = state.stage_data.get(PipelineStage.PERSPECTIVE_MAPPING.value, {})
        stage3 = state.stage_data.get(PipelineStage.DATABASE_RETRIEVAL.value, {})
        graph = state.stage_data.get(PipelineStage.GRAPH_TRAVERSAL.value, {})
        stage5 = state.stage_data.get(PipelineStage.IMPACT_ANALYSIS.value, {})
        impact = stage2.get("impact", {}) if isinstance(stage2, dict) else {}
        ecosystem = _pf_safe_list(stage2.get("ecosystem")) if isinstance(stage2, dict) else []
        targets = _pf_safe_list(stage3.get("targets")) if isinstance(stage3, dict) else []
        unresolved = _pf_safe_list(stage3.get("unresolved")) if isinstance(stage3, dict) else []
        consequences = stage5.get("consequences", {}) if isinstance(stage5, dict) else {}
        impact_chain = _pf_safe_list(stage5.get("impact_chain")) if isinstance(stage5, dict) else []
        dashboard = engine._fallback_dashboard(
            article_state if isinstance(article_state, dict) else {"event": {}, "facts": [], "meaning": [], "actors": [], "uncertainties": []},
            impact if isinstance(impact, dict) else {"impact_domains": []},
            graph if isinstance(graph, dict) else {"nodes": [], "edges": [], "paths": []},
            consequences if isinstance(consequences, dict) else {"consequences": [], "gaps": []},
            impact_chain, str(exc)
        )
        dashboard["research_required"] = (
            unresolved
            + _pf_safe_list(dashboard.get("gaps"))
            + _pf_safe_list(graph.get("research_required"))
        )
        dashboard["status"] = "partial"
        dashboard["partial"] = True
        dashboard["pipeline_checkpoint"] = {
            "job_id": job_id,
            "completed_stages": list(state.completed_stages),
            "current_stage": state.current_stage,
            "resume_available": True,
            "errors": list(state.error_log),
        }

    dashboard = _pf_finalize_dashboard(dashboard, perspective, engine, reasoning_log, vault)

    # Cross-border bridges are deterministic DB output.  Crucially, an unknown
    # source/event country must NOT be replaced with the perspective country,
    # because that would fabricate a cross-border relationship.
    source_country = str(dashboard.get("source_country") or "").strip()
    event_country = str(dashboard.get("event_country") or "").strip()
    bridge_source = source_country or event_country
    bridges: List[Dict[str, Any]] = []
    if bridge_source and _pf_norm(bridge_source) != _pf_norm(perspective.country):
        try:
            bridges = vault.build_cross_border_bridge_context(perspective, bridge_source)
        except Exception as exc:
            logger.warning("[BRIDGE] deterministic bridge lookup failed: %s", exc)
    dashboard["cross_border_analysis"] = {
        "event_country": event_country,
        "source_country": source_country,
        "cross_border_bridges": bridges,
        "cross_border_bridges_count": len(bridges),
        "source": "database",
        "status": "SUPPORTED" if bridges else "NONE_FOUND" if bridge_source else "UNKNOWN_SOURCE_COUNTRY",
    }

    dashboard["pipeline_metadata"] = _pf_build_reasoning_metadata(
        reasoning_log, engine, vault, perspective,
        {"event": {"title": dashboard.get("trigger_event", ""), "source_country": source_country, "event_country": event_country}, "facts": dashboard.get("facts", [])},
        dashboard, knowledge_state
    )
    dashboard["pipeline_metadata"]["source_article"] = source_label
    dashboard["pipeline_metadata"]["reasoning_log"] = dict(dashboard["pipeline_metadata"].get("reasoning_log", {}))
    dashboard["pipeline_metadata"]["checkpointing"] = {
        "job_id": state.intelligence_id,
        "status": state.status,
        "current_stage": state.current_stage,
        "completed_stages": list(state.completed_stages),
        "checkpoint_store": str(persistence.storage_dir),
        "resume_available": state.status != "COMPLETED",
        "updated_at": state.updated_at,
    }
    dashboard["pipeline_metadata"]["durable_execution"] = {
        "queue_db": str(NEWS_QUEUE_DB),
        "execution_model": "durable_worker",
        "lease_seconds": NEWS_QUEUE_LEASE_SECONDS,
        "max_attempts": NEWS_QUEUE_MAX_ATTEMPTS,
        "http_request_independent": True,
    }

    # Starlette's JSONResponse uses allow_nan=False semantics. Sanitize all
    # dashboard telemetry before both disk persistence and the API return so a
    # legitimate "unbounded" execution budget can never become JSON Infinity.
    dashboard = StatePersistenceManager._json_safe(dashboard)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    safe_job = re.sub(r"[^A-Za-z0-9_-]", "_", str(state.intelligence_id))[-40:]
    output_path = DASHBOARDS_DIR / f"atis_dashboard_{timestamp}_{safe_job}.json"
    try:
        output_path.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        dashboard["pipeline_metadata"]["dashboard_path"] = str(output_path)
        logger.info("[FINAL] Dashboard persisted: %s", output_path.resolve())
    except Exception as exc:
        logger.error("[FINAL] Failed to persist dashboard: %s", exc)

    logger.info("\n%s", reasoning_log.log_tree())
    logger.info("[EXECUTION] elapsed=%.2fs | llm_calls=%d | timeouts=%d | truncation_retries=%d", time.monotonic() - started_at, engine.calls, engine.timeouts, engine.truncated_retries)
    logger.info("=" * 78)
    logger.info("ATIS NEWS PIPELINE COMPLETE")
    logger.info("=" * 78)
    return dashboard


def process_article_pipeline(article_path: str, perspective: Any | None = None, job_id: str | None = None) -> Dict[str, Any]:
    article_file = Path(article_path)
    if not article_file.exists():
        raise FileNotFoundError(f"Article not found: {article_path}")
    article_text = article_file.read_text(encoding="utf-8")
    if os.getenv("ATIS_NEWS_EXECUTION_MODE", "sync").strip().lower() == "async":
        receipt = submit_news_job(article_text, perspective, "file_upload", job_id)
        return {
            "status": receipt["status"].lower(),
            "job_id": receipt["job_id"],
            "resume_available": True,
            "execution_model": "durable_worker",
        }
    return _run_perspective_first_news(article_text, perspective, source_label="file_upload", job_id=job_id)


def run_news_pipeline(article_text: str, perspective: Any | None = None, job_id: str | None = None) -> Dict[str, Any]:
    """Backward-compatible synchronous API with optional durable submission mode.

    Set ATIS_NEWS_EXECUTION_MODE=async to submit and return a durable job receipt.
    The default remains synchronous so existing callers are not silently broken.
    """
    if os.getenv("ATIS_NEWS_EXECUTION_MODE", "sync").strip().lower() == "async":
        receipt = submit_news_job(article_text, perspective, "web_upload", job_id)
        return {
            "status": receipt["status"].lower(),
            "job_id": receipt["job_id"],
            "resume_available": True,
            "execution_model": "durable_worker",
        }
    return _run_perspective_first_news(article_text, perspective, source_label="web_upload", job_id=job_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="ATIS News — Perspective-First, Graph-Grounded Intelligence Engine")
    subparsers = parser.add_subparsers(dest="command")

    article_parser = subparsers.add_parser("process", help="Process an article synchronously (backward-compatible mode).")
    article_parser.add_argument("article_path", metavar="ARTICLE", help="Path to the plain-text news article to process.")

    worker_parser = subparsers.add_parser("worker", help="Run the durable News worker.")
    worker_parser.add_argument("--worker-id", default=None)

    status_parser = subparsers.add_parser("status", help="Get durable News job status.")
    status_parser.add_argument("job_id")

    args = parser.parse_args()
    try:
        if args.command == "worker":
            run_news_worker_forever(args.worker_id)
            return
        if args.command == "status":
            print(json.dumps(get_news_job_status(args.job_id), indent=2, ensure_ascii=False))
            return
        article_path = args.article_path if args.command == "process" else getattr(args, "article_path", None)
        if not article_path:
            parser.print_help()
            return
        print(json.dumps(process_article_pipeline(article_path), indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.critical("Pipeline terminated with fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
