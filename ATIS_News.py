#!/usr/bin/env python3
"""
ATIS News Intelligence Engine - Production-Grade Rebuild
======================================================

Architecture: Perspective-First, Graph-Grounded Intelligence Pipeline

Core Principles:
  1. PERSPECTIVE-FIRST: The selected country is the analytical lens, not a filter
  2. EVIDENCE-BASED: All inferences must be traceable to vault evidence or article facts
  3. DETERMINISTIC: Same inputs produce same outputs (temperature=0.0, sorted iterations)
  4. RESILIENT: Graceful degradation, checkpointing, resumable execution
  5. SAFE: Never truncate source material, never invent facts

Pipeline Stages:
  0. Input Validation & Normalization
  1. Article Understanding (LLM)
  2. Perspective Ecosystem Loading (DB)
  3. Perspective Impact Mapping (LLM)
  4. Target Resolution (DB)
  5. Graph Traversal (DB)
  6. Impact Analysis (Deterministic)
  7. Final Synthesis (LLM)
  8. Validation & Grounding
  9. Output Assembly & Persistence

Public API (backward compatible):
  - process_article_pipeline(article_path, perspective=None, job_id=None) -> Dict
  - run_news_pipeline(article_text, perspective=None, job_id=None) -> Dict
  - submit_news_job(article_text, perspective=None, source_label=None, job_id=None) -> Dict
  - get_news_job_status(job_id) -> Dict
  - run_news_worker_once(worker_id=None) -> Optional[Dict]
  - run_news_worker_forever(worker_id=None) -> None

Author: Vibe Code (Mistral AI)
Version: 3.0.0-production-rebuild
Date: 2025
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Imports from ATIS modules
# ---------------------------------------------------------------------------

from llm_client import (
    LLMClient,
    get_client,
    LLMTokenLimitError,
    LLMProviderError,
    LLMConfigError,
    ModelCapabilities,
)

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NewsConfig:
    """
    Central configuration for ATIS News pipeline.
    """
    llm_provider: str = "mistral"
    llm_model: str = "labs-leanstral-1-5"
    llm_fallback_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_timeout: float = 45.0
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 30.0
    safety_margin_tokens: int = 2000
    system_prompt_overhead: int = 500
    max_output_tokens: int = 4096
    stage_output_tokens: int = 3072
    final_output_tokens: int = 3072
    max_article_chars: int = 500000
    article_batch_chars: int = 12000
    max_graph_depth: int = 2
    max_graph_nodes: int = 0
    max_graph_paths: int = 0
    max_retrieval_targets: int = 0
    max_perspective_ecosystem_nodes: int = 0
    max_ecosystem_batch_nodes: int = 40
    max_evidence_batch_records: int = 25
    vault_dir: Path = field(default_factory=lambda: Path("./vault"))
    dashboards_dir: Path = field(default_factory=lambda: Path("./dashboards"))
    job_store_dir: Path = field(default_factory=lambda: Path("./job_store"))
    queue_db_path: Path = field(default_factory=lambda: Path("./job_store/news_jobs.sqlite3"))
    queue_lease_seconds: float = 180.0
    queue_poll_seconds: float = 1.0
    queue_max_attempts: int = 5
    temperature: float = 0.0
    seed: Optional[int] = None
    min_node_resolution_score: float = 0.72
    min_opportunity_graph_score: float = 0.55
    enable_checkpointing: bool = True
    enable_caching: bool = True
    enable_telemetry: bool = True

    @classmethod
    def from_env(cls) -> "NewsConfig":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "mistral"),
            llm_model=os.getenv("LLM_MODEL", "labs-leanstral-1-5"),
            llm_fallback_model=os.getenv("LLM_FALLBACK_MODEL"),
            llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("MISTRAL_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_timeout=float(os.getenv("ATIS_NEWS_LLM_TIMEOUT", "45")),
            llm_max_retries=int(os.getenv("ATIS_NEWS_LLM_MAX_RETRIES", "3")),
            llm_retry_base_delay=float(os.getenv("ATIS_NEWS_RETRY_BASE_DELAY", "1.0")),
            llm_retry_max_delay=float(os.getenv("ATIS_NEWS_RETRY_MAX_DELAY", "30.0")),
            safety_margin_tokens=int(os.getenv("ATIS_NEWS_SAFETY_MARGIN", "2000")),
            system_prompt_overhead=int(os.getenv("ATIS_NEWS_SYSTEM_OVERHEAD", "500")),
            max_output_tokens=int(os.getenv("ATIS_NEWS_MAX_OUTPUT_TOKENS", "4096")),
            stage_output_tokens=int(os.getenv("ATIS_NEWS_STAGE_OUTPUT_TOKENS", "3072")),
            final_output_tokens=int(os.getenv("ATIS_NEWS_FINAL_OUTPUT_TOKENS", "3072")),
            max_article_chars=int(os.getenv("ATIS_NEWS_MAX_ARTICLE_CHARS", "500000")),
            article_batch_chars=int(os.getenv("ATIS_NEWS_ARTICLE_BATCH_CHARS", "12000")),
            max_graph_depth=int(os.getenv("ATIS_NEWS_MAX_GRAPH_DEPTH", "2")),
            max_graph_nodes=int(os.getenv("ATIS_NEWS_MAX_GRAPH_NODES", "0")),
            max_graph_paths=int(os.getenv("ATIS_NEWS_MAX_GRAPH_PATHS", "0")),
            max_retrieval_targets=int(os.getenv("ATIS_NEWS_MAX_RETRIEVAL_TARGETS", "0")),
            max_perspective_ecosystem_nodes=int(os.getenv("ATIS_NEWS_MAX_PERSPECTIVE_NODES", "0")),
            max_ecosystem_batch_nodes=int(os.getenv("ATIS_NEWS_MAX_ECOSYSTEM_BATCH", "40")),
            max_evidence_batch_records=int(os.getenv("ATIS_NEWS_MAX_EVIDENCE_BATCH", "25")),
            vault_dir=Path(os.getenv("ATIS_VAULT_DIR", "./vault")),
            dashboards_dir=Path(os.getenv("ATIS_DASHBOARDS_DIR", "./dashboards")),
            job_store_dir=Path(os.getenv("ATIS_NEWS_JOB_STORE", "./job_store")),
            queue_db_path=Path(os.getenv("ATIS_NEWS_QUEUE_DB", os.getenv("ATIS_NEWS_JOB_STORE", "./job_store") + "/news_jobs.sqlite3")),
            queue_lease_seconds=float(os.getenv("ATIS_NEWS_QUEUE_LEASE_SECONDS", "180")),
            queue_poll_seconds=float(os.getenv("ATIS_NEWS_QUEUE_POLL_SECONDS", "1.0")),
            queue_max_attempts=int(os.getenv("ATIS_NEWS_QUEUE_MAX_ATTEMPTS", "5")),
            temperature=float(os.getenv("ATIS_NEWS_TEMPERATURE", "0.0")),
            seed=int(os.getenv("ATIS_NEWS_SEED")) if os.getenv("ATIS_NEWS_SEED") else None,
            min_node_resolution_score=float(os.getenv("ATIS_NEWS_MIN_NODE_RESOLUTION_SCORE", "0.72")),
            min_opportunity_graph_score=float(os.getenv("ATIS_NEWS_MIN_OPPORTUNITY_GRAPH_SCORE", "0.55")),
        )


_config: Optional[NewsConfig] = None


def get_config() -> NewsConfig:
    global _config
    if _config is None:
        _config = NewsConfig.from_env()
    return _config


def set_config(config: NewsConfig) -> None:
    global _config
    _config = config


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("atis_news")


class ContextLogger:
    def __init__(self, job_id: str, stage: str):
        self.job_id = job_id
        self.stage = stage
        self._logger = logging.getLogger("atis_news")

    def _format(self, level: str, message: str) -> str:
        return f"[{self.job_id}] [{self.stage}] {message}"

    def debug(self, message: str, **kwargs) -> None:
        self._logger.debug(self._format("DEBUG", message), extra={**kwargs, "job_id": self.job_id, "stage": self.stage})

    def info(self, message: str, **kwargs) -> None:
        self._logger.info(self._format("INFO", message), extra={**kwargs, "job_id": self.job_id, "stage": self.stage})

    def warning(self, message: str, **kwargs) -> None:
        self._logger.warning(self._format("WARNING", message), extra={**kwargs, "job_id": self.job_id, "stage": self.stage})

    def error(self, message: str, **kwargs) -> None:
        self._logger.error(self._format("ERROR", message), extra={**kwargs, "job_id": self.job_id, "stage": self.stage})

    def exception(self, message: str, **kwargs) -> None:
        self._logger.exception(self._format("EXCEPTION", message), extra={**kwargs, "job_id": self.job_id, "stage": self.stage})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NewsPipelineError(Exception):
    def __init__(self, message: str, stage: str = "", job_id: str = "", details: Optional[Dict[str, Any]] = None):
        self.stage = stage
        self.job_id = job_id
        self.details = details or {}
        super().__init__(f"[{stage}] {message}" if stage else message)


class NewsPipelineContextOverflow(NewsPipelineError):
    pass


class NewsPipelineDeadline(NewsPipelineError):
    pass


class NewsValidationError(NewsPipelineError):
    pass


class NewsConfigError(NewsPipelineError):
    pass


# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    INPUT_VALIDATION = "INPUT_VALIDATION"
    ARTICLE_UNDERSTANDING = "ARTICLE_UNDERSTANDING"
    PERSPECTIVE_ECOSYSTEM_LOADING = "PERSPECTIVE_ECOSYSTEM_LOADING"
    PERSPECTIVE_IMPACT_MAPPING = "PERSPECTIVE_IMPACT_MAPPING"
    TARGET_RESOLUTION = "TARGET_RESOLUTION"
    GRAPH_TRAVERSAL = "GRAPH_TRAVERSAL"
    IMPACT_ANALYSIS = "IMPACT_ANALYSIS"
    FINAL_SYNTHESIS = "FINAL_SYNTHESIS"
    VALIDATION_GROUNDING = "VALIDATION_GROUNDING"
    OUTPUT_ASSEMBLY = "OUTPUT_ASSEMBLY"
    COMPLETE = "COMPLETE"


class EvidenceCategory(str, Enum):
    DIRECT = "direct"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"
    PERSPECTIVE = "perspective"
    BRIDGE = "bridge"
    GLOBAL = "global"


class OpportunityType(str, Enum):
    EXPLICIT = "explicit"
    DERIVED = "derived"
    POTENTIAL = "potential"


class NodeStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    RESEARCH_REQUIRED = "RESEARCH_REQUIRED"
    EXCLUDED = "EXCLUDED"


class ImpactPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class JobState:
    intelligence_id: str
    status: str = "IN_PROGRESS"
    current_stage: str = PipelineStage.INPUT_VALIDATION.value
    completed_stages: List[str] = field(default_factory=list)
    stage_data: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    perspective: Dict[str, str] = field(default_factory=dict)
    pipeline_version: str = "3.0.0"

    def mark_stage_complete(self, stage: PipelineStage, data: Any) -> None:
        self.stage_data[stage.value] = data
        if stage.value not in self.completed_stages:
            self.completed_stages.append(stage.value)
        self.updated_at = time.time()

    def is_completed(self, stage: PipelineStage) -> bool:
        return stage.value in self.completed_stages and stage.value in self.stage_data

    def get_stage_data(self, stage: PipelineStage) -> Any:
        return self.stage_data.get(stage.value)

    def add_error(self, error: str) -> None:
        self.error_log.append(error)
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ReasoningLog:
    article_chars: int = 0
    entities_extracted: int = 0
    perspective_nodes: int = 0
    candidate_nodes: int = 0
    impact_domains: int = 0
    retrieval_targets: int = 0
    unresolved_targets: int = 0
    direct_graph_nodes: int = 0
    first_order_graph_nodes: int = 0
    second_order_graph_nodes: int = 0
    backlink_candidates: int = 0
    graph_paths: int = 0
    relevant_nodes: int = 0
    selected_evidence: int = 0
    estimated_tokens: int = 0
    safe_budget: int = 0
    reasoning_mode: str = ""
    evidence_calls: int = 0
    synthesis_calls: int = 0
    final_call: int = 0
    total_llm_calls: int = 0
    retry_calls: int = 0
    gaps: int = 0
    opportunities: int = 0
    validated_opportunities: int = 0
    research_required: int = 0
    partitions: int = 0
    deduplicated_nodes: int = 0
    weak_backlinks_filtered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def log_tree(self) -> str:
        return (
            f"NEWS REASONING\n"
            f"├── article chars: {self.article_chars:,}\n"
            f"├── entities extracted: {self.entities_extracted}\n"
            f"├── perspective nodes: {self.perspective_nodes}\n"
            f"├── candidate nodes: {self.candidate_nodes}\n"
            f"├── impact domains: {self.impact_domains}\n"
            f"├── retrieval targets: {self.retrieval_targets}\n"
            f"├── unresolved targets: {self.unresolved_targets}\n"
            f"├── direct graph nodes: {self.direct_graph_nodes}\n"
            f"├── first-order graph nodes: {self.first_order_graph_nodes}\n"
            f"├── second-order graph nodes: {self.second_order_graph_nodes}\n"
            f"├── backlink candidates: {self.backlink_candidates}\n"
            f"├── graph paths: {self.graph_paths}\n"
            f"├── selected evidence: {self.selected_evidence}\n"
            f"├── estimated tokens: {self.estimated_tokens:,}\n"
            f"├── safe budget: {self.safe_budget:,}\n"
            f"├── reasoning mode: {self.reasoning_mode}\n"
            f"├── evidence calls: {self.evidence_calls}\n"
            f"├── synthesis calls: {self.synthesis_calls}\n"
            f"├── final call: {self.final_call}\n"
            f"├── total LLM calls: {self.total_llm_calls}\n"
            f"├── partitions: {self.partitions}\n"
            f"├── deduplicated nodes: {self.deduplicated_nodes}\n"
            f"├── weak backlinks filtered: {self.weak_backlinks_filtered}\n"
            f"├── gaps: {self.gaps}\n"
            f"├── opportunities: {self.opportunities}\n"
            f"├── validated opportunities: {self.validated_opportunities}\n"
            f"└── research required: {self.research_required}"
        )


@dataclass
class TokenBudget:
    provider_context_limit: int
    max_output_tokens: int
    safety_margin: int = 2000
    system_overhead: int = 500

    @property
    def usable_input_budget(self) -> int:
        return self.provider_context_limit - self.max_output_tokens - self.safety_margin

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def estimate_message_tokens(messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += TokenBudget.estimate_tokens(content)
        total += len(messages) * 4
        return total

    def validate(self, input_tokens: int, output_tokens: int, stage: str) -> None:
        total = input_tokens + output_tokens + self.safety_margin
        if total > self.provider_context_limit:
            raise LLMTokenLimitError(
                f"[{stage}] Estimated total tokens ({total:,}) exceeds provider "
                f"context limit ({self.provider_context_limit:,}). "
                f"Input: ~{input_tokens:,}, Output: {output_tokens:,}, "
                f"Safety margin: {self.safety_margin:,}."
            )
        if output_tokens > self.max_output_tokens:
            raise LLMTokenLimitError(
                f"[{stage}] Requested output tokens ({output_tokens:,}) exceeds "
                f"provider maximum ({self.max_output_tokens:,})."
            )

    def compute_safe_output(self, input_tokens: int) -> int:
        available = self.provider_context_limit - input_tokens - self.safety_margin
        return min(available, self.max_output_tokens)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _pf_norm(value: Any) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _pf_tokens(value: Any) -> Set[str]:
    return {x for x in re.findall(r"[a-z0-9]{3,}", _pf_norm(value))}


def _pf_similarity(a: Any, b: Any) -> float:
    aa, bb = _pf_tokens(a), _pf_tokens(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def _pf_country_matches(value: str, country: str) -> bool:
    a = _pf_norm(value)
    b = _pf_norm(country)
    if not b:
        return False
    if a == b:
        return True
    code = COUNTRY_CODES.get(b, b)
    return _pf_norm(code) == a


def _pf_safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


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


def _pf_text(value: Any, limit: int = 1200) -> str:
    return str(value or "").strip()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_str(value: Any) -> str:
    return str(value or "")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


# =============================================================================
# PART 2: VAULT LAYER
# =============================================================================


class ObsidianVaultManager:
    """
    Unified vault manager for ATIS News pipeline.
    
    Responsibilities:
      - Index all markdown files in the vault
      - Build bidirectional link maps (outbound links + backlinks)
      - Extract metadata (country, type, sector, summary)
      - Canonical name matching for fuzzy lookups
      - Cross-border bridge detection
      - Perspective-side node retrieval
    """

    def __init__(self, vault_dir: Optional[Path] = None) -> None:
        config = get_config()
        self.vault_dir: Path = vault_dir or config.vault_dir
        self._ensure_directories()
        
        self.file_map: Dict[str, str] = {}
        self.backlink_map: Dict[str, Set[str]] = {}
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        self.node_content: Dict[str, str] = {}
        self.outbound_links: Dict[str, List[str]] = {}
        
        self._index_vault()
        logger.info("Vault index built: %d nodes, %d backlink targets",
                    len(self.file_map), len(self.backlink_map))

    def _ensure_directories(self) -> None:
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        config = get_config()
        config.dashboards_dir.mkdir(parents=True, exist_ok=True)
        config.job_store_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _canonicalize(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "", str(name or "")).lower()

    def _index_vault(self) -> None:
        self.file_map.clear()
        self.backlink_map.clear()
        self.node_metadata.clear()
        self.node_content.clear()
        self.outbound_links.clear()

        md_files = sorted(self.vault_dir.rglob("*.md"), key=lambda p: str(p))
        logger.debug("Indexing %d vault files...", len(md_files))

        for file_path in md_files:
            actual_stem = file_path.stem
            canonical_stem = self._canonicalize(actual_stem)
            self.file_map[canonical_stem] = actual_stem

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to read %s: %s", actual_stem, exc)
                continue

            country = ""
            node_type = ""
            sector = ""
            summary = ""
            
            fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
            if fm_match:
                try:
                    import yaml
                    front = yaml.safe_load(fm_match.group(1)) or {}
                    country = front.get("country", "") or front.get("location", "")
                    node_type = (front.get("node_type", "") or 
                                front.get("type", "") or 
                                front.get("entity_type", ""))
                    sector = front.get("sector", "") or front.get("industry", "")
                except Exception:
                    pass

            if not country:
                path_parts = [p.lower() for p in file_path.relative_to(self.vault_dir).parts]
                for part in path_parts:
                    if part in COUNTRY_CODES:
                        country = part.title()
                        break

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

            for link in cleaned_links:
                canon_target = self._canonicalize(link)
                if canon_target not in self.backlink_map:
                    self.backlink_map[canon_target] = set()
                self.backlink_map[canon_target].add(actual_stem)

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

    def get_perspective_nodes(self, perspective: PerspectiveContext) -> List[Dict[str, Any]]:
        country_norm = perspective.country.lower()
        nodes = []
        
        for canon, meta in sorted(self.node_metadata.items()):
            node_country = meta.get("country", "").lower()
            if node_country == country_norm:
                nodes.append({
                    "node_id": self.file_map.get(canon, canon),
                    "canonical_id": canon,
                    "country": meta.get("country", ""),
                    "type": meta.get("type", ""),
                    "sector": meta.get("sector", ""),
                    "summary": meta.get("summary", ""),
                    "path": meta.get("path", ""),
                })
        
        return sorted(nodes, key=lambda n: (n["sector"], n["type"], n["node_id"]))

    def get_cross_border_bridges(
        self, 
        perspective: PerspectiveContext, 
        source_country: str
    ) -> List[Dict[str, Any]]:
        perspective_norm = perspective.country.lower()
        source_norm = source_country.lower()
        
        if perspective_norm == source_norm:
            return []
        
        # Use a canonical set to track unique bridges
        # Key: (from_node_canon, to_node_canon) - relationship type doesn't matter for uniqueness
        seen: Set[Tuple[str, str]] = set()
        bridges: List[Dict[str, Any]] = []
        
        for canon, meta in sorted(self.node_metadata.items()):
            node_country = meta.get("country", "").lower()
            if node_country not in (perspective_norm, source_norm):
                continue
            
            actual = self.file_map.get(canon, canon)
            actual_canon = self._canonicalize(actual)
            
            # Check outbound links
            for link_target in self.outbound_links.get(canon, []):
                link_canon = self._canonicalize(link_target)
                link_meta = self.node_metadata.get(link_canon, {})
                link_country = link_meta.get("country", "").lower()
                
                if node_country == perspective_norm and link_country == source_norm:
                    from_node = actual
                    to_node = self.file_map.get(link_canon, link_target)
                    from_canon = self._canonicalize(from_node)
                    to_canon = self._canonicalize(to_node)
                    key = (from_canon, to_canon)
                    if key not in seen:
                        seen.add(key)
                        bridges.append({
                            "from_node": from_node,
                            "from_country": perspective.country,
                            "to_node": to_node,
                            "to_country": source_country,
                            "relationship_type": "outbound_link",
                        })
                elif node_country == source_norm and link_country == perspective_norm:
                    from_node = actual
                    to_node = self.file_map.get(link_canon, link_target)
                    from_canon = self._canonicalize(from_node)
                    to_canon = self._canonicalize(to_node)
                    key = (from_canon, to_canon)
                    if key not in seen:
                        seen.add(key)
                        bridges.append({
                            "from_node": from_node,
                            "from_country": source_country,
                            "to_node": to_node,
                            "to_country": perspective.country,
                            "relationship_type": "outbound_link",
                        })
            
            # Check backlinks
            for inbound in self.backlink_map.get(canon, set()):
                inbound_canon = self._canonicalize(inbound)
                inbound_meta = self.node_metadata.get(inbound_canon, {})
                inbound_country = inbound_meta.get("country", "").lower()
                
                if node_country == perspective_norm and inbound_country == source_norm:
                    from_node = inbound
                    to_node = actual
                    from_canon = self._canonicalize(from_node)
                    to_canon = actual_canon
                    key = (from_canon, to_canon)
                    if key not in seen:
                        seen.add(key)
                        bridges.append({
                            "from_node": from_node,
                            "from_country": source_country,
                            "to_node": to_node,
                            "to_country": perspective.country,
                            "relationship_type": "backlink",
                        })
                elif node_country == source_norm and inbound_country == perspective_norm:
                    from_node = inbound
                    to_node = actual
                    from_canon = self._canonicalize(from_node)
                    to_canon = actual_canon
                    key = (from_canon, to_canon)
                    if key not in seen:
                        seen.add(key)
                        bridges.append({
                            "from_node": from_node,
                            "from_country": perspective.country,
                            "to_node": to_node,
                            "to_country": source_country,
                            "relationship_type": "backlink",
                        })
        
        return sorted(bridges, key=lambda b: (b.get("from_node", ""), b.get("to_node", "")))

    def build_cross_border_bridge_context(
        self,
        perspective: PerspectiveContext,
        source_country: str,
        max_bridges: int = 15
    ) -> str:
        bridges = self.get_cross_border_bridges(perspective, source_country)
        
        if not bridges:
            return (
                "=== CROSS-BORDER BRIDGE CONTEXT ===\n"
                f"No evidenced cross-border relationships found between "
                f"{perspective.country} and {source_country}."
            )
        
        lines = ["=== CROSS-BORDER BRIDGE CONTEXT ==="]
        for bridge in bridges[:max_bridges]:
            lines.append(
                f"- {bridge['from_node']} ({bridge['from_country']}) "
                f"\u2192 {bridge['to_node']} ({bridge['to_country']}) "
                f"via {bridge['relationship_type']}"
            )
        return "\n".join(lines)

    def build_graph_context(
        self,
        entities: List[Dict[str, Any]],
        perspective: PerspectiveContext,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Build graph context for entities with cross-border bridge discovery.
        
        Backward compatible method for existing tests.
        
        Returns:
            Tuple of (graph_context_string, perspective_nodes, cross_border_bridges)
        """
        # Get perspective nodes
        perspective_nodes = self.get_perspective_nodes(perspective)
        
        # Extract source country from entities
        source_countries: Set[str] = set()
        for entity in entities:
            context = entity.get("context", "")
            for country in COUNTRY_CODES.values():
                if _pf_country_matches(context, country):
                    source_countries.add(country)
        
        # Get cross-border bridges
        cross_border_bridges: List[Dict[str, Any]] = []
        if source_countries:
            for source_country in source_countries:
                bridges = self.get_cross_border_bridges(perspective, source_country)
                cross_border_bridges.extend(bridges)
        
        # Build graph context string
        lines = ["=== GRAPH CONTEXT ==="]
        
        # Add perspective nodes
        lines.append(f"\nPERSPECTIVE NODES ({perspective.country}):")
        for node in perspective_nodes:
            lines.append(f"  - {node.get('node_id')} ({node.get('sector')})")
        
        # Add cross-border bridges
        if cross_border_bridges:
            lines.append(f"\nCROSS-BORDER BRIDGES:")
            for bridge in cross_border_bridges:
                lines.append(
                    f"  - {bridge.get('from_node')} ({bridge.get('from_country')}) "
                    f"\u2192 {bridge.get('to_node')} ({bridge.get('to_country')}) "
                    f"via {bridge.get('relationship_type')}"
                )
        
        graph_context = "\n".join(lines)
        
        return graph_context, perspective_nodes, cross_border_bridges

    def resolve_node(self, hint: str) -> Optional[str]:
        target = _pf_norm(hint)
        if not target:
            return None
        
        if target in self.file_map:
            return self.file_map[target]
        
        candidates: List[Tuple[float, str]] = []
        for canon, meta in self.node_metadata.items():
            node_id = self.file_map.get(canon, canon)
            score = max(
                _pf_similarity(hint, node_id),
                _pf_similarity(hint, meta.get("summary", ""))
            )
            if score >= get_config().min_node_resolution_score:
                candidates.append((score, canon))
        
        candidates.sort(key=lambda x: (-x[0], x[1]))
        if candidates and (len(candidates) == 1 or 
                          candidates[0][0] - candidates[1][0] >= 0.12):
            return self.file_map.get(candidates[0][1], candidates[0][1])
        
        return None

    def get_node_record(self, canon: str) -> Dict[str, Any]:
        meta = self.node_metadata.get(canon, {})
        return {
            "node_id": self.file_map.get(canon, canon),
            "canonical_id": canon,
            "country": meta.get("country", ""),
            "type": meta.get("type", ""),
            "sector": meta.get("sector", ""),
            "summary": meta.get("summary", ""),
            "path": meta.get("path", ""),
            "content": self.node_content.get(canon, ""),
        }


_VAULT_CACHE: Dict[str, Tuple[tuple, ObsidianVaultManager]] = {}


def _vault_signature(vault_dir: Path) -> tuple:
    rows: List[Tuple[str, int, int]] = []
    try:
        for path in sorted(vault_dir.rglob("*.md"), key=lambda p: str(p)):
            try:
                stat = path.stat()
                rows.append((
                    str(path.relative_to(vault_dir)),
                    int(stat.st_mtime_ns),
                    int(stat.st_size)
                ))
            except OSError:
                continue
    except OSError:
        return tuple()
    return tuple(rows)


def get_vault(vault_dir: Optional[Path] = None) -> ObsidianVaultManager:
    config = get_config()
    target_dir = vault_dir or config.vault_dir
    key = str(target_dir.resolve())
    
    if not config.enable_caching:
        return ObsidianVaultManager(target_dir)
    
    signature = _vault_signature(target_dir)
    cached = _VAULT_CACHE.get(key)
    
    if cached and cached[0] == signature:
        logger.debug("[VAULT] Reusing cached index: %d nodes", len(cached[1].file_map))
        return cached[1]
    
    vault = ObsidianVaultManager(target_dir)
    _VAULT_CACHE[key] = (signature, vault)
    return vault


# =============================================================================
# PART 3: PERSISTENCE LAYER
# =============================================================================


class StatePersistenceManager:
    """
    Durable, atomic checkpoint storage for resumable News jobs.
    
    Features:
      - Atomic writes (temp file → fsync → rename)
      - JSON serialization with sanitization
      - Automatic directory creation
      - Safe filename handling
    """

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        config = get_config()
        configured = storage_dir or config.job_store_dir
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
        
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=str(self.storage_dir)
        )
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
            return JobState.from_dict(raw)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("[STATE] Ignoring unreadable checkpoint %s: %s", path, exc)
            return None


class DurableNewsJobQueue:
    """
    Cross-process durable queue for News jobs.
    
    SQLite-backed with atomic leases.
    
    Features:
      - Atomic claim/lease via BEGIN IMMEDIATE transactions
      - WAL mode for concurrent access
      - Exponential backoff on failures
      - Max attempts per job
      - Immutable payloads (article + perspective cannot change)
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        config = get_config()
        self.db_path = db_path or config.queue_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> Any:
        import sqlite3
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        import sqlite3
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
                "pipeline_version": "3.0.0",
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
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        article_text = article_text.strip()
        if not article_text:
            raise ValueError("News article text is empty")
        
        config = get_config()
        if len(article_text) > config.max_article_chars:
            raise ValueError(
                f"Article contains {len(article_text):,} characters, exceeding "
                f"{config.max_article_chars:,}. ATIS refuses to truncate the source article."
            )
        
        job_id = job_id or self.make_job_id(article_text, perspective)
        now = time.time()
        payload = json.dumps(perspective.as_dict(), sort_keys=True, ensure_ascii=False)
        
        import sqlite3
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
        
        if row["article_text"] != article_text or row["perspective_json"] != payload:
            raise ValueError(
                f"News job ID {job_id} already belongs to different immutable input"
            )
        return self._row_to_status(row)

    def claim(self, worker_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        import sqlite3
        worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        now = time.time()
        config = get_config()
        lease_until = now + config.queue_lease_seconds
        
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
                (config.queue_max_attempts, now, now),
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
        import sqlite3
        now = time.time()
        config = get_config()
        
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE news_jobs
                SET lease_until=?, updated_at=?
                WHERE job_id=? AND status='RUNNING' AND worker_id=?
                """,
                (now + config.queue_lease_seconds, now, job_id, worker_id),
            )
        return cur.rowcount == 1

    def complete(self, job_id: str, worker_id: str) -> None:
        import sqlite3
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
        import sqlite3
        now = time.time()
        config = get_config()
        
        with self._connect() as conn:
            row = conn.execute(
                "SELECT attempts FROM news_jobs WHERE job_id=? AND worker_id=?",
                (job_id, worker_id),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"News job {job_id} is not owned by worker {worker_id}")
            
            attempts = int(row["attempts"])
            terminal = attempts >= config.queue_max_attempts
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
        import sqlite3
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM news_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return self._row_to_status(row) if row else None

    @staticmethod
    def _row_to_status(row: Any) -> Dict[str, Any]:
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
    def _row_to_job(cls, row: Any) -> Dict[str, Any]:
        result = cls._row_to_status(row)
        result.update({
            "article_text": row["article_text"],
            "perspective": json.loads(row["perspective_json"]),
            "source_label": row["source_label"],
        })
        return result


# =============================================================================
# PART 4: LLM LAYER
# =============================================================================


class LLMCaller:
    """
    Wrapper around LLMClient with pipeline-specific functionality.
    
    Provides:
      - Token budget management
      - Transport timeout protection
      - JSON parsing with fallbacks
      - Retry logic
      - Call counting and metrics
    """

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client = client or get_client()
        self.config = self.client.config
        self.capabilities = self.client.adapter.capabilities
        self.budget = TokenBudget(
            provider_context_limit=self.capabilities.max_context_tokens,
            max_output_tokens=self.capabilities.max_output_tokens,
        )
        self.calls = 0
        self.timeouts = 0
        self.truncated_retries = 0
        self._stage_durations: Dict[str, float] = {}

    def _stage_timeout(self, stage: str) -> float:
        config = get_config()
        stage_lower = stage.lower()
        if "article" in stage_lower:
            return config.llm_timeout
        if "impact" in stage_lower:
            return config.llm_timeout
        if "final" in stage_lower or "synthesis" in stage_lower:
            return config.llm_timeout
        return config.llm_timeout

    def _call_provider(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        stage: str,
    ) -> str:
        """
        Execute provider call with transport timeout protection.
        
        Uses threading to enforce timeout without affecting the actual
        intelligence execution lifetime.
        """
        config = get_config()
        timeout = max(0.1, self._stage_timeout(stage))
        
        import threading
        result: Dict[str, Any] = {}
        error: Dict[str, BaseException] = {}
        
        def worker() -> None:
            try:
                result["value"] = self.client.chat(
                    messages,
                    temperature=config.temperature,
                    max_tokens=max_tokens,
                )
            except BaseException as exc:
                error["value"] = exc
        
        started = time.monotonic()
        thread = threading.Thread(
            target=worker,
            name=f"atis-news-{stage[:24]}",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=timeout)
        elapsed = time.monotonic() - started
        self._stage_durations[f"{stage}#transport"] = round(elapsed, 3)
        
        if thread.is_alive():
            self.timeouts += 1
            raise NewsPipelineDeadline(
                f"[{stage}] provider transport timeout after {timeout:.1f}s",
                stage=stage,
            )
        
        if "value" in error:
            raise error["value"]
        
        if "value" not in result:
            raise RuntimeError(f"[{stage}] provider call returned no result")
        
        return str(result["value"])

    def _is_truncated(self, raw: str) -> bool:
        """Check if LLM response is truncated."""
        text = raw.strip()
        if not text:
            return False
        if text.endswith("..."):
            return True
        
        last_char = text[-1]
        if last_char not in {"}", "]", "\"", ">", "'"}:
            if not (last_char.isdigit() or last_char.lower() in {"e", "l"}):
                return True
        
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

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        requested_output: int,
        stage: str,
    ) -> Dict[str, Any]:
        """
        Execute LLM call and parse JSON response.
        
        Features:
          - Pre-flight token validation
          - Transport timeout protection
          - JSON parsing with layered fallbacks
          - Automatic retry on truncated/malformed JSON
        """
        config = get_config()
        cap = max(512, self.budget.max_output_tokens)
        requested = min(max(512, int(requested_output)), cap)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_tokens = TokenBudget.estimate_message_tokens(messages)
        
        if not self.budget.fits_in_budget(input_tokens, requested):
            safe_output = self.budget.compute_safe_output(input_tokens)
            if safe_output < 512:
                raise LLMTokenLimitError(
                    f"[{stage}] complete input cannot fit provider context without truncation: "
                    f"input~{input_tokens}, context={self.budget.provider_context_limit}"
                )
            requested = min(requested, safe_output)
        
        logger.info(
            "[LLM] %s | input~%d | output<=%d | budget=%d",
            stage, input_tokens, requested, self.budget.provider_context_limit
        )
        
        self.calls += 1
        
        raw = self._call_provider(messages, requested, stage)
        
        try:
            parsed = self._safe_json_loads(raw, stage_name=stage)
            if not isinstance(parsed, dict):
                raise NewsValidationError(
                    f"{stage} returned JSON that is not an object",
                    stage=stage,
                )
            return parsed
        except (NewsValidationError, RuntimeError) as first_error:
            retry_output = min(cap, max(requested + 1024, requested * 2))
            if retry_output <= requested:
                raise first_error
            if not self.budget.fits_in_budget(input_tokens, retry_output):
                retry_output = self.budget.compute_safe_output(input_tokens)
            if retry_output <= requested or retry_output < 512:
                raise first_error
            
            self.truncated_retries += 1
            self.calls += 1
            logger.warning(
                "[LLM] %s returned unusable JSON; retrying SAME complete input with output<=%d",
                stage, retry_output
            )
            
            raw_retry = self._call_provider(messages, retry_output, f"{stage} retry")
            parsed = self._safe_json_loads(raw_retry, stage_name=f"{stage} retry")
            if not isinstance(parsed, dict):
                raise NewsValidationError(
                    f"{stage} retry returned JSON that is not an object",
                    stage=stage,
                )
            return parsed

    @staticmethod
    def _safe_json_loads(raw_text: str, stage_name: str) -> Dict[str, Any]:
        """Safely parse JSON with layered fallback strategies."""
        if not raw_text or not raw_text.strip():
            raise NewsValidationError(
                f"Empty response received from {stage_name}",
                stage=stage_name,
            )

        original = raw_text.strip()
        
        cleaned = LLMCaller._strip_markdown_fences(original)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        balanced = LLMCaller._extract_balanced_json(cleaned)
        if balanced:
            try:
                return json.loads(balanced)
            except json.JSONDecodeError:
                pass
            fixed = LLMCaller._fix_common_json_errors(balanced)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        heuristic = re.sub(r',(\s*[\}\]])', r'\1', cleaned)
        try:
            return json.loads(heuristic)
        except json.JSONDecodeError:
            pass

        logger.error("All JSON parsing strategies exhausted for %s.", stage_name)
        logger.error("Raw response excerpt (first 1000 chars):\n%s", original[:1000])
        raise NewsValidationError(
            f"Failed to parse JSON response from {stage_name} after all recovery strategies.",
            stage=stage_name,
        )

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^~~~(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n?~~~\s*$', '', text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _extract_balanced_json(text: str) -> str:
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

    @staticmethod
    def _fix_common_json_errors(text: str) -> str:
        text = re.sub(r',(\s*[\}\]])', r'\1', text)
        text = re.sub(r',+(\s*)', r',', text)
        return text

    def fits_in_budget(self, input_tokens: int, output_tokens: int, stage: str) -> bool:
        total = input_tokens + output_tokens + self.budget.safety_margin
        return total <= self.budget.provider_context_limit


# =============================================================================
# PART 5: PIPELINE STAGES
# =============================================================================


class PipelineStages:
    """
    Implementation of all pipeline stages.
    Each stage is a separate method with clear inputs, outputs, and error handling.
    """

    def __init__(
        self,
        vault: ObsidianVaultManager,
        llm: LLMCaller,
        config: NewsConfig,
    ) -> None:
        self.vault = vault
        self.llm = llm
        self.config = config

    # ------------------------------------------------------------------
    # Stage 0: Input Validation
    # ------------------------------------------------------------------

    def stage_0_validate_input(
        self,
        article_text: str,
        perspective: PerspectiveContext,
    ) -> Tuple[str, PerspectiveContext]:
        """
        Validate and normalize inputs.
        
        Raises:
            NewsValidationError: If inputs are invalid
        """
        # Validate article text
        if not article_text:
            raise NewsValidationError(
                "Article text cannot be empty",
                stage=PipelineStage.INPUT_VALIDATION.value,
            )
        
        if not isinstance(article_text, str):
            raise NewsValidationError(
                f"Article text must be a string, not {type(article_text)}",
                stage=PipelineStage.INPUT_VALIDATION.value,
            )
        
        article_text = article_text.strip()
        
        if len(article_text) > self.config.max_article_chars:
            raise NewsValidationError(
                f"Article contains {len(article_text):,} characters, "
                f"exceeding limit of {self.config.max_article_chars:,}. "
                f"ATIS refuses to truncate source material.",
                stage=PipelineStage.INPUT_VALIDATION.value,
            )
        
        # Validate perspective
        if not isinstance(perspective, PerspectiveContext):
            try:
                perspective = PerspectiveContext.from_payload(perspective)
            except (ValueError, TypeError, KeyError) as e:
                raise NewsValidationError(
                    f"Invalid perspective: {e}",
                    stage=PipelineStage.INPUT_VALIDATION.value,
                )
        
        return article_text, perspective

    # ------------------------------------------------------------------
    # Stage 1: Article Understanding
    # ------------------------------------------------------------------

    def stage_1_understand_article(
        self,
        article_text: str,
        perspective: PerspectiveContext,
    ) -> Dict[str, Any]:
        """
        Extract structured understanding from the article.
        
        Uses LLM to extract:
          - Event metadata (title, summary, countries)
          - Facts with evidence
          - Actors with roles
          - Mechanisms
          - Meaning/interpretation
          - Uncertainties
        
        Returns:
            Structured article understanding
        """
        system_prompt = """You are ATIS News Stage 1: Article Understanding.
Understand ONLY what is contained in the supplied article segment.
Do not use external knowledge. Do not invent entities, relationships, causes, or facts.
Preserve every material fact, actor, mechanism, meaning and uncertainty present in the supplied segment.
Return ONLY valid JSON.

Schema:
{
  "event": {"title": "", "summary": "", "event_country": "", "source_country": ""},
  "facts": [{"fact": "", "evidence": "", "importance": "high|medium|low"}],
  "actors": [{"name": "", "role": "", "evidence": ""}],
  "mechanisms": [{"mechanism": "", "evidence": ""}],
  "meaning": [{"interpretation": "", "based_on_fact_indexes": [0]}],
  "uncertainties": ["..."]
}"""

        requested = self.config.stage_output_tokens
        
        # Check if we need batching
        full_user = "ARTICLE SEGMENT (COMPLETE):\n" + article_text
        full_tokens = TokenBudget.estimate_message_tokens([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user},
        ])
        
        if self.llm.budget.fits_in_budget(full_tokens, requested):
            result = self.llm.call_json(
                system_prompt, full_user, requested,
                PipelineStage.ARTICLE_UNDERSTANDING.value
            )
        else:
            # Batch the article
            batches = self._partition_paragraphs(article_text)
            logger.warning(
                "[ARTICLE] complete article does not fit; processing %d lossless batches",
                len(batches)
            )
            
            results: List[Dict[str, Any]] = []
            for index, batch in enumerate(batches, start=1):
                user = (
                    f"ARTICLE SEGMENT {index}/{len(batches)} "
                    f"(COMPLETE; DO NOT INFER BEYOND THIS SEGMENT):\n{batch}"
                )
                results.append(self.llm.call_json(
                    system_prompt, user, requested,
                    f"{PipelineStage.ARTICLE_UNDERSTANDING.value} [{index}/{len(batches)}]"
                ))
            
            result = self._merge_article_results(results)
        
        # Ensure all expected fields exist
        result.setdefault("event", {})
        result.setdefault("facts", [])
        result.setdefault("actors", [])
        result.setdefault("mechanisms", [])
        result.setdefault("meaning", [])
        result.setdefault("uncertainties", [])
        
        logger.info(
            "[FACTS] %d facts | %d actors | %d mechanisms | %d meanings",
            len(_pf_safe_list(result.get("facts"))),
            len(_pf_safe_list(result.get("actors"))),
            len(_pf_safe_list(result.get("mechanisms"))),
            len(_pf_safe_list(result.get("meaning"))),
        )
        
        return result

    def _partition_paragraphs(self, text: str) -> List[str]:
        """Partition text into paragraphs without data loss."""
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        max_chars = self.config.article_batch_chars
        
        for paragraph in paragraphs:
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

    def _merge_article_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from multiple article batches."""
        merged: Dict[str, Any] = {
            "event": {},
            "facts": [],
            "actors": [],
            "mechanisms": [],
            "meaning": [],
            "uncertainties": [],
        }
        
        for result in results:
            event = _safe_dict(result.get("event"))
            for key in ("title", "summary", "event_country", "source_country"):
                if not merged["event"].get(key) and event.get(key):
                    merged["event"][key] = event[key]
            
            for key in ("facts", "actors", "mechanisms", "uncertainties"):
                for item in _pf_safe_list(result.get(key)):
                    identity = _json(item)
                    if not any(_json(existing) == identity for existing in merged[key]):
                        merged[key].append(item)
        
        # Clear fact indexes from meaning (they're batch-local)
        for item in merged["meaning"]:
            if isinstance(item, dict):
                item["based_on_fact_indexes"] = []
        
        return merged

    # ------------------------------------------------------------------
    # Stage 2: Perspective Ecosystem Loading
    # ------------------------------------------------------------------

    def stage_2_load_ecosystem(
        self,
        perspective: PerspectiveContext,
    ) -> List[Dict[str, Any]]:
        """
        Load all nodes for the perspective country from the vault.
        
        This is a database-only operation - no LLM calls.
        
        Returns:
            List of node records for the perspective country
        """
        country_norm = _pf_norm(perspective.country)
        candidates: List[Dict[str, Any]] = []
        
        for canon in sorted(self.vault.node_metadata):
            meta = self.vault.node_metadata[canon]
            node_country = _pf_norm(meta.get("country", ""))
            
            if not node_country:
                path = str(meta.get("path", "")).lower()
                if country_norm and country_norm in path:
                    node_country = country_norm
            
            if node_country != country_norm:
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
        
        config = get_config()
        if config.max_perspective_ecosystem_nodes > 0 and \
                len(candidates) > config.max_perspective_ecosystem_nodes:
            logger.warning(
                "Perspective ecosystem capped at %d nodes (found %d)",
                config.max_perspective_ecosystem_nodes, len(candidates)
            )
            candidates = candidates[:config.max_perspective_ecosystem_nodes]
        
        logger.info(
            "[PERSPECTIVE] %s ecosystem nodes available: %d",
            perspective.country, len(candidates)
        )
        
        return candidates

    @staticmethod
    def build_ecosystem_context(
        ecosystem: List[Dict[str, Any]],
        country: str,
        code: str,
    ) -> str:
        """Build context string for ecosystem."""
        lines = [
            f"PERSPECTIVE COUNTRY: {country} ({code})",
            "Every NODE ID below is an existing database node. "
            "This registry is retrieval guidance, not proof of impact.",
        ]
        for n in ecosystem:
            lines.append(
                f"NODE={n['node_id']} | type={n.get('type') or 'unknown'} | "
                f"sector={n.get('sector') or 'unknown'} | summary={n.get('summary') or ''}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 3: Perspective Impact Mapping
    # ------------------------------------------------------------------

    def stage_3_map_impact(
        self,
        article: Dict[str, Any],
        ecosystem: List[Dict[str, Any]],
        perspective: PerspectiveContext,
    ) -> Dict[str, Any]:
        """
        Map the article onto the perspective ecosystem.
        
        Uses LLM to identify which ecosystem domains should be investigated
        based on the article's facts and meaning.
        
        Returns:
            Impact domains with node hints
        """
        system_prompt = """You are ATIS News Stage 2: Perspective Impact Mapper.
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
  "impact_domains": [
    {
      "domain": "",
      "why_relevant": "",
      "mechanism": "",
      "priority": "high|medium|low",
      "ecosystem_node_hints": ["exact NODE ID"]
    }
  ],
  "excluded_domains": [{"domain": "", "reason": ""}]
}"""

        article_json = _json(article)
        requested = self.config.stage_output_tokens
        
        def normalize(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            domains: List[Dict[str, Any]] = []
            for item in _pf_safe_list(result.get("impact_domains")):
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                item["ecosystem_node_hints"] = sorted({
                    str(x).strip()
                    for x in _pf_safe_list(item.get("ecosystem_node_hints"))
                    if not isinstance(x, (dict, list)) and str(x).strip()
                })
                domains.append(item)
            excluded = [x for x in _pf_safe_list(result.get("excluded_domains")) if isinstance(x, dict)]
            return domains, excluded
        
        # Try with full ecosystem first
        registry = self.build_ecosystem_context(ecosystem, perspective.country, perspective.country_code)
        full_user = (
            f"ARTICLE UNDERSTANDING:\n{article_json}\n\n"
            f"PERSPECTIVE ECOSYSTEM (COMPLETE):\n{registry}"
        )
        full_tokens = TokenBudget.estimate_message_tokens([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user},
        ])
        
        if self.llm.budget.fits_in_budget(full_tokens, requested):
            result = self.llm.call_json(
                system_prompt, full_user, requested,
                PipelineStage.PERSPECTIVE_IMPACT_MAPPING.value
            )
            domains, excluded = normalize(result)
        else:
            # Batch the ecosystem
            batches = self._partition_records(ecosystem, self.config.max_ecosystem_batch_nodes)
            logger.warning(
                "[PERSPECTIVE] complete registry is ~%d tokens; partitioning into %d batches",
                full_tokens, len(batches)
            )
            
            all_domains: List[Dict[str, Any]] = []
            all_excluded: List[Dict[str, Any]] = []
            
            for batch_index, batch in enumerate(batches, start=1):
                if not batch:
                    continue
                batch_registry = self.build_ecosystem_context(
                    batch, perspective.country, perspective.country_code
                )
                user = (
                    f"ARTICLE UNDERSTANDING:\n{article_json}\n\n"
                    f"PERSPECTIVE ECOSYSTEM BATCH {batch_index}/{len(batches)}:\n"
                    f"{batch_registry}"
                )
                
                # Check if batch fits
                input_tokens = TokenBudget.estimate_message_tokens([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ])
                
                if not self.llm.budget.fits_in_budget(input_tokens, requested):
                    # Try sub-batching
                    sub_batches = self._partition_records(batch, max(1, len(batch) // 2))
                    if len(sub_batches) > 1:
                        for sub_index, sub_batch in enumerate(sub_batches, start=1):
                            sub_registry = self.build_ecosystem_context(
                                sub_batch, perspective.country, perspective.country_code
                            )
                            sub_user = (
                                f"ARTICLE UNDERSTANDING:\n{article_json}\n\n"
                                f"PERSPECTIVE ECOSYSTEM SUB-BATCH {batch_index}.{sub_index}:\n"
                                f"{sub_registry}"
                            )
                            sub_result = self.llm.call_json(
                                system_prompt, sub_user, requested,
                                f"{PipelineStage.PERSPECTIVE_IMPACT_MAPPING.value} [{batch_index}.{sub_index}]"
                            )
                            d, e = normalize(sub_result)
                            all_domains.extend(d)
                            all_excluded.extend(e)
                        continue
                    else:
                        raise NewsPipelineContextOverflow(
                            f"Perspective ecosystem batch {batch_index} cannot fit provider context"
                        )
                
                result = self.llm.call_json(
                    system_prompt, user, requested,
                    f"{PipelineStage.PERSPECTIVE_IMPACT_MAPPING.value} [{batch_index}/{len(batches)}]"
                )
                domains, excluded = normalize(result)
                all_domains.extend(domains)
                all_excluded.extend(excluded)
            
            domains = all_domains
            excluded = all_excluded
        
        # Merge equivalent domains
        merged: Dict[str, Dict[str, Any]] = {}
        for domain in domains:
            key = _pf_norm(domain.get("domain", "")) or f"domain-{len(merged)}"
            if key not in merged:
                merged[key] = dict(domain)
                merged[key]["ecosystem_node_hints"] = sorted(set(domain.get("ecosystem_node_hints", [])))
                continue
            current = merged[key]
            current["ecosystem_node_hints"] = sorted(set(current.get("ecosystem_node_hints", [])) | 
                                                   set(domain.get("ecosystem_node_hints", [])))
            for field_name in ("why_relevant", "mechanism"):
                domain_val = domain.get(field_name, "")
                current_val = current.get(field_name, "")
                if domain_val and domain_val not in current_val:
                    current[field_name] = f"{current_val}; {domain_val}".strip("; ")
            
            priority_rank = {"high": 0, "medium": 1, "low": 2}
            domain_priority = str(domain.get("priority", "medium")).lower()
            current_priority = str(current.get("priority", "medium")).lower()
            if priority_rank.get(domain_priority, 1) < priority_rank.get(current_priority, 1):
                current["priority"] = domain.get("priority")
        
        sorted_domains = sorted(
            merged.values(),
            key=lambda d: (
                0 if str(d.get("priority", "medium")).lower() == "high" 
                else 1 if str(d.get("priority", "medium")).lower() == "medium" else 2,
                _pf_norm(d.get("domain", "")),
            ),
        )
        
        return {"impact_domains": sorted_domains, "excluded_domains": excluded}

    def _partition_records(
        self,
        records: List[Any],
        max_per_batch: int,
    ) -> List[List[Any]]:
        if not records:
            return []
        if max_per_batch <= 0:
            raise ValueError("max_per_batch must be positive")
        return [records[i:i + max_per_batch] for i in range(0, len(records), max_per_batch)]

    # ------------------------------------------------------------------
    # Stage 4: Target Resolution
    # ------------------------------------------------------------------

    def stage_4_resolve_targets(
        self,
        impact: Dict[str, Any],
        ecosystem: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Resolve impact domain hints to actual vault nodes.
        
        This is a database-only operation - no LLM calls.
        
        Returns:
            Tuple of (resolved target nodes, unresolved hints)
        """
        resolved: Dict[str, Dict[str, Any]] = {}
        unresolved: List[Dict[str, Any]] = []
        
        for domain in _pf_safe_list(impact.get("impact_domains")):
            if not isinstance(domain, dict):
                continue
            
            hints = _pf_safe_list(domain.get("ecosystem_node_hints"))
            if not hints:
                # Try to match by sector
                sector = _pf_norm(domain.get("domain", ""))
                if sector:
                    hints = [
                        n["node_id"]
                        for n in ecosystem
                        if sector in _pf_norm(n.get("sector", ""))
                    ]
            
            found = False
            for hint in hints:
                resolved_node = self.vault.resolve_node(str(hint))
                if resolved_node:
                    canon = self.vault._canonicalize(resolved_node)
                    if canon not in resolved:
                        rec = self.vault.get_node_record(canon)
                        domains_set = set(rec.get("target_domains", []))
                        domains_set.add(str(domain.get("domain", "")))
                        rec["target_domains"] = sorted(x for x in domains_set if x)
                        resolved[canon] = rec
                    found = True
                else:
                    unresolved.append({
                        "hint": str(hint),
                        "domain": domain.get("domain", ""),
                        "status": "UNRESOLVED",
                    })
            
            if not found:
                unresolved.append({
                    "hint": domain.get("domain", ""),
                    "domain": domain.get("domain", ""),
                    "status": "NO_TARGET_NODE_RESOLVED",
                })
        
        nodes = sorted(
            resolved.values(),
            key=lambda n: (str(n.get("sector", "")), str(n.get("node_id", "")))
        )
        
        config = get_config()
        if config.max_retrieval_targets > 0 and len(nodes) > config.max_retrieval_targets:
            overflow = nodes[config.max_retrieval_targets:]
            nodes = nodes[:config.max_retrieval_targets]
            unresolved.extend({
                "hint": n.get("node_id", ""),
                "domain": "retrieval_cap",
                "status": "RESEARCH_REQUIRED",
                "reason": "operator_configured_target_cap",
            } for n in overflow)
        
        logger.info(
            "[RETRIEVAL] targeted nodes=%d unresolved targets=%d",
            len(nodes), len(unresolved)
        )
        
        return nodes, unresolved

    # ------------------------------------------------------------------
    # Stage 5: Graph Traversal
    # ------------------------------------------------------------------

    def stage_5_traverse_graph(
        self,
        target_nodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Traverse the vault graph from target nodes.
        
        Uses bidirectional links (outbound + backlinks) to build a graph
        of related nodes, edges, and paths.
        
        This is a database-only operation - no LLM calls.
        
        Returns:
            Graph structure with nodes, edges, paths, and metadata
        """
        config = get_config()
        seeds = sorted({
            n["canonical_id"]
            for n in target_nodes
            if n.get("canonical_id") in self.vault.file_map
        })
        
        visited: Dict[str, int] = {s: 0 for s in seeds}
        queue: List[str] = list(seeds)
        all_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        backlink_candidates = 0
        
        while queue:
            current = queue.pop(0)
            depth = visited[current]
            
            if config.max_graph_depth > 0 and depth >= config.max_graph_depth:
                continue
            
            outgoing = sorted(
                self.vault.outbound_links.get(current, []),
                key=lambda x: self.vault._canonicalize(x)
            )
            incoming = sorted(
                self.vault.backlink_map.get(current, set()),
                key=lambda x: self.vault._canonicalize(x)
            )
            
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
            key=lambda e: (
                int(e.get("depth") or 0),
                str(e.get("from_node")),
                str(e.get("to_node")),
                str(e.get("relationship_type")),
            ),
        )
        
        graph_nodes: List[Dict[str, Any]] = []
        overflow_nodes: List[str] = []
        ordered_node_ids = sorted(visited.items(), key=lambda x: (x[1], x[0]))
        
        for index, (canon, depth) in enumerate(ordered_node_ids):
            if config.max_graph_nodes > 0 and index >= config.max_graph_nodes:
                overflow_nodes.append(self.vault.file_map.get(canon, canon))
                continue
            
            rec = self.vault.get_node_record(canon)
            rec["graph_depth"] = depth
            rec["graph_category"] = (
                "target" if canon in seeds
                else "first_order" if depth == 1
                else "second_order"
            )
            graph_nodes.append(rec)
        
        allowed_ids = {str(n.get("node_id")) for n in graph_nodes}
        filtered_edges = [
            e for e in ordered_edges
            if str(e.get("from_node")) in allowed_ids and str(e.get("to_node")) in allowed_ids
        ]
        
        overflow_edges = max(0, len(ordered_edges) - len(filtered_edges))
        if config.max_graph_paths > 0 and len(filtered_edges) > config.max_graph_paths:
            overflow_edges += len(filtered_edges) - config.max_graph_paths
            filtered_edges = filtered_edges[:config.max_graph_paths]
        
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
    # Stage 6: Impact Analysis
    # ------------------------------------------------------------------

    def stage_6_analyze_impact(
        self,
        article: Dict[str, Any],
        impact: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert verified graph into machine-grounded evidence.
        
        This is a deterministic operation - no LLM calls.
        Relationships come from the database, not from LLM invention.
        
        Returns:
            Consequences, gaps, and signals derived from the graph
        """
        nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
        edges = [e for e in graph.get("edges", []) if isinstance(e, dict)]
        
        consequences: List[Dict[str, Any]] = []
        for edge in edges:
            consequences.append({
                "statement": (
                    f"Verified database relationship: {edge.get('from_node')} -> {edge.get('to_node')}. "
                    f"Type: {edge.get('relationship_type')}"
                ),
                "order": "direct" if int(edge.get("depth") or 1) == 1 else "second_order",
                "supporting_nodes": [str(edge.get("from_node")), str(edge.get("to_node"))],
                "supporting_edges": [f"{edge.get('from_node')}->{edge.get('to_node')}"],
                "confidence": 1.0,
                "source": "database",
            })
        
        gaps: List[Dict[str, Any]] = []
        if not nodes:
            gaps.append({
                "gap": "No targeted database graph nodes were resolved",
                "status": "RESEARCH_REQUIRED",
                "related_nodes": [],
            })
        
        if graph.get("overflow_nodes"):
            gaps.append({
                "gap": f"Graph traversal hit node limit ({len(graph.get('overflow_nodes', []))} nodes overflow)",
                "status": "RESEARCH_REQUIRED",
                "related_nodes": graph.get("overflow_nodes", []),
            })
        
        if graph.get("overflow_edge_count", 0) > 0:
            gaps.append({
                "gap": f"Graph traversal hit edge/path limit ({graph.get('overflow_edge_count')} edges overflow)",
                "status": "RESEARCH_REQUIRED",
                "related_nodes": [],
            })
        
        return {
            "consequences": consequences,
            "gaps": gaps,
            "opportunity_signals": [],
            "risk_signals": [],
            "source": "database_deterministic",
        }

    # ------------------------------------------------------------------
    # Stage 7: Final Synthesis
    # ------------------------------------------------------------------

    def stage_7_final_synthesis(
        self,
        article: Dict[str, Any],
        impact: Dict[str, Any],
        graph: Dict[str, Any],
        consequences: Dict[str, Any],
        impact_chain: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produce final intelligence dashboard from all previous outputs.
        
        Uses LLM to synthesize the complete analysis.
        
        Returns:
            Complete intelligence dashboard
        """
        system_prompt = """You are the ATIS News Final Synthesis Engine.
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
  "trigger_event": "",
  "market_equilibrium_shift": "",
  "executive_summary": "",
  "analytical_perspective": {"country": "", "country_code": "", "description": ""},
  "facts": [],
  "meaning": [],
  "impact_domains": [],
  "impact_chain": [],
  "findings": [{"text": "", "source_nodes": [], "graph_paths": [], "status": "SUPPORTED|RESEARCH_REQUIRED"}],
  "opportunities": [
    {
      "opportunity_id": "",
      "title": "",
      "status": "SUPPORTED|RESEARCH_REQUIRED",
      "perspective_country": "",
      "opportunity_country": "",
      "source_country": "",
      "event_country": "",
      "cross_border": false,
      "cross_border_countries": [],
      "perspective_actor": "",
      "perspective_capability": "",
      "pathway": "",
      "justification": "",
      "urgency_score": 0.0,
      "feasibility_score": 0.0,
      "source_nodes": [],
      "graph_paths": [],
      "required_missing_nodes": []
    }
  ],
  "risks": [
    {"text": "", "status": "SUPPORTED|RESEARCH_REQUIRED", "severity": "high|medium|low", "source_nodes": [], "graph_paths": []}
  ],
  "gaps": [{"gap": "", "status": "RESEARCH_REQUIRED", "related_nodes": [], "missing_relationship": ""}],
  "key_entities": [
    {"entity_name": "", "entity_type": "", "country": "", "sector": "", "significance_score": 0, "summary": "", "source_node": ""}
  ]
}"""

        requested = self.config.final_output_tokens
        
        # Build base context (immutable across all batches)
        base = (
            f"PERSPECTIVE={self.llm.client.config.model}\n"
            f"ARTICLE UNDERSTANDING={_json(article)}\n"
            f"PERSPECTIVE IMPACT={_json(impact)}\n"
        )
        
        # Pack evidence records into context-safe batches
        consequence_records: List[Dict[str, Any]] = []
        for key in ("consequences", "opportunity_signals", "risk_signals", "gaps"):
            for item in _pf_safe_list(consequences.get(key)):
                consequence_records.append({"collection": key, "record": item})
        
        narrative_records = [
            {"collection": "impact_chain", "record": item}
            for item in impact_chain
            if item.get("stage") != "graph_relationship"
        ]
        
        node_records = [
            {"collection": "graph_nodes", "record": dict(n)}
            for n in _pf_safe_list(graph.get("nodes"))
        ]
        edge_records = [
            {"collection": "graph_edges", "record": dict(e)}
            for e in _pf_safe_list(graph.get("edges"))
        ]
        path_records = [
            {"collection": "graph_paths", "record": dict(path)}
            for path in _pf_safe_list(graph.get("paths"))
        ]
        
        atomic = consequence_records + narrative_records + node_records + edge_records + path_records
        
        # Calculate available budget
        base_tokens = TokenBudget.estimate_message_tokens([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base},
        ])
        usable = self.llm.budget.usable_input_budget - requested - base_tokens
        
        if usable <= 0:
            raise NewsPipelineContextOverflow(
                f"Final synthesis immutable context alone consumes {base_tokens} tokens; "
                f"no room remains for evidence records."
            )
        
        # Pack records into batches
        def record_cost(record: Dict[str, Any]) -> int:
            return max(1, TokenBudget.estimate_tokens(_json(record)))
        
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
            cost = record_cost(record)
            if cost > usable:
                raise NewsPipelineContextOverflow(
                    f"A complete final-synthesis evidence record cannot fit within the "
                    f"provider context budget ({cost} > {usable} tokens). "
                    f"ATIS refuses to truncate the record."
                )
            if current and current_tokens + cost > usable:
                flush()
            current.append(record)
            current_tokens += cost
        flush()
        
        if not batches:
            batches = [[]]
        
        logger.info(
            "[FINAL] lossless final synthesis: %d batches | base_tokens=%d | evidence_budget=%d | records=%d",
            len(batches), base_tokens, usable, len(atomic),
        )
        
        # Execute batches
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
                + f"VERIFIED EVIDENCE BATCH {index}/{len(batches)}={_json(payload)}\n"
                + "Analyze ONLY the supplied evidence. Preserve provenance and mark "
                + "anything not supported by this batch as RESEARCH_REQUIRED."
            )
            
            input_tokens = TokenBudget.estimate_message_tokens([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ])
            
            if not self.llm.budget.fits_in_budget(input_tokens, requested):
                raise NewsPipelineContextOverflow(
                    f"Final synthesis batch {index} exceeded the provider budget "
                    f"after serialization ({input_tokens} tokens)."
                )
            
            results.append(self.llm.call_json(
                system_prompt, user, requested,
                f"{PipelineStage.FINAL_SYNTHESIS.value} [{index}/{len(batches)}]"
            ))
        
        if not results:
            raise RuntimeError("Final synthesis produced no batch results")
        
        # Merge results
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
    # Stage 8: Validation & Grounding
    # ------------------------------------------------------------------

    def stage_8_validate_and_ground(
        self,
        dashboard: Dict[str, Any],
        graph: Dict[str, Any],
        ecosystem: List[Dict[str, Any]],
        article: Dict[str, Any],
        impact: Dict[str, Any],
        consequences: Dict[str, Any],
        perspective: PerspectiveContext,
    ) -> Dict[str, Any]:
        """
        Validate all outputs against vault evidence.
        
        Ensures:
          - Opportunities have valid source nodes and graph paths
          - Risks have valid source nodes
          - All references point to real vault nodes
          - Perspective actor is from the perspective country
        
        Returns:
            Validated dashboard with RESEARCH_REQUIRED markers where appropriate
        """
        valid_nodes = {str(n.get("node_id")) for n in graph.get("nodes", []) if n.get("node_id")}
        valid_edges = {
            f"{e.get('from_node')}->{e.get('to_node')}"
            for e in graph.get("edges", [])
            if e.get("from_node") and e.get("to_node")
        }
        perspective_nodes = {str(n.get("node_id")) for n in ecosystem if n.get("node_id")}
        config = get_config()
        
        # Validate opportunities
        grounded_opportunities: List[Dict[str, Any]] = []
        for raw in _pf_safe_list(dashboard.get("opportunities")):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            source_nodes = [str(x) for x in _pf_safe_list(item.get("source_nodes")) if str(x) in valid_nodes]
            graph_paths = [str(x) for x in _pf_safe_list(item.get("graph_paths")) if str(x) in valid_edges]
            actor = str(item.get("perspective_actor", ""))
            actor_supported = bool(actor) and actor in perspective_nodes
            perspective_country = str(item.get("perspective_country") or perspective.country)
            
            path_endpoints = {
                part for path in graph_paths
                for part in str(path).split("->", 1)
            }
            
            supported = bool(
                source_nodes
                and graph_paths
                and actor_supported
                and _pf_norm(perspective_country) == _pf_norm(perspective.country)
                and _pf_norm(str(item.get("opportunity_country") or perspective.country)) == _pf_norm(perspective.country)
                and any(node in path_endpoints for node in source_nodes)
            )
            
            item["source_nodes"] = source_nodes
            item["graph_paths"] = graph_paths
            item["perspective_country"] = perspective.country
            item["perspective_country_code"] = perspective.country_code
            
            if supported:
                item["status"] = "SUPPORTED"
                item["opportunity_type"] = OpportunityType.EXPLICIT.value
                item["opportunity_confidence"] = max(
                    _pf_safe_float(item.get("opportunity_confidence")),
                    config.min_opportunity_graph_score
                )
            else:
                item["status"] = "RESEARCH_REQUIRED"
                item["opportunity_type"] = OpportunityType.POTENTIAL.value
                item["opportunity_confidence"] = min(
                    _pf_safe_float(item.get("opportunity_confidence")),
                    0.49
                )
                missing = list(_pf_safe_list(item.get("required_missing_nodes")))
                if not source_nodes:
                    missing.append("verified source node")
                if not graph_paths:
                    missing.append("verified database graph path")
                if not actor_supported:
                    missing.append("verified perspective-country actor")
                item["required_missing_nodes"] = sorted(set(str(x) for x in missing if x))
            
            # Generate stable ID
            stable_id = compute_opportunity_identity(
                title=item.get("title", ""),
                perspective_country=perspective.country,
                source_country=item.get("source_country", ""),
                event_country=item.get("event_country", ""),
                opportunity_country=item.get("opportunity_country", ""),
                perspective_actor=item.get("perspective_actor", ""),
                perspective_capability=item.get("perspective_capability", ""),
                pathway=item.get("pathway", ""),
                source_nodes=item.get("source_nodes", []),
            )
            item["opportunity_id"] = stable_id
            item["stable_opportunity_id"] = stable_id
            
            grounded_opportunities.append(item)
        
        dashboard["opportunities"] = grounded_opportunities
        
        # Validate risks
        grounded_risks: List[Dict[str, Any]] = []
        for raw in _pf_safe_list(dashboard.get("risks")):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item["source_nodes"] = [
                str(x) for x in _pf_safe_list(item.get("source_nodes"))
                if str(x) in valid_nodes
            ]
            item["graph_paths"] = [
                str(x) for x in _pf_safe_list(item.get("graph_paths"))
                if str(x) in valid_edges
            ]
            item.setdefault("status", "SUPPORTED" if item["source_nodes"] else "RESEARCH_REQUIRED")
            grounded_risks.append(item)
        dashboard["risks"] = grounded_risks
        
        # Ensure all list fields exist
        dashboard.setdefault("findings", [])
        dashboard.setdefault("gaps", [])
        dashboard.setdefault("entities", [])
        dashboard.setdefault("facts", [])
        dashboard.setdefault("meaning", [])
        dashboard.setdefault("impact_domains", [])
        dashboard.setdefault("impact_chain", [])
        
        # Add analytical perspective
        dashboard["analytical_perspective"] = {
            "country": perspective.country,
            "country_code": perspective.country_code,
            "description": "Country through which this event is interpreted.",
        }
        
        # Add graph analysis metadata
        dashboard["graph_analysis"] = {
            "direct": graph.get("direct_nodes", 0),
            "first_order": graph.get("first_order_nodes", 0),
            "second_order": graph.get("second_order_nodes", 0),
            "backlink_candidates": graph.get("backlink_candidates", 0),
            "paths": graph.get("paths", []),
        }
        
        # Extract source/event countries from article
        event = article.get("event", {}) if isinstance(article.get("event"), dict) else {}
        dashboard["source_country"] = event.get("source_country", "") or ""
        dashboard["event_country"] = event.get("event_country", "") or ""
        
        return dashboard

    def build_impact_chain(
        self,
        article: Dict[str, Any],
        impact: Dict[str, Any],
        graph: Dict[str, Any],
        consequences: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build the complete impact chain from all stages."""
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
            if not isinstance(edge, dict):
                continue
            chain.append({
                "stage": "graph_relationship",
                "label": f"{edge.get('from_node')} \u2192 {edge.get('to_node')}",
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
    # Stage 9: Output Assembly
    # ------------------------------------------------------------------

    def stage_9_assemble_output(
        self,
        dashboard: Dict[str, Any],
        article: Dict[str, Any],
        impact: Dict[str, Any],
        graph: Dict[str, Any],
        consequences: Dict[str, Any],
        impact_chain: List[Dict[str, Any]],
        ecosystem: List[Dict[str, Any]],
        unresolved: List[Dict[str, Any]],
        perspective: PerspectiveContext,
        reasoning_log: ReasoningLog,
        job_state: JobState,
    ) -> Dict[str, Any]:
        """
        Assemble the final output with all metadata and persistence.
        
        Returns:
            Complete dashboard ready for return
        """
        # Ensure authoritative fields are preserved
        dashboard = dict(dashboard or {})
        dashboard["facts"] = _pf_safe_list(article.get("facts"))
        dashboard["meaning"] = _pf_safe_list(article.get("meaning"))
        dashboard["impact_domains"] = _pf_safe_list(impact.get("impact_domains"))
        dashboard["impact_chain"] = impact_chain
        
        # Add source nodes from graph
        dashboard["source_nodes"] = [
            {k: n.get(k, "") for k in (
                "node_id", "canonical_id", "country", "type", "sector", 
                "summary", "path", "graph_depth", "graph_category"
            )}
            for n in graph.get("nodes", []) if isinstance(n, dict)
        ]
        
        # Add perspective nodes
        dashboard["perspective_nodes"] = [
            {k: n.get(k, "") for k in (
                "node_id", "canonical_id", "country", "type", "sector", 
                "summary", "path"
            )}
            for n in ecosystem if isinstance(n, dict)
        ]
        
        # Add cross-border bridges
        event = article.get("event", {}) if isinstance(article.get("event"), dict) else {}
        source_country = str(dashboard.get("source_country") or event.get("source_country", "")).strip()
        event_country = str(dashboard.get("event_country") or event.get("event_country", "")).strip()
        bridge_source = source_country or event_country
        
        bridges: List[Dict[str, Any]] = []
        if bridge_source and _pf_norm(bridge_source) != _pf_norm(perspective.country):
            try:
                bridges = self.vault.get_cross_border_bridges(perspective, bridge_source)
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
        
        # Add structured intelligence
        dashboard["structured_intelligence"] = {
            "facts": dashboard["facts"],
            "meaning": dashboard["meaning"],
            "impact_domains": dashboard["impact_domains"],
            "graph": {
                "nodes": dashboard["source_nodes"],
                "edges": graph.get("edges", []),
                "paths": graph.get("paths", []),
            },
            "consequences": consequences.get("consequences", []),
            "opportunities": dashboard.get("opportunities", []),
            "risks": dashboard.get("risks", []),
            "gaps": dashboard.get("gaps", []),
        }
        
        # Add research required
        dashboard["research_required"] = (
            unresolved
            + _pf_safe_list(consequences.get("gaps"))
            + _pf_safe_list(graph.get("research_required"))
        )
        
        # Add pipeline metadata
        dashboard["pipeline_metadata"] = self._build_pipeline_metadata(
            dashboard, perspective, reasoning_log, job_state, article, graph
        )
        
        # Set status
        dashboard["status"] = dashboard.get("status", "complete")
        dashboard["partial"] = bool(dashboard.get("partial", False))
        
        # Add perspective
        dashboard["perspective"] = perspective.as_dict()
        
        return dashboard

    def _build_pipeline_metadata(
        self,
        dashboard: Dict[str, Any],
        perspective: PerspectiveContext,
        reasoning_log: ReasoningLog,
        job_state: JobState,
        article: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build comprehensive pipeline metadata."""
        config = get_config()
        event = article.get("event", {}) if isinstance(article.get("event"), dict) else {}
        core_event = event.get("title") or event.get("summary") or dashboard.get("trigger_event") or "Unknown event"
        
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
            knowledge_state = KnowledgeState(vault_path=self.vault.vault_dir)
            knowledge_state.compute()
            knowledge_hash = knowledge_state.knowledge_state_hash
        except Exception:
            knowledge_hash = ""
        
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
        
        return {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "architecture": "perspective_first_graph_grounded",
            "architecture_version": "3.0.0",
            "pipeline_version": job_state.pipeline_version,
            "source_article": "web_upload",
            "core_event": core_event,
            "source_country": dashboard.get("source_country", ""),
            "event_country": dashboard.get("event_country", ""),
            "perspective_country": perspective.country,
            "perspective_country_code": perspective.country_code,
            "model_primary": config.llm_model,
            "model_fallback": config.llm_fallback_model or "",
            "seed_sent_to_provider": False,
            "selected_evidence_nodes": (
                dashboard.get("graph_analysis", {}).get("direct", 0) +
                dashboard.get("graph_analysis", {}).get("first_order", 0) +
                dashboard.get("graph_analysis", {}).get("second_order", 0)
            ),
            "cross_border_bridges_found": dashboard.get("cross_border_analysis", {}).get("cross_border_bridges_count", 0),
            "analysis_version": ANALYSIS_VERSION,
            "schema_version": SCHEMA_VERSION,
            "analysis_fingerprint": fingerprint,
            "knowledge_state": {"hash": knowledge_hash},
            "reasoning_log": reasoning_data,
            "llm_calls": self.llm.calls,
            "truncated_retries": self.llm.truncated_retries,
            "pipeline_execution": {
                "elapsed_seconds": round(time.monotonic() - job_state.created_at, 3),
                "transport_timeout_only": True,
                "llm_calls": self.llm.calls,
                "llm_timeouts": self.llm.timeouts,
                "truncated_retries": self.llm.truncated_retries,
                "retry_calls": reasoning_log.retry_calls,
                "deadline_exhausted": False,
                "transport_failures": self.llm.timeouts,
                "stage_durations": dict(self.llm._stage_durations),
                "graph_llm_calls": 0,
                "planned_llm_stages": [
                    "Article Understanding",
                    "Perspective Impact Mapping",
                    "Final News Synthesis"
                ],
                "max_normal_llm_calls": None,
                "llm_call_policy": "stage calls are bounded by provider transport timeouts; partitioning may increase call count",
                "character_truncation": False,
                "structured_record_truncation": False,
                "durable_checkpointing": config.enable_checkpointing,
                "checkpoint_job_id": job_state.intelligence_id,
                "completed_stages": list(job_state.completed_stages),
                "current_stage": job_state.current_stage,
            },
            "durable_execution": {
                "queue_db": str(config.queue_db_path),
                "execution_model": "durable_worker",
                "lease_seconds": config.queue_lease_seconds,
                "max_attempts": config.queue_max_attempts,
                "http_request_independent": True,
            },
            "checkpointing": {
                "job_id": job_state.intelligence_id,
                "status": job_state.status,
                "current_stage": job_state.current_stage,
                "completed_stages": list(job_state.completed_stages),
                "checkpoint_store": str(config.job_store_dir),
                "resume_available": job_state.status != "COMPLETED",
                "updated_at": job_state.updated_at,
            },
        }


# =============================================================================
# PART 6: MAIN ENGINE
# =============================================================================


class PerspectiveFirstNewsEngine:
    """
    Main production engine for ATIS News pipeline.
    
    Orchestrates all pipeline stages with:
      - Checkpointing for resumability
      - Error handling with safe fallbacks
      - Token budgeting
      - Durable execution
    """

    def __init__(
        self,
        vault: Optional[ObsidianVaultManager] = None,
        perspective: Optional[PerspectiveContext] = None,
        started_at: Optional[float] = None,
    ) -> None:
        self.vault = vault or get_vault()
        self.perspective = perspective or PerspectiveContext()
        self.llm = LLMCaller()
        self.config = get_config()
        self.stages = PipelineStages(self.vault, self.llm, self.config)
        self.started_at = started_at if started_at is not None else time.monotonic()
        self.calls = 0
        self.timeouts = 0
        self.truncated_retries = 0

    def run(
        self,
        article_text: str,
        perspective: Optional[PerspectiveContext] = None,
        reasoning_log: Optional[ReasoningLog] = None,
        state: Optional[JobState] = None,
        persistence: Optional[StatePersistenceManager] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete News pipeline.
        
        Args:
            article_text: The news article text
            perspective: The analytical perspective (country + code)
            reasoning_log: Optional log for observability
            state: Optional job state for resumption
            persistence: Optional state persistence manager
            job_id: Optional job ID for checkpointing
            
        Returns:
            Complete intelligence dashboard
        """
        config = get_config()
        
        if perspective:
            self.perspective = perspective
        
        if reasoning_log is None:
            reasoning_log = ReasoningLog()
        
        if persistence is None:
            persistence = StatePersistenceManager()
        
        if state is None:
            job_id = job_id or DurableNewsJobQueue.make_job_id(article_text, self.perspective)
            state = JobState(intelligence_id=job_id)
            state.perspective = self.perspective.as_dict()
            state.pipeline_version = "3.0.0"
        
        # Update state with current perspective if not set
        if not state.perspective:
            state.perspective = self.perspective.as_dict()
        
        def set_stage(stage: PipelineStage) -> None:
            state.current_stage = stage.value
            state.status = "IN_PROGRESS"
            if config.enable_checkpointing:
                persistence.save_state(state)

        def checkpoint(stage: PipelineStage, data: Any) -> None:
            state.mark_stage_complete(stage, data)
            state.status = "IN_PROGRESS"
            state.updated_at = time.time()
            if config.enable_checkpointing:
                persistence.save_state(state)
            logger.info("[CHECKPOINT] %s complete | job=%s", stage.value, state.intelligence_id)

        try:
            # Stage 0: Input Validation
            if state.is_completed(PipelineStage.INPUT_VALIDATION):
                article_text, self.perspective = state.get_stage_data(PipelineStage.INPUT_VALIDATION)
            else:
                set_stage(PipelineStage.INPUT_VALIDATION)
                article_text, self.perspective = self.stages.stage_0_validate_input(
                    article_text, self.perspective
                )
                checkpoint(PipelineStage.INPUT_VALIDATION, (article_text, self.perspective))
            
            reasoning_log.article_chars = len(article_text)
            
            # Stage 1: Article Understanding
            if state.is_completed(PipelineStage.ARTICLE_UNDERSTANDING):
                article = state.get_stage_data(PipelineStage.ARTICLE_UNDERSTANDING)
            else:
                set_stage(PipelineStage.ARTICLE_UNDERSTANDING)
                article = self.stages.stage_1_understand_article(article_text, self.perspective)
                checkpoint(PipelineStage.ARTICLE_UNDERSTANDING, article)
            
            reasoning_log.entities_extracted = len(_pf_safe_list(article.get("actors")))
            reasoning_log.estimated_tokens = TokenBudget.estimate_tokens(article_text)
            reasoning_log.safe_budget = self.llm.budget.provider_context_limit - self.llm.budget.safety_margin
            reasoning_log.evidence_calls = self.llm.calls
            
            # Stage 2: Perspective Ecosystem Loading
            if state.is_completed(PipelineStage.PERSPECTIVE_ECOSYSTEM_LOADING):
                stage2_data = state.get_stage_data(PipelineStage.PERSPECTIVE_ECOSYSTEM_LOADING)
                ecosystem = _pf_safe_list(stage2_data)
            else:
                set_stage(PipelineStage.PERSPECTIVE_ECOSYSTEM_LOADING)
                ecosystem = self.stages.stage_2_load_ecosystem(self.perspective)
                checkpoint(PipelineStage.PERSPECTIVE_ECOSYSTEM_LOADING, ecosystem)
            
            reasoning_log.perspective_nodes = len(ecosystem)
            reasoning_log.candidate_nodes = len(self.vault.file_map)
            
            # Stage 3: Perspective Impact Mapping
            if state.is_completed(PipelineStage.PERSPECTIVE_IMPACT_MAPPING):
                stage3_data = state.get_stage_data(PipelineStage.PERSPECTIVE_IMPACT_MAPPING)
                impact = stage3_data
            else:
                set_stage(PipelineStage.PERSPECTIVE_IMPACT_MAPPING)
                impact = self.stages.stage_3_map_impact(article, ecosystem, self.perspective)
                checkpoint(PipelineStage.PERSPECTIVE_IMPACT_MAPPING, impact)
            
            reasoning_log.impact_domains = len(_pf_safe_list(impact.get("impact_domains")))
            
            # Stage 4: Target Resolution
            if state.is_completed(PipelineStage.TARGET_RESOLUTION):
                stage4_data = state.get_stage_data(PipelineStage.TARGET_RESOLUTION)
                targets = stage4_data.get("targets", [])
                unresolved = stage4_data.get("unresolved", [])
            else:
                set_stage(PipelineStage.TARGET_RESOLUTION)
                targets, unresolved = self.stages.stage_4_resolve_targets(impact, ecosystem)
                checkpoint(PipelineStage.TARGET_RESOLUTION, {"targets": targets, "unresolved": unresolved})
            
            reasoning_log.retrieval_targets = len(targets)
            reasoning_log.unresolved_targets = len(unresolved)
            
            # Stage 5: Graph Traversal
            if state.is_completed(PipelineStage.GRAPH_TRAVERSAL):
                graph = state.get_stage_data(PipelineStage.GRAPH_TRAVERSAL)
            else:
                set_stage(PipelineStage.GRAPH_TRAVERSAL)
                graph = self.stages.stage_5_traverse_graph(targets)
                checkpoint(PipelineStage.GRAPH_TRAVERSAL, graph)
            
            reasoning_log.direct_graph_nodes = graph.get("direct_nodes", 0)
            reasoning_log.first_order_graph_nodes = graph.get("first_order_nodes", 0)
            reasoning_log.second_order_graph_nodes = graph.get("second_order_nodes", 0)
            reasoning_log.backlink_candidates = graph.get("backlink_candidates", 0)
            reasoning_log.graph_paths = len(graph.get("paths", []))
            reasoning_log.relevant_nodes = len(graph.get("nodes", []))
            reasoning_log.selected_evidence = len(graph.get("nodes", []))
            reasoning_log.reasoning_mode = (
                "perspective_first_graph_grounded"
                if graph.get("nodes") else "perspective_only_no_graph_evidence"
            )
            
            # Stage 6: Impact Analysis
            if state.is_completed(PipelineStage.IMPACT_ANALYSIS):
                stage6_data = state.get_stage_data(PipelineStage.IMPACT_ANALYSIS)
                consequences = stage6_data.get("consequences", {})
                impact_chain = _pf_safe_list(stage6_data.get("impact_chain"))
            else:
                set_stage(PipelineStage.IMPACT_ANALYSIS)
                consequences = self.stages.stage_6_analyze_impact(article, impact, graph)
                if not graph.get("nodes"):
                    consequences.setdefault("gaps", []).append({
                        "gap": "No evidence-backed relationship was found between the selected perspective targets and the current database graph.",
                        "status": "RESEARCH_REQUIRED",
                        "related_nodes": [n.get("node_id", "") for n in targets],
                    })
                impact_chain = self.stages.build_impact_chain(article, impact, graph, consequences)
                checkpoint(PipelineStage.IMPACT_ANALYSIS, {"consequences": consequences, "impact_chain": impact_chain})
            
            reasoning_log.gaps = len(_pf_safe_list(consequences.get("gaps")))
            
            # Stage 7: Final Synthesis
            if state.is_completed(PipelineStage.FINAL_SYNTHESIS):
                dashboard = state.get_stage_data(PipelineStage.FINAL_SYNTHESIS)
            else:
                set_stage(PipelineStage.FINAL_SYNTHESIS)
                try:
                    dashboard = self.stages.stage_7_final_synthesis(
                        article, impact, graph, consequences, impact_chain
                    )
                except (NewsPipelineDeadline, LLMTokenLimitError, NewsPipelineContextOverflow, RuntimeError) as exc:
                    logger.warning("[FINAL] %s", exc)
                    dashboard = self._fallback_dashboard(
                        article, impact, graph, consequences, impact_chain, str(exc)
                    )
                
                if dashboard.get("status") != "partial":
                    checkpoint(PipelineStage.FINAL_SYNTHESIS, dashboard)
            
            # Stage 8: Validation & Grounding
            if state.is_completed(PipelineStage.VALIDATION_GROUNDING):
                dashboard = state.get_stage_data(PipelineStage.VALIDATION_GROUNDING)
            else:
                set_stage(PipelineStage.VALIDATION_GROUNDING)
                dashboard = self.stages.stage_8_validate_and_ground(
                    dashboard, graph, ecosystem, article, impact, consequences, self.perspective
                )
                checkpoint(PipelineStage.VALIDATION_GROUNDING, dashboard)
            
            reasoning_log.opportunities = len(_pf_safe_list(dashboard.get("opportunities")))
            reasoning_log.validated_opportunities = sum(
                1 for o in _pf_safe_list(dashboard.get("opportunities"))
                if o.get("status") == "SUPPORTED"
            )
            reasoning_log.research_required = sum(
                1 for o in _pf_safe_list(dashboard.get("opportunities"))
                if o.get("status") == "RESEARCH_REQUIRED"
            ) + len(_pf_safe_list(dashboard.get("gaps")))
            
            # Stage 9: Output Assembly
            if state.is_completed(PipelineStage.OUTPUT_ASSEMBLY):
                dashboard = state.get_stage_data(PipelineStage.OUTPUT_ASSEMBLY)
            else:
                set_stage(PipelineStage.OUTPUT_ASSEMBLY)
                dashboard = self.stages.stage_9_assemble_output(
                    dashboard, article, impact, graph, consequences, 
                    impact_chain, ecosystem, unresolved, self.perspective,
                    reasoning_log, state
                )
                checkpoint(PipelineStage.OUTPUT_ASSEMBLY, dashboard)
            
            # Finalize
            self._finalize_dashboard(dashboard, self.perspective, reasoning_log, state)
            
            # Mark complete
            state.status = "COMPLETED"
            state.current_stage = PipelineStage.COMPLETE.value
            if PipelineStage.COMPLETE.value not in state.completed_stages:
                state.completed_stages.append(PipelineStage.COMPLETE.value)
            state.stage_data[PipelineStage.OUTPUT_ASSEMBLY.value] = dashboard
            if config.enable_checkpointing:
                persistence.save_state(state)
            
            # Update metrics
            reasoning_log.total_llm_calls = self.llm.calls
            reasoning_log.retry_calls = self.llm.truncated_retries
            reasoning_log.final_call = 1
            
            logger.info("\n%s", reasoning_log.log_tree())
            
            return dashboard
            
        except Exception as exc:
            logger.exception("[NEWS] Pipeline failed: %s", exc)
            state.status = "PARTIAL"
            state.current_stage = state.current_stage or PipelineStage.INPUT_VALIDATION.value
            state.add_error(str(exc))
            if config.enable_checkpointing:
                persistence.save_state(state)
            
            # Build partial result
            article_state = state.get_stage_data(PipelineStage.ARTICLE_UNDERSTANDING) if state.is_completed(PipelineStage.ARTICLE_UNDERSTANDING) else {"event": {}, "facts": [], "meaning": [], "actors": [], "uncertainties": []}
            stage2 = state.get_stage_data(PipelineStage.PERSPECTIVE_ECOSYSTEM_LOADING) if state.is_completed(PipelineStage.PERSPECTIVE_ECOSYSTEM_LOADING) else {}
            stage3 = state.get_stage_data(PipelineStage.PERSPECTIVE_IMPACT_MAPPING) if state.is_completed(PipelineStage.PERSPECTIVE_IMPACT_MAPPING) else {}
            stage4 = state.get_stage_data(PipelineStage.TARGET_RESOLUTION) if state.is_completed(PipelineStage.TARGET_RESOLUTION) else {}
            stage5 = state.get_stage_data(PipelineStage.GRAPH_TRAVERSAL) if state.is_completed(PipelineStage.GRAPH_TRAVERSAL) else {"nodes": [], "edges": [], "paths": []}
            stage6 = state.get_stage_data(PipelineStage.IMPACT_ANALYSIS) if state.is_completed(PipelineStage.IMPACT_ANALYSIS) else {"consequences": [], "gaps": []}
            
            impact = stage3 if isinstance(stage3, dict) else {"impact_domains": []}
            ecosystem = _pf_safe_list(stage2) if isinstance(stage2, list) else []
            targets = _pf_safe_list(stage4.get("targets")) if isinstance(stage4, dict) else []
            unresolved = _pf_safe_list(stage4.get("unresolved")) if isinstance(stage4, dict) else []
            graph = stage5 if isinstance(stage5, dict) else {"nodes": [], "edges": [], "paths": []}
            consequences = stage6 if isinstance(stage6, dict) else {"consequences": [], "gaps": []}
            impact_chain = _pf_safe_list(stage6.get("impact_chain")) if isinstance(stage6, dict) else []
            
            dashboard = self._fallback_dashboard(
                article_state, impact, graph, consequences, impact_chain, str(exc)
            )
            dashboard["research_required"] = (
                unresolved
                + _pf_safe_list(dashboard.get("gaps"))
                + _pf_safe_list(graph.get("research_required"))
            )
            dashboard["status"] = "partial"
            dashboard["partial"] = True
            dashboard["pipeline_checkpoint"] = {
                "job_id": state.intelligence_id,
                "completed_stages": list(state.completed_stages),
                "current_stage": state.current_stage,
                "resume_available": True,
                "errors": list(state.error_log),
            }
            
            self._finalize_dashboard(dashboard, self.perspective, reasoning_log, state)
            
            return dashboard

    def _fallback_dashboard(
        self,
        article: Dict[str, Any],
        impact: Dict[str, Any],
        graph: Dict[str, Any],
        consequences: Dict[str, Any],
        impact_chain: List[Dict[str, Any]],
        reason: str,
    ) -> Dict[str, Any]:
        """Create a truthful partial dashboard when pipeline fails."""
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

    def _finalize_dashboard(
        self,
        dashboard: Dict[str, Any],
        perspective: PerspectiveContext,
        reasoning_log: ReasoningLog,
        state: JobState,
    ) -> None:
        """Finalize dashboard with metadata and persistence."""
        config = get_config()
        
        dashboard = dict(dashboard or {})
        dashboard.setdefault("intelligence_id", 
            f"ATIS-INT-{hashlib.sha256(str(dashboard.get('trigger_event','')).encode()).hexdigest()[:12].upper()}")
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
        
        for key in ("opportunities", "risks", "gaps", "entities", "findings", "facts", 
                   "meaning", "impact_domains", "impact_chain", "structured_intelligence"):
            dashboard.setdefault(key, [])
        
        # Sanitize for JSON
        dashboard = sanitize_for_json(dashboard)
        
        # Persist to file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_job = re.sub(r"[^A-Za-z0-9_-]", "_", str(state.intelligence_id))[-40:]
        output_path = config.dashboards_dir / f"atis_dashboard_{timestamp}_{safe_job}.json"
        
        try:
            output_path.write_text(
                json.dumps(dashboard, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8"
            )
            dashboard["pipeline_metadata"]["dashboard_path"] = str(output_path)
            logger.info("[FINAL] Dashboard persisted: %s", output_path.resolve())
        except Exception as exc:
            logger.error("[FINAL] Failed to persist dashboard: %s", exc)


# =============================================================================
# PART 7: ENTRY POINTS
# =============================================================================


def process_article_pipeline(
    article_path: str,
    perspective: Any | None = None,
    job_id: str | None = None,
) -> Dict[str, Any]:
    """
    Process an article from a file path.
    
    This is the primary entry point for file-based article processing.
    
    Args:
        article_path: Path to the article file
        perspective: Optional perspective (country + code)
        job_id: Optional job ID for idempotency
        
    Returns:
        Complete intelligence dashboard
        
    Raises:
        FileNotFoundError: If article file doesn't exist
        NewsValidationError: If inputs are invalid
    """
    article_file = Path(article_path)
    if not article_file.exists():
        raise FileNotFoundError(f"Article not found: {article_path}")
    
    article_text = article_file.read_text(encoding="utf-8")
    
    config = get_config()
    if os.getenv("ATIS_NEWS_EXECUTION_MODE", "sync").strip().lower() == "async":
        queue = DurableNewsJobQueue()
        perspective_ctx = perspective or PerspectiveContext()
        if isinstance(perspective, dict):
            perspective_ctx = PerspectiveContext.from_payload(perspective)
        receipt = queue.submit(article_text, perspective_ctx, "file_upload", job_id)
        return {
            "status": receipt["status"].lower(),
            "job_id": receipt["job_id"],
            "resume_available": True,
            "execution_model": "durable_worker",
        }
    
    return _run_pipeline(article_text, perspective, "file_upload", job_id)


def run_news_pipeline(
    article_text: str,
    perspective: Any | None = None,
    job_id: str | None = None,
) -> Dict[str, Any]:
    """
    Process an article from text string.
    
    This is the primary entry point for text-based article processing.
    Backward compatible with existing callers.
    
    Args:
        article_text: The news article text
        perspective: Optional perspective (country + code)
        job_id: Optional job ID for idempotency
        
    Returns:
        Complete intelligence dashboard
        
    Raises:
        NewsValidationError: If inputs are invalid
    """
    config = get_config()
    if os.getenv("ATIS_NEWS_EXECUTION_MODE", "sync").strip().lower() == "async":
        queue = DurableNewsJobQueue()
        perspective_ctx = perspective or PerspectiveContext()
        if isinstance(perspective, dict):
            perspective_ctx = PerspectiveContext.from_payload(perspective)
        receipt = queue.submit(article_text, perspective_ctx, "web_upload", job_id)
        return {
            "status": receipt["status"].lower(),
            "job_id": receipt["job_id"],
            "resume_available": True,
            "execution_model": "durable_worker",
        }
    
    return _run_pipeline(article_text, perspective, "web_upload", job_id)


def _run_pipeline(
    article_text: str,
    perspective: Any | None = None,
    source_label: str = "web_upload",
    job_id: str | None = None,
) -> Dict[str, Any]:
    """
    Internal pipeline execution.
    
    Args:
        article_text: The news article text
        perspective: Optional perspective
        source_label: Label for the source (web_upload, file_upload)
        job_id: Optional job ID
        
    Returns:
        Complete intelligence dashboard
    """
    started_at = time.monotonic()
    config = get_config()
    
    # Normalize perspective
    perspective_ctx = perspective or PerspectiveContext()
    if isinstance(perspective, dict):
        perspective_ctx = PerspectiveContext.from_payload(perspective)
    
    reasoning_log = ReasoningLog()
    
    logger.info("=" * 78)
    logger.info(
        "ATIS NEWS v3.0.0 | PERSPECTIVE-FIRST | %s (%s)",
        perspective_ctx.country, perspective_ctx.country_code
    )
    logger.info("=" * 78)
    
    # Initialize durable state
    persistence = StatePersistenceManager()
    if not job_id:
        job_id = DurableNewsJobQueue.make_job_id(article_text, perspective_ctx)
    
    state = persistence.load_state(job_id)
    if state is not None and state.stage_data.get("raw_article") != article_text.strip():
        logger.warning(
            "[STATE] Checkpoint %s belongs to different article input; starting fresh",
            job_id
        )
        state = None
    if state is not None and state.pipeline_version != "3.0.0":
        logger.info(
            "[STATE] Checkpoint %s was created by older version; starting fresh",
            job_id
        )
        state = None
    if state is not None:
        saved_perspective = state.stage_data.get("perspective", {})
        if isinstance(saved_perspective, dict):
            saved_country = _pf_norm(saved_perspective.get("country", ""))
            saved_code = _pf_norm(saved_perspective.get("country_code", ""))
            if (saved_country != _pf_norm(perspective_ctx.country) or 
                saved_code != _pf_norm(perspective_ctx.country_code)):
                logger.warning(
                    "[STATE] Checkpoint %s belongs to different perspective; starting fresh",
                    job_id
                )
                state = None
    
    if state is None:
        state = JobState(intelligence_id=job_id)
        state.stage_data["raw_article"] = article_text.strip()
        state.stage_data["perspective"] = perspective_ctx.as_dict()
        state.stage_data["_pipeline_version"] = "3.0.0"
        state.pipeline_version = "3.0.0"
        persistence.save_state(state)
    
    # Get vault (cached)
    vault = get_vault()
    
    # Create and run engine
    engine = PerspectiveFirstNewsEngine(
        vault=vault,
        perspective=perspective_ctx,
        started_at=started_at,
    )
    
    try:
        dashboard = engine.run(
            article_text,
            perspective=perspective_ctx,
            reasoning_log=reasoning_log,
            state=state,
            persistence=persistence,
            job_id=job_id,
        )
    except Exception as exc:
        logger.exception("[NEWS] Pipeline failed: %s", exc)
        raise
    
    # Add cross-border bridges to metadata
    event = article.get("event", {}) if isinstance(dashboard.get("event"), dict) else dashboard.get("event", {})
    source_country = str(dashboard.get("source_country") or event.get("source_country", "")).strip()
    event_country = str(dashboard.get("event_country") or event.get("event_country", "")).strip()
    bridge_source = source_country or event_country
    
    if bridge_source and _pf_norm(bridge_source) != _pf_norm(perspective_ctx.country):
        try:
            bridges = vault.get_cross_border_bridges(perspective_ctx, bridge_source)
            dashboard["cross_border_analysis"]["cross_border_bridges"] = bridges
            dashboard["cross_border_analysis"]["cross_border_bridges_count"] = len(bridges)
        except Exception as exc:
            logger.warning("[BRIDGE] deterministic bridge lookup failed: %s", exc)
    
    # Log completion
    elapsed = time.monotonic() - started_at
    logger.info(
        "[EXECUTION] elapsed=%.2fs | llm_calls=%d | timeouts=%d | truncation_retries=%d",
        elapsed, engine.llm.calls, engine.llm.timeouts, engine.llm.truncated_retries
    )
    logger.info("=" * 78)
    logger.info("ATIS NEWS PIPELINE COMPLETE")
    logger.info("=" * 78)
    
    return dashboard


def submit_news_job(
    article_text: str,
    perspective: Any | None = None,
    source_label: str = "web_upload",
    job_id: str | None = None,
) -> Dict[str, Any]:
    """
    Submit a News job to the durable queue.
    
    The job will be processed asynchronously by a worker.
    
    Args:
        article_text: The news article text
        perspective: Optional perspective (country + code)
        source_label: Label for the source
        job_id: Optional job ID for idempotency
        
    Returns:
        Job receipt with status and job_id
    """
    perspective_ctx = perspective or PerspectiveContext()
    if isinstance(perspective, dict):
        perspective_ctx = PerspectiveContext.from_payload(perspective)
    
    queue = DurableNewsJobQueue()
    return queue.submit(article_text, perspective_ctx, source_label, job_id)


def get_news_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a News job.
    
    Args:
        job_id: The job ID to check
        
    Returns:
        Job status with checkpoint information if available
        
    Raises:
        KeyError: If job not found
    """
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
            result["result"] = state.stage_data.get(PipelineStage.OUTPUT_ASSEMBLY.value)
    
    return result


def run_news_worker_once(worker_id: str | None = None) -> Optional[Dict[str, Any]]:
    """
    Claim and execute one job from the durable queue.
    
    Args:
        worker_id: Optional worker identifier
        
    Returns:
        Job result or None if no jobs available
    """
    queue = DurableNewsJobQueue()
    claimed = queue.claim(worker_id)
    if claimed is None:
        return None
    
    owner = claimed["worker_id"]
    stop_heartbeat = threading.Event()
    lease_lost = threading.Event()
    
    def heartbeat() -> None:
        config = get_config()
        interval = max(2.0, config.queue_lease_seconds / 3)
        while not stop_heartbeat.is_set():
            time.sleep(interval)
            if stop_heartbeat.is_set():
                break
            try:
                if not queue.renew(claimed["job_id"], owner):
                    lease_lost.set()
                    break
            except Exception:
                lease_lost.set()
                break
    
    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"atis-news-lease-{claimed['job_id'][-12:]}",
        daemon=True,
    )
    heartbeat_thread.start()
    
    try:
        perspective_payload = claimed["perspective"] if isinstance(claimed["perspective"], dict) else {}
        perspective_ctx = PerspectiveContext(
            country=str(perspective_payload.get("country") or "Zimbabwe"),
            country_code=str(perspective_payload.get("country_code") or "ZW"),
        )
        
        dashboard = _run_pipeline(
            claimed["article_text"],
            perspective=perspective_ctx,
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
    """
    Run the durable worker loop indefinitely.
    
    This should be run as a separate process/service.
    
    Args:
        worker_id: Optional worker identifier
    """
    resolved_worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
    logger.info("[WORKER] ATIS News durable worker started: %s", resolved_worker_id)
    
    while True:
        job = run_news_worker_once(resolved_worker_id)
        if job is None:
            config = get_config()
            time.sleep(config.queue_poll_seconds)


# =============================================================================
# PART 8: CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATIS News - Perspective-First, Graph-Grounded Intelligence Engine"
    )
    subparsers = parser.add_subparsers(dest="command")
    
    article_parser = subparsers.add_parser(
        "process",
        help="Process an article synchronously (backward-compatible mode)."
    )
    article_parser.add_argument(
        "article_path",
        metavar="ARTICLE",
        help="Path to the plain-text news article to process."
    )
    
    worker_parser = subparsers.add_parser(
        "worker",
        help="Run the durable News worker."
    )
    worker_parser.add_argument("--worker-id", default=None)
    
    status_parser = subparsers.add_parser(
        "status",
        help="Get durable News job status."
    )
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
