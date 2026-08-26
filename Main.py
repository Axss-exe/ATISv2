#!/usr/bin/env python3
"""
ATIS Intelligence API — Render Web Service Entry Point

Features:
  - CORS configured for Vercel frontend
  - Vault entity listing with recursive subfolder discovery & full profile content
  - SINGLE ENTITY LOOKUP with slug-based IDs, backlink resolution & related vault traversal
  - 60-second hard timeout on all LLM pipelines
  - Global request lock (prevents concurrent pipeline runs)
  - Emergency kill endpoint
  - Perspective-First Deterministic Architecture v2.2
  - Analysis fingerprinting and knowledge-state-aware caching
  - Investigation / Query String backend layer
  - INVESTIGATION REPORT GENERATION (payload-based, no local persistence)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ATIS pipeline modules
from ATIS_News import run_news_pipeline
from ATIS_Execute import run_execute_pipeline
from ATIS_Query import run_query_pipeline, ObsidianVaultManager as QueryVaultManager
from atis_context import (
    PerspectiveContext,
    KnowledgeState,
    AnalysisCache,
    ANALYSIS_VERSION,
    SCHEMA_VERSION,
)

# Investigation layer
from investigation_manager import (
    create_investigation,
    add_query_to_investigation,
    get_investigation,
    list_investigations,
    generate_investigation_report,
    INVESTIGATIONS_DIR,
)

# NEW: Payload-based report generation (no local persistence)
from report_generator import generate_investigation_report as generate_report_from_payload

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ATIS_API")

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="ATIS Intelligence API",
    description="Africa-wide Intelligence System with Perspective-First Deterministic Architecture",
    version="2.2.0-perspective-deterministic",
)

# =============================================================================
# CORS — specific origins (NOT wildcard)
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://av2-fkq2sfy2c-tmakiriyado1-4301s-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global state
# =============================================================================
_vault_path = Path(os.getenv("VAULT_PATH", "./vault"))

# Cache vault indexes at startup (skip re-indexing on every request)
_query_vault: QueryVaultManager | None = None

def _get_query_vault() -> QueryVaultManager:
    global _query_vault
    if _query_vault is None:
        logger.info("Building query vault index...")
        _query_vault = QueryVaultManager(_vault_path)
        _query_vault.build_index()
        logger.info("Query vault index complete: %d nodes", _query_vault.indexed_count)
    return _query_vault

# Request lock — prevents concurrent LLM pipeline runs
_pipeline_lock = threading.Lock()
_pipeline_in_progress = False

# Simple in-memory cache for query results
_query_cache: Dict[str, Any] = {}
_analysis_cache = AnalysisCache()

def _query_cache_key(question: str | None, perspective: PerspectiveContext, knowledge_state_hash: str = "") -> str:
    """Perspective-aware AND knowledge-state-aware cache key."""
    material = f"{perspective.country_code}:{question or 'full_scan'}:{knowledge_state_hash}"
    return hashlib.sha256(material.encode()).hexdigest()[:16]

# Entity cache — rebuilds only when files change
_entities_cache: Dict[str, Any] | None = None
_entities_cache_mtime: float = 0.0

# =============================================================================
# Request models
# =============================================================================
class PerspectiveModel(BaseModel):
    country: str = "Zimbabwe"
    country_code: str = "ZW"

class NewsRequest(BaseModel):
    article_text: str
    perspective_country: str | None = None
    perspective_country_code: str | None = None

class ExecuteRequest(BaseModel):
    dashboard_json: dict
    opportunity_id: str
    perspective_country: str | None = None
    perspective_country_code: str | None = None

class QueryRequest(BaseModel):
    question: str | None = None
    perspective_country: str | None = None
    perspective_country_code: str | None = None

class EntityRequest(BaseModel):
    entity_name: str
    perspective: Optional[PerspectiveModel] = None

# --- Investigation request models ---
class CreateInvestigationRequest(BaseModel):
    question: str
    perspective_country: str | None = None
    perspective_country_code: str | None = None

class AddQueryRequest(BaseModel):
    question: str
    parent_query_id: str | None = None

# NEW: Payload-based report generation request model
class GenerateReportRequest(BaseModel):
    investigation: dict

# =============================================================================
# Helper: Run pipeline with timeout and lock
# =============================================================================
def _acquire_pipeline_lock() -> bool:
    """Try to acquire the global pipeline lock. Returns False if busy."""
    global _pipeline_in_progress
    if _pipeline_in_progress:
        return False
    _pipeline_in_progress = True
    return True

def _release_pipeline_lock():
    """Release the global pipeline lock."""
    global _pipeline_in_progress
    _pipeline_in_progress = False

async def _run_with_timeout(func, *args, timeout: float = 60.0, **kwargs):
    """
    Run a synchronous function in a thread pool with a hard timeout.
    Kills the function if it exceeds the timeout.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Pipeline timed out after {timeout} seconds")

# =============================================================================
# Slug & decode utilities (fixes 404s and double-encoding)
# =============================================================================
def slugify(text: str) -> str:
    """Create a stable, URL-safe slug from any vault filename or title."""
    if not text:
        return ""
    text = urllib.parse.unquote(text)
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
    text = re.sub(r"-+", "-", text)
    return text.strip("-")

def decode_entity_id(raw: str) -> str:
    """Recursively URL-decode until stable (handles double-encoded URLs)."""
    prev = ""
    decoded = raw
    while decoded != prev:
        prev = decoded
        decoded = urllib.parse.unquote(prev)
    return decoded

def _infer_entity_type(rel_path: Path) -> str:
    """
    Guess entity category from folder structure.
    """
    parts = [p.lower() for p in rel_path.parts]
    type_hints = {
        "businesses": "business",
        "companies": "business",
        "commodities": "commodity",
        "minerals": "commodity",
        "government": "government",
        "agencies": "government",
        "infrastructure": "infrastructure",
        "people": "person",
        "contacts": "person",
        "region": "region",
    }
    for part in parts:
        if part in type_hints:
            return type_hints[part]
    return "unknown"

# =============================================================================
# Entity Discovery Helpers
# =============================================================================

def _discover_entity_directories(vault: Path) -> List[Path]:
    """
    Auto-discover directories containing entity profiles.
    Scans country folders and their category subfolders.
    """
    discovered: List[Path] = []
    seen: Set[str] = set()

    if not vault.exists():
        logger.error("Vault path does not exist: %s", vault)
        return discovered

    country_dirs = [d for d in vault.iterdir() if d.is_dir() and not d.name.startswith('.')]

    for country_dir in country_dirs:
        if any(country_dir.glob("*.md")):
            key = str(country_dir.resolve())
            if key not in seen:
                seen.add(key)
                discovered.append(country_dir)

        for category_dir in country_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            md_count = len(list(category_dir.rglob("*.md")))
            if md_count > 0:
                key = str(category_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    discovered.append(category_dir)

    if not discovered:
        for root, dirs, files in os.walk(vault):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            root_path = Path(root)
            if any(f.endswith('.md') for f in files):
                key = str(root_path.resolve())
                if key not in seen:
                    seen.add(key)
                    discovered.append(root_path)

    return discovered

def _load_entities_from_dirs(dirs: List[Path]) -> List[Dict[str, Any]]:
    """Load entity profiles from discovered directories."""
    entities = []
    seen_ids = set()

    for directory in dirs:
        for f in sorted(directory.rglob("*.md")):
            if not f.is_file():
                continue
            if f.stem in seen_ids:
                continue
            seen_ids.add(f.stem)

            try:
                content = f.read_text(encoding="utf-8")
                rel = f.relative_to(_vault_path)
                entities.append({
                    "id": f.stem,
                    "slug": slugify(f.stem),
                    "name": f.stem.replace("_", " ").replace("-", " "),
                    "filename": f.name,
                    "path": str(rel),
                    "content": content,
                    "size_bytes": f.stat().st_size,
                })
            except Exception as exc:
                logger.warning("Could not read %s: %s", f.name, exc)
                continue

    return entities

def _resolve_related_entities(node, vault: QueryVaultManager) -> List[Dict[str, Any]]:
    """Resolve outbound links and backlinks into traversable entity profiles."""
    related: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for link in getattr(node, "outbound_links", []):
        if link in vault.nodes:
            target = vault.nodes[link]
            if target.uid in seen:
                continue
            seen.add(target.uid)
            try:
                rel_path = target.absolute_path.relative_to(_vault_path)
            except ValueError:
                rel_path = Path(target.absolute_path.name)
            related.append({
                "slug": slugify(target.uid),
                "name": getattr(target, "front_matter", {}).get("title") or target.stem,
                "entity_type": _infer_entity_type(rel_path),
                "relation_type": "outbound",
                "summary": getattr(target, "summary", "")[:150],
            })

    for back in getattr(node, "backlink_uids", []):
        if back in vault.nodes:
            source = vault.nodes[back]
            if source.uid in seen:
                continue
            seen.add(source.uid)
            try:
                rel_path = source.absolute_path.relative_to(_vault_path)
            except ValueError:
                rel_path = Path(source.absolute_path.name)
            related.append({
                "slug": slugify(source.uid),
                "name": getattr(source, "front_matter", {}).get("title") or source.stem,
                "entity_type": _infer_entity_type(rel_path),
                "relation_type": "backlink",
                "summary": getattr(source, "summary", "")[:150],
            })

    return related

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check with detailed system status."""
    ks = KnowledgeState(vault_path=_vault_path)
    ks.compute()
    return {
        "status": "ok",
        "service": "ATIS API",
        "version": "2.2.0-perspective-deterministic",
        "pipeline_busy": _pipeline_in_progress,
        "cached_queries": len(_query_cache),
        "cached_entities": _entities_cache is not None,
        "entity_count": _entities_cache["count"] if _entities_cache else 0,
        "knowledge_state": ks.as_dict(),
    }

# -----------------------------------------------------------------------------
# Entity listing (lightweight, no LLM, cached, recursive subfolder support)
# -----------------------------------------------------------------------------
@app.get("/api/entities")
async def list_entities():
    """
    Returns all entity profiles from the vault.
    Auto-discovers directories and recursively searches subfolders for .md files.
    Cached — rebuilds only on file changes.
    """
    global _entities_cache, _entities_cache_mtime

    all_md_files = list(_vault_path.rglob("*.md"))
    current_mtime = max(
        (f.stat().st_mtime for f in all_md_files),
        default=0.0
    )

    if _entities_cache is not None and current_mtime <= _entities_cache_mtime:
        logger.info("Serving entities from cache (%d profiles)", _entities_cache["count"])
        return _entities_cache

    start = time.time()
    discovered_dirs = _discover_entity_directories(_vault_path)

    if not discovered_dirs:
        logger.error("No entity directories discovered in vault: %s", _vault_path)
        raise HTTPException(
            status_code=404,
            detail="No entity profiles found in vault. Checked paths and scanned subdirectories."
        )

    entities = _load_entities_from_dirs(discovered_dirs)
    elapsed = time.time() - start

    logger.info("Entity cache built in %.3fs — %d profiles loaded from %d directories",
                elapsed, len(entities), len(discovered_dirs))

    _entities_cache = {
        "status": "success",
        "count": len(entities),
        "directories": [str(d.relative_to(_vault_path)) for d in discovered_dirs],
        "entities": entities,
    }
    _entities_cache_mtime = current_mtime

    return _entities_cache

# -----------------------------------------------------------------------------
# Single entity profile with backlink & outbound resolution
# -----------------------------------------------------------------------------
@app.get("/api/entity/{entity_id}")
async def get_entity(entity_id: str):
    """
    Retrieve a single entity profile by ID or slug.
    Resolves outbound links and backlinks across the entire vault.
    """
    clean_id = decode_entity_id(entity_id)
    target_slug = slugify(clean_id)
    logger.info("Entity lookup | raw=%s | decoded=%s | slug=%s", entity_id, clean_id, target_slug)

    if _entities_cache:
        for ent in _entities_cache.get("entities", []):
            if ent.get("slug") == target_slug or ent["id"] == clean_id:
                result = dict(ent)
                try:
                    vault = _get_query_vault()
                    node = vault.nodes.get(ent["id"])
                    if node:
                        result["front_matter"] = getattr(node, "front_matter", {})
                        result["summary"] = getattr(node, "summary", "")
                        result["outbound_links"] = getattr(node, "outbound_links", [])
                        result["backlink_uids"] = getattr(node, "backlink_uids", [])
                        result["related_entities"] = _resolve_related_entities(node, vault)
                    else:
                        result["outbound_links"] = []
                        result["backlink_uids"] = []
                        result["related_entities"] = []
                except Exception as exc:
                    logger.warning("Failed to enrich entity %s: %s", ent["id"], exc)
                    result["outbound_links"] = []
                    result["backlink_uids"] = []
                    result["related_entities"] = []
                return result

    try:
        vault = _get_query_vault()
        node = None

        if clean_id in vault.nodes:
            node = vault.nodes[clean_id]
        else:
            for stem, n in vault.nodes.items():
                if slugify(stem) == target_slug:
                    node = n
                    break

        if not node:
            raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")

        rel_path = node.absolute_path.relative_to(_vault_path)

        return {
            "id": node.uid,
            "slug": slugify(node.uid),
            "name": getattr(node, "front_matter", {}).get("title") or node.stem,
            "filename": node.absolute_path.name,
            "path": str(rel_path),
            "content": getattr(node, "raw_content", ""),
            "size_bytes": node.absolute_path.stat().st_size,
            "front_matter": getattr(node, "front_matter", {}),
            "summary": getattr(node, "summary", ""),
            "entity_type": _infer_entity_type(rel_path),
            "outbound_links": getattr(node, "outbound_links", []),
            "backlink_uids": getattr(node, "backlink_uids", []),
            "related_entities": _resolve_related_entities(node, vault),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Entity lookup failed: %s", exc)
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")

# -----------------------------------------------------------------------------
# Search across all vault nodes
# -----------------------------------------------------------------------------
@app.get("/api/search")
async def search_entities(q: str):
    """Full-text search across the entire vault index."""
    vault = _get_query_vault()
    q_clean = q.lower()
    results = []
    seen_slugs: Set[str] = set()

    for stem, node in vault.nodes.items():
        haystack = f"{stem} {getattr(node, 'summary', '')} {getattr(node, 'body', '')}".lower()
        if q_clean in haystack:
            slug = slugify(stem)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            try:
                rel_path = node.absolute_path.relative_to(_vault_path)
            except ValueError:
                rel_path = Path(node.absolute_path.name)

            results.append({
                "id": stem,
                "slug": slug,
                "name": getattr(node, "front_matter", {}).get("title") or stem,
                "entity_type": _infer_entity_type(rel_path),
                "summary": getattr(node, "summary", "")[:200],
            })

    return {"query": q, "count": len(results), "results": results}

# -----------------------------------------------------------------------------
# News pipeline
# -----------------------------------------------------------------------------
@app.post("/api/news")
async def news_endpoint(request: NewsRequest):
    if not _acquire_pipeline_lock():
        return {
            "status": "busy",
            "detail": "Another pipeline is running. Please wait and retry."
        }

    try:
        start = time.time()
        perspective = PerspectiveContext.from_values(
            request.perspective_country, request.perspective_country_code
        )
        result = await _run_with_timeout(
            run_news_pipeline,
            request.article_text,
            perspective,
            timeout=60.0
        )
        elapsed = time.time() - start
        logger.info("News pipeline completed in %.1fs", elapsed)
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 1),
            "analysis_version": ANALYSIS_VERSION,
            "schema_version": SCHEMA_VERSION,
            "data": result
        }
    except RuntimeError as exc:
        logger.error("News pipeline failed: %s", exc)
        return {"status": "error", "detail": str(exc)}
    except Exception as exc:
        logger.error("News pipeline unexpected error: %s", exc)
        return {"status": "error", "detail": f"Pipeline failed: {str(exc)}"}
    finally:
        _release_pipeline_lock()

# -----------------------------------------------------------------------------
# Execute pipeline
# -----------------------------------------------------------------------------
@app.post("/api/execute")
async def execute_endpoint(request: ExecuteRequest):
    if not _acquire_pipeline_lock():
        return {
            "status": "busy",
            "detail": "Another pipeline is running. Please wait and retry."
        }

    try:
        start = time.time()
        perspective = PerspectiveContext.from_values(
            request.perspective_country, request.perspective_country_code
        ) if request.perspective_country or request.perspective_country_code else None

        result = await _run_with_timeout(
            run_execute_pipeline,
            request.dashboard_json,
            request.opportunity_id,
            perspective,
            timeout=60.0
        )
        elapsed = time.time() - start
        logger.info("Execute pipeline completed in %.1fs", elapsed)
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 1),
            "analysis_version": ANALYSIS_VERSION,
            "schema_version": SCHEMA_VERSION,
            "data": result
        }
    except RuntimeError as exc:
        logger.error("Execute pipeline failed: %s", exc)
        return {"status": "error", "detail": str(exc)}
    except Exception as exc:
        logger.error("Execute pipeline unexpected error: %s", exc)
        return {"status": "error", "detail": f"Pipeline failed: {str(exc)}"}
    finally:
        _release_pipeline_lock()

# -----------------------------------------------------------------------------
# Query pipeline (with knowledge-state-aware caching)
# -----------------------------------------------------------------------------
@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    perspective = PerspectiveContext.from_values(
        request.perspective_country, request.perspective_country_code
    )

    # Compute knowledge state for cache key
    knowledge_state = KnowledgeState(vault_root=_vault_path)
    knowledge_state.compute()

    cache_key = _query_cache_key(request.question, perspective, knowledge_state.knowledge_state_hash)

    if cache_key in _query_cache:
        logger.info("Query cache HIT: %s", cache_key)
        cached_result = _query_cache[cache_key]
        cached_result["cache_hit"] = True
        cached_result["cache_key"] = cache_key
        cached_result["knowledge_state"] = knowledge_state.as_dict()
        return {
            "status": "success",
            "cached": True,
            "analysis_version": ANALYSIS_VERSION,
            "schema_version": SCHEMA_VERSION,
            "data": cached_result
        }

    if not _acquire_pipeline_lock():
        return {
            "status": "busy",
            "detail": "Another pipeline is running. Please wait and retry."
        }

    try:
        start = time.time()
        result = await _run_with_timeout(
            run_query_pipeline,
            request.question,
            _vault_path,
            perspective,
            timeout=60.0
        )
        elapsed = time.time() - start
        logger.info("Query pipeline completed in %.1fs", elapsed)

        result["cache_hit"] = False
        result["cache_key"] = cache_key
        result["knowledge_state"] = knowledge_state.as_dict()
        result["analysis_version"] = ANALYSIS_VERSION
        result["schema_version"] = SCHEMA_VERSION

        _query_cache[cache_key] = result

        return {
            "status": "success",
            "cached": False,
            "elapsed_seconds": round(elapsed, 1),
            "analysis_version": ANALYSIS_VERSION,
            "schema_version": SCHEMA_VERSION,
            "data": result
        }
    except RuntimeError as exc:
        logger.error("Query pipeline failed: %s", exc)
        return {"status": "error", "detail": str(exc)}
    except Exception as exc:
        logger.error("Query pipeline unexpected error: %s", exc)
        return {"status": "error", "detail": f"Pipeline failed: {str(exc)}"}
    finally:
        _release_pipeline_lock()

# =============================================================================
# Investigation endpoints
# =============================================================================

@app.post("/api/investigations")
async def create_investigation_endpoint(request: CreateInvestigationRequest):
    """
    Create a new investigation and execute the initial query.
    The initial question becomes Query #1.
    """
    if not _acquire_pipeline_lock():
        return {
            "status": "busy",
            "detail": "Another pipeline is running. Please wait and retry."
        }

    try:
        start = time.time()
        investigation = await _run_with_timeout(
            create_investigation,
            request.question,
            request.perspective_country,
            request.perspective_country_code,
            _vault_path,
            timeout=60.0,
        )
        elapsed = time.time() - start
        logger.info("Investigation created in %.1fs | id=%s", elapsed, investigation["investigation_id"])
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 1),
            "investigation": investigation,
        }
    except ValueError as exc:
        logger.warning("Invalid investigation request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Investigation creation timed out: %s", exc)
        return {"status": "error", "detail": str(exc)}
    except Exception as exc:
        logger.error("Investigation creation failed: %s", exc)
        return {"status": "error", "detail": f"Investigation creation failed: {str(exc)}"}
    finally:
        _release_pipeline_lock()


@app.post("/api/investigations/{investigation_id}/queries")
async def add_query_to_investigation_endpoint(investigation_id: str, request: AddQueryRequest):
    """
    Add a new query to an existing investigation.
    Executes the existing ATIS query engine and updates aggregated knowledge.
    """
    if not _acquire_pipeline_lock():
        return {
            "status": "busy",
            "detail": "Another pipeline is running. Please wait and retry."
        }

    try:
        start = time.time()
        investigation = await _run_with_timeout(
            add_query_to_investigation,
            investigation_id,
            request.question,
            request.parent_query_id,
            _vault_path,
            timeout=60.0,
        )
        elapsed = time.time() - start
        logger.info("Query added to investigation %s in %.1fs | total_queries=%d",
                    investigation_id, elapsed, len(investigation["queries"]))
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 1),
            "investigation": investigation,
        }
    except FileNotFoundError as exc:
        logger.warning("Investigation not found: %s", investigation_id)
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        logger.warning("Invalid query request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Query addition timed out: %s", exc)
        return {"status": "error", "detail": str(exc)}
    except Exception as exc:
        logger.error("Query addition failed: %s", exc)
        return {"status": "error", "detail": f"Query addition failed: {str(exc)}"}
    finally:
        _release_pipeline_lock()


@app.get("/api/investigations/{investigation_id}")
async def get_investigation_endpoint(investigation_id: str):
    """Retrieve a complete investigation by ID."""
    investigation = get_investigation(investigation_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail=f"Investigation not found: {investigation_id}")
    return {
        "status": "success",
        "investigation": investigation,
    }


@app.post("/api/investigations/{investigation_id}/report")
async def generate_report_endpoint(investigation_id: str):
    """Generate a Knowledge Report for an investigation (local file-based)."""
    if not _acquire_pipeline_lock():
        return {
            "status": "busy",
            "detail": "Another pipeline is running. Please wait and retry."
        }

    try:
        start = time.time()
        report = await _run_with_timeout(
            generate_investigation_report,
            investigation_id,
            timeout=60.0,
        )
        elapsed = time.time() - start
        logger.info("Report generated for investigation %s in %.1fs", investigation_id, elapsed)
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 1),
            "report": report,
        }
    except FileNotFoundError as exc:
        logger.warning("Investigation not found for report: %s", investigation_id)
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        logger.warning("Invalid report request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Report generation timed out: %s", exc)
        return {"status": "error", "detail": str(exc)}
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        return {"status": "error", "detail": f"Report generation failed: {str(exc)}"}
    finally:
        _release_pipeline_lock()


# =============================================================================
# NEW: Payload-based report generation endpoint (no local persistence)
# =============================================================================

@app.post("/api/investigation/report")
async def generate_report_from_payload_endpoint(request: GenerateReportRequest):
    """
    Generate a Knowledge Report from a complete Investigation payload.

    This endpoint does NOT read from local persistence.
    The frontend sends the complete investigation state.
    """
    if not _acquire_pipeline_lock():
        return {
            "status": "busy",
            "detail": "Another pipeline is running. Please wait and retry."
        }

    try:
        start = time.time()
        report = await _run_with_timeout(
            generate_report_from_payload,
            request.investigation,
            timeout=120.0,
        )
        elapsed = time.time() - start
        logger.info("Report generated from payload in %.1fs", elapsed)
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 1),
            "report": report,
        }
    except ValueError as exc:
        logger.warning("Invalid report payload: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Report generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error("Report generation unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(exc)}")
    finally:
        _release_pipeline_lock()


@app.get("/api/investigations")
async def list_investigations_endpoint():
    """List all investigations (lightweight summary)."""
    investigations = list_investigations()
    return {
        "status": "success",
        "count": len(investigations),
        "investigations": investigations,
    }

# =============================================================================
# Cache management endpoints
# =============================================================================
@app.post("/cache/invalidate")
async def invalidate_cache(evidence_id: Optional[str] = None):
    """Invalidate cached analyses by evidence ID or clear all."""
    if evidence_id:
        removed = _analysis_cache.invalidate_by_evidence(evidence_id)
        return {
            "status": "success",
            "invalidated": removed,
            "evidence_id": evidence_id,
            "analysis_version": ANALYSIS_VERSION,
        }
    else:
        removed = _analysis_cache.invalidate_all()
        _query_cache.clear()
        return {
            "status": "success",
            "invalidated": removed,
            "scope": "all",
            "analysis_version": ANALYSIS_VERSION,
        }

@app.get("/cache/stats")
async def cache_stats():
    """Cache statistics endpoint."""
    return {
        "query_cache_entries": len(_query_cache),
        "analysis_cache_dir": str(_analysis_cache.cache_dir),
        "analysis_cache_files": len(list(_analysis_cache.cache_dir.glob("analysis_*.json"))),
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SCHEMA_VERSION,
    }

# -----------------------------------------------------------------------------
# Knowledge state endpoint
# -----------------------------------------------------------------------------
@app.get("/knowledge-state")
async def knowledge_state():
    """Return current vault knowledge state."""
    ks = KnowledgeState(vault_path=_vault_path)
    ks.compute()
    return ks.as_dict()

# -----------------------------------------------------------------------------
# Emergency kill endpoint
# -----------------------------------------------------------------------------
@app.post("/admin/kill")
async def kill_pipeline():
    """Emergency endpoint to terminate the container and kill all running processes."""
    logger.critical("KILL endpoint triggered — restarting container")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "killed"}

# =============================================================================
# Startup event — pre-warm vault index
# =============================================================================
@app.on_event("startup")
async def startup_event():
    logger.info("ATIS API v2.2.0 starting up...")
    try:
        vault = _get_query_vault()
        ks = KnowledgeState(vault_path=_vault_path)
        ks.compute()
        logger.info("Startup complete. Vault ready with %d nodes. Knowledge state: %s",
                    vault.indexed_count, ks.knowledge_state_hash[:16])
    except Exception as exc:
        logger.error("Failed to index vault on startup: %s", exc)
