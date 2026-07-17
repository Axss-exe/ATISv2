#!/usr/bin/env python3
"""
ATIS Intelligence API — Render Web Service Entry Point

Features:
  - CORS configured for Vercel frontend
  - Vault entity listing with recursive subfolder discovery & full profile content
  - 60-second hard timeout on all LLM pipelines
  - Global request lock (prevents concurrent pipeline runs)
  - Emergency kill endpoint
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Import ATIS pipeline modules
# -----------------------------------------------------------------------------
from ATIS_News import run_news_pipeline
from ATIS_Execute import run_execute_pipeline
from ATIS_Query import run_query_pipeline, ObsidianVaultManager as QueryVaultManager

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
app = FastAPI(title="ATIS Intelligence API")

# =============================================================================
# CORS
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

# Entity cache — rebuilds only when files change
_entities_cache: Dict[str, Any] | None = None
_entities_cache_mtime: float = 0.0

# =============================================================================
# Request models
# =============================================================================
class NewsRequest(BaseModel):
    article_text: str

class ExecuteRequest(BaseModel):
    dashboard_json: dict
    opportunity_id: str

class QueryRequest(BaseModel):
    question: str | None = None

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
# Entity Discovery Helpers
# =============================================================================

def _discover_entity_directories(vault: Path) -> List[Path]:
    """
    Auto-discover directories that likely contain entity profiles.
    Tries known paths first, then scans the vault structure recursively.
    """
    discovered: List[Path] = []
    
    # --- Candidate paths to try (in order) ---
    candidates = [
        vault / "Zimbabwe" / "Zimbabwe Businesses" / "Companies",
        vault / "Zimbabwe" / "Zimbabwe_Businesses" / "Companies",
        vault / "Zimbabwe" / "Businesses" / "Companies",
        vault / "Zimbabwe" / "Companies",
        vault / "Businesses" / "Companies",
        vault / "Entities" / "Companies",
        vault / "Companies",
        vault / "Entities",
        vault / "Profiles",
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Recursively count .md files in this directory tree
            md_count = len(list(candidate.rglob("*.md")))
            logger.info("Candidate path exists: %s (%d .md files in tree)", candidate, md_count)
            if md_count > 0:
                discovered.append(candidate)
        else:
            logger.debug("Candidate path not found: %s", candidate)
    
    # --- Fallback: scan vault for any directory named Companies/Businesses/Entities/Profiles ---
    if not discovered:
        logger.info("No candidate paths matched. Scanning vault recursively for entity directories...")
        for root, dirs, files in os.walk(vault):
            root_path = Path(root)
            dir_name = root_path.name.lower()
            if dir_name in ("companies", "businesses", "entities", "profiles"):
                md_count = len(list(root_path.rglob("*.md")))
                if md_count > 0:
                    logger.info("Discovered via scan: %s (%d .md files in tree)", root_path, md_count)
                    discovered.append(root_path)
    
    # --- Last resort: if vault has .md files directly, treat root as entity dir ---
    if not discovered:
        root_md = list(vault.rglob("*.md"))
        if root_md:
            logger.info("Using vault root as entity directory (%d .md files)", len(root_md))
            discovered.append(vault)
    
    return discovered

def _load_entities_from_dirs(dirs: List[Path]) -> List[Dict[str, Any]]:
    """
    Load entity profiles from discovered directories.
    RECURSIVELY searches subfolders for .md files.
    """
    entities = []
    seen_ids = set()
    
    for directory in dirs:
        # rglob = recursive glob — walks into every subfolder
        for f in sorted(directory.rglob("*.md")):
            if not f.is_file():
                continue
            # Prevent duplicates by ID
            if f.stem in seen_ids:
                continue
            seen_ids.add(f.stem)
            
            try:
                content = f.read_text(encoding="utf-8")
                entities.append({
                    "id": f.stem,
                    "name": f.stem.replace("_", " ").replace("-", " "),
                    "filename": f.name,
                    "path": str(f.relative_to(_vault_path)),
                    "content": content,
                    "size_bytes": f.stat().st_size,
                })
            except Exception as exc:
                logger.warning("Could not read %s: %s", f.name, exc)
                continue
    
    return entities

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "ATIS API",
        "pipeline_busy": _pipeline_in_progress,
        "cached_queries": len(_query_cache),
        "cached_entities": _entities_cache is not None,
        "entity_count": _entities_cache["count"] if _entities_cache else 0,
    }

# -----------------------------------------------------------------------------
# Entity listing (lightweight, no LLM, cached, recursive subfolder support)
# -----------------------------------------------------------------------------
@app.get("/api/entities")
async def list_entities():
    """
    Returns all business entity profiles from the vault.
    Auto-discovers directories and recursively searches subfolders for .md files.
    Cached — rebuilds only on file changes.
    """
    global _entities_cache, _entities_cache_mtime

    # --- Cache invalidation: check if any .md file in vault changed ---
    all_md_files = list(_vault_path.rglob("*.md"))
    current_mtime = max(
        (f.stat().st_mtime for f in all_md_files),
        default=0.0
    )

    if _entities_cache is not None and current_mtime <= _entities_cache_mtime:
        logger.info("Serving entities from cache (%d profiles)", _entities_cache["count"])
        return _entities_cache

    # --- Discover and build ---
    start = time.time()
    discovered_dirs = _discover_entity_directories(_vault_path)
    
    if not discovered_dirs:
        logger.error("No entity directories discovered in vault: %s", _vault_path)
        try:
            root_contents = [p.name for p in _vault_path.iterdir()]
            logger.error("Vault root contains: %s", root_contents)
        except Exception:
            pass
        raise HTTPException(
            status_code=404,
            detail=f"No entity profiles found in vault. Checked paths and scanned subdirectories."
        )

    logger.info("Discovered %d entity directories", len(discovered_dirs))
    for d in discovered_dirs:
        logger.info("  -> %s", d)

    entities = _load_entities_from_dirs(discovered_dirs)
    elapsed = time.time() - start

    logger.info("Entity cache built in %.3fs — %d profiles loaded from %d directories (recursive)", 
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
        result = await _run_with_timeout(
            run_news_pipeline,
            request.article_text,
            timeout=60.0
        )
        elapsed = time.time() - start
        logger.info("News pipeline completed in %.1fs", elapsed)
        return {"status": "success", "elapsed_seconds": round(elapsed, 1), "data": result}
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
        result = await _run_with_timeout(
            run_execute_pipeline,
            request.dashboard_json,
            request.opportunity_id,
            timeout=60.0
        )
        elapsed = time.time() - start
        logger.info("Execute pipeline completed in %.1fs", elapsed)
        return {"status": "success", "elapsed_seconds": round(elapsed, 1), "data": result}
    except RuntimeError as exc:
        logger.error("Execute pipeline failed: %s", exc)
        return {"status": "error", "detail": str(exc)}
    except Exception as exc:
        logger.error("Execute pipeline unexpected error: %s", exc)
        return {"status": "error", "detail": f"Pipeline failed: {str(exc)}"}
    finally:
        _release_pipeline_lock()

# -----------------------------------------------------------------------------
# Query pipeline (with caching)
# -----------------------------------------------------------------------------
@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    cache_key = hashlib.sha256((request.question or "full_scan").encode()).hexdigest()[:16]

    if cache_key in _query_cache:
        logger.info("Cache hit for query: %s", cache_key)
        return {
            "status": "success",
            "cached": True,
            "data": _query_cache[cache_key]
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
            timeout=60.0
        )
        elapsed = time.time() - start
        logger.info("Query pipeline completed in %.1fs", elapsed)

        _query_cache[cache_key] = result

        return {
            "status": "success",
            "cached": False,
            "elapsed_seconds": round(elapsed, 1),
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
    logger.info("ATIS API starting up...")
    try:
        vault = _get_query_vault()
        logger.info("Startup complete. Vault ready with %d nodes.", vault.indexed_count)
    except Exception as exc:
        logger.error("Failed to index vault on startup: %s", exc)