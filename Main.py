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
"""

from __future__ import annotations

import asyncio
import hashlib
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
_vault_path = Path(os.getenv("VAULT_PATH", r"C:\Users\tmaki\Documents\ATIS\vault"))

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
# NEW: Slug & decode utilities (fixes 404s and double-encoding)
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
    Examples:
      Zimbabwe/Zimbabwe Businesses/...  -> 'business'
      Zimbabwe/Zimbabwe Commodities/... -> 'commodity'
      Zimbabwe/Zimbabwe Government/...  -> 'government'
      Zimbabwe/Zimbabwe Infrastructure/... -> 'infrastructure'
      Zimbabwe/Zimbabwe People/...      -> 'person'
      Zimbabwe/Zimbabwe Region/...      -> 'region'
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
    Scans country folders (Zimbabwe, China, etc.) and their category subfolders
    (Zimbabwe Businesses, Zimbabwe Commodities, Zimbabwe Government, etc.).
    """
    discovered: List[Path] = []
    seen: Set[str] = set()
    
    if not vault.exists():
        logger.error("Vault path does not exist: %s", vault)
        return discovered
    
    # --- Strategy 1: Country -> Category structure ---
    # Vault root contains country folders. Each country has category folders.
    country_dirs = [d for d in vault.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for country_dir in country_dirs:
        # Check country root for .md files (e.g., Zimbabwe.md)
        if any(country_dir.glob("*.md")):
            key = str(country_dir.resolve())
            if key not in seen:
                seen.add(key)
                discovered.append(country_dir)
                logger.info("Discovered country root: %s", country_dir.relative_to(vault))
        
        # Check category subfolders (e.g., Zimbabwe Businesses, Zimbabwe Commodities)
        for category_dir in country_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            
            md_count = len(list(category_dir.rglob("*.md")))
            if md_count > 0:
                key = str(category_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    discovered.append(category_dir)
                    logger.info("Discovered category: %s (%d .md files)", category_dir.relative_to(vault), md_count)
    
    # --- Strategy 2: Full recursive scan fallback ---
    if not discovered:
        logger.info("No structured directories found. Scanning vault recursively...")
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
    """
    Load entity profiles from discovered directories.
    RECURSIVELY searches subfolders for .md files.
    """
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

# =============================================================================
# NEW: Relationship resolver (uses the graph index for backlinks & outbound links)
# =============================================================================
def _resolve_related_entities(node, vault: QueryVaultManager) -> List[Dict[str, Any]]:
    """
    Given a vault node, resolve its outbound links and backlinks into
    traversable entity profiles with slugs, names, types, and summaries.
    """
    related: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    # Outbound links
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

    # Backlinks
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
    Returns all entity profiles from the vault.
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
        logger.info("  -> %s", d.relative_to(_vault_path))

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
# NEW: Single entity profile with backlink & outbound resolution
# -----------------------------------------------------------------------------
@app.get("/api/entity/{entity_id}")
async def get_entity(entity_id: str):
    """
    Retrieve a single entity profile by ID or slug.
    Resolves outbound links and backlinks across the entire vault
    (businesses, laws, commodities, infrastructure, people, government, etc.).
    """
    clean_id = decode_entity_id(entity_id)
    target_slug = slugify(clean_id)
    logger.info("Entity lookup | raw=%s | decoded=%s | slug=%s", entity_id, clean_id, target_slug)

    # --- Strategy 1: Look in entity cache (fast path) ---
    if _entities_cache:
        for ent in _entities_cache.get("entities", []):
            if ent.get("slug") == target_slug or ent["id"] == clean_id:
                result = dict(ent)

                # Enrich with graph data from query vault
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
                    logger.warning("Failed to enrich entity %s from vault index: %s", ent["id"], exc)
                    result["outbound_links"] = []
                    result["backlink_uids"] = []
                    result["related_entities"] = []

                return result

    # --- Strategy 2: Direct vault index lookup (fallback for non-cached or cross-folder files) ---
    try:
        vault = _get_query_vault()
        node = None

        # Try exact stem match
        if clean_id in vault.nodes:
            node = vault.nodes[clean_id]
        else:
            # Try slug match across all nodes
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
# NEW: Search across all vault nodes (not just entity cache)
# -----------------------------------------------------------------------------
@app.get("/api/search")
async def search_entities(q: str):
    """
    Full-text search across the entire vault index.
    Returns matching entities with slugs for direct navigation.
    """
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