#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATIS_API.py
FastAPI backend for the ATIS Intelligence Suite.

Fixes:
  - Stable slug-based entity IDs (no more display-name lookups).
  - Proper URL decode + normalize on all path parameters.
  - Full backlink resolution: entity profiles expose outbound links AND backlinks.
  - Related-vault discovery: infrastructure, laws, commodities, etc. are traversable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
# Config
# =============================================================================
VAULT_ROOT = Path(os.getenv("VAULT_PATH", r"C:\Users\tmaki\Documents\ATIS\vault\Zimbabwe")).resolve()

app = FastAPI(title="ATIS API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Vault Indexing (reuses proven logic from ATIS_Execute.py)
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
    backlink_uids: List[str] = field(default_factory=list)
    entity_type: str = "unknown"  # inferred from folder path

    def to_dict(self, resolve_links: bool = True) -> Dict[str, Any]:
        d = {
            "id": self.uid,
            "slug": slugify(self.uid),
            "name": self.front_matter.get("title") or self.front_matter.get("name") or self.stem,
            "stem": self.stem,
            "summary": self.summary,
            "entity_type": self.entity_type,
            "front_matter": self.front_matter,
            "outbound_links": self.outbound_links,
            "backlink_uids": self.backlink_uids,
            "absolute_path": str(self.absolute_path),
        }
        if resolve_links:
            d["related_entities"] = []  # populated at serve time
        return d


class VaultIndex:
    _WIKILINK_PATTERN = re.compile(r'\[\[(.*?)\]\]')

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = Path(vault_root).resolve()
        self.nodes: Dict[str, VaultNode] = {}          # key = stem (exact)
        self.slug_map: Dict[str, str] = {}             # key = slug -> stem
        self.backlink_map: Dict[str, List[str]] = {}   # target -> [source stems]
        self.link_resolver: Dict[str, str] = {}        # canonical -> stem
        self._build()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def canonicalize(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "", str(text)).lower()

    @staticmethod
    def _parse_markdown(raw: str) -> Tuple[Dict[str, Any], str]:
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.DOTALL)
        if m:
            try:
                front = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError:
                front = {}
            body = m.group(2)
        else:
            front, body = {}, raw
        return front, body

    @staticmethod
    def _extract_summary(front: Dict[str, Any], body: str) -> str:
        for key in ("summary", "Summary", "description", "Description", "abstract", "Abstract"):
            val = front.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            elif isinstance(val, list) and val and isinstance(val[0], str):
                return val[0].strip()
        m = re.search(r"(?im)^#{1,6}\s*Summary\s*\n(.*?)(?=\n#{1,6}\s|\Z)", body, re.DOTALL)
        if m:
            return m.group(1).strip()
        return body[:400].strip()

    @classmethod
    def _extract_outbound_links(cls, raw: str) -> List[str]:
        raw_links = cls._WIKILINK_PATTERN.findall(raw)
        cleaned: List[str] = []
        seen: Set[str] = set()
        for link in raw_links:
            target = link.split("|")[0]
            core = target.split("/")[-1].strip()
            if core and core.lower() not in seen:
                seen.add(core.lower())
                cleaned.append(core)
        return cleaned

    @staticmethod
    def _extract_aliases(front: Dict[str, Any]) -> List[str]:
        aliases: List[str] = []
        for key in ("aliases", "alias", "title", "name"):
            val = front.get(key)
            if isinstance(val, str):
                aliases.append(val)
            elif isinstance(val, list):
                aliases.extend([i for i in val if isinstance(i, str)])
        return aliases

    def _infer_entity_type(self, rel_path: Path) -> str:
        """
        Guess entity category from folder structure, e.g.
        vault/Zimbabwe/Infrastructure/Dams  -> 'infrastructure'
        vault/Zimbabwe/Laws/Statutes        -> 'laws'
        vault/Zimbabwe/Commodities          -> 'commodities'
        """
        parts = [p.lower() for p in rel_path.parts]
        type_hints = {
            "businesses": "business",
            "companies": "business",
            "infrastructure": "infrastructure",
            "laws": "law",
            "statutes": "law",
            "commodities": "commodity",
            "minerals": "commodity",
            "agriculture": "agriculture",
            "energy": "energy",
            "people": "person",
            "contacts": "person",
            "government": "government",
            "agencies": "government",
        }
        for part in parts:
            if part in type_hints:
                return type_hints[part]
        return "unknown"

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build(self) -> None:
        logger.info("Building vault index from: %s", self.vault_root)
        if not self.vault_root.exists():
            raise FileNotFoundError(f"Vault root missing: {self.vault_root}")

        md_files = list(self.vault_root.rglob("*.md"))
        logger.info("Found %d markdown files.", len(md_files))

        for md_path in md_files:
            try:
                raw = md_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Skipping unreadable file %s: %s", md_path, exc)
                continue

            front, body = self._parse_markdown(raw)
            summary = self._extract_summary(front, body)
            outbound = self._extract_outbound_links(raw)
            rel = md_path.relative_to(self.vault_root)
            entity_type = self._infer_entity_type(rel)

            stem = md_path.stem
            node = VaultNode(
                uid=stem,
                absolute_path=md_path,
                stem=stem,
                front_matter=front,
                summary=summary,
                outbound_links=outbound,
                raw_content=raw,
                body=body,
                entity_type=entity_type,
            )
            self.nodes[stem] = node

            # Resolver maps
            self.link_resolver[self.canonicalize(stem)] = stem
            for alias in self._extract_aliases(front):
                self.link_resolver[self.canonicalize(alias)] = stem

        # Resolve outbound links to stems
        for node in self.nodes.values():
            resolved: List[str] = []
            seen: Set[str] = set()
            for link in node.outbound_links:
                canon = self.canonicalize(link)
                if canon in self.link_resolver:
                    target = self.link_resolver[canon]
                    if target not in seen:
                        seen.add(target)
                        resolved.append(target)
                else:
                    # dangling link — keep raw for display, but don't backlink
                    if link.lower() not in seen:
                        seen.add(link.lower())
                        resolved.append(link)
            node.outbound_links = resolved

        # Build backlink map
        for stem, node in self.nodes.items():
            for target in node.outbound_links:
                if target in self.nodes:
                    self.backlink_map.setdefault(target, []).append(stem)

        # Attach backlinks to nodes
        for stem, node in self.nodes.items():
            node.backlink_uids = list(dict.fromkeys(self.backlink_map.get(stem, [])))

        # Slug map for stable URL lookups
        for stem in self.nodes:
            self.slug_map[slugify(stem)] = stem

        logger.info(
            "Index ready: %d nodes, %d backlinks, %d slugs.",
            len(self.nodes),
            len(self.backlink_map),
            len(self.slug_map),
        )

    # ------------------------------------------------------------------
    # Public lookup API
    # ------------------------------------------------------------------
    def get_by_slug(self, slug: str) -> Optional[VaultNode]:
        """Lookup by URL-safe slug (stable ID)."""
        stem = self.slug_map.get(slug)
        if stem:
            return self.nodes.get(stem)
        return None

    def get_by_stem(self, stem: str) -> Optional[VaultNode]:
        """Direct stem lookup (fallback)."""
        return self.nodes.get(stem)

    def list_entities(self) -> List[Dict[str, Any]]:
        """Lightweight list for the directory view."""
        return [
            {
                "id": n.uid,
                "slug": slugify(n.uid),
                "name": n.front_matter.get("title") or n.front_matter.get("name") or n.stem,
                "entity_type": n.entity_type,
                "summary": n.summary[:200],
            }
            for n in self.nodes.values()
        ]

    def enrich_node(self, node: VaultNode) -> Dict[str, Any]:
        """
        Full entity profile with resolved related entities (outbound + backlinks).
        """
        data = node.to_dict(resolve_links=True)

        # Resolve outbound links to full mini-profiles
        related: List[Dict[str, Any]] = []
        seen_slugs: Set[str] = set()

        def add_relation(target_stem: str, relation_type: str):
            s = slugify(target_stem)
            if s in seen_slugs:
                return
            seen_slugs.add(s)
            target = self.nodes.get(target_stem)
            if target:
                related.append({
                    "slug": s,
                    "name": target.front_matter.get("title") or target.stem,
                    "entity_type": target.entity_type,
                    "relation_type": relation_type,
                    "summary": target.summary[:150],
                })
            else:
                # Dangling link
                related.append({
                    "slug": s,
                    "name": target_stem,
                    "entity_type": "unknown",
                    "relation_type": relation_type,
                    "summary": "",
                })

        for out in node.outbound_links:
            add_relation(out, "outbound")

        for back in node.backlink_uids:
            add_relation(back, "backlink")

        data["related_entities"] = related
        data["backlink_count"] = len(node.backlink_uids)
        data["outbound_count"] = len([o for o in node.outbound_links if o in self.nodes])
        return data


# =============================================================================
# Slug utility — THE FIX for URL lookups
# =============================================================================
def slugify(text: str) -> str:
    """
    Create a stable, URL-safe slug from any vault filename or title.
    Normalizes: lowercase, strips punctuation, collapses whitespace to hyphens.
    """
    if not text:
        return ""
    # URL-decode first (in case raw %20 slips through)
    text = urllib.parse.unquote(text)
    text = text.strip()
    # Replace any non-alphanumeric run with a single hyphen
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
    # Collapse multiple hyphens
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def decode_path_param(raw: str) -> str:
    """
    Decode a FastAPI path parameter that may have been double-encoded by the frontend.
    """
    # FastAPI auto-decodes once. If frontend double-encoded, we still see %20.
    # We unquote until stable.
    prev = ""
    decoded = raw
    while decoded != prev:
        prev = decoded
        decoded = urllib.parse.unquote(prev)
    return decoded


# =============================================================================
# Global index (loaded once at startup)
# =============================================================================
_vault_index: Optional[VaultIndex] = None

def get_index() -> VaultIndex:
    global _vault_index
    if _vault_index is None:
        _vault_index = VaultIndex(VAULT_ROOT)
    return _vault_index


# =============================================================================
# API Routes
# =============================================================================
@app.get("/api/health")
def health():
    idx = get_index()
    return {
        "status": "ok",
        "vault_root": str(VAULT_ROOT),
        "entities_loaded": len(idx.nodes),
        "backlinks_tracked": len(idx.backlink_map),
    }


@app.get("/api/entities")
def list_entities():
    """
    Returns every entity with a stable slug.
    Frontend should use `slug` (not `name`) for detail links.
    """
    idx = get_index()
    return {"entities": idx.list_entities()}


@app.get("/api/entity/{entity_slug}")
def get_entity(entity_slug: str, request: Request):
    """
    Retrieve a single entity profile by its stable slug.
    Includes full backlink and outbound-link resolution.
    """
    idx = get_index()

    # Defensive decode in case raw %20 still arrives
    clean_slug = slugify(decode_path_param(entity_slug))
    logger.info("Lookup | raw=%s | decoded=%s | slug=%s", entity_slug, decode_path_param(entity_slug), clean_slug)

    node = idx.get_by_slug(clean_slug)
    if not node:
        # Fallback: try direct stem match after decode
        stem_guess = decode_path_param(entity_slug)
        node = idx.get_by_stem(stem_guess)

    if not node:
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_slug}")

    return idx.enrich_node(node)


@app.get("/api/entity/{entity_slug}/related")
def get_related(entity_slug: str):
    """
    Convenience endpoint: returns ONLY the related-entity graph for a given slug.
    Useful for sidebar/link panels.
    """
    idx = get_index()
    clean_slug = slugify(decode_path_param(entity_slug))
    node = idx.get_by_slug(clean_slug)
    if not node:
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_slug}")

    # Return just the relation graph
    enriched = idx.enrich_node(node)
    return {
        "entity": {"slug": clean_slug, "name": enriched["name"]},
        "related_entities": enriched["related_entities"],
    }


@app.get("/api/search")
def search_entities(q: str):
    """
    Simple text search across names, summaries, and entity types.
    """
    idx = get_index()
    q_clean = q.lower()
    results = []
    for node in idx.nodes.values():
        haystack = f"{node.stem} {node.summary} {node.entity_type}".lower()
        if q_clean in haystack:
            results.append({
                "slug": slugify(node.uid),
                "name": node.front_matter.get("title") or node.stem,
                "entity_type": node.entity_type,
                "summary": node.summary[:200],
            })
    return {"query": q, "results": results}


# =============================================================================
# Startup
# =============================================================================
@app.on_event("startup")
def startup_event():
    logger.info("ATIS API starting up...")
    get_index()  # eager load
    logger.info("Vault index eager-loaded.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)