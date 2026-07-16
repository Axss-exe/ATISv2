from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path

from ATIS_News import run_news_pipeline
from ATIS_Execute import run_execute_pipeline
from ATIS_Query import run_query_pipeline

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

# -----------------------------------------------------------------------------
# Request models
# -----------------------------------------------------------------------------
class NewsRequest(BaseModel):
    article_text: str

class ExecuteRequest(BaseModel):
    dashboard_json: dict
    opportunity_id: str

class QueryRequest(BaseModel):
    question: str | None = None

# -----------------------------------------------------------------------------
# NEW: Entity listing endpoint
# -----------------------------------------------------------------------------
@app.get("/api/entities")
async def list_entities():
    """
    Returns all business entity names from the vault.
    Scans vault/Zimbabwe/Zimbabwe Businesses/Companies for .md files.
    """
    vault_base = Path(os.getenv("VAULT_PATH", "./vault"))
    entities_dir = vault_base / "Zimbabwe" / "Zimbabwe Businesses" / "Companies"
    
    if not entities_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Entities directory not found: {entities_dir}"
        )
    
    # Get all .md files, strip extension, return as list
    md_files = sorted(entities_dir.glob("*.md"))
    entities = [
        {
            "id": f.stem,
            "name": f.stem.replace("_", " ").replace("-", " "),
            "filename": f.name,
            "path": str(f.relative_to(vault_base))
        }
        for f in md_files
    ]
    
    return {
        "status": "success",
        "count": len(entities),
        "directory": str(entities_dir),
        "entities": entities
    }

# -----------------------------------------------------------------------------
# Existing endpoints
# -----------------------------------------------------------------------------
@app.post("/api/news")
async def news_endpoint(request: NewsRequest):
    result = run_news_pipeline(request.article_text)
    return {"status": "success", "data": result}

@app.post("/api/execute")
async def execute_endpoint(request: ExecuteRequest):
    result = run_execute_pipeline(request.dashboard_json, request.opportunity_id)
    return {"status": "success", "data": result}

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    result = run_query_pipeline(request.question)
    return {"status": "success", "data": result}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ATIS API"}