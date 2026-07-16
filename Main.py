from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ← ADD THIS
from pydantic import BaseModel
import os

from ATIS_News import run_news_pipeline
from ATIS_Execute import run_execute_pipeline
from ATIS_Query import run_query_pipeline

app = FastAPI(title="ATIS Intelligence API")

# =============================================================================
# CORS — Allow your frontend to call this API
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",      # Local dev
    "https://av2-fkq2sfy2c-tmakiriyado1-4301s-projects.vercel.app"], # Your production frontend
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
# Endpoints
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