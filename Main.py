from fastapi import FastAPI
from pydantic import BaseModel
import os

from ATIS_News import run_news_pipeline
from ATIS_Execute import run_execute_pipeline
from ATIS_Query import run_query_pipeline

app = FastAPI(title="ATIS Intelligence API")

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
    """
    Trigger: Frontend "News" button
    Input: Raw article text
    Output: ATIS dashboard JSON
    """
    result = run_news_pipeline(request.article_text)
    return {"status": "success", "data": result}

@app.post("/api/execute")
async def execute_endpoint(request: ExecuteRequest):
    """
    Trigger: Frontend "Execute" button
    Input: Dashboard JSON + Opportunity ID
    Output: Tactical roadmap + reasoning graph
    """
    result = run_execute_pipeline(request.dashboard_json, request.opportunity_id)
    return {"status": "success", "data": result}

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    """
    Trigger: Frontend "Query" button
    Input: Optional natural language question
    Output: Vault intelligence dashboard
    """
    result = run_query_pipeline(request.question)
    return {"status": "success", "data": result}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ATIS API"}