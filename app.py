"""
World-as-Oracle REST API
Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""
import time, os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from oracle import WorldOracle, EventCategory
from oracle.impact_scorer import create_article_from_dict

app = FastAPI(title="World-as-Oracle API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_oracles: dict = {}
_histories: dict = {}

class CreateEventRequest(BaseModel):
    event_id: str
    event_description: str
    resolution_criteria: str
    category: str = "corporate_legal"
    historical_base_rate: float = 0.20
    structural_prior: float = 0.20
    market_implied_prior: Optional[float] = None

class ArticleRequest(BaseModel):
    article_id: str
    source_name: str
    tier: int = 4
    publication_time: Optional[float] = None
    headline: str
    content_summary: str = ""
    url: str = ""
    raw_impact: float = 0.0
    direction: int = 0
    reasoning_chain: str = ""

class ResolveRequest(BaseModel):
    outcome: int

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0",
            "affiliation": "Verslan / predictmarkets.finance / verslan.xyz",
            "active_events": len(_oracles)}

@app.get("/events")
def list_events():
    result = []
    for eid, oracle in _oracles.items():
        try:
            out = oracle.compute()
            result.append({"event_id": eid, "description": oracle.event_description,
                           "category": oracle.config.category.value,
                           "probability": round(out.probability, 4),
                           "state": out.state.value,
                           "articles_processed": out.total_articles_processed})
        except Exception:
            result.append({"event_id": eid, "error": "compute failed"})
    return {"events": result, "count": len(result)}

@app.post("/events")
def create_event(req: CreateEventRequest):
    if req.event_id in _oracles:
        raise HTTPException(400, f"Event '{req.event_id}' already exists")
    category_map = {c.value: c for c in EventCategory}
    category = category_map.get(req.category, EventCategory.CORPORATE_LEGAL)
    oracle = WorldOracle(event_id=req.event_id, event_description=req.event_description,
                         resolution_criteria=req.resolution_criteria, category=category)
    oracle.set_prior(req.historical_base_rate, req.structural_prior, req.market_implied_prior)
    _oracles[req.event_id] = oracle
    _histories[req.event_id] = []
    return {"event_id": req.event_id, "prior": round(oracle._prior, 4),
            "category": category.value, "status": "created"}

@app.post("/events/{event_id}/articles")
def ingest_articles(event_id: str, articles: List[ArticleRequest]):
    if event_id not in _oracles:
        raise HTTPException(404, f"Event '{event_id}' not found")
    oracle = _oracles[event_id]
    ingested = []
    for req in articles:
        data = req.model_dump()
        if data["publication_time"] is None:
            data["publication_time"] = time.time()
        ingested.append(create_article_from_dict(data))
    oracle.ingest_articles(ingested, auto_score_independence=True, auto_score_impact=False)
    out = oracle.compute()
    _histories[event_id].append({"timestamp": time.time(), "probability": round(out.probability, 4),
                                  "state": out.state.value, "fast_shock": round(out.fast_shock, 4),
                                  "persistent_signal": round(out.persistent_signal, 4),
                                  "independent_sources": out.independent_source_count})
    return {"event_id": event_id, "articles_ingested": len(ingested),
            "total_articles": out.total_articles_processed,
            "current_probability": round(out.probability, 4), "state": out.state.value}

@app.get("/events/{event_id}/probability")
def get_probability(event_id: str):
    if event_id not in _oracles:
        raise HTTPException(404, f"Event '{event_id}' not found")
    out = _oracles[event_id].compute()
    return {"event_id": event_id, "probability": round(out.probability, 4),
            "ci_lower": round(out.ci_lower, 4), "ci_upper": round(out.ci_upper, 4),
            "state": out.state.value,
            "signal_decomposition": {"prior": round(out.prior, 4),
                                      "fast_shock": round(out.fast_shock, 4),
                                      "persistent_signal": round(out.persistent_signal, 4)},
            "independent_sources": out.independent_source_count,
            "total_articles": out.total_articles_processed,
            "audit_trail": out.audit_trail[:5], "timestamp": out.timestamp}

@app.get("/events/{event_id}/history")
def get_history(event_id: str):
    if event_id not in _oracles:
        raise HTTPException(404, f"Event '{event_id}' not found")
    return {"event_id": event_id, "history": _histories.get(event_id, []),
            "count": len(_histories.get(event_id, []))}

@app.post("/events/{event_id}/resolve")
def resolve_event(event_id: str, req: ResolveRequest):
    if event_id not in _oracles:
        raise HTTPException(404, f"Event '{event_id}' not found")
    if req.outcome not in (0, 1):
        raise HTTPException(400, "Outcome must be 0 or 1")
    oracle = _oracles[event_id]
    oracle.record_resolution(req.outcome)
    report = oracle.calibration_report()
    return {"event_id": event_id, "outcome": req.outcome,
            "brier_score": report["overall_brier_score"], "calibration": report}

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
