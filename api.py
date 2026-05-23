"""
World-as-Oracle REST API — Production version with SQLite persistence.
Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import time, os
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from oracle import WorldOracle, EventCategory
from oracle.models import SourceTier, Article
from oracle.impact_scorer import create_article_from_dict
from db import (init_db, save_event, get_event, get_all_events,
                save_article, get_articles, save_history_snapshot,
                get_history, save_resolution, event_exists)

# ── In-memory oracle cache (rebuilt from DB on startup) ───────────────────
_oracle_cache: dict = {}


def _rebuild_oracle_from_db(event_id: str) -> WorldOracle:
    """Reconstruct a WorldOracle from persisted DB state."""
    ev = get_event(event_id)
    if not ev:
        raise KeyError(f"Event {event_id} not in DB")

    cat_map = {c.value: c for c in EventCategory}
    oracle = WorldOracle(
        event_id=event_id,
        event_description=ev["description"],
        resolution_criteria=ev["resolution"],
        category=cat_map.get(ev["category"], EventCategory.CORPORATE_LEGAL),
        platt_a=ev.get("platt_a", 1.0),
        platt_b=ev.get("platt_b", 0.0),
    )
    oracle._prior = ev["prior"]

    # Reload articles
    rows = get_articles(event_id)
    tier_map = {i: SourceTier(i) for i in range(1, 7)}
    for row in rows:
        a = Article(
            article_id=row["article_id"],
            source_name=row["source_name"],
            tier=tier_map.get(row["tier"], SourceTier.TIER4_WIRE),
            publication_time=row["publication_time"],
            headline=row["headline"],
            content_summary=row.get("content_summary", ""),
            url=row.get("url", ""),
            raw_impact=row.get("raw_impact", 0.0),
            direction=row.get("direction", 0),
            independence_score=row.get("independence_score", 0.0),
            reasoning_chain=row.get("reasoning_chain", ""),
        )
        oracle._articles.append(a)

    return oracle


def get_oracle(event_id: str) -> WorldOracle:
    """Get from cache or rebuild from DB."""
    if event_id not in _oracle_cache:
        if not event_exists(event_id):
            raise HTTPException(404, f"Event '{event_id}' not found")
        _oracle_cache[event_id] = _rebuild_oracle_from_db(event_id)
    return _oracle_cache[event_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: init DB and warm oracle cache for all events."""
    init_db()
    for ev in get_all_events():
        try:
            _oracle_cache[ev["event_id"]] = _rebuild_oracle_from_db(ev["event_id"])
        except Exception as e:
            print(f"Could not rebuild oracle for {ev['event_id']}: {e}")
    print(f"Oracle cache warmed: {len(_oracle_cache)} events loaded")
    yield


app = FastAPI(title="World-as-Oracle API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Pydantic models ────────────────────────────────────────────────────────

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


# ── API Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0",
            "active_events": len(_oracle_cache),
            "affiliation": "Verslan / predictmarkets.finance / verslan.xyz"}


@app.get("/events")
def list_events():
    result = []
    for ev in get_all_events():
        try:
            oracle = get_oracle(ev["event_id"])
            out = oracle.compute()
            result.append({
                "event_id": ev["event_id"],
                "description": ev["description"],
                "category": ev["category"],
                "probability": round(out.probability, 4),
                "state": out.state.value,
                "articles_processed": out.total_articles_processed,
                "resolved": ev.get("outcome") is not None,
            })
        except Exception:
            result.append({"event_id": ev["event_id"], "error": "compute failed"})
    return {"events": result, "count": len(result)}


@app.post("/events")
def create_event(req: CreateEventRequest):
    if event_exists(req.event_id):
        raise HTTPException(400, f"Event '{req.event_id}' already exists")

    cat_map = {c.value: c for c in EventCategory}
    category = cat_map.get(req.category, EventCategory.CORPORATE_LEGAL)

    oracle = WorldOracle(
        event_id=req.event_id,
        event_description=req.event_description,
        resolution_criteria=req.resolution_criteria,
        category=category,
    )
    oracle.set_prior(req.historical_base_rate, req.structural_prior, req.market_implied_prior)

    save_event(
        req.event_id, req.event_description, req.resolution_criteria,
        category.value, oracle._prior,
        req.historical_base_rate, req.structural_prior, req.market_implied_prior,
    )
    _oracle_cache[req.event_id] = oracle

    return {"event_id": req.event_id, "prior": round(oracle._prior, 4),
            "category": category.value, "status": "created"}


@app.post("/events/{event_id}/articles")
def ingest_articles(event_id: str, articles: List[ArticleRequest]):
    oracle = get_oracle(event_id)
    ingested = []

    for req in articles:
        data = req.model_dump()
        if data["publication_time"] is None:
            data["publication_time"] = time.time()
        article = create_article_from_dict(data)
        ingested.append(article)

    oracle.ingest_articles(ingested, auto_score_independence=True, auto_score_impact=False)

    # Persist each article
    for a in ingested:
        save_article(event_id, a)

    out = oracle.compute()
    snapshot = {
        "timestamp": time.time(),
        "probability": round(out.probability, 4),
        "state": out.state.value,
        "fast_shock": round(out.fast_shock, 4),
        "persistent_signal": round(out.persistent_signal, 4),
        "independent_sources": out.independent_source_count,
    }
    save_history_snapshot(event_id, snapshot)

    return {"event_id": event_id, "articles_ingested": len(ingested),
            "total_articles": out.total_articles_processed,
            "current_probability": round(out.probability, 4),
            "state": out.state.value}


@app.get("/events/{event_id}/probability")
def get_probability(event_id: str):
    oracle = get_oracle(event_id)
    out = oracle.compute()
    return {
        "event_id": event_id,
        "probability": round(out.probability, 4),
        "ci_lower": round(out.ci_lower, 4),
        "ci_upper": round(out.ci_upper, 4),
        "state": out.state.value,
        "signal_decomposition": {
            "prior": round(out.prior, 4),
            "fast_shock": round(out.fast_shock, 4),
            "persistent_signal": round(out.persistent_signal, 4),
        },
        "independent_sources": out.independent_source_count,
        "total_articles": out.total_articles_processed,
        "audit_trail": out.audit_trail[:5],
        "timestamp": out.timestamp,
    }


@app.get("/events/{event_id}/history")
def get_event_history(event_id: str):
    if not event_exists(event_id):
        raise HTTPException(404, f"Event '{event_id}' not found")
    history = get_history(event_id)
    return {"event_id": event_id, "history": history, "count": len(history)}


@app.post("/events/{event_id}/resolve")
def resolve_event(event_id: str, req: ResolveRequest):
    if req.outcome not in (0, 1):
        raise HTTPException(400, "Outcome must be 0 or 1")
    oracle = get_oracle(event_id)
    oracle.record_resolution(req.outcome)
    report = oracle.calibration_report()
    save_resolution(event_id, req.outcome,
                    oracle._history[-1].probability if oracle._history else 0.5,
                    report.get("overall_brier_score"))
    return {"event_id": event_id, "outcome": req.outcome,
            "brier_score": report["overall_brier_score"], "calibration": report}


# ── Frontend ───────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
