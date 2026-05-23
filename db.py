"""
Persistence layer — SQLite-backed storage for oracle events and history.
Replaces the in-memory dict store so data survives server restarts.

In production: swap SQLite for Postgres by changing DATABASE_URL env var.
Railway provides managed Postgres as a one-click add-on.

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import json
import sqlite3
import os
import time
from typing import Optional

DATABASE_URL = os.getenv("DATABASE_URL", "/tmp/oracle.db")  # /tmp is writable on Streamlit Cloud


def get_conn():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            event_id        TEXT PRIMARY KEY,
            description     TEXT NOT NULL,
            resolution      TEXT NOT NULL,
            category        TEXT NOT NULL,
            prior           REAL NOT NULL,
            platt_a         REAL DEFAULT 1.0,
            platt_b         REAL DEFAULT 0.0,
            hist_base_rate  REAL NOT NULL,
            structural_prior REAL NOT NULL,
            market_prior    REAL,
            created_at      REAL NOT NULL,
            resolved_at     REAL,
            outcome         INTEGER
        );

        CREATE TABLE IF NOT EXISTS articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        TEXT NOT NULL,
            article_id      TEXT NOT NULL,
            source_name     TEXT NOT NULL,
            tier            INTEGER NOT NULL,
            publication_time REAL NOT NULL,
            headline        TEXT NOT NULL,
            content_summary TEXT,
            url             TEXT,
            raw_impact      REAL DEFAULT 0.0,
            direction       INTEGER DEFAULT 0,
            independence_score REAL DEFAULT 0.0,
            reasoning_chain TEXT,
            ingested_at     REAL NOT NULL,
            FOREIGN KEY (event_id) REFERENCES events(event_id),
            UNIQUE(event_id, article_id)
        );

        CREATE TABLE IF NOT EXISTS probability_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        TEXT NOT NULL,
            timestamp       REAL NOT NULL,
            probability     REAL NOT NULL,
            state           TEXT NOT NULL,
            fast_shock      REAL,
            persistent_signal REAL,
            independent_sources INTEGER,
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        );

        CREATE TABLE IF NOT EXISTS resolutions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id    TEXT NOT NULL,
            outcome     INTEGER NOT NULL,
            p_oracle    REAL NOT NULL,
            brier_score REAL,
            resolved_at REAL NOT NULL,
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        );

        CREATE INDEX IF NOT EXISTS idx_articles_event ON articles(event_id);
        CREATE INDEX IF NOT EXISTS idx_history_event ON probability_history(event_id);
    """)
    conn.commit()
    conn.close()


def save_event(event_id, description, resolution, category, prior,
               hist_base_rate, structural_prior, market_prior=None):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO events
        (event_id, description, resolution, category, prior,
         hist_base_rate, structural_prior, market_prior, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (event_id, description, resolution, category, prior,
          hist_base_rate, structural_prior, market_prior, time.time()))
    conn.commit()
    conn.close()


def get_event(event_id) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_events() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM events ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_article(event_id, article):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO articles
        (event_id, article_id, source_name, tier, publication_time,
         headline, content_summary, url, raw_impact, direction,
         independence_score, reasoning_chain, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (event_id, article.article_id, article.source_name, article.tier.value,
          article.publication_time, article.headline, article.content_summary,
          article.url, article.raw_impact, article.direction,
          article.independence_score, article.reasoning_chain, time.time()))
    conn.commit()
    conn.close()


def get_articles(event_id) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM articles WHERE event_id = ? ORDER BY publication_time",
        (event_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_history_snapshot(event_id, snapshot: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO probability_history
        (event_id, timestamp, probability, state, fast_shock, persistent_signal, independent_sources)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (event_id, snapshot["timestamp"], snapshot["probability"], snapshot["state"],
          snapshot.get("fast_shock", 0), snapshot.get("persistent_signal", 0),
          snapshot.get("independent_sources", 0)))
    conn.commit()
    conn.close()


def get_history(event_id) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM probability_history WHERE event_id = ? ORDER BY timestamp",
        (event_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_resolution(event_id, outcome, p_oracle, brier_score=None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO resolutions (event_id, outcome, p_oracle, brier_score, resolved_at)
        VALUES (?, ?, ?, ?, ?)
    """, (event_id, outcome, p_oracle, brier_score, time.time()))
    conn.execute(
        "UPDATE events SET resolved_at = ?, outcome = ? WHERE event_id = ?",
        (time.time(), outcome, event_id)
    )
    conn.commit()
    conn.close()


def event_exists(event_id) -> bool:
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM events WHERE event_id = ?", (event_id,)).fetchone()
    conn.close()
    return row is not None
