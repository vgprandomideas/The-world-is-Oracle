"""
Live Data Pipeline — GDELT + NewsAPI + RSS

Auto-ingests articles in real time and scores them for oracle events.
No manual article entry needed.

Sources:
  GDELT    — free, real-time global event database (no key needed)
  GNews    — free tier 100 req/day (GNEWS_API_KEY env var)
  NewsAPI  — free tier 100 req/day (NEWSAPI_KEY env var)

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import os
import time
import json
import requests
from typing import List, Optional
from datetime import datetime, timedelta
from oracle.models import Article, SourceTier


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GNEWS_API     = "https://gnews.io/api/v4/search"
NEWSAPI_API   = "https://newsapi.org/v2/everything"


def _source_tier_from_domain(domain: str) -> SourceTier:
    """Infer source tier from domain name."""
    tier1 = ["sec.gov", "doj.gov", "rbi.org.in", "federalreserve.gov",
             "judiciary.gov", "eci.gov.in", "sebi.gov.in"]
    tier2 = ["economictimes.com", "business-standard.com", "livemint.com",
             "financialexpress.com", "thehindu.com", "moneycontrol.com"]
    tier3 = ["ft.com", "nikkei.com", "wsj.com", "scmp.com"]
    tier4 = ["bloomberg.com", "reuters.com", "apnews.com", "bbc.com",
             "cnbc.com", "theguardian.com"]

    domain = domain.lower()
    for d in tier1:
        if d in domain: return SourceTier.TIER1_PRIMARY
    for d in tier2:
        if d in domain: return SourceTier.TIER2_LOCAL_PRESS
    for d in tier3:
        if d in domain: return SourceTier.TIER3_REGIONAL
    for d in tier4:
        if d in domain: return SourceTier.TIER4_WIRE
    return SourceTier.TIER4_WIRE  # Default


def fetch_gdelt(
    query: str,
    max_results: int = 10,
    hours_back: int = 24,
) -> List[Article]:
    """
    Fetch articles from GDELT Document API.
    Free, no API key required. Rate limit: 1 req/sec.
    """
    articles = []
    try:
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": max_results,
            "format": "json",
            "timespan": f"{hours_back}h",
            "sort": "DateDesc",
        }
        resp = requests.get(GDELT_DOC_API, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        data = resp.json()
        articles_raw = data.get("articles", [])

        for i, art in enumerate(articles_raw):
            url = art.get("url", "")
            domain = url.split("/")[2] if url.startswith("http") else ""
            tier = _source_tier_from_domain(domain)

            # Parse date
            date_str = art.get("seendate", "")
            try:
                pub_time = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ").timestamp()
            except:
                pub_time = time.time() - (i * 3600)

            article = Article(
                article_id=f"gdelt_{abs(hash(url)) % 999999}",
                source_name=art.get("domain", domain),
                tier=tier,
                publication_time=pub_time,
                headline=art.get("title", "")[:300],
                content_summary=art.get("seendate", ""),
                url=url,
            )
            articles.append(article)

    except Exception as e:
        print(f"GDELT fetch error: {e}")

    return articles


def fetch_gnews(
    query: str,
    max_results: int = 10,
    hours_back: int = 24,
) -> List[Article]:
    """
    Fetch articles from GNews API.
    Free tier: 100 requests/day. Set GNEWS_API_KEY env var.
    """
    api_key = os.getenv("GNEWS_API_KEY", "")
    if not api_key:
        return []

    articles = []
    try:
        from_dt = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "q": query,
            "lang": "en",
            "max": max_results,
            "from": from_dt,
            "apikey": api_key,
            "sortby": "publishedAt",
        }
        resp = requests.get(GNEWS_API, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        for art in resp.json().get("articles", []):
            source_url = art.get("url", "")
            domain = source_url.split("/")[2] if source_url.startswith("http") else ""
            tier = _source_tier_from_domain(domain)

            try:
                pub_time = datetime.strptime(
                    art.get("publishedAt", ""), "%Y-%m-%dT%H:%M:%SZ"
                ).timestamp()
            except:
                pub_time = time.time()

            article = Article(
                article_id=f"gnews_{abs(hash(source_url)) % 999999}",
                source_name=art.get("source", {}).get("name", domain),
                tier=tier,
                publication_time=pub_time,
                headline=art.get("title", "")[:300],
                content_summary=(art.get("description") or "")[:500],
                url=source_url,
            )
            articles.append(article)

    except Exception as e:
        print(f"GNews fetch error: {e}")

    return articles


def fetch_newsapi(
    query: str,
    max_results: int = 10,
    hours_back: int = 24,
) -> List[Article]:
    """
    Fetch articles from NewsAPI.
    Free tier: 100 requests/day. Set NEWSAPI_KEY env var.
    """
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        return []

    articles = []
    try:
        from_dt = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "q": query,
            "from": from_dt,
            "sortBy": "publishedAt",
            "pageSize": max_results,
            "language": "en",
            "apiKey": api_key,
        }
        resp = requests.get(NEWSAPI_API, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        for art in resp.json().get("articles", []):
            source_url = art.get("url", "")
            domain = source_url.split("/")[2] if source_url.startswith("http") else ""
            tier = _source_tier_from_domain(domain)

            try:
                pub_time = datetime.strptime(
                    art.get("publishedAt", ""), "%Y-%m-%dT%H:%M:%SZ"
                ).timestamp()
            except:
                pub_time = time.time()

            article = Article(
                article_id=f"newsapi_{abs(hash(source_url)) % 999999}",
                source_name=art.get("source", {}).get("name", domain),
                tier=tier,
                publication_time=pub_time,
                headline=art.get("title", "")[:300],
                content_summary=(art.get("description") or "")[:500],
                url=source_url,
            )
            articles.append(article)

    except Exception as e:
        print(f"NewsAPI fetch error: {e}")

    return articles


def live_ingest(
    query: str,
    hours_back: int = 24,
    max_per_source: int = 10,
) -> List[Article]:
    """
    Pull from all available sources and deduplicate by headline similarity.
    Returns combined, deduplicated article list.
    """
    all_articles = []

    # GDELT — always available
    gdelt = fetch_gdelt(query, max_per_source, hours_back)
    all_articles.extend(gdelt)

    # GNews — if key available
    gnews = fetch_gnews(query, max_per_source, hours_back)
    all_articles.extend(gnews)

    # NewsAPI — if key available
    newsapi = fetch_newsapi(query, max_per_source, hours_back)
    all_articles.extend(newsapi)

    # Deduplicate by headline (Jaccard similarity > 0.7)
    deduped = []
    seen_headlines = []

    for article in all_articles:
        headline_tokens = set(article.headline.lower().split())
        is_duplicate = False

        for seen in seen_headlines:
            if not seen or not headline_tokens:
                continue
            jaccard = len(headline_tokens & seen) / len(headline_tokens | seen)
            if jaccard > 0.70:
                is_duplicate = True
                break

        if not is_duplicate:
            deduped.append(article)
            seen_headlines.append(set(article.headline.lower().split()))

    return sorted(deduped, key=lambda a: a.publication_time, reverse=True)


def build_query_from_event(event_description: str) -> str:
    """
    Extract key search terms from event description.
    Simple keyword extraction — upgrade with NLP for production.
    """
    stopwords = {"the", "a", "an", "is", "at", "or", "and", "of", "in",
                 "to", "for", "on", "with", "that", "this", "it", "by",
                 "from", "as", "are", "was", "were", "be", "been", "will"}
    words = event_description.lower().split()
    keywords = [w.strip(".,;:") for w in words if w not in stopwords and len(w) > 3]
    return " ".join(keywords[:6])  # Top 6 keywords
