"""
LLM Impact Scorer — Layer 3.

Uses Claude to score each article's causal impact on event probability
through a structured 6-step chain-of-thought reasoning process.

KEY DESIGN PRINCIPLE: The LLM does not output a probability.
It outputs an impact score (1-10) and direction (+1/-1/0).
The probability is computed deterministically in Layer 5.
This isolates LLM overconfidence from pricing-grade output.

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import json
import anthropic
from oracle.models import Article, SourceTier

client = anthropic.Anthropic()

IMPACT_SCORING_PROMPT = """You are an expert event probability analyst for a financial oracle system.

Your task: Score the causal impact of this article on the probability that the following event resolves YES.

EVENT: {event_description}
EVENT RESOLUTION CRITERIA: {resolution_criteria}

ARTICLE TO SCORE:
Source: {source_name} (Tier {tier}: {tier_description})
Headline: {headline}
Content: {content}

Follow these 6 steps EXACTLY. Output ONLY valid JSON.

STEP 1 — FACTUAL CLAIM: What is the primary factual claim in this article?
STEP 2 — SOURCE TYPE: Is this primary reporting (new facts) or derivative (restating prior sources)?
STEP 3 — DIRECTION: Does this increase (+1), decrease (-1), or not affect (0) the probability the event resolves YES?
STEP 4 — MAGNITUDE: On a scale of 1-10, how strongly does this move the probability? (1=negligible, 10=near-certain resolution)
STEP 5 — COUNTER-EVIDENCE: What is the strongest counter-argument against this article's implied direction?
STEP 6 — CONFIDENCE: How confident are you in your assessment? (low/medium/high)

Return ONLY this JSON, no other text:
{{
  "factual_claim": "...",
  "is_primary_reporting": true,
  "direction": 1,
  "magnitude": 7,
  "reasoning": "...",
  "counter_evidence": "...",
  "confidence": "medium"
}}"""

TIER_DESCRIPTIONS = {
    SourceTier.TIER1_PRIMARY: "Primary source — court filing, regulatory order, official statement",
    SourceTier.TIER2_LOCAL_PRESS: "Local financial press — accountable to regional readership",
    SourceTier.TIER3_REGIONAL: "Regional international press",
    SourceTier.TIER4_WIRE: "International wire — high volume, geopolitical editorial filters apply",
    SourceTier.TIER5_CAPITAL_FLOW: "Capital flow signal — real money expressing real conviction",
    SourceTier.TIER6_UNVERIFIED: "Unverified single-source or social media",
}


def score_article_impact(
    article: Article,
    event_description: str,
    resolution_criteria: str,
    model: str = "claude-sonnet-4-20250514",
) -> Article:
    """
    Score a single article's causal impact using Claude.
    Populates article.raw_impact, article.direction, article.reasoning_chain.
    Returns the article with scores populated.
    """
    prompt = IMPACT_SCORING_PROMPT.format(
        event_description=event_description,
        resolution_criteria=resolution_criteria,
        source_name=article.source_name,
        tier=article.tier.value,
        tier_description=TIER_DESCRIPTIONS.get(article.tier, ""),
        headline=article.headline,
        content=article.content_summary,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)

        article.raw_impact = float(result.get("magnitude", 3))
        article.direction = int(result.get("direction", 0))
        article.reasoning_chain = result.get("reasoning", "")
        article.counter_evidence = result.get("counter_evidence", "")

    except Exception as e:
        # Fallback: neutral score on error
        article.raw_impact = 0.0
        article.direction = 0
        article.reasoning_chain = f"Scoring error: {e}"

    return article


def batch_score_articles(
    articles: list,
    event_description: str,
    resolution_criteria: str,
) -> list:
    """Score a batch of articles. Returns articles with impact scores populated."""
    return [
        score_article_impact(a, event_description, resolution_criteria)
        for a in articles
    ]


def create_article_from_dict(data: dict) -> Article:
    """
    Helper to create Article objects for manual testing without live API.

    data = {
        "article_id": "art_001",
        "source_name": "DOJ Press Office",
        "tier": 1,
        "publication_time": 1732060000.0,
        "headline": "...",
        "content_summary": "...",
        "raw_impact": 9,       # Optional: pre-scored
        "direction": -1,
    }
    """
    from oracle.models import SourceTier
    tier_map = {i: SourceTier(i) for i in range(1, 7)}

    article = Article(
        article_id=data["article_id"],
        source_name=data["source_name"],
        tier=tier_map.get(data.get("tier", 4), SourceTier.TIER4_WIRE),
        publication_time=data["publication_time"],
        headline=data["headline"],
        content_summary=data.get("content_summary", ""),
        url=data.get("url", ""),
        raw_impact=data.get("raw_impact", 0.0),
        direction=data.get("direction", 0),
        reasoning_chain=data.get("reasoning_chain", "Pre-scored"),
    )
    return article
