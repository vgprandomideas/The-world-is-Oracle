"""
Source Independence Scorer — Layer 2 / Section 3.3.

Implements: ι(a_i) = 1 - max_{j ∈ A_{t_i}, j≠i} sim(a_i, a_j)

The core mechanism that prevents citation amplification attacks.
Three outlets citing one primary source count as ONE signal, not three.

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import re
import math
from typing import List, Optional
from oracle.models import Article


def _tokenize(text: str) -> set:
    """Simple tokenization for TF-IDF-style similarity."""
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'has',
        'have', 'had', 'was', 'were', 'are', 'its', 'not', 'but',
        'said', 'says', 'will', 'would', 'could', 'also', 'after',
        'been', 'than', 'more', 'they', 'their', 'about', 'into',
    }
    return set(t for t in tokens if t not in stopwords)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Jaccard similarity between two texts.
    Production systems should use sentence transformers or TF-IDF cosine.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def compute_independence_score(
    article: Article,
    existing_articles: List[Article],
    similarity_fn=jaccard_similarity,
) -> float:
    """
    ι(a_i) = 1 - max_{j ∈ A_{t_i}, j≠i} sim(a_i, a_j)

    Returns ι(a_i) ∈ [0, 1].
    Score of 1.0 = completely independent (no similar prior article).
    Score of 0.0 = exact duplicate of an existing article.

    Threshold θ_ind = 0.40 required to contribute to persistent signal.
    """
    if not existing_articles:
        return 1.0  # First article is always independent

    text_i = f"{article.headline} {article.content_summary}"

    max_similarity = 0.0
    for other in existing_articles:
        if other.article_id == article.article_id:
            continue
        text_j = f"{other.headline} {other.content_summary}"
        sim = similarity_fn(text_i, text_j)
        if sim > max_similarity:
            max_similarity = sim

    independence = 1.0 - max_similarity
    return round(independence, 4)


def score_all_articles(articles: List[Article]) -> List[Article]:
    """
    Score independence for a list of articles in chronological order.
    Each article is scored against all prior articles.
    """
    articles_sorted = sorted(articles, key=lambda a: a.publication_time)
    scored = []

    for i, article in enumerate(articles_sorted):
        prior_articles = articles_sorted[:i]
        article.independence_score = compute_independence_score(
            article, prior_articles
        )
        scored.append(article)

    return scored


def flag_citation_amplification(
    articles: List[Article],
    theta_ind: float = 0.40,
) -> dict:
    """
    Identify citation amplification patterns.
    Returns a report of which articles are derivative vs independent.

    This is the mechanism that caught the Hindenburg case:
    15 Bloomberg articles → 2 genuinely independent sources.
    """
    total = len(articles)
    independent = [a for a in articles if a.independence_score >= theta_ind]
    derivative = [a for a in articles if a.independence_score < theta_ind]

    amplification_ratio = len(derivative) / total if total > 0 else 0

    return {
        "total_articles": total,
        "independent_count": len(independent),
        "derivative_count": len(derivative),
        "amplification_ratio": round(amplification_ratio, 3),
        "genuine_signal_density": len(independent),
        "warning": amplification_ratio > 0.6,
        "independent_articles": [
            {"id": a.article_id, "source": a.source_name,
             "score": a.independence_score}
            for a in independent
        ],
        "derivative_articles": [
            {"id": a.article_id, "source": a.source_name,
             "score": a.independence_score}
            for a in derivative
        ],
    }
