"""
Temporal Persistence Filter — Layer 4 / Sections 3.5 and 3.6.

The paper's core contribution. Distinguishes genuine sustained information
flows from manufactured narrative bursts.

Two components:
  F(t) = Fast Shock — decays exponentially without confirmation, capped at F_max
  P(t) = Persistent Signal — threshold-gated, requires ρ_min independent sources

Formula 3.5 — Fast Shock with exponential decay:
  F(t) = min( Σ s_i · δ(t, t_i), F_max )
  δ(t, t_i) = exp( -λ · Δt · (1 - γ_c · min(n_c(t_i, t), 1)) )

Formula 3.6 — Persistent Signal Accumulator:
  P(t) = 0                                    if ρ(t) < ρ_min
  P(t) = min( Σ s_i · ω(t - t_i), P_max )    if ρ(t) ≥ ρ_min

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import math
import time
from typing import List, Optional
from oracle.models import Article
from oracle.config import CategoryConfig


def decay_factor(
    delta_t_days: float,
    decay_rate: float,
    n_confirming: int,
    gamma_c: float = 0.95,
) -> float:
    """
    δ(t, t_i) = exp( -λ · Δt · (1 - γ_c · min(n_c, 1)) )

    When n_confirming = 0: full decay at rate λ per day
    When n_confirming ≥ 1: decay effectively arrested (rate drops to λ · 0.05)
    """
    confirmation_factor = gamma_c * min(n_confirming, 1)
    effective_rate = decay_rate * (1.0 - confirmation_factor)
    return math.exp(-effective_rate * delta_t_days)


def count_confirming_signals(
    article: Article,
    all_articles: List[Article],
    theta_ind: float,
    theta_imp: float,
) -> int:
    """
    Count independent confirming signals published AFTER article.publication_time.
    A confirming signal must:
    - Be published after the focal article
    - Have independence_score ≥ θ_ind
    - Have |signed_impact| ≥ θ_imp
    - Have same direction as focal article (confirming, not contradicting)
    """
    count = 0
    for other in all_articles:
        if other.article_id == other.article_id and other is article:
            continue
        if other.publication_time <= article.publication_time:
            continue
        if other.independence_score < theta_ind:
            continue
        if abs(other.signed_impact) < theta_imp:
            continue
        # Must confirm (same direction)
        if other.direction == article.direction and article.direction != 0:
            count += 1
    return count


def compute_fast_shock(
    articles: List[Article],
    current_time: float,
    config: CategoryConfig,
) -> float:
    """
    F(t) = min( Σ_{a_i ∈ A_t} s_i · δ(t, t_i), F_max )

    Sums all articles' signed impacts weighted by their decay factor.
    Note: F(t) ∈ [-F_max, F_max] — can be negative when counter-signals dominate.
    """
    total = 0.0

    for article in articles:
        delta_t_days = (current_time - article.publication_time) / 86400.0
        if delta_t_days < 0:
            continue

        n_confirming = count_confirming_signals(
            article, articles, config.theta_ind, config.theta_imp
        )
        d = decay_factor(
            delta_t_days,
            config.decay_rate_lambda,
            n_confirming,
            config.gamma_c,
        )
        total += article.signed_impact * d

    # Cap: [-F_max, F_max]
    return max(-config.f_max, min(config.f_max, total))


def get_qualifying_articles(
    articles: List[Article],
    current_time: float,
    config: CategoryConfig,
) -> List[Article]:
    """
    I(t) = { a_i : ι(a_i) ≥ θ_ind  AND  |s_i| ≥ θ_imp  AND  t - W ≤ t_i ≤ t }
    """
    window_start = current_time - (config.persistence_window_days * 86400.0)
    qualifying = []

    for article in articles:
        if article.publication_time < window_start:
            continue
        if article.publication_time > current_time:
            continue
        if article.independence_score < config.theta_ind:
            continue
        if abs(article.signed_impact) < config.theta_imp:
            continue
        qualifying.append(article)

    return qualifying


def recency_weight(age_days: float, all_ages: List[float]) -> float:
    """
    ω(τ) = exp(-0.05τ) normalised over I(t)
    Gentle recency weighting within the persistence window.
    """
    raw = math.exp(-0.05 * age_days)
    return raw  # Normalisation applied over the full set in compute_persistent_signal


def compute_persistent_signal(
    articles: List[Article],
    current_time: float,
    config: CategoryConfig,
) -> tuple[float, int, List[Article]]:
    """
    P(t) = 0                                   if ρ(t) < ρ_min
    P(t) = min( Σ s_i · ω(t - t_i), P_max )   if ρ(t) ≥ ρ_min

    Returns: (P(t), signal_density ρ(t), qualifying_articles)

    Contradiction override: if a Tier 1 source fires a strong counter-signal
    (|s_i| ≥ 7) in the OPPOSITE direction to the dominant signal, the
    persistent accumulator is immediately reduced.
    """
    qualifying = get_qualifying_articles(articles, current_time, config)
    rho = len(qualifying)

    if rho < config.rho_min:
        return 0.0, rho, qualifying

    # Compute normalised recency weights
    ages = [
        (current_time - a.publication_time) / 86400.0
        for a in qualifying
    ]
    raw_weights = [math.exp(-0.05 * age) for age in ages]
    total_weight = sum(raw_weights)
    if total_weight == 0:
        return 0.0, rho, qualifying

    # Weighted sum of signed impacts
    total = sum(
        a.signed_impact * (w / total_weight)
        for a, w in zip(qualifying, raw_weights)
    )

    # Contradiction handling: Tier 1 strong counter-signal override
    from oracle.models import SourceTier
    for a in qualifying:
        if a.tier == SourceTier.TIER1_PRIMARY and abs(a.signed_impact) >= 7:
            # Check if it's opposing the dominant direction
            dominant_positive = total > 0
            if (dominant_positive and a.direction < 0) or \
               (not dominant_positive and a.direction > 0):
                # Immediately reduce persistent accumulator
                total *= 0.5

    # P(t) ∈ [-P_max, P_max]
    p_t = max(-config.p_max, min(config.p_max, total))
    return p_t, rho, qualifying
