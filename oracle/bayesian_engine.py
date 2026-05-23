"""
Bayesian Probability Engine — Layer 5 (Upgraded)

Replaces linear composition with proper Bayesian posterior updating.
Each signal updates the posterior via likelihood ratio scoring.

P(E|s₁,s₂,...,sₙ) ∝ P(E) × Π L(sᵢ|E) / Π L(sᵢ|¬E)

Likelihood ratios are computed from:
  - Source tier credibility c(τ)
  - Independence score ι(aᵢ)
  - Signed impact sᵢ
  - Temporal decay δ(t,tᵢ)

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from oracle.models import Article
from oracle.config import CategoryConfig


def logit(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def likelihood_ratio(article: Article, decay: float) -> float:
    """
    L(s|E) / L(s|¬E) — likelihood ratio for a single signal.

    Positive direction → LR > 1 (evidence for E)
    Negative direction → LR < 1 (evidence against E)

    LR = exp( s_i · ι(a_i) · δ(t,t_i) · scale )
    where scale maps [1,10] impact to meaningful probability updates.
    """
    if article.direction == 0:
        return 1.0  # Neutral signal — no update

    # Scaled log-likelihood ratio
    log_lr = (
        article.raw_impact *
        article.credibility *
        article.independence_score *
        decay *
        article.direction *
        0.15  # Scale factor: impact=10, tier1, full independence → LR ≈ 4.5x
    )
    return math.exp(log_lr)


def bayesian_update(
    prior: float,
    articles: List[Article],
    decays: List[float],
    min_p: float = 0.02,
    max_p: float = 0.98,
) -> Tuple[float, float, List[float]]:
    """
    Sequential Bayesian update across all articles.

    Returns: (posterior, log_odds_final, [individual_lrs])

    Sequential update is equivalent to joint update when signals
    are conditionally independent given E — which our independence
    scorer enforces.
    """
    # Start from prior in log-odds space
    log_odds = logit(prior)
    individual_lrs = []

    for article, decay in zip(articles, decays):
        lr = likelihood_ratio(article, decay)
        individual_lrs.append(lr)
        if lr > 0:
            log_odds += math.log(lr)  # Sequential Bayesian update

    posterior = sigmoid(log_odds)
    posterior = max(min_p, min(max_p, posterior))
    return posterior, log_odds, individual_lrs


def compute_posterior_with_uncertainty(
    prior: float,
    articles: List[Article],
    decays: List[float],
    n_bootstrap: int = 500,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap uncertainty quantification around the Bayesian posterior.

    Resamples the article set with replacement to estimate:
    - Mean posterior
    - Posterior standard deviation (epistemic uncertainty)
    - 5th and 95th percentile CI

    Returns: (mean_posterior, std_posterior, ci_5, ci_95)
    """
    if not articles:
        return prior, 0.20, max(0.02, prior - 0.20), min(0.98, prior + 0.20)

    posteriors = []
    n = len(articles)
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        # Resample articles with replacement
        indices = rng.integers(0, n, size=n)
        sampled_articles = [articles[i] for i in indices]
        sampled_decays = [decays[i] for i in indices]

        p, _, _ = bayesian_update(prior, sampled_articles, sampled_decays)
        posteriors.append(p)

    posteriors = np.array(posteriors)
    return (
        float(np.mean(posteriors)),
        float(np.std(posteriors)),
        float(np.percentile(posteriors, 5)),
        float(np.percentile(posteriors, 95)),
    )


class BayesianOracleEngine:
    """
    Full Bayesian oracle engine with uncertainty quantification.
    Replaces the linear P₀ + F(t) + P(t) composition.
    """

    def __init__(self, prior: float, config: CategoryConfig):
        self.prior = prior
        self.config = config
        self._posterior_history: List[dict] = []

    def compute(
        self,
        articles: List[Article],
        current_time: float,
        use_bootstrap: bool = True,
    ) -> dict:
        import time as _time
        from oracle.temporal_filter import decay_factor, count_confirming_signals

        # Compute temporal decay for each article
        qualifying = []
        decays = []

        for article in articles:
            delta_t = (current_time - article.publication_time) / 86400.0
            if delta_t < 0:
                continue
            if article.independence_score < self.config.theta_ind:
                continue
            if abs(article.signed_impact) < self.config.theta_imp:
                continue

            n_confirming = count_confirming_signals(
                article, articles,
                self.config.theta_ind, self.config.theta_imp
            )
            d = decay_factor(
                delta_t, self.config.decay_rate_lambda,
                n_confirming, self.config.gamma_c
            )
            qualifying.append(article)
            decays.append(d)

        if not qualifying:
            return {
                "posterior": self.prior,
                "std": 0.20,
                "ci_5": max(0.02, self.prior - 0.20),
                "ci_95": min(0.98, self.prior + 0.20),
                "log_odds": logit(self.prior),
                "n_signals": 0,
                "dominant_lr": 1.0,
            }

        if use_bootstrap:
            mean_p, std_p, ci5, ci95 = compute_posterior_with_uncertainty(
                self.prior, qualifying, decays
            )
        else:
            mean_p, log_odds, lrs = bayesian_update(self.prior, qualifying, decays)
            std_p = 0.10
            ci5 = max(0.02, mean_p - 1.645 * std_p)
            ci95 = min(0.98, mean_p + 1.645 * std_p)

        # Dominant likelihood ratio (most influential signal)
        _, _, lrs = bayesian_update(self.prior, qualifying, decays)
        dominant_lr = max(lrs, key=abs) if lrs else 1.0

        result = {
            "posterior": mean_p,
            "std": std_p,
            "ci_5": ci5,
            "ci_95": ci95,
            "log_odds": logit(mean_p),
            "n_signals": len(qualifying),
            "dominant_lr": dominant_lr,
        }

        self._posterior_history.append({
            "timestamp": current_time,
            **result
        })
        return result
