"""
Probability Composition Engine — Layer 5 / Section 3.7.
Calibration with Platt Scaling — Layer 6 / Section 3.7 and 3.11.

Full probability composition:
  P_raw(t) = P₀(E) + F(t) + P(t)
  P_raw_c(t) = clamp(P_raw(t), 0.01, 0.99)
  P_oracle(t) = σ( a · logit(P_raw_c(t)) + b ), constrained to [0.02, 0.98]

Platt scaling parameters (a, b) are estimated from historical event data.
Default (a=1.0, b=0.0) means no calibration correction applied.

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import math
import json
from typing import Optional, List, Tuple
from oracle.config import CategoryConfig


ORACLE_P_MIN = 0.02
ORACLE_P_MAX = 0.98


def logit(p: float) -> float:
    """logit(p) = log(p / (1 - p))"""
    p = max(1e-9, min(1 - 1e-9, p))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """σ(x) = 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + math.exp(-x))


def compute_prior(
    historical_base_rate: float,
    structural_prior: float,
    market_implied_prior: Optional[float],
    config: CategoryConfig,
) -> float:
    """
    P₀(E) = α · H(E) + β · S(E) + γ · M(E), where α + β + γ = 1

    α = historical base rate weight
    β = structural prior weight
    γ = market-implied prior weight (down-weighted or excluded for single-name events)
    """
    if market_implied_prior is None:
        # Redistribute γ weight to α and β proportionally
        total = config.alpha + config.beta
        alpha_adj = config.alpha / total
        beta_adj = config.beta / total
        return alpha_adj * historical_base_rate + beta_adj * structural_prior
    else:
        return (
            config.alpha * historical_base_rate
            + config.beta * structural_prior
            + config.gamma * market_implied_prior
        )


def compose_probability(
    prior: float,
    fast_shock: float,
    persistent_signal: float,
    platt_a: float = 1.0,
    platt_b: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Full probability composition with Platt scaling.

    Returns: (p_oracle, p_raw, p_raw_clamped)

    Step 1: P_raw = P₀ + F(t) + P(t)
    Step 2: P_raw_c = clamp(P_raw, 0.01, 0.99)   ← domain safety for logit
    Step 3: P_oracle = σ(a · logit(P_raw_c) + b)
    Step 4: Constrain to [ORACLE_P_MIN, ORACLE_P_MAX]
    """
    p_raw = prior + fast_shock + persistent_signal

    # Clamp before logit (domain validity — Section 3.7 fix)
    p_raw_c = max(0.01, min(0.99, p_raw))

    # Platt scaling
    p_oracle_raw = sigmoid(platt_a * logit(p_raw_c) + platt_b)

    # Final constraint to oracle output range
    p_oracle = max(ORACLE_P_MIN, min(ORACLE_P_MAX, p_oracle_raw))

    return p_oracle, p_raw, p_raw_c


def compute_confidence_interval(
    n_independent: int,
    quality_factor: float,
    is_contested: bool = False,
    sigma_base: float = 0.20,
) -> Tuple[float, float]:
    """
    CI_width(t) = σ_base / √( n_ind(t) · q(t) )

    sigma_base = 0.20 (default; category-specific calibration required)
    Under CONTESTED status: CI_width is doubled.

    Returns: (ci_half_width)
    """
    if n_independent == 0 or quality_factor == 0:
        ci_half = sigma_base  # Maximum uncertainty
    else:
        ci_half = sigma_base / math.sqrt(n_independent * quality_factor)

    if is_contested:
        ci_half = min(ci_half * 2.0, 0.49)  # Double but cap at valid range

    return ci_half


def compute_quality_factor(articles: list) -> float:
    """
    q(t) = weighted average credibility of contributing independent sources.
    """
    if not articles:
        return 0.0
    return sum(a.credibility for a in articles) / len(articles)


class PlattScaler:
    """
    Platt scaling calibration layer.

    Fits: P_calibrated = σ(a · logit(P_raw) + b)
    Parameters (a, b) are estimated via maximum likelihood on historical data.

    Recalibration triggered when 60-day rolling Brier Score exceeds 0.18.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, category: str = "default"):
        self.a = a
        self.b = b
        self.category = category
        self._calibration_data: List[Tuple[float, int]] = []  # (p_raw, outcome)

    def fit(self, p_raw_scores: List[float], outcomes: List[int]):
        """
        Fit Platt scaling parameters using sklearn's LogisticRegression.
        p_raw_scores: list of raw oracle probabilities
        outcomes: list of binary outcomes (0 or 1)
        """
        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            logits = [logit(p) for p in p_raw_scores]
            X = np.array(logits).reshape(-1, 1)
            y = np.array(outcomes)

            lr = LogisticRegression(C=1e10, fit_intercept=True)
            lr.fit(X, y)

            self.a = float(lr.coef_[0][0])
            self.b = float(lr.intercept_[0])

        except ImportError:
            print("scikit-learn required for Platt scaling fitting. Using defaults.")

    def transform(self, p_raw: float) -> float:
        """Apply Platt scaling to a raw probability."""
        p_raw_c = max(0.01, min(0.99, p_raw))
        scaled = sigmoid(self.a * logit(p_raw_c) + self.b)
        return max(ORACLE_P_MIN, min(ORACLE_P_MAX, scaled))

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({"a": self.a, "b": self.b, "category": self.category}, f)

    @classmethod
    def load(cls, path: str) -> "PlattScaler":
        with open(path) as f:
            data = json.load(f)
        return cls(a=data["a"], b=data["b"], category=data.get("category", "default"))


class BrierScoreTracker:
    """
    Tracks Brier Score and triggers recalibration when threshold exceeded.

    BS = (1/T) · Σ (P_oracle(t) - O_t)²
    Target: BS < 0.12
    Recalibration trigger: 60-day rolling BS > 0.18
    """

    def __init__(self, recalibration_threshold: float = 0.18, window_days: int = 60):
        self.recalibration_threshold = recalibration_threshold
        self.window_days = window_days
        self._records: List[dict] = []

    def record(self, timestamp: float, p_oracle: float, outcome: int):
        """Record a resolved prediction."""
        self._records.append({
            "timestamp": timestamp,
            "p_oracle": p_oracle,
            "outcome": outcome,
            "squared_error": (p_oracle - outcome) ** 2,
        })

    def rolling_brier_score(self, current_time: float) -> Optional[float]:
        """Compute Brier Score over the last window_days."""
        cutoff = current_time - (self.window_days * 86400)
        recent = [r for r in self._records if r["timestamp"] >= cutoff]
        if not recent:
            return None
        return sum(r["squared_error"] for r in recent) / len(recent)

    def overall_brier_score(self) -> Optional[float]:
        """Compute Brier Score over all recorded predictions."""
        if not self._records:
            return None
        return sum(r["squared_error"] for r in self._records) / len(self._records)

    def needs_recalibration(self, current_time: float) -> bool:
        """True if rolling Brier Score exceeds recalibration threshold."""
        score = self.rolling_brier_score(current_time)
        if score is None:
            return False
        return score > self.recalibration_threshold
