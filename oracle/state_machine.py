"""
Oracle State Machine — Section 3.9.

States:
  BASELINE     — ρ(t) < ρ_min AND F(t) < 0.05. Prior dominates.
  SHOCK_ACTIVE — F(t) ≥ 0.05 AND ρ(t) < ρ_min.
  BUILDING     — ρ(t) ≥ ρ_min AND 0 < P(t) < 0.40.
  SUSTAINED    — P(t) ≥ 0.40 AND signal density maintained ≥ 14 days.
  CONTESTED    — |P_oracle(t) - P_tier1(t)| > 0.30 for ≥ 3 consecutive days.

The CONTESTED state is the paper's most distinctive contribution to oracle design.
It surfaces information warfare risk as a first-class output rather than
suppressing it behind a forced probability estimate.

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import time
from typing import List, Optional
from oracle.models import Article, OracleState, SourceTier
from oracle.config import CategoryConfig


SHOCK_ACTIVE_THRESHOLD = 0.05       # F(t) ≥ 0.05 triggers SHOCK_ACTIVE
BUILDING_TO_SUSTAINED_THRESHOLD = 0.40  # P(t) ≥ 0.40 for SUSTAINED
SUSTAINED_MIN_DAYS = 14


def compute_tier1_probability(
    articles: List[Article],
    current_time: float,
    config: CategoryConfig,
) -> Optional[float]:
    """
    Extract implied probability from Tier 1 sources only.
    Used for CONTESTED state detection.
    """
    tier1_articles = [
        a for a in articles
        if a.tier == SourceTier.TIER1_PRIMARY
        and a.publication_time <= current_time
        and (current_time - a.publication_time) / 86400.0 <= config.persistence_window_days
    ]

    if not tier1_articles:
        return None

    # Simple signed-impact average for Tier 1 only
    total_impact = sum(a.signed_impact for a in tier1_articles)
    # Normalise to a probability shift (rough heuristic)
    return max(0.02, min(0.98, 0.5 + total_impact * 0.05))


def determine_state(
    fast_shock: float,
    persistent_signal: float,
    signal_density: int,
    p_oracle: float,
    articles: List[Article],
    current_time: float,
    config: CategoryConfig,
    contested_history: List[dict],
) -> OracleState:
    """
    Determine oracle state based on current signal environment.

    contested_history: list of {"timestamp": t, "divergence": d}
    for tracking CONTESTED duration.
    """
    # Check for CONTESTED state first (overrides others)
    tier1_p = compute_tier1_probability(articles, current_time, config)
    if tier1_p is not None:
        divergence = abs(p_oracle - tier1_p)
        if divergence > config.contested_divergence_threshold:
            # Check how long divergence has persisted
            cutoff = current_time - (config.contested_duration_days * 86400)
            sustained_divergence = [
                h for h in contested_history
                if h["timestamp"] >= cutoff
                and h["divergence"] > config.contested_divergence_threshold
            ]
            if len(sustained_divergence) >= config.contested_duration_days:
                return OracleState.CONTESTED

    # SUSTAINED: strong persistent signal held for ≥ 14 days
    if abs(persistent_signal) >= BUILDING_TO_SUSTAINED_THRESHOLD:
        return OracleState.SUSTAINED

    # BUILDING: persistent signal accumulating
    if signal_density >= config.rho_min and abs(persistent_signal) > 0:
        return OracleState.BUILDING

    # SHOCK_ACTIVE: fast shock fired but no persistent confirmation yet
    if abs(fast_shock) >= SHOCK_ACTIVE_THRESHOLD:
        return OracleState.SHOCK_ACTIVE

    # INSUFFICIENT SIGNAL
    if signal_density == 0 and abs(fast_shock) < 0.02:
        return OracleState.INSUFFICIENT

    # Default: BASELINE — prior dominates
    return OracleState.BASELINE


class ContestationTracker:
    """Tracks CONTESTED state history for duration checks."""

    def __init__(self):
        self._history: List[dict] = []

    def record(self, timestamp: float, p_oracle: float, tier1_p: Optional[float]):
        if tier1_p is not None:
            divergence = abs(p_oracle - tier1_p)
        else:
            divergence = 0.0
        self._history.append({"timestamp": timestamp, "divergence": divergence})

    def get_history(self) -> List[dict]:
        return self._history

    def is_contested(
        self,
        current_time: float,
        threshold: float = 0.30,
        min_days: int = 3,
    ) -> bool:
        cutoff = current_time - (min_days * 86400)
        recent = [
            h for h in self._history
            if h["timestamp"] >= cutoff
            and h["divergence"] > threshold
        ]
        return len(recent) >= min_days
