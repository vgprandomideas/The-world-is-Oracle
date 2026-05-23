"""
Cross-Event Correlation Engine

Models correlation between events and propagates probability shocks
across correlated events.

Example correlation pairs:
  Adani US action ↔ India sovereign credit ↔ USD/INR
  Fed rate hike ↔ SOFR ↔ Treasury yields ↔ EM credit spreads
  China tech regulation ↔ Hong Kong listings ↔ ADR spreads

Uses Cholesky decomposition for joint simulation of correlated
probability paths — directly feeds Monte Carlo VaR for OTC desk.

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# Pre-defined correlation templates for common event pairs
CORRELATION_TEMPLATES = {
    ("adani_us_action", "india_sovereign_credit"):    0.65,
    ("adani_us_action", "usd_inr_move"):              0.45,
    ("adani_us_action", "india_equity_selloff"):      0.55,
    ("fed_rate_hike", "sofr_rate_move"):              0.92,
    ("fed_rate_hike", "usd_strength"):                0.70,
    ("fed_rate_hike", "em_credit_spread"):            0.60,
    ("fed_rate_hike", "treasury_yield_move"):         0.88,
    ("china_tech_regulation", "hk_listings"):         0.75,
    ("china_tech_regulation", "adr_spreads"):         0.68,
    ("sovereign_default", "cds_spread"):              0.90,
    ("electoral_outcome", "currency_move"):           0.55,
}


class CorrelationMatrix:
    """
    Manages a dynamic correlation matrix across active oracle events.
    Supports real-time correlation updates as new evidence arrives.
    """

    def __init__(self):
        self._events: List[str] = []
        self._matrix: np.ndarray = np.array([[]])
        self._custom_correlations: Dict[Tuple, float] = {}

    def add_event(self, event_id: str):
        if event_id in self._events:
            return
        n = len(self._events)
        self._events.append(event_id)

        # Expand matrix
        new_matrix = np.eye(n + 1)
        if n > 0:
            new_matrix[:n, :n] = self._matrix
        self._matrix = new_matrix

        # Apply known correlations
        for i, other in enumerate(self._events[:-1]):
            rho = self._lookup_correlation(event_id, other)
            self._matrix[n, i] = rho
            self._matrix[i, n] = rho

    def set_correlation(self, event_a: str, event_b: str, rho: float):
        """Manually set correlation between two events."""
        rho = max(-0.99, min(0.99, rho))
        self._custom_correlations[(event_a, event_b)] = rho
        self._custom_correlations[(event_b, event_a)] = rho

        if event_a in self._events and event_b in self._events:
            i = self._events.index(event_a)
            j = self._events.index(event_b)
            self._matrix[i, j] = rho
            self._matrix[j, i] = rho

    def _lookup_correlation(self, event_a: str, event_b: str) -> float:
        """Look up correlation from custom overrides or template library."""
        if (event_a, event_b) in self._custom_correlations:
            return self._custom_correlations[(event_a, event_b)]

        # Fuzzy match against templates
        for (a_key, b_key), rho in CORRELATION_TEMPLATES.items():
            if (a_key in event_a and b_key in event_b) or \
               (a_key in event_b and b_key in event_a):
                return rho

        return 0.0  # Default: uncorrelated

    def get_correlation(self, event_a: str, event_b: str) -> float:
        if event_a not in self._events or event_b not in self._events:
            return 0.0
        i = self._events.index(event_a)
        j = self._events.index(event_b)
        return float(self._matrix[i, j])

    def is_positive_definite(self) -> bool:
        if len(self._events) == 0:
            return True
        try:
            np.linalg.cholesky(self._matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    def make_positive_definite(self):
        """Higham nearest PD matrix algorithm — ensures Cholesky decomposable."""
        if len(self._events) <= 1:
            return
        eigvals, eigvecs = np.linalg.eigh(self._matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        self._matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-normalise diagonal to 1
        d = np.sqrt(np.diag(self._matrix))
        self._matrix = self._matrix / np.outer(d, d)

    def propagate_shock(
        self,
        source_event: str,
        shock_magnitude: float,
        current_probabilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Propagate a probability shock from source_event to correlated events.

        shock_magnitude: change in P_oracle for source_event (e.g. +0.15)
        Returns: {event_id: implied_probability_change} for all events
        """
        if source_event not in self._events:
            return {}

        source_idx = self._events.index(source_event)
        propagated = {}

        for i, event_id in enumerate(self._events):
            if event_id == source_event:
                propagated[event_id] = shock_magnitude
                continue

            rho = self._matrix[source_idx, i]
            if abs(rho) < 0.05:
                continue

            # Implied shock via correlation
            p_current = current_probabilities.get(event_id, 0.50)
            p_source = current_probabilities.get(source_event, 0.50)

            # Vol-adjusted propagation
            source_vol = math.sqrt(p_source * (1 - p_source))
            target_vol = math.sqrt(p_current * (1 - p_current))

            if source_vol > 0:
                implied_change = rho * (target_vol / source_vol) * shock_magnitude
            else:
                implied_change = rho * shock_magnitude

            propagated[event_id] = round(implied_change, 4)

        return propagated

    def joint_simulation(
        self,
        current_probabilities: Dict[str, float],
        daily_vols: Dict[str, float],
        n_paths: int = 10000,
        n_days: int = 30,
    ) -> Dict[str, dict]:
        """
        Monte Carlo joint simulation of correlated probability paths.
        Uses Cholesky decomposition to generate correlated shocks.

        Returns percentile distribution for each event.
        """
        if len(self._events) == 0:
            return {}

        if not self.is_positive_definite():
            self.make_positive_definite()

        try:
            L = np.linalg.cholesky(self._matrix)
        except np.linalg.LinAlgError:
            self.make_positive_definite()
            L = np.linalg.cholesky(self._matrix)

        n_events = len(self._events)
        results = {}

        # Simulate paths
        rng = np.random.default_rng(42)
        final_probs = np.zeros((n_paths, n_events))

        for path in range(n_paths):
            p_current = np.array([
                current_probabilities.get(e, 0.5) for e in self._events
            ])

            for day in range(n_days):
                # Generate correlated standard normals
                z = rng.standard_normal(n_events)
                correlated_z = L @ z

                # Daily probability moves
                vols = np.array([daily_vols.get(e, 0.08) for e in self._events])
                dp = vols * correlated_z

                # Update in logit space for bounded moves
                logits = np.array([logit_safe(p) for p in p_current])
                logits += dp * 3.0  # Scale to logit space
                p_current = np.array([sigmoid_safe(l) for l in logits])

            final_probs[path] = p_current

        for i, event_id in enumerate(self._events):
            probs = final_probs[:, i]
            results[event_id] = {
                "mean": round(float(np.mean(probs)), 4),
                "std": round(float(np.std(probs)), 4),
                "p5": round(float(np.percentile(probs, 5)), 4),
                "p25": round(float(np.percentile(probs, 25)), 4),
                "p50": round(float(np.percentile(probs, 50)), 4),
                "p75": round(float(np.percentile(probs, 75)), 4),
                "p95": round(float(np.percentile(probs, 95)), 4),
            }

        return results

    def to_dict(self) -> dict:
        return {
            "events": self._events,
            "matrix": self._matrix.tolist() if len(self._events) > 0 else [],
        }


import math

def logit_safe(p: float) -> float:
    p = max(0.02, min(0.98, p))
    return math.log(p / (1 - p))

def sigmoid_safe(x: float) -> float:
    x = max(-10, min(10, x))
    return 1.0 / (1.0 + math.exp(-x))
