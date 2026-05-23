"""
The World-as-Oracle — Main Oracle Class.

Integrates all 7 layers into a single interface.

Usage:
    oracle = WorldOracle(
        event_id="fed_rate_jun2023",
        event_description="Federal Reserve raises interest rates at June 2023 FOMC meeting",
        resolution_criteria="Fed funds rate increases by 25bps or more at June 14 2023 meeting",
        category=EventCategory.CENTRAL_BANK,
    )

    oracle.set_prior(
        historical_base_rate=0.65,
        structural_prior=0.60,
        market_implied_prior=0.72,
    )

    oracle.ingest_articles(articles)
    output = oracle.compute(current_time=time.time())
    print(output)

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import time
import json
from typing import List, Optional
from oracle.models import Article, OracleOutput, OracleState, EventCategory
from oracle.config import CategoryConfig, CATEGORY_CONFIGS
from oracle.independence import score_all_articles
from oracle.impact_scorer import score_article_impact
from oracle.temporal_filter import compute_fast_shock, compute_persistent_signal
from oracle.probability import (
    compute_prior, compose_probability,
    compute_confidence_interval, compute_quality_factor,
    PlattScaler, BrierScoreTracker,
)
from oracle.state_machine import (
    determine_state, ContestationTracker, compute_tier1_probability,
)


class WorldOracle:
    """
    The World-as-Oracle probability estimation engine.

    Architecture: 7-layer pipeline as specified in the paper.
    Layer 1: Data ingestion (external — articles fed via ingest_articles)
    Layer 2: Independence scoring (automatic)
    Layer 3: Impact scoring (LLM via Claude — or pre-scored)
    Layer 4: Temporal persistence filter
    Layer 5: Probability composition engine
    Layer 6: Platt scaling calibration
    Layer 7: Structured output with audit trail
    """

    def __init__(
        self,
        event_id: str,
        event_description: str,
        resolution_criteria: str,
        category: EventCategory = EventCategory.CORPORATE_LEGAL,
        platt_a: float = 1.0,
        platt_b: float = 0.0,
    ):
        self.event_id = event_id
        self.event_description = event_description
        self.resolution_criteria = resolution_criteria
        self.config: CategoryConfig = CATEGORY_CONFIGS[category]
        self.platt_scaler = PlattScaler(a=platt_a, b=platt_b, category=category.value)
        self.brier_tracker = BrierScoreTracker()
        self.contestation_tracker = ContestationTracker()

        # State
        self._articles: List[Article] = []
        self._prior: float = 0.30  # Default prior
        self._history: List[OracleOutput] = []

    def set_prior(
        self,
        historical_base_rate: float,
        structural_prior: float,
        market_implied_prior: Optional[float] = None,
    ) -> float:
        """Set P₀(E) — the base rate before any signals."""
        self._prior = compute_prior(
            historical_base_rate,
            structural_prior,
            market_implied_prior,
            self.config,
        )
        return self._prior

    def ingest_articles(
        self,
        articles: List[Article],
        auto_score_independence: bool = True,
        auto_score_impact: bool = False,  # Set True to call Claude LLM scoring
    ) -> List[Article]:
        """
        Layer 1 → Layer 2 → Layer 3 pipeline.

        auto_score_independence: compute ι(a_i) for each article
        auto_score_impact: use Claude to score raw_impact and direction
        """
        all_articles = self._articles + articles

        if auto_score_independence:
            all_articles = score_all_articles(all_articles)

        if auto_score_impact:
            for article in articles:
                if article.raw_impact == 0.0:
                    score_article_impact(
                        article,
                        self.event_description,
                        self.resolution_criteria,
                    )

        self._articles = all_articles
        return self._articles

    def compute(
        self,
        current_time: Optional[float] = None,
    ) -> OracleOutput:
        """
        Run the full oracle computation for a given moment in time.

        Executes Layers 4 → 5 → 6 → 7.
        Returns a structured OracleOutput with full audit trail.
        """
        if current_time is None:
            current_time = time.time()

        # Layer 4: Temporal Persistence Filter
        fast_shock = compute_fast_shock(
            self._articles, current_time, self.config
        )
        persistent_signal, signal_density, qualifying_articles = \
            compute_persistent_signal(
                self._articles, current_time, self.config
            )

        # Layer 5: Probability Composition
        p_oracle, p_raw, p_raw_c = compose_probability(
            prior=self._prior,
            fast_shock=fast_shock,
            persistent_signal=persistent_signal,
            platt_a=self.platt_scaler.a,
            platt_b=self.platt_scaler.b,
        )

        # Quality factor for CI
        quality = compute_quality_factor(qualifying_articles)
        n_independent = len(qualifying_articles)

        # Contestation tracking
        tier1_p = compute_tier1_probability(
            self._articles, current_time, self.config
        )
        self.contestation_tracker.record(current_time, p_oracle, tier1_p)

        # Determine state
        state = determine_state(
            fast_shock=fast_shock,
            persistent_signal=persistent_signal,
            signal_density=signal_density,
            p_oracle=p_oracle,
            articles=self._articles,
            current_time=current_time,
            config=self.config,
            contested_history=self.contestation_tracker.get_history(),
        )

        # Layer 6: Confidence interval
        is_contested = (state == OracleState.CONTESTED)
        ci_half = compute_confidence_interval(
            n_independent=n_independent,
            quality_factor=quality if quality > 0 else 0.1,
            is_contested=is_contested,
        )
        ci_lower = max(0.0, p_oracle - ci_half)
        ci_upper = min(1.0, p_oracle + ci_half)

        # Layer 7: Structured output
        output = OracleOutput(
            event_id=self.event_id,
            timestamp=current_time,
            probability=p_oracle,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            state=state,
            prior=self._prior,
            fast_shock=fast_shock,
            persistent_signal=persistent_signal,
            p_raw=p_raw,
            independent_source_count=n_independent,
            total_articles_processed=len(self._articles),
            audit_trail=[
                {
                    "article_id": a.article_id,
                    "source": a.source_name,
                    "tier": a.tier.value,
                    "independence_score": a.independence_score,
                    "signed_impact": round(a.signed_impact, 3),
                    "reasoning": a.reasoning_chain[:200] if a.reasoning_chain else "",
                }
                for a in qualifying_articles
            ],
        )

        self._history.append(output)
        return output

    def record_resolution(self, outcome: int, timestamp: Optional[float] = None):
        """
        Record the actual event resolution (0 or 1).
        Updates Brier score tracking for calibration monitoring.
        """
        if not self._history:
            return
        if timestamp is None:
            timestamp = time.time()
        last_output = self._history[-1]
        self.brier_tracker.record(timestamp, last_output.probability, outcome)

    def calibration_report(self) -> dict:
        """Return current calibration metrics."""
        return {
            "event_id": self.event_id,
            "total_outputs": len(self._history),
            "overall_brier_score": self.brier_tracker.overall_brier_score(),
            "needs_recalibration": self.brier_tracker.needs_recalibration(
                time.time()
            ),
            "platt_a": self.platt_scaler.a,
            "platt_b": self.platt_scaler.b,
            "category": self.config.category.value,
        }

    def to_json(self) -> str:
        """Export oracle state for persistence."""
        return json.dumps({
            "event_id": self.event_id,
            "event_description": self.event_description,
            "category": self.config.category.value,
            "prior": self._prior,
            "platt_a": self.platt_scaler.a,
            "platt_b": self.platt_scaler.b,
            "total_articles": len(self._articles),
            "history_count": len(self._history),
        }, indent=2)

    def history_dataframe(self):
        """Export history as pandas DataFrame if available."""
        try:
            import pandas as pd
            rows = []
            for o in self._history:
                rows.append({
                    "timestamp": o.timestamp,
                    "probability": o.probability,
                    "ci_lower": o.ci_lower,
                    "ci_upper": o.ci_upper,
                    "state": o.state.value,
                    "fast_shock": o.fast_shock,
                    "persistent_signal": o.persistent_signal,
                    "n_independent": o.independent_source_count,
                })
            return pd.DataFrame(rows)
        except ImportError:
            print("pandas not installed. Returning raw list.")
            return self._history
