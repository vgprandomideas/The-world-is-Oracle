"""
Tests for the World-as-Oracle core engine.
Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import math
import time
import unittest
from oracle.models import Article, OracleState, SourceTier, EventCategory, SwapContract
from oracle.independence import compute_independence_score, flag_citation_amplification
from oracle.temporal_filter import decay_factor, compute_fast_shock, compute_persistent_signal
from oracle.probability import logit, sigmoid, compose_probability, compute_prior, PlattScaler
from oracle.config import CATEGORY_CONFIGS
from oracle.mtm import daily_variation_margin, terminal_settlement, initial_margin_schedule
from oracle.oracle import WorldOracle


def make_article(article_id, tier=4, impact=5.0, direction=1,
                 independence=0.8, pub_time=None, content="test article content"):
    if pub_time is None:
        pub_time = time.time() - 86400
    a = Article(
        article_id=article_id,
        source_name=f"Source_{article_id}",
        tier=SourceTier(tier),
        publication_time=pub_time,
        headline=f"Headline {article_id}",
        content_summary=content,
    )
    a.raw_impact = impact
    a.direction = direction
    a.independence_score = independence
    return a


class TestMathFunctions(unittest.TestCase):

    def test_logit_sigmoid_inverse(self):
        """σ(logit(p)) = p"""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            self.assertAlmostEqual(sigmoid(logit(p)), p, places=6)

    def test_logit_bounds(self):
        """logit handles boundary values safely."""
        self.assertIsNotNone(logit(0.01))
        self.assertIsNotNone(logit(0.99))

    def test_decay_factor_no_confirmation(self):
        """Without confirmation, decay follows exp(-λΔt)."""
        d = decay_factor(delta_t_days=7.0, decay_rate=0.40, n_confirming=0)
        expected = math.exp(-0.40 * 7.0)
        self.assertAlmostEqual(d, expected, places=5)

    def test_decay_factor_with_confirmation(self):
        """With confirmation (n_c >= 1), decay is nearly arrested."""
        d_no_confirm = decay_factor(7.0, 0.40, n_confirming=0)
        d_confirmed = decay_factor(7.0, 0.40, n_confirming=1, gamma_c=0.95)
        self.assertGreater(d_confirmed, d_no_confirm)
        # With gamma_c=0.95, effective rate = 0.40 * 0.05 = 0.02
        expected_arrested = math.exp(-0.40 * 0.05 * 7.0)
        self.assertAlmostEqual(d_confirmed, expected_arrested, places=5)

    def test_prior_weights_sum_to_one(self):
        """α + β + γ = 1 holds for all category configs."""
        for category, config in CATEGORY_CONFIGS.items():
            total = config.alpha + config.beta + config.gamma
            self.assertAlmostEqual(total, 1.0, places=6,
                msg=f"Weights don't sum to 1.0 for {category}")

    def test_compose_probability_output_range(self):
        """P_oracle always in [0.02, 0.98]."""
        for prior in [0.05, 0.30, 0.50, 0.80]:
            for shock in [-0.20, 0.0, 0.20]:
                for persistent in [-0.80, 0.0, 0.80]:
                    p, _, _ = compose_probability(prior, shock, persistent)
                    self.assertGreaterEqual(p, 0.02)
                    self.assertLessEqual(p, 0.98)

    def test_p_raw_can_exceed_bounds_but_clamped(self):
        """P_raw can exceed [0,1] but clamping ensures logit is defined."""
        # Prior + F_max + P_max = 0.20 + 0.20 + 0.80 > 1.0
        p, p_raw, p_raw_c = compose_probability(
            prior=0.20, fast_shock=0.20, persistent_signal=0.80
        )
        self.assertGreater(p_raw, 1.0)        # P_raw exceeds 1
        self.assertLessEqual(p_raw_c, 0.99)   # But clamped
        self.assertLessEqual(p, 0.98)          # Output bounded


class TestIndependenceScorer(unittest.TestCase):

    def test_first_article_is_independent(self):
        """First article has no prior — independence = 1.0"""
        a = make_article("art1", content="Unique content about regulatory action")
        score = compute_independence_score(a, [])
        self.assertEqual(score, 1.0)

    def test_duplicate_article_low_independence(self):
        """Near-duplicate article has low independence score."""
        text = "Federal Reserve raises interest rates bribery fraud investigation"
        a1 = make_article("art1", content=text, pub_time=time.time() - 7200)
        a2 = make_article("art2", content=text + " additional words",
                          pub_time=time.time() - 3600)
        score = compute_independence_score(a2, [a1])
        self.assertLess(score, 0.40)  # Below θ_ind

    def test_independent_article_high_score(self):
        """Genuinely independent article has high independence score."""
        a1 = make_article("art1", content="court filing bribery charges conviction fraud")
        a2 = make_article("art2", content="capital flow bond yield spread investment portfolio")
        score = compute_independence_score(a2, [a1])
        self.assertGreater(score, 0.40)

    def test_citation_amplification_detection(self):
        """Detects when many articles cite one source."""
        hindenburg = make_article("hind", content="adani stock manipulation fraud allegations")
        bloom1 = make_article("b1", content="adani stock manipulation fraud allegations reuters")
        bloom2 = make_article("b2", content="adani fraud allegations stock manipulation report")
        ft1 = make_article("ft1", content="adani manipulation allegations stock fraud findings")

        articles = [hindenburg, bloom1, bloom2, ft1]
        for i, a in enumerate(articles[1:], 1):
            a.independence_score = compute_independence_score(a, articles[:i])

        report = flag_citation_amplification(articles, theta_ind=0.40)
        self.assertGreater(report["amplification_ratio"], 0.4)


class TestTemporalFilter(unittest.TestCase):

    def test_fast_shock_capped(self):
        """Fast shock cannot exceed F_max = 0.20."""
        config = CATEGORY_CONFIGS[EventCategory.CORPORATE_LEGAL]
        now = time.time()
        articles = [
            make_article(f"a{i}", tier=1, impact=10.0, direction=1,
                         independence=0.9, pub_time=now - 3600)
            for i in range(10)
        ]
        f = compute_fast_shock(articles, now, config)
        self.assertLessEqual(f, config.f_max)

    def test_fast_shock_negative_on_countersignals(self):
        """Counter-signals can make fast shock negative."""
        config = CATEGORY_CONFIGS[EventCategory.CORPORATE_LEGAL]
        now = time.time()
        articles = [
            make_article("counter1", tier=5, impact=8.0, direction=-1,
                         independence=0.9, pub_time=now - 3600),
            make_article("counter2", tier=5, impact=8.0, direction=-1,
                         independence=0.9, pub_time=now - 7200),
        ]
        f = compute_fast_shock(articles, now, config)
        self.assertLess(f, 0)

    def test_persistent_signal_gated_by_rho_min(self):
        """Persistent signal is 0 if fewer than ρ_min independent sources."""
        config = CATEGORY_CONFIGS[EventCategory.CORPORATE_LEGAL]
        now = time.time()
        # Only 2 articles — below ρ_min = 3
        articles = [
            make_article("a1", impact=7.0, direction=1, independence=0.9,
                         pub_time=now - 86400),
            make_article("a2", impact=6.0, direction=1, independence=0.8,
                         pub_time=now - 43200),
        ]
        p, rho, _ = compute_persistent_signal(articles, now, config)
        self.assertEqual(p, 0.0)
        self.assertLess(rho, config.rho_min)

    def test_persistent_signal_activates_at_rho_min(self):
        """Persistent signal activates when ρ(t) ≥ ρ_min."""
        config = CATEGORY_CONFIGS[EventCategory.CORPORATE_LEGAL]
        now = time.time()
        articles = [
            make_article(f"a{i}", impact=6.0, direction=1, independence=0.9,
                         pub_time=now - (i * 43200))
            for i in range(3)  # exactly ρ_min = 3
        ]
        p, rho, _ = compute_persistent_signal(articles, now, config)
        self.assertGreaterEqual(rho, config.rho_min)
        self.assertNotEqual(p, 0.0)


class TestMarkToMarket(unittest.TestCase):

    def test_variation_margin_long_floating(self):
        """Party A (long floating) gains when probability rises."""
        vm = daily_variation_margin(0.60, 0.50, 1_000_000, long_floating=True)
        self.assertAlmostEqual(vm, 100_000.0)

    def test_variation_margin_short_floating(self):
        """Party B (short floating) gains when probability falls."""
        vm = daily_variation_margin(0.40, 0.50, 1_000_000, long_floating=False)
        self.assertAlmostEqual(vm, 100_000.0)

    def test_initial_margin_deterministic(self):
        """Total margin = (P_max - P_min) × N regardless of P_fixed."""
        for p_fixed in [0.20, 0.40, 0.60, 0.80]:
            m = initial_margin_schedule(p_fixed, 1_000_000)
            expected_total = (0.98 - 0.02) * 1_000_000
            self.assertAlmostEqual(m["total_margin"], expected_total, places=1)

    def test_terminal_settlement(self):
        """Terminal settlement pays (outcome - last_p) × N."""
        vm = terminal_settlement(outcome=1, p_previous=0.60, notional=1_000_000)
        self.assertAlmostEqual(vm, 400_000.0)

    def test_vm_suspended_in_contested_state(self):
        """VM returns None when oracle is CONTESTED."""
        from oracle.models import OracleState
        vm = daily_variation_margin(
            0.60, 0.50, 1_000_000,
            oracle_state=OracleState.CONTESTED
        )
        self.assertIsNone(vm)


class TestOracleIntegration(unittest.TestCase):

    def test_full_pipeline(self):
        """End-to-end oracle computation runs without error."""
        oracle = WorldOracle(
            event_id="test_event",
            event_description="Test regulatory action",
            resolution_criteria="Regulator issues enforcement action",
            category=EventCategory.CORPORATE_LEGAL,
        )
        oracle.set_prior(0.15, 0.20, None)

        now = time.time()
        articles = [
            make_article(f"a{i}", tier=i % 4 + 1, impact=float(5 + i),
                         direction=1, independence=0.8 - i * 0.05,
                         pub_time=now - (i * 86400), content=f"unique content {i} regulatory action")
            for i in range(4)
        ]
        oracle.ingest_articles(articles)
        out = oracle.compute(now)

        self.assertIsInstance(out.state, OracleState)
        self.assertGreaterEqual(out.probability, 0.02)
        self.assertLessEqual(out.probability, 0.98)
        self.assertGreater(out.ci_upper, out.ci_lower)

    def test_platt_scaler_identity(self):
        """Default Platt params (a=1, b=0) are identity transform."""
        ps = PlattScaler(a=1.0, b=0.0)
        for p in [0.10, 0.30, 0.50, 0.70, 0.90]:
            result = ps.transform(p)
            self.assertAlmostEqual(result, max(0.02, min(0.98, p)), places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
