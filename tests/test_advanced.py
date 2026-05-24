import unittest

from oracle.car_presets import get_car_preset
from oracle.correlation import CorrelationMatrix
from oracle.deterministic_car import run_deterministic_car
from oracle.portfolio import PortfolioPosition, analyze_portfolio_risk, summarize_positions


class TestDeterministicCAR(unittest.TestCase):

    def test_hormuz_case_applies_material_haircut(self):
        preset = get_car_preset("hormuz_2027")
        result = run_deterministic_car(
            question=preset["question"],
            p_naive=preset["naive_probability"],
            actor_profiles=preset["actors"],
        )

        self.assertLess(result["final"]["p_car"], result["final"]["p_naive"])
        self.assertGreater(result["final"]["adversarial_haircut"], 0.20)
        impact_map = {item["actor"]: item for item in result["marginal_impacts"]}
        self.assertIn("Russia", impact_map)
        actor_map = {item["actor"]: item for item in result["actor_profiles"]}
        self.assertGreater(actor_map["Russia"]["structural_silence_score"], 2.0)

    def test_fed_case_keeps_haircut_near_zero(self):
        preset = get_car_preset("fed_jun_2023")
        result = run_deterministic_car(
            question=preset["question"],
            p_naive=preset["naive_probability"],
            actor_profiles=preset["actors"],
        )

        self.assertAlmostEqual(result["final"]["adversarial_haircut"], 0.0, places=3)
        self.assertAlmostEqual(result["final"]["p_car"], result["final"]["p_naive"], places=3)


class TestPortfolioRisk(unittest.TestCase):

    def test_portfolio_summary_and_risk(self):
        positions = [
            PortfolioPosition("p1", "event_a", 1_000_000, 0.45, True, "A long"),
            PortfolioPosition("p2", "event_b", 750_000, 0.60, False, "B short"),
        ]
        summary = summarize_positions(
            positions,
            {"event_a": 0.50, "event_b": 0.55},
        )

        self.assertEqual(len(summary["positions"]), 2)
        self.assertGreater(summary["totals"]["gross_notional"], 0)

        corr = CorrelationMatrix()
        corr.add_event("event_a")
        corr.add_event("event_b")
        corr.set_correlation("event_a", "event_b", 0.35)

        result = analyze_portfolio_risk(
            positions=positions,
            current_probabilities={"event_a": 0.50, "event_b": 0.55},
            daily_vols={"event_a": 0.08, "event_b": 0.10},
            correlation_matrix=corr,
            n_paths=1000,
            n_days=10,
        )

        self.assertIsNotNone(result["risk"])
        self.assertIn("var_95", result["risk"])
        self.assertEqual(len(result["tail_contributors"]), 2)
        self.assertTrue(result["histogram"])


if __name__ == "__main__":
    unittest.main()
