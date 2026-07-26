from __future__ import annotations

import unittest

from src.evaluate_prompt_ablation_posneg import auc, pairwise_concordance, primary_score


class PromptAblationEvaluationTests(unittest.TestCase):
    def test_v1_composite_score(self) -> None:
        judgment = {
            "verdict": "score",
            "saliency_market_1_5": 3,
            "checks": {
                "hook_within_3s": 1,
                "surprise_or_twist": 1,
                "emotional_peak": 1,
                "quotable_moment": 1,
                "payoff_or_conclusion": 1,
                "natural_start": 1,
                "natural_end": 1,
            },
        }
        score = primary_score("v1", judgment)
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score["primary_score"], 50.0)

    def test_v2_composite_score(self) -> None:
        judgment = {
            "verdict": "score",
            "saliency_market_0_100": 80,
            "checks": {
                "hook_within_3s": 2,
                "surprise_or_twist": 2,
                "emotional_peak": 2,
                "quotable_moment": 2,
                "payoff_or_conclusion": 2,
                "natural_start": 2,
                "natural_end": 2,
            },
        }
        score = primary_score("v2", judgment)
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score["primary_score"], 65.0)

    def test_v5_joint_funnel_probability(self) -> None:
        judgment = {
            "verdict": "score",
            "p_stop": 50,
            "p_watch": 80,
            "p_share": 10,
        }
        score = primary_score("v5", judgment)
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score["primary_score"], 4.0)

    def test_auc_uses_half_credit_for_ties(self) -> None:
        self.assertAlmostEqual(auc([1, 0], [10.0, 10.0]), 0.5)
        self.assertAlmostEqual(auc([1, 0], [11.0, 10.0]), 1.0)

    def test_cell_pairwise_concordance_uses_continuous_percentile(self) -> None:
        rows = [
            {"score": 30.0, "percentile": 10.0},
            {"score": 20.0, "percentile": 20.0},
            {"score": 10.0, "percentile": 30.0},
        ]
        self.assertAlmostEqual(pairwise_concordance(rows), 0.0)


if __name__ == "__main__":
    unittest.main()
