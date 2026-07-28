from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_performance_calibrator_v14_dev import (  # noqa: E402
    candidate_specs,
    development_metrics,
)


class PerformanceCalibratorV14DevelopmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (
                ROOT / "config" / "performance_calibrator_v14_dev.json"
            ).read_text(encoding="utf-8")
        )

    def test_candidate_specs_use_normalized_text_and_mid_weighting(self) -> None:
        specs = candidate_specs(self.config)
        self.assertEqual(set(specs), set(self.config["candidate_specs"]))
        for spec in specs.values():
            self.assertNotEqual(spec.representation, "concat_raw_char")
            self.assertEqual(spec.mid_pair_boost, 3.0)
            self.assertEqual(spec.extreme_pair_weight, 0.25)
            self.assertTrue(spec.channel_balanced_pairs)

    def test_perfect_order_scores_perfect_targeted_metrics(self) -> None:
        y = np.array(
            [0.1, 0.3, 0.5, 0.7, 0.9] * 3,
            dtype=float,
        )
        channels = np.array(
            ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
            dtype=object,
        )
        metrics = development_metrics(
            y,
            y.copy(),
            channels,
            0.2,
            0.8,
            0.03,
            self.config["inner_selection_weights"],
        )

        self.assertAlmostEqual(
            metrics["mid_only_channel_centered_spearman"],
            1.0,
        )
        self.assertAlmostEqual(metrics["mid_only_pairwise_accuracy"], 1.0)
        self.assertAlmostEqual(
            metrics["same_channel_local_pairwise_accuracy"],
            1.0,
        )
        self.assertAlmostEqual(metrics["extremes_pos_neg_auc"], 1.0)
        self.assertAlmostEqual(metrics["selection_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
