from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from performance_judge_v1 import (  # noqa: E402
    binary_auc,
    extract_structure_features,
    fit_logistic,
    interval_coverage,
    predict_logistic,
    timed_cues,
    timestamp_to_seconds,
)
from predict_performance_judge_v1 import (  # noqa: E402
    classify_score,
    ensure_deployable,
)
from train_performance_judge_v1 import (  # noqa: E402
    choose_decision_thresholds,
    choose_deploy_model,
)


class PerformanceJudgeV1Test(unittest.TestCase):
    def test_timestamp_parser_supports_minutes_and_hours(self) -> None:
        self.assertEqual(timestamp_to_seconds("23:18"), 1398.0)
        self.assertEqual(timestamp_to_seconds("1:02:03"), 3723.0)

    def test_timed_cues_and_union_coverage(self) -> None:
        cues = timed_cues("[0:10-0:20] A\n[0:18-0:25] B")
        self.assertEqual(cues, [(10.0, 20.0), (18.0, 25.0)])
        self.assertAlmostEqual(interval_coverage(cues, 10.0, 30.0), 0.75)

    def test_structure_features_use_absolute_candidate_times(self) -> None:
        candidate = {
            "start_ms": 10000,
            "end_ms": 30000,
            "scene_ids": ["s1", "s2"],
            "transcript": "[0:10-0:20] A: 질문?\n[0:20-0:30] B: 답!",
            "before_context": "[0:05-0:10] A: 전",
            "after_context": "[0:30-0:35] B: 후",
            "longform_overview": [{"end_ms": 100000}],
        }
        features = extract_structure_features(candidate)
        self.assertAlmostEqual(features["duration_sec"], 20.0)
        self.assertAlmostEqual(features["position_ratio"], 0.1)
        self.assertAlmostEqual(features["scene_rate_per_min"], 6.0)
        self.assertAlmostEqual(features["speech_coverage_ratio"], 1.0)
        self.assertAlmostEqual(features["start_boundary_distance_sec"], 0.0)
        self.assertAlmostEqual(features["end_boundary_distance_sec"], 0.0)

    def test_regularized_logistic_learns_order(self) -> None:
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = fit_logistic(x, y, alpha=0.5)
        scores = predict_logistic(model, x)
        self.assertGreater(scores[-1], scores[0])
        self.assertEqual(binary_auc(y.astype(int), scores), 1.0)
        self.assertTrue(all(math.isfinite(value) for value in scores))

    def test_decision_thresholds_are_derived_from_blind_predictions(self) -> None:
        rows = [{"target": value} for value in [0, 0, 0, 1, 0, 1, 1, 1]]
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.8, 0.9])
        policy = choose_decision_thresholds(
            rows,
            scores,
            min_precision=0.75,
            min_count=2,
        )
        self.assertLess(
            policy["low_max_score_0_100"],
            policy["high_min_score_0_100"],
        )
        self.assertEqual(
            classify_score(
                100.0,
                low_max=policy["low_max_score_0_100"],
                high_min=policy["high_min_score_0_100"],
            ),
            "high_signal",
        )

    def test_non_isolated_codex_is_excluded_from_deployment(self) -> None:
        comparison = [
            {
                "model": "codex_plus_claude",
                "deployment_eligible": False,
                "lolo_macro_channel_auc": 0.66,
                "lolo_pooled_auc": 0.69,
                "lolo_pooled_auc_ci95_low": 0.53,
                "loco_pooled_auc": 0.64,
            },
            {
                "model": "codex_only",
                "deployment_eligible": False,
                "lolo_macro_channel_auc": 0.645,
                "lolo_pooled_auc": 0.674,
                "lolo_pooled_auc_ci95_low": 0.52,
                "loco_pooled_auc": 0.728,
            },
            {
                "model": "claude_only",
                "deployment_eligible": True,
                "lolo_macro_channel_auc": 0.516,
                "lolo_pooled_auc": 0.443,
                "lolo_pooled_auc_ci95_low": 0.30,
                "loco_pooled_auc": 0.385,
            },
        ]
        model, status, reason = choose_deploy_model(comparison)
        self.assertEqual(model, "claude_only")
        self.assertEqual(status, "rejected")
        self.assertIn("Codex 세션", reason)

    def test_rejected_artifact_is_blocked_by_default(self) -> None:
        artifact = {
            "deployment_status": "rejected",
            "deployment_block_reason": "failed validation",
        }
        with self.assertRaises(ValueError):
            ensure_deployable(artifact, allow_unvalidated=False)
        ensure_deployable(artifact, allow_unvalidated=True)


if __name__ == "__main__":
    unittest.main()
