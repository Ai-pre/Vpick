from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_shortform_success_holdout_v14 import (  # noqa: E402
    evaluate,
    read_targets,
)


class PerformanceJudgeValidationV14Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (
                ROOT / "config" / "performance_judge_validation_v14.json"
            ).read_text(encoding="utf-8")
        )
        self.config["bootstrap_repetitions"] = 50

    def test_exact_predictions_pass_all_gates_with_sufficient_mid_data(self) -> None:
        targets = []
        predictions = []
        candidate_index = 0
        for channel_index in range(10):
            percentiles = [5.0, 15.0, 85.0] + [
                25.0 + offset * 3.5 for offset in range(15)
            ]
            for percentile in percentiles:
                candidate_id = f"C{candidate_index:03d}"
                targets.append(
                    {
                        "candidate_id": candidate_id,
                        "longform_id": f"L{candidate_index:03d}",
                        "channel_name": f"channel_{channel_index}",
                        "channel_performance_percentile_PRIVATE": percentile,
                        "evaluation_label_PRIVATE": (
                            "neg"
                            if percentile <= 20
                            else "pos"
                            if percentile >= 80
                            else "mid"
                        ),
                    }
                )
                predictions.append(
                    {
                        "candidate_id": candidate_id,
                        "shortform_success_potential_0_100": percentile,
                    }
                )
                candidate_index += 1

        result = evaluate(
            pd.DataFrame(predictions),
            pd.DataFrame(targets),
            self.config,
            "synthetic_holdout",
            True,
        )

        self.assertEqual(result["metrics"]["candidate_count"], 180)
        self.assertEqual(result["metrics"]["mid_candidate_count"], 150)
        self.assertTrue(
            result["accepted_as_continuous_performance_judge"]
        )

    def test_target_labels_are_derived_when_private_label_is_absent(self) -> None:
        path = ROOT / "tests" / "_temporary_v14_targets.csv"
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "low",
                    "longform_id": "L1",
                    "channel_name": "channel",
                    "channel_performance_percentile_PRIVATE": 20,
                },
                {
                    "candidate_id": "middle",
                    "longform_id": "L2",
                    "channel_name": "channel",
                    "channel_performance_percentile_PRIVATE": 50,
                },
                {
                    "candidate_id": "high",
                    "longform_id": "L3",
                    "channel_name": "channel",
                    "channel_performance_percentile_PRIVATE": 80,
                },
            ]
        )
        try:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            loaded = read_targets(path, self.config)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(
            loaded["evaluation_label_PRIVATE"].tolist(),
            ["neg", "mid", "pos"],
        )


if __name__ == "__main__":
    unittest.main()
