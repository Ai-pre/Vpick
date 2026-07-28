from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_performance_calibrator_v13 import (  # noqa: E402
    MEMBER_SPECS,
    weighted_average,
)
from evaluate_shortform_success_holdout_v13 import evaluate  # noqa: E402
from predict_shortform_success import validate_record  # noqa: E402


class PerformanceCalibratorV13Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (
                ROOT / "config" / "performance_calibrator_v13.json"
            ).read_text(encoding="utf-8")
        )

    def test_ensemble_is_frozen_and_weights_sum_to_one(self) -> None:
        self.assertEqual(
            self.config["ensemble_members"],
            ["numeric_050", "channel_balanced", "numeric_025"],
        )
        self.assertAlmostEqual(
            sum(self.config["ensemble_weights"]),
            1.0,
            places=8,
        )
        self.assertEqual(
            set(self.config["ensemble_members"]),
            set(MEMBER_SPECS),
        )

    def test_weighted_average(self) -> None:
        values = [
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
        ]
        actual = weighted_average(values, [0.25, 0.75])
        np.testing.assert_allclose(actual, np.array([0.75, 0.25]))

    def test_gate_and_model_inputs_exclude_source(self) -> None:
        self.assertFalse(
            any(
                "source" in key.lower()
                for key in self.config["acceptance_gates"]
            )
        )
        self.assertIn(
            "transcript_source",
            self.config["forbidden_model_inputs"],
        )

    def test_internal_pass_does_not_equal_final_acceptance(self) -> None:
        policy = self.config["claim_policy"]
        self.assertTrue(policy["internal_gate_pass_is_not_final_validation"])
        self.assertTrue(
            policy["accepted_as_performance_judge_requires_fresh_holdout"]
        )

    def test_holdout_requires_explicit_fresh_confirmation(self) -> None:
        candidate_ids = [f"c{index}" for index in range(12)]
        percentiles = [10, 25, 40, 55, 70, 85] * 2
        predictions = pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "shortform_success_potential_0_100": percentiles,
            }
        )
        targets = pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "longform_id": [f"l{index}" for index in range(12)],
                "channel_name": ["a"] * 6 + ["b"] * 6,
                "channel_performance_percentile_PRIVATE": percentiles,
            }
        )
        config = {
            **self.config,
            "bootstrap_repetitions": 50,
        }
        unconfirmed = evaluate(
            predictions,
            targets,
            config,
            "test_holdout",
            False,
        )
        confirmed = evaluate(
            predictions,
            targets,
            config,
            "test_holdout",
            True,
        )
        self.assertTrue(unconfirmed["internal_gate_pass"])
        self.assertFalse(unconfirmed["accepted_as_performance_judge"])
        self.assertTrue(confirmed["accepted_as_performance_judge"])

    def test_private_performance_alias_is_rejected_at_prediction(self) -> None:
        record = {
            "candidate_id": "leaked",
            "description": "설명",
            "transcript": "충분한 후보 대사입니다.",
            "performance_label_PRIVATE": "pos",
            **{column: 2 for column in [
                "self_contained_clarity",
                "progression_payoff",
                "boundary_integrity",
                "opening_pull",
                "change_or_surprise",
                "emotional_or_information_gain",
                "memorable_specificity",
            ]},
        }
        with self.assertRaises(ValueError):
            validate_record(
                record,
                [
                    "self_contained_clarity",
                    "progression_payoff",
                    "boundary_integrity",
                    "opening_pull",
                    "change_or_surprise",
                    "emotional_or_information_gain",
                    "memorable_specificity",
                ],
                ["performance_label_PRIVATE"],
            )


if __name__ == "__main__":
    unittest.main()
