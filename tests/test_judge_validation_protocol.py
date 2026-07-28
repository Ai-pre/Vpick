from __future__ import annotations

import unittest

from build_within_video_judge_eval import auto_interval
from evaluate_judge_performance_consistency import qwk
from evaluate_within_video_judge import (
    exact_random_position_p_value,
    expected_random_mrr,
    random_hit_at_k,
    tie_aware_metrics,
)
from prepare_judge_validation_94 import assign_group_split, normalize_context


class JudgeValidationProtocolTest(unittest.TestCase):
    def test_context_normalization_uses_nearest_boundary_text(self) -> None:
        text = "0123456789"
        self.assertEqual(normalize_context(text, "before", 4), "6789")
        self.assertEqual(normalize_context(text, "after", 4), "0123")

    def test_group_split_never_crosses_longform(self) -> None:
        rows = [
            {
                "candidate_id": "A1",
                "longform_id": "L1",
                "channel_name": "BDNS",
                "performance_label_PRIVATE": "pos",
                "dataset_role_v2": "dev",
            },
            {
                "candidate_id": "A2",
                "longform_id": "L1",
                "channel_name": "BDNS",
                "performance_label_PRIVATE": "mid",
                "dataset_role_v2": "mid_percentile_expansion",
            },
            {
                "candidate_id": "B1",
                "longform_id": "L2",
                "channel_name": "OOTB",
                "performance_label_PRIVATE": "neg",
                "dataset_role_v2": "locked_test",
            },
            {
                "candidate_id": "C1",
                "longform_id": "L3",
                "channel_name": "OOTB",
                "performance_label_PRIVATE": "mid",
                "dataset_role_v2": "mid_percentile_expansion",
            },
        ]
        roles, summary = assign_group_split(
            rows,
            dev_fraction=0.5,
            seed="test",
        )
        self.assertEqual(roles["A1"], roles["A2"])
        self.assertEqual(roles["A1"], "dev")
        self.assertEqual(roles["B1"], "locked_test")
        self.assertEqual(summary["longform_overlap_count"], 0)

    def test_qwk_is_one_for_identical_labels(self) -> None:
        self.assertEqual(qwk([0, 1, 2, 2], [0, 1, 2, 2]), 1.0)

    def test_auto_interval_rejects_discontinuous_edit(self) -> None:
        contiguous = {
            "generation_metadata": {
                "scenes": [
                    {"source_start_ms": 1000, "source_end_ms": 5000},
                    {"source_start_ms": 5500, "source_end_ms": 9000},
                ]
            }
        }
        discontinuous = {
            "generation_metadata": {
                "scenes": [
                    {"source_start_ms": 1000, "source_end_ms": 5000},
                    {"source_start_ms": 8000, "source_end_ms": 9000},
                ]
            }
        }
        self.assertEqual(auto_interval(contiguous), (1.0, 9.0))
        self.assertIsNone(auto_interval(discontinuous))

    def test_exact_random_controls(self) -> None:
        self.assertAlmostEqual(random_hit_at_k(9, 1, 1), 1 / 9)
        self.assertAlmostEqual(random_hit_at_k(9, 1, 3), 1 / 3)
        self.assertEqual(round(expected_random_mrr(2, 1), 4), 0.75)

    def test_tie_aware_credit_does_not_use_candidate_id(self) -> None:
        metrics = tie_aware_metrics(
            [
                (80.0, True),
                (80.0, False),
                (70.0, False),
            ]
        )
        self.assertEqual(metrics["hit_at_1"], 0.5)
        self.assertEqual(metrics["hit_at_3"], 1.0)
        self.assertEqual(metrics["mrr"], 0.75)

    def test_exact_random_position_p_value_uses_score_ties(self) -> None:
        pool = [[(90.0, True), (80.0, False), (70.0, False)]]
        self.assertAlmostEqual(
            exact_random_position_p_value(pool, [1.0], "hit_at_1"),
            1 / 3,
        )


if __name__ == "__main__":
    unittest.main()
