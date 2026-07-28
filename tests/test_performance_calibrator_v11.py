from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_performance_calibrator_v11 import (  # noqa: E402
    QUALITY_COLUMNS,
    assemble_fully_nested_selection,
    grouped_splits,
    performance_metrics,
)
from predict_shortform_success import validate_record  # noqa: E402


class ContinuousPerformanceCalibratorTests(unittest.TestCase):
    def test_config_excludes_pos_neg_controls(self) -> None:
        config = json.loads(
            (ROOT / "config" / "performance_calibrator_v11.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn(
            "label_oracle_sanity_only",
            config["diagnostic_controls"],
        )
        self.assertIn(
            "Pos/Neg labels and AUC are not used in splitting, training, model selection, or reporting.",
            config["notes"],
        )
        self.assertNotIn("source_residual_spearman", config["selection_metric"])

    def test_grouped_splits_never_split_a_longform(self) -> None:
        groups = np.array(
            ["a", "a", "b", "c", "d", "d", "e", "f", "g", "h"],
            dtype=object,
        )
        for train, test in grouped_splits(groups, 4, 20260728):
            self.assertFalse(set(groups[train]).intersection(groups[test]))

    def test_metrics_use_continuous_targets_without_buckets(self) -> None:
        y = np.array([0.1, 0.4, 0.7, 0.9, 0.2, 0.8])
        scores = np.array([0.2, 0.3, 0.8, 0.9, 0.1, 0.7])
        channels = np.array(["a", "a", "a", "a", "b", "b"])
        sources = np.array(["x", "x", "y", "y", "x", "y"])
        metrics = performance_metrics(y, scores, channels, sources)
        self.assertGreater(metrics["channel_centered_spearman"], 0.8)
        self.assertNotIn("within_performance_bucket_spearman", metrics)
        self.assertGreater(metrics["same_channel_pairwise_accuracy"], 0.8)

    def test_fully_nested_selector_uses_inner_score_only(self) -> None:
        fold_outputs = {
            "model_a": [
                {
                    "repeat_index": 0,
                    "outer_fold": 0,
                    "test_indices": np.array([0, 1]),
                    "predictions": np.array([0.2, 0.4]),
                    "selected_parameter": 1.0,
                    "inner_selection_score": 0.3,
                },
                {
                    "repeat_index": 0,
                    "outer_fold": 1,
                    "test_indices": np.array([2, 3]),
                    "predictions": np.array([0.6, 0.8]),
                    "selected_parameter": 1.0,
                    "inner_selection_score": 0.5,
                },
            ],
            "model_b": [
                {
                    "repeat_index": 0,
                    "outer_fold": 0,
                    "test_indices": np.array([0, 1]),
                    "predictions": np.array([0.9, 0.7]),
                    "selected_parameter": 10.0,
                    "inner_selection_score": 0.4,
                },
                {
                    "repeat_index": 0,
                    "outer_fold": 1,
                    "test_indices": np.array([2, 3]),
                    "predictions": np.array([0.1, 0.3]),
                    "selected_parameter": 10.0,
                    "inner_selection_score": 0.2,
                },
            ],
        }
        predictions, log = assemble_fully_nested_selection(
            fold_outputs,
            ["model_a", "model_b"],
            sample_count=4,
            repeat_count=1,
        )
        np.testing.assert_allclose(predictions, [0.9, 0.7, 0.6, 0.8])
        self.assertEqual(
            [item["selected_model"] for item in log],
            ["model_b", "model_a"],
        )

    def test_new_candidate_rejects_performance_leakage(self) -> None:
        record = {
            "candidate_id": "new",
            "description": "후보 설명",
            "transcript": "후보 자막",
            "channel_name": "forbidden",
            "codex_features": {column: 2 for column in QUALITY_COLUMNS},
        }
        with self.assertRaisesRegex(ValueError, "forbidden inputs"):
            validate_record(record, QUALITY_COLUMNS, ["channel_name"])

    def test_new_candidate_accepts_only_codex_quality_features(self) -> None:
        record = {
            "candidate_id": "new",
            "description": "후보 설명",
            "transcript": "후보 자막",
            "codex_features": {column: 2 for column in QUALITY_COLUMNS},
        }
        values = validate_record(record, QUALITY_COLUMNS, ["channel_name"])
        self.assertEqual(set(values), set(QUALITY_COLUMNS))


if __name__ == "__main__":
    unittest.main()
