from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_performance_calibrator_v12 import (  # noqa: E402
    RankerSpec,
    SPECS,
    normalize_semantic_text,
    pair_region_weight,
    pair_infos,
)


class PerformanceCalibratorV12Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (
                ROOT / "config" / "performance_calibrator_v12.json"
            ).read_text(encoding="utf-8")
        )
        self.registered = next(
            spec
            for spec in SPECS
            if spec.name == self.config["registered_primary_spec"]
        )

    def test_acceptance_gates_have_no_source_or_label_metrics(self) -> None:
        gates = set(self.config["acceptance_gates"])
        self.assertEqual(
            gates,
            {
                "channel_centered_spearman_min",
                "channel_macro_spearman_min",
                "same_channel_pairwise_accuracy_min",
                "same_channel_local_pairwise_accuracy_min",
                "bootstrap_primary_ci_lower_min",
            },
        )

    def test_forbidden_inputs_cover_performance_and_source_metadata(self) -> None:
        forbidden = set(self.config["forbidden_model_inputs"])
        self.assertTrue(
            {
                "channel_name",
                "performance_label_PRIVATE",
                "channel_performance_percentile_PRIVATE",
                "transcript_source",
                "short_video_url",
            }.issubset(forbidden)
        )

    def test_text_normalization_removes_timestamp_and_normalizes_speaker(self) -> None:
        normalized = normalize_semantic_text(
            "[0:01-0:04] S2: 안녕하세요   반갑습니다"
        )
        self.assertNotIn("0:01", normalized)
        self.assertNotIn("S2:", normalized)
        self.assertIn("화자:", normalized)
        self.assertEqual(normalized.count("  "), 0)

    def test_registered_spec_uses_fixed_improvements(self) -> None:
        self.assertEqual(
            self.registered.representation,
            "field_aware_char_word",
        )
        self.assertEqual(
            self.registered.score_calibration,
            "train_ecdf",
        )
        self.assertTrue(self.registered.channel_balanced_pairs)
        self.assertTrue(self.registered.reliability_weighting)
        self.assertGreater(self.registered.local_boost, 1.0)
        self.assertEqual(self.registered.cross_channel_weight, 0.0)

    def test_channel_balancing_equalizes_pair_weight_totals(self) -> None:
        y = np.array([0.1, 0.4, 0.8, 0.2, 0.7], dtype=float)
        channels = np.array(["a", "a", "a", "b", "b"], dtype=object)
        reliability = np.ones(len(y), dtype=float)
        infos = pair_infos(y, channels, reliability, self.registered)
        totals = {}
        for info in infos:
            totals.setdefault(info["channel"], 0.0)
            totals[info["channel"]] += float(info["weight"])
        self.assertAlmostEqual(totals["a"], totals["b"], places=8)

    def test_default_pair_region_weight_preserves_v13_behavior(self) -> None:
        spec = RankerSpec(
            "default",
            "concat_raw_char",
            "raw",
            False,
            False,
            0.05,
            2.0,
            False,
        )
        self.assertEqual(pair_region_weight(0.3, 0.5, 0.2, spec), 2.0)
        self.assertEqual(pair_region_weight(0.1, 0.9, 0.8, spec), 1.0)

    def test_mid_sensitive_pair_region_weights(self) -> None:
        spec = RankerSpec(
            "mid_sensitive",
            "concat_normalized_char",
            "train_ecdf",
            False,
            True,
            0.03,
            2.0,
            False,
            mid_pair_boost=3.0,
            extreme_pair_weight=0.25,
        )
        self.assertEqual(pair_region_weight(0.3, 0.5, 0.2, spec), 6.0)
        self.assertEqual(pair_region_weight(0.1, 0.9, 0.8, spec), 0.25)


if __name__ == "__main__":
    unittest.main()
