from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortform_judge_v9 import (  # noqa: E402
    EDITORIAL_DIMENSIONS,
    ENGAGEMENT_DIMENSIONS,
    EVIDENCE_DIMENSIONS,
    load_config,
    normalize_judgment,
    weighted_score,
)


class ShortformJudgeV9Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(
            ROOT / "config" / "shortform_judge_v9_opus.json"
        )

    def response(self) -> dict:
        def axis(dimensions: tuple[str, ...], score: int) -> dict:
            return {
                dimension: {"score": score, "reason": "specific evidence"}
                for dimension in dimensions
            }

        return {
            "candidate_id": "candidate-1",
            "verdict": "score",
            "evidence": {
                dimension: 4 for dimension in EVIDENCE_DIMENSIONS
            },
            "editorial": axis(EDITORIAL_DIMENSIONS, 4),
            "engagement": axis(ENGAGEMENT_DIMENSIONS, 2),
            "confidence_1_5": 4,
            "failure_flags": [],
            "reason": "specific evidence supports the judgment",
        }

    def test_axis_scores_and_total_use_fixed_formula(self) -> None:
        row = normalize_judgment(
            self.response(),
            "candidate-1",
            self.config,
        )
        self.assertEqual(100.0, row["editorial_score_100"])
        self.assertEqual(50.0, row["engagement_score_100"])
        self.assertEqual(75.0, row["judge_score_100"])

    def test_abstain_does_not_turn_missing_evidence_into_zero(self) -> None:
        raw = self.response()
        raw["verdict"] = "abstain"
        raw["editorial"] = None
        raw["engagement"] = None
        raw["failure_flags"] = []
        row = normalize_judgment(raw, "candidate-1", self.config)
        self.assertEqual("", row["judge_score_100"])
        self.assertIn("insufficient_evidence", row["failure_flags"])

    def test_weighted_score_rejects_missing_dimension(self) -> None:
        weights = self.config["dimension_weights"]["editorial"]
        scores = {dimension: 4 for dimension in EDITORIAL_DIMENSIONS[:-1]}
        with self.assertRaises(ValueError):
            weighted_score(scores, weights, EDITORIAL_DIMENSIONS)


if __name__ == "__main__":
    unittest.main()
