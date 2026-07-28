from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_predictions import deduplicate_ranked_predictions  # noqa: E402


class EvaluatePredictionsTests(unittest.TestCase):
    def test_duplicate_intervals_are_removed_and_later_ranks_refill_top5(self) -> None:
        rows = []
        intervals = [
            (10.0, 20.0),
            (10.0, 20.0),
            (30.0, 40.0),
            (30.0, 40.0),
            (50.0, 60.0),
            (70.0, 80.0),
            (90.0, 100.0),
        ]
        for rank, (start, end) in enumerate(intervals, start=1):
            rows.append(
                {
                    "pair_id": "G001",
                    "run_id": "baseline",
                    "selector_type": "vpick",
                    "prompt_id": "auto",
                    "model_name": "vpick",
                    "rank": str(rank),
                    "pred_start_sec": str(start),
                    "pred_end_sec": str(end),
                }
            )

        deduplicated, audit = deduplicate_ranked_predictions(rows)

        self.assertEqual(len(deduplicated), 5)
        self.assertEqual(
            [int(row["rank"]) for row in deduplicated],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [
                (float(row["pred_start_sec"]), float(row["pred_end_sec"]))
                for row in deduplicated
            ],
            [
                (10.0, 20.0),
                (30.0, 40.0),
                (50.0, 60.0),
                (70.0, 80.0),
                (90.0, 100.0),
            ],
        )
        self.assertEqual(audit[0]["duplicate_rows_removed"], 2)
        self.assertEqual(audit[0]["original_top5_unique_count"], 3)
        self.assertEqual(audit[0]["final_top5_unique_count"], 5)
        self.assertFalse(audit[0]["top5_underfilled"])


if __name__ == "__main__":
    unittest.main()
