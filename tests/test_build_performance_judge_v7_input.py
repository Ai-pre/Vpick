from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_performance_judge_v7_input import BLIND_FIELDS, build_rows  # noqa: E402


class BuildPerformanceJudgeV7InputTest(unittest.TestCase):
    def test_output_is_blind_and_maps_to_source_candidate_id(self) -> None:
        candidate = {
            "candidate_id": "PJ_1",
            "start_ms": 1000,
            "end_ms": 31000,
            "description": "설명",
            "transcript": "대사",
            "before_context": "전",
            "after_context": "후",
        }
        target = {
            "candidate_id": "PJ_1",
            "source_candidate_id": "C_1",
            "performance_label": "pos",
            "channel_name": "hidden",
        }
        with self.assertRaises(ValueError):
            build_rows([candidate], [target])

        candidates = [dict(candidate, candidate_id=f"PJ_{index}") for index in range(60)]
        targets = [
            dict(
                target,
                candidate_id=f"PJ_{index}",
                source_candidate_id=f"C_{index}",
            )
            for index in range(60)
        ]
        rows = build_rows(candidates, targets)
        self.assertEqual(tuple(rows[0]), BLIND_FIELDS)
        self.assertEqual(rows[0]["candidate_id"], "C_0")
        self.assertNotIn("performance_label", rows[0])
        self.assertNotIn("channel_name", rows[0])


if __name__ == "__main__":
    unittest.main()
