from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_shortform_judge_v9_human_sheet import (  # noqa: E402
    build_rows,
    select_anchors,
)


class BuildShortformJudgeV9HumanSheetTests(unittest.TestCase):
    def test_selects_one_extreme_per_label_and_channel(self) -> None:
        targets = [
            {
                "candidate_id": "p1",
                "channel_name": "A",
                "performance_label": "pos",
                "channel_performance_percentile": "80",
                "longform_id": "long",
                "start_sec": "10",
                "end_sec": "20",
            },
            {
                "candidate_id": "p2",
                "channel_name": "A",
                "performance_label": "pos",
                "channel_performance_percentile": "99",
                "longform_id": "long",
                "start_sec": "20",
                "end_sec": "30",
            },
            {
                "candidate_id": "n1",
                "channel_name": "A",
                "performance_label": "neg",
                "channel_performance_percentile": "20",
                "longform_id": "long",
                "start_sec": "30",
                "end_sec": "40",
            },
            {
                "candidate_id": "n2",
                "channel_name": "A",
                "performance_label": "neg",
                "channel_performance_percentile": "1",
                "longform_id": "long",
                "start_sec": "40",
                "end_sec": "50",
            },
        ]
        selected = select_anchors(targets)
        self.assertEqual({"p2", "n2"}, {row["candidate_id"] for row in selected})
        rows = build_rows(targets, ["H1", "H2"], 1)
        self.assertEqual(4, len(rows))
        self.assertNotIn("performance_label", rows[0])
        self.assertNotIn("channel_name", rows[0])


if __name__ == "__main__":
    unittest.main()
