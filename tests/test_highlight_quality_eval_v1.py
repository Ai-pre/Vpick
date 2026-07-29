from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_highlight_quality_eval_v1 import (  # noqa: E402
    assert_blind,
    pair_rows,
)


class HighlightQualityEvalV1Test(unittest.TestCase):
    def test_pair_builder_keeps_same_longform_and_no_duplicate_pair(self) -> None:
        blind = {
            candidate_id: {
                "candidate_id": candidate_id,
                "longform_id": "L1",
                "longform_overview": [],
                "start_ms": index * 1_000,
                "end_ms": (index + 1) * 1_000,
                "description": "",
                "transcript": "",
            }
            for index, candidate_id in enumerate(("A", "B"))
        }
        private = {
            "A": {"candidate_source": "published_short"},
            "B": {"candidate_source": "random"},
        }
        import random

        pairs, mappings = pair_rows(
            "A",
            ["B", "B"],
            blind,
            private,
            random.Random(1),
            1,
        )
        self.assertEqual(1, len(pairs))
        self.assertEqual(1, len(mappings))
        self.assertEqual("L1", pairs[0]["longform_id"])

    def test_blind_guard_rejects_source_and_performance_leakage(self) -> None:
        assert_blind([{"candidate_id": "A", "description": "ok"}])
        with self.assertRaises(ValueError):
            assert_blind([{"candidate_id": "A", "candidate_source": "published_short"}])
        with self.assertRaises(ValueError):
            assert_blind([{"candidate_id": "A", "performance_label": "pos"}])


if __name__ == "__main__":
    unittest.main()
