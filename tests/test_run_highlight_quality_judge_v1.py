from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_highlight_quality_judge_v1 import (  # noqa: E402
    mock_pairwise,
    pairwise_consensus,
    restore_swapped_result,
    swapped_pair,
)
from highlight_quality_judge_v1 import normalize_pairwise  # noqa: E402


class RunHighlightQualityJudgeV1Test(unittest.TestCase):
    def test_swapped_pair_and_result_restore_physical_candidate(self) -> None:
        pair = {
            "pair_id": "P1",
            "candidate_a": {"candidate_id": "left"},
            "candidate_b": {"candidate_id": "right"},
        }
        swapped = swapped_pair(pair)
        self.assertEqual("right", swapped["candidate_a"]["candidate_id"])
        normalized = normalize_pairwise(mock_pairwise("P1"), "P1")
        normalized["winner"] = "A"
        for name in (
            "source_salience",
            "hook",
            "payoff",
            "self_contained",
            "density",
            "boundary",
        ):
            normalized[f"{name}_winner"] = "B"
        restored = restore_swapped_result(normalized)
        self.assertEqual("B", restored["winner"])
        self.assertEqual("A", restored["hook_winner"])

    def test_consensus_accepts_only_matching_restored_winners(self) -> None:
        accepted = pairwise_consensus(
            [
                {"winner": "A", "confidence_1_5": 5},
                {"winner": "A", "confidence_1_5": 4},
            ]
        )
        self.assertEqual(accepted["consensus_status"], "accepted")
        self.assertEqual(accepted["consensus_winner"], "A")
        self.assertEqual(accepted["consensus_confidence_1_5"], 4)

        abstained = pairwise_consensus(
            [
                {"winner": "A", "confidence_1_5": 5},
                {"winner": "B", "confidence_1_5": 5},
            ]
        )
        self.assertEqual(
            abstained["consensus_status"],
            "abstain_order_inconsistent",
        )
        self.assertEqual(abstained["consensus_winner"], "abstain")


if __name__ == "__main__":
    unittest.main()
