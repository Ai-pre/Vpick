from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_highlight_quality_judge_v1 import (  # noqa: E402
    comparison_summary,
    consensus_judgment_rows,
    pairwise_summary,
    pairwise_preference_rows,
    pointwise_pair_rows,
)


class AnalyzeHighlightQualityJudgeV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = {
            "gold": {"candidate_id": "gold", "candidate_source": "published_short"},
            "alt": {"candidate_id": "alt", "candidate_source": "boundary_shift"},
        }
        self.pairs = [
            {
                "pair_id": "p1",
                "longform_id": "l1",
                "candidate_a_id": "alt",
                "candidate_b_id": "gold",
                "published_anchor_id": "gold",
            }
        ]

    def test_pointwise_comparison_restores_published_anchor(self) -> None:
        rows = pointwise_pair_rows(
            [
                {"candidate_id": "gold", "highlight_quality_score_100": 80},
                {"candidate_id": "alt", "highlight_quality_score_100": 60},
            ],
            self.pairs,
            self.candidates,
        )
        self.assertEqual(rows[0]["outcome"], "win")
        self.assertEqual(rows[0]["score_delta"], 20)
        summary = comparison_summary(rows, "alternative_source")
        self.assertEqual(summary[0]["strict_win_rate"], 1.0)

    def test_pairwise_winner_is_mapped_to_physical_candidate(self) -> None:
        rows = pairwise_preference_rows(
            [{"pair_id": "p1", "winner": "B", "order_inconsistent": False}],
            {"p1": self.pairs[0]},
            self.candidates,
        )
        self.assertEqual(rows[0]["outcome"], "win")

    def test_order_inconsistent_pair_is_abstained_from_preference_rate(self) -> None:
        summary = pairwise_summary(
            [
                {
                    "alternative_source": "boundary_shift",
                    "outcome": "win",
                    "order_inconsistent": True,
                },
                {
                    "alternative_source": "boundary_shift",
                    "outcome": "loss",
                    "order_inconsistent": False,
                },
            ]
        )
        self.assertEqual(summary[0]["consistent_count"], 1)
        self.assertEqual(summary[0]["order_abstain_count"], 1)
        self.assertEqual(summary[0]["strict_win_rate"], 0.0)

    def test_consensus_output_selects_candidate_only_when_orders_agree(self) -> None:
        rows = consensus_judgment_rows(
            [
                {
                    "pair_id": "p1",
                    "winner": "B",
                    "swapped_winner_restored": "B",
                    "confidence_1_5": 4,
                }
            ],
            {"p1": self.pairs[0]},
        )
        self.assertEqual(rows[0]["consensus_status"], "accepted")
        self.assertEqual(rows[0]["selected_candidate_id"], "gold")

        abstained = consensus_judgment_rows(
            [
                {
                    "pair_id": "p1",
                    "winner": "B",
                    "swapped_winner_restored": "A",
                    "confidence_1_5": 4,
                }
            ],
            {"p1": self.pairs[0]},
        )
        self.assertEqual(abstained[0]["consensus_winner"], "abstain")
        self.assertEqual(abstained[0]["selected_candidate_id"], "")


if __name__ == "__main__":
    unittest.main()
