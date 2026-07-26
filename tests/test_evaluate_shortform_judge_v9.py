from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_shortform_judge_v9 import (  # noqa: E402
    aggregate_scores,
    performance_metrics,
    repeat_metrics,
)


def score_row(
    candidate_id: str,
    repeat_index: int,
    editorial: float,
    engagement: float,
) -> dict[str, str]:
    return {
        "judge_run_id": "judge",
        "candidate_id": candidate_id,
        "longform_id": candidate_id,
        "repeat_index": str(repeat_index),
        "verdict": "score",
        "editorial_score_100": str(editorial),
        "engagement_score_100": str(engagement),
        "judge_score_100": str((editorial + engagement) / 2),
        "confidence_1_5": "4",
    }


class EvaluateShortformJudgeV9Tests(unittest.TestCase):
    def test_repeat_metrics_measure_both_axes(self) -> None:
        rows = [
            score_row("a", 1, 25, 25),
            score_row("b", 1, 75, 75),
            score_row("a", 2, 30, 30),
            score_row("b", 2, 80, 80),
        ]
        result = repeat_metrics(rows, 2)[0]
        self.assertEqual(1.0, result["candidate_scoring_coverage"])
        self.assertEqual(1.0, result["editorial_repeat_spearman"])
        self.assertEqual(5.0, result["engagement_repeat_mae"])

    def test_performance_validity_uses_engagement_axis(self) -> None:
        rows = [
            score_row("high", 1, 10, 90),
            score_row("low", 1, 90, 10),
        ]
        aggregates = aggregate_scores(rows)
        targets = [
            {
                "candidate_id": "high",
                "channel_name": "channel",
                "performance_label": "pos",
                "channel_performance_percentile": "90",
            },
            {
                "candidate_id": "low",
                "channel_name": "channel",
                "performance_label": "neg",
                "channel_performance_percentile": "10",
            },
        ]
        summary, channels = performance_metrics(aggregates, targets)
        self.assertEqual(1.0, summary[0]["pooled_auc_supplementary"])
        self.assertEqual(1.0, channels[0]["channel_auc"])


if __name__ == "__main__":
    unittest.main()
