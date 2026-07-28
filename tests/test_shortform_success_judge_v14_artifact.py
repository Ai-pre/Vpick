from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit_shortform_success_judge_v14_dev import compose_fields  # noqa: E402
from predict_shortform_success_v14_dev import validate_record  # noqa: E402


class ShortformSuccessJudgeV14ArtifactTests(unittest.TestCase):
    def test_composed_fields_remove_timestamp_formatting(self) -> None:
        semantic, context = compose_fields(
            [
                SimpleNamespace(
                    description="핵심 설명",
                    transcript="[00:01-00:03] S1: 중요한 대사",
                    before_context="[00:00-00:01] 준비",
                    after_context="[00:03-00:04] 결론",
                )
            ]
        )

        self.assertNotIn("00:01", semantic[0])
        self.assertIn("중요한 대사", semantic[0])
        self.assertNotIn("00:03", context[0])

    def test_prediction_rejects_private_performance_inputs(self) -> None:
        record = {
            "candidate_id": "candidate",
            "description": "설명",
            "transcript": "대사",
            "channel_performance_percentile_PRIVATE": 90,
        }
        with self.assertRaises(ValueError):
            validate_record(
                record,
                ["channel_performance_percentile_PRIVATE"],
            )


if __name__ == "__main__":
    unittest.main()
