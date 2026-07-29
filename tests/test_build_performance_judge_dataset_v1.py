from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_highlight_quality_eval_v1 import assert_blind  # noqa: E402


class BuildPerformanceJudgeDatasetV1Test(unittest.TestCase):
    def test_performance_targets_are_rejected_from_blind_payload(self) -> None:
        with self.assertRaises(ValueError):
            assert_blind(
                [
                    {
                        "candidate_id": "c1",
                        "performance_label": "pos",
                    }
                ]
            )

    def test_evidence_provider_is_allowed_in_blind_payload(self) -> None:
        assert_blind(
            [
                {
                    "candidate_id": "c1",
                    "evidence_provider": "vpick_scene_api",
                    "visual_evidence_available": True,
                }
            ]
        )


if __name__ == "__main__":
    unittest.main()
