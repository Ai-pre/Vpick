from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_same_longform_hard_negative_eval import (  # noqa: E402
    choose_hard_negative,
    generate_scene_windows,
    overlaps_known_short,
)
from evaluate_same_longform_hard_negatives import build_results  # noqa: E402


def scene(index: int, start: float, end: float, text: str) -> dict:
    return {
        "scene_id": str(index),
        "start_sec": start,
        "end_sec": end,
        "duration_sec": end - start,
        "name": f"scene {index}",
        "description": text,
        "speeches": [
            {
                "speech_id": str(index),
                "start_sec": start + 1,
                "end_sec": end - 1,
                "speaker_id": str(index % 2),
                "text": text,
            }
        ],
        "raw": {"persons": [{"person_id": str(index % 2)}]},
    }


class SameLongformHardNegativeTests(unittest.TestCase):
    def test_overlap_rejects_any_intersection_and_guard(self) -> None:
        known = [(20.0, 40.0)]
        self.assertTrue(overlaps_known_short(10.0, 21.0, known))
        self.assertFalse(overlaps_known_short(5.0, 15.0, known))
        self.assertTrue(overlaps_known_short(5.0, 19.0, known, guard_sec=2.0))

    def test_scene_windows_match_target_duration_band(self) -> None:
        scenes = [
            scene(1, 0.0, 10.0, "첫 장면"),
            scene(2, 10.0, 22.0, "둘째 장면"),
            scene(3, 22.0, 40.0, "셋째 장면"),
        ]
        windows = generate_scene_windows(scenes, 20.0, 0.75, 1.25)
        self.assertTrue(windows)
        self.assertTrue(all(15.0 <= row["duration_sec"] <= 25.0 for row in windows))

    def test_selected_negative_never_overlaps_known_short(self) -> None:
        scenes = [
            scene(1, 0.0, 20.0, "도입 대화가 이어진다"),
            scene(2, 20.0, 40.0, "질문과 답변이 이어진다"),
            scene(3, 40.0, 60.0, "반응과 결론이 나온다"),
            scene(4, 60.0, 80.0, "새로운 사건과 반응이 나온다"),
        ]
        selected, _pool = choose_hard_negative(
            scenes,
            {},
            20.0,
            [(0.0, 40.0)],
            guard_sec=0.0,
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(
            0.0,
            max(
                0.0,
                min(float(selected["end_sec"]), 40.0)
                - max(float(selected["start_sec"]), 0.0),
            ),
        )

    def test_pairwise_evaluation_uses_private_roles_after_scoring(self) -> None:
        private = [
            {
                "eval_pair_id": "HN001",
                "candidate_id": "blind_a",
                "reference_role": "hard_negative",
                "channel_name": "A",
                "long_video_id": "long",
            },
            {
                "eval_pair_id": "HN001",
                "candidate_id": "blind_b",
                "reference_role": "positive",
                "channel_name": "A",
                "long_video_id": "long",
            },
        ]
        scores = [
            {
                "candidate_id": "blind_a",
                "verdict": "score",
                "quality_score_100": "40",
            },
            {
                "candidate_id": "blind_b",
                "verdict": "score",
                "quality_score_100": "70",
            },
        ]
        rows, summary = build_results(private, scores)
        self.assertEqual(1, rows[0]["strict_correct"])
        self.assertEqual(1.0, summary["micro_tie_aware_pairwise_accuracy"])


if __name__ == "__main__":
    unittest.main()
