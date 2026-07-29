from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highlight_quality_judge_v1 import (  # noqa: E402
    DIMENSIONS,
    normalize_pointwise,
    normalize_pairwise,
    validate_candidate,
    validate_longform,
    validate_pairwise_annotation,
    weighted_total,
)


class HighlightQualityJudgeV1Test(unittest.TestCase):
    def test_weighted_total_uses_configured_zero_to_four_scale(self) -> None:
        weights = {
            "source_salience": 0.20,
            "hook": 0.20,
            "payoff": 0.20,
            "self_contained": 0.15,
            "density": 0.15,
            "boundary": 0.10,
        }
        scores = {name: 4 for name in weights}
        self.assertEqual(100.0, weighted_total(scores, weights))
        scores["boundary"] = 0
        self.assertEqual(90.0, weighted_total(scores, weights))

    def test_standard_schemas_accept_minimal_valid_objects(self) -> None:
        scene = {
            "scene_id": "s1",
            "start_ms": 0,
            "end_ms": 1_000,
            "scene_name": "시작",
            "description": "상황을 소개한다.",
            "transcript": "안녕하세요.",
            "speaker": "",
            "person_ids": [],
        }
        validate_longform(
            {
                "longform_id": "L1",
                "channel_id": "",
                "title": "제목",
                "duration_ms": 1_000,
                "upload_date": "",
                "view_count": 0,
                "scenes": [scene],
            }
        )
        validate_candidate(
            {
                "candidate_id": "C1",
                "longform_id": "L1",
                "start_ms": 0,
                "end_ms": 1_000,
                "scene_ids": ["s1"],
                "candidate_source": "published_short",
                "is_published": True,
            }
        )
        validate_pairwise_annotation(
            {
                "pair_id": "P1",
                "longform_id": "L1",
                "candidate_a_id": "C1",
                "candidate_b_id": "C2",
                "display_order": "A-B",
                "annotator_id": "human-1",
                "winner": "tie",
                "confidence_1_5": 3,
                "reason": "",
                "created_at": "",
            }
        )

    def test_pairwise_annotation_rejects_cross_identity_pair(self) -> None:
        with self.assertRaises(ValueError):
            validate_pairwise_annotation(
                {
                    "pair_id": "P1",
                    "longform_id": "L1",
                    "candidate_a_id": "C1",
                    "candidate_b_id": "C1",
                    "display_order": "A-B",
                    "annotator_id": "human-1",
                    "winner": "A",
                    "reason": "",
                    "created_at": "",
                }
            )

    def test_numeric_verdict_is_repaired_when_dimensions_are_valid(self) -> None:
        raw = {
            "candidate_id": "candidate-1",
            "verdict": "3",
            "dimensions": {
                name: {
                    "score": 3,
                    "reason": "근거가 충분하다.",
                    "scene_ids": ["scene-1"],
                    "insufficient_information": False,
                }
                for name in DIMENSIONS
            },
            "fatal_flags": [],
            "confidence_1_5": 4,
            "overall_reason": "유효한 후보이다.",
        }
        weights = {
            "source_salience": 0.20,
            "hook": 0.20,
            "payoff": 0.20,
            "self_contained": 0.15,
            "density": 0.15,
            "boundary": 0.10,
        }

        normalized = normalize_pointwise(raw, "candidate-1", weights)

        self.assertEqual(normalized["verdict"], "score")
        self.assertEqual(normalized["schema_repairs"], "numeric_verdict_to_score")

    def test_pairwise_fatal_flag_alias_is_normalized(self) -> None:
        raw = {
            "pair_id": "pair-1",
            "dimension_comparisons": {
                name: {
                    "winner": "A",
                    "reason": "A가 더 완결적이다.",
                    "scene_ids": ["scene-1"],
                }
                for name in DIMENSIONS
            },
            "winner": "A",
            "fatal_flags_a": [],
            "fatal_flags_b": ["context_missing"],
            "confidence_1_5": 4,
            "reason": "B는 앞 문맥이 부족하다.",
        }

        normalized = normalize_pairwise(raw, "pair-1")

        self.assertEqual(normalized["fatal_flags_b"], "missing_context")
        self.assertEqual(
            normalized["schema_repairs"],
            "context_missing_to_missing_context",
        )

    def test_unsupported_pairwise_flag_is_recorded_and_dropped(self) -> None:
        raw = {
            "pair_id": "pair-1",
            "dimension_comparisons": {
                name: {
                    "winner": "A",
                    "reason": "A가 더 완결적이다.",
                    "scene_ids": ["scene-1"],
                }
                for name in DIMENSIONS
            },
            "winner": "A",
            "fatal_flags_a": [],
            "fatal_flags_b": ["disjointed"],
            "confidence_1_5": 4,
            "reason": "B의 흐름이 단절된다.",
        }

        normalized = normalize_pairwise(raw, "pair-1")

        self.assertEqual(normalized["fatal_flags_b"], "")
        self.assertEqual(
            normalized["schema_repairs"],
            "dropped_unsupported_fatal_flag:disjointed",
        )


if __name__ == "__main__":
    unittest.main()
