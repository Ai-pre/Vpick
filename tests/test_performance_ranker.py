from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_vpick_candidate_features import (  # noqa: E402
    build_judge_transcript,
    load_scene_payload,
    text_is_usable,
    vpick_features,
    vpick_transcript_is_usable,
)
from evaluate_performance_ranker import (  # noqa: E402
    FEATURE_SETS,
    binary_auc,
    lolo_predictions,
)
from fetch_vpick_account_inventory import infer_long_video_id  # noqa: E402


class VpickFeatureTests(unittest.TestCase):
    def test_account_inventory_prefers_target_longform_id(self) -> None:
        targets = {"p1WHu-Vhjxk"}
        self.assertEqual(
            infer_long_video_id("OOTB_Studio_p1WHu-Vhjxk", targets),
            "p1WHu-Vhjxk",
        )

    def test_extracts_window_structure_and_boundaries(self) -> None:
        scenes = [
            {
                "scene_id": "s1",
                "start_sec": 0.0,
                "end_sec": 10.0,
                "name": "시작",
                "description": "사람이 질문을 시작한다.",
                "speeches": [
                    {
                        "speech_id": "p1",
                        "start_sec": 2.0,
                        "end_sec": 6.0,
                        "speaker_id": "1",
                        "text": "질문입니다.",
                    }
                ],
                "raw": {
                    "is_fallback": False,
                    "persons": [{"person_id": "person-1"}],
                },
            },
            {
                "scene_id": "s2",
                "start_sec": 10.0,
                "end_sec": 20.0,
                "name": "대답",
                "description": "다른 사람이 웃으며 답한다.",
                "speeches": [
                    {
                        "speech_id": "p2",
                        "start_sec": 11.0,
                        "end_sec": 16.0,
                        "speaker_id": "2",
                        "text": "대답입니다.",
                    }
                ],
                "raw": {
                    "is_fallback": False,
                    "persons": [{"person_id": "person-2"}],
                },
            },
        ]
        features, description, transcript = vpick_features(
            scenes,
            {"duration_ms": 20_000, "resolution": 1080, "persons": [{}, {}]},
            0.0,
            20.0,
        )
        self.assertEqual(features["vpick_available"], 1)
        self.assertEqual(features["vpick_scene_count"], 2)
        self.assertEqual(features["vpick_person_count"], 2)
        self.assertEqual(features["vpick_unique_speakers"], 2)
        self.assertEqual(features["vpick_start_aligned_2s"], 1)
        self.assertEqual(features["vpick_end_aligned_2s"], 1)
        self.assertTrue(description)
        self.assertTrue(transcript)
        self.assertEqual(features["vpick_transcript_usable"], 1)

    def test_rejects_mojibake_description(self) -> None:
        self.assertFalse(text_is_usable("寃쎌긽?? ?쒖슱???"))
        self.assertTrue(text_is_usable("사람이 질문하고 상대방이 웃으며 대답한다."))

    def test_vpick_transcript_requires_enough_speech_evidence(self) -> None:
        text = "[8:21-8:28] S5: 질문에 답하며 상황을 구체적으로 설명합니다."
        self.assertFalse(vpick_transcript_is_usable(text, 1))
        self.assertTrue(vpick_transcript_is_usable(text + "\n" + text, 2))

    def test_can_preserve_low_confidence_transcript_for_judge_input(self) -> None:
        scenes = [
            {
                "scene_id": "s1",
                "start_sec": 0.0,
                "end_sec": 10.0,
                "name": "scene",
                "description": "candidate scene description",
                "speeches": [
                    {
                        "speech_id": "p1",
                        "start_sec": 2.0,
                        "end_sec": 4.0,
                        "speaker_id": "1",
                        "text": "okay",
                    }
                ],
                "raw": {"is_fallback": True, "persons": []},
            }
        ]
        features, _, filtered = vpick_features(scenes, {}, 0.0, 10.0)
        _, _, preserved = vpick_features(
            scenes,
            {},
            0.0,
            10.0,
            preserve_raw_transcript=True,
        )
        self.assertEqual(features["vpick_transcript_usable"], 0)
        self.assertEqual(filtered, "")
        self.assertIn("okay", preserved)

    def test_judge_transcript_keeps_both_usable_sources(self) -> None:
        transcript, source = build_judge_transcript("Vpick 대사", "YouTube 자막")
        self.assertIn("[VPICK_ASR]", transcript)
        self.assertIn("[YT_DLP_CAPTIONS]", transcript)
        self.assertEqual(
            source,
            "vpick_scene_api_asr+yt_dlp_candidate_transcript",
        )

    def test_judge_transcript_labels_subtitle_fallback(self) -> None:
        transcript, source = build_judge_transcript(
            "fallback 대사",
            "후보 자막",
            "yt_dlp_transcript_fallback",
        )
        self.assertIn("[SUBTITLE_FALLBACK]", transcript)
        self.assertNotIn("[VPICK_ASR]", transcript)
        self.assertEqual(
            "yt_dlp_full_longform_scene_transcript+yt_dlp_candidate_transcript",
            source,
        )

    def test_scene_loader_prefers_vpick_then_uses_fallback(self) -> None:
        import json
        import tempfile

        payload = {
            "data": [
                {
                    "scene_id": "s1",
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "description": "설명",
                    "speeches": [],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "vpick"
            fallback = Path(tmp) / "fallback"
            raw.mkdir()
            fallback.mkdir()
            (fallback / "video_scenes.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            scenes, _, provider, _ = load_scene_payload(raw, "video", fallback)
            self.assertEqual(1, len(scenes))
            self.assertEqual("yt_dlp_transcript_fallback", provider)
            (raw / "video_scenes.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            _, _, provider, _ = load_scene_payload(raw, "video", fallback)
            self.assertEqual("vpick_scene_api", provider)


class PerformanceRankerTests(unittest.TestCase):
    def test_auc_counts_ties_as_half(self) -> None:
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.8, 0.5, 0.5, 0.2])
        self.assertAlmostEqual(binary_auc(labels, scores), 0.875)

    def test_feature_sets_exclude_outcome_columns(self) -> None:
        forbidden = {"short_views", "short_likes", "performance_label", "channel_name"}
        for feature_names in FEATURE_SETS.values():
            self.assertFalse(forbidden.intersection(feature_names))
            self.assertNotIn("channel_performance_percentile", feature_names)
            self.assertNotIn("vpick_available", feature_names)

    def test_lolo_predictions_hold_out_complete_longform(self) -> None:
        rows = [
            {
                "candidate_id": "a",
                "long_video_id": "long-1",
                "channel_performance_percentile": "90",
                "duration_sec": "30",
            },
            {
                "candidate_id": "b",
                "long_video_id": "long-1",
                "channel_performance_percentile": "80",
                "duration_sec": "35",
            },
            {
                "candidate_id": "c",
                "long_video_id": "long-2",
                "channel_performance_percentile": "10",
                "duration_sec": "60",
            },
            {
                "candidate_id": "d",
                "long_video_id": "long-3",
                "channel_performance_percentile": "20",
                "duration_sec": "55",
            },
        ]
        predictions = lolo_predictions(rows, ["duration_sec"], alpha=1.0)
        self.assertEqual(len(predictions), 4)
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
