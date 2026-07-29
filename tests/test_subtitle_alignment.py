from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_short_long_alignment import (  # noqa: E402
    Cue,
    SubtitleCollector,
    align_transcripts,
    display_timestamp,
    interval_iou,
    parse_json3,
    row_video_id,
)


class SubtitleAlignmentTests(unittest.TestCase):
    def test_parse_json3_ignores_layout_only_events(self) -> None:
        payload = {
            "events": [
                {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hello world"}]},
                {"tStartMs": 1000, "dDurationMs": 500, "segs": [{"utf8": "\n"}]},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json3"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cues = parse_json3(path)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "hello world")

    def test_continuous_excerpt_is_detected(self) -> None:
        long_cues = [
            Cue(index * 3.0, index * 3.0 + 2.5, f"unique phrase number {index} alpha beta")
            for index in range(20)
        ]
        short_cues = [
            Cue(index * 3.0, index * 3.0 + 2.5, f"unique phrase number {index + 6} alpha beta")
            for index in range(6)
        ]
        result = align_transcripts(short_cues, long_cues)
        self.assertEqual(result["status"], "continuous")
        self.assertLess(abs(float(result["predicted_start"]) - 18.0), 4.0)

    def test_large_source_jump_is_flagged_as_edit(self) -> None:
        long_cues = [
            Cue(0.0, 3.0, "opening unique alpha one"),
            Cue(3.0, 6.0, "opening unique alpha two"),
            Cue(100.0, 103.0, "closing unique omega one"),
            Cue(103.0, 106.0, "closing unique omega two"),
        ]
        short_cues = [
            Cue(0.0, 3.0, "opening unique alpha one"),
            Cue(3.0, 6.0, "opening unique alpha two"),
            Cue(6.0, 9.0, "closing unique omega one"),
            Cue(9.0, 12.0, "closing unique omega two"),
        ]
        result = align_transcripts(short_cues, long_cues)
        self.assertIn(result["status"], {"light_edit", "heavy_edit"})
        self.assertGreater(float(result["source_span"]), float(result["short_span"]) * 2.0)

    def test_monotonic_edit_under_thirty_seconds_is_light(self) -> None:
        long_cues = [
            Cue(0.0, 3.0, "opening unique alpha one"),
            Cue(3.0, 6.0, "opening unique alpha two"),
            Cue(30.0, 33.0, "closing unique omega one"),
            Cue(33.0, 36.0, "closing unique omega two"),
        ]
        short_cues = [
            Cue(0.0, 3.0, "opening unique alpha one"),
            Cue(3.0, 6.0, "opening unique alpha two"),
            Cue(6.0, 9.0, "closing unique omega one"),
            Cue(9.0, 12.0, "closing unique omega two"),
        ]
        result = align_transcripts(short_cues, long_cues)
        self.assertEqual(result["status"], "light_edit")

    def test_interval_iou(self) -> None:
        self.assertAlmostEqual(interval_iou(0.0, 10.0, 5.0, 15.0), 1.0 / 3.0)

    def test_url_only_rows_are_supported(self) -> None:
        row = {
            "long_video_url": "https://youtu.be/ZLWqgD03tFU?si=example",
            "short_video_url": "https://www.youtube.com/shorts/UoEqF2_aIJk",
        }
        self.assertEqual(row_video_id(row, "long"), "ZLWqgD03tFU")
        self.assertEqual(row_video_id(row, "short"), "UoEqF2_aIJk")

    def test_display_timestamp(self) -> None:
        self.assertEqual(display_timestamp(1338.919), "22:18.919")

    def test_korean_original_auto_caption_is_preferred_for_alignment(self) -> None:
        info = {
            "subtitles": {"ko": [{"ext": "json3", "url": "manual"}]},
            "automatic_captions": {"ko-orig": [{"ext": "json3", "url": "automatic"}]},
        }
        selected = SubtitleCollector._preferred_track(info)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0:2], ("automatic", "ko-orig"))

    def test_original_english_caption_beats_korean_translation(self) -> None:
        info = {
            "automatic_captions": {
                "ko": [{"ext": "json3", "url": "translated"}],
                "en-orig": [{"ext": "json3", "url": "original"}],
            }
        }
        selected = SubtitleCollector._preferred_track(info)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0:2], ("automatic", "en-orig"))
        self.assertEqual(selected[2]["url"], "original")


if __name__ == "__main__":
    unittest.main()
