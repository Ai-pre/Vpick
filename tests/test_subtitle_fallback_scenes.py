from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_subtitle_fallback_scenes import (  # noqa: E402
    add_llm_descriptions,
    base_scenes,
    build_payload,
    chunk_speeches,
    parse_json3,
)


class SubtitleFallbackScenesTest(unittest.TestCase):
    def test_parse_chunk_and_payload_preserve_timestamps_and_provenance(self) -> None:
        events = {
            "events": [
                {
                    "tStartMs": 1_000,
                    "dDurationMs": 10_000,
                    "segs": [{"utf8": "첫 번째 설명입니다."}],
                },
                {
                    "tStartMs": 12_000,
                    "dDurationMs": 11_000,
                    "segs": [{"utf8": "두 번째 반응입니다."}],
                },
                {
                    "tStartMs": 24_000,
                    "dDurationMs": 10_000,
                    "segs": [{"utf8": "결론입니다."}],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.json3"
            path.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
            speeches = parse_json3(path)
            chunks = chunk_speeches(speeches)
            scenes = base_scenes("VIDEO123456", chunks)
            payload = build_payload(
                "VIDEO123456",
                scenes,
                path,
                "automatic",
                "ko-orig",
                {"provider": "extractive", "model": "none"},
            )

        self.assertEqual(3, len(speeches))
        self.assertEqual(1_000, scenes[0]["start_ms"])
        self.assertEqual(34_000, scenes[-1]["end_ms"])
        self.assertTrue(all(scene["is_fallback"] for scene in scenes))
        self.assertFalse(payload["summary"]["visual_evidence_available"])
        self.assertEqual("yt_dlp_transcript_fallback", payload["summary"]["evidence_provider"])
        self.assertIn("Not a Vpick", payload["summary"]["warning"])

    def test_llm_description_batches_and_retries_missing_scene(self) -> None:
        scenes = [
            {
                "scene_id": f"s{index}",
                "start_ms": index * 1_000,
                "end_ms": (index + 1) * 1_000,
                "speeches": [{"text": f"대사 {index}"}],
                "scene_name": "old",
                "description": "old",
            }
            for index in range(5)
        ]
        calls: list[list[str]] = []

        def fake_call(
            provider: str,
            model: str,
            system_prompt: str,
            user_prompt: str,
            max_tokens: int,
        ) -> dict[str, object]:
            ids = [
                row["scene_id"]
                for row in json.loads(user_prompt)["scenes"]
            ]
            calls.append(ids)
            returned = ids[:-1] if len(calls) == 1 else ids
            return {
                "json": {
                    "scenes": [
                        {
                            "scene_id": scene_id,
                            "scene_name": f"name-{scene_id}",
                            "description": f"description-{scene_id}",
                        }
                        for scene_id in returned
                    ]
                },
                "usage": {"calls": 1},
            }

        import build_subtitle_fallback_scenes as module

        original = module.call_llm
        module.call_llm = fake_call
        try:
            enriched, summary = add_llm_descriptions(
                scenes,
                provider="test",
                model="test",
                batch_size=3,
            )
        finally:
            module.call_llm = original

        self.assertEqual(5, summary["generated_scene_count"])
        self.assertEqual(3, summary["call_count"])
        self.assertEqual("description-s2", enriched[2]["description"])


if __name__ == "__main__":
    unittest.main()
