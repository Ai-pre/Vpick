from __future__ import annotations

import json
from pathlib import Path

from webapp.server import (
    SAMPLE_SCENES,
    autofill_short_candidate,
    extract_scene_list,
    generate_candidate_pool,
    interval_overlap_ratio,
    parse_json3_transcript,
    run_highlight_pipeline,
    score_package_formula,
)


def test_package_formula_matches_frozen_weights() -> None:
    score = score_package_formula(
        {
            "change_or_surprise_0_4": 4,
            "title_packaging_0_4": 2,
            "thumbnail_packaging_0_4": 3,
        }
    )
    assert score == 81.2


def test_url_only_candidate_is_filled_from_youtube_collector() -> None:
    def fake_collector(video_url: str) -> dict[str, object]:
        assert video_url.endswith("demoShort1")
        return {
            "title": "마지막 한 개로 성공한 미션",
            "description": "종료 직전에 마지막 주문이 들어온다.",
            "transcript": "[0:00-0:05] 한 개만 더 오면 성공이에요.",
            "thumbnail_url": "https://example.com/thumbnail.jpg",
            "start_time": "0:00",
            "end_time": "0:53",
            "duration_sec": 53,
            "metadata_source": "yt_dlp",
            "transcript_source": "youtube_auto_captions",
            "caption_language": "ko-orig",
        }

    enriched, summary = autofill_short_candidate(
        {"video_url": "https://youtube.com/shorts/demoShort1"},
        collector=fake_collector,
    )

    assert enriched["title"] == "마지막 한 개로 성공한 미션"
    assert enriched["end_time"] == "0:53"
    assert "한 개만 더" in str(enriched["transcript"])
    assert summary["auto_filled"] is True
    assert summary["transcript_source"] == "youtube_auto_captions"


def test_json3_captions_are_converted_to_timestamped_transcript() -> None:
    transcript = parse_json3_transcript(
        {
            "events": [
                {
                    "tStartMs": 1200,
                    "dDurationMs": 2300,
                    "segs": [{"utf8": "첫 번째 문장"}],
                },
                {
                    "tStartMs": 4100,
                    "dDurationMs": 1800,
                    "segs": [{"utf8": "결국 성공했습니다"}],
                },
            ]
        }
    )

    assert "[0:01-0:03] 첫 번째 문장" in transcript
    assert "[0:04-0:05] 결국 성공했습니다" in transcript


def test_demo_scenes_generate_a_diverse_candidate_pool() -> None:
    payload = json.loads(Path(SAMPLE_SCENES).read_text(encoding="utf-8"))
    scenes = extract_scene_list(payload)
    candidates = generate_candidate_pool(scenes)

    assert len(scenes) == 12
    assert len(candidates) >= 10
    for index, candidate in enumerate(candidates):
        for previous in candidates[:index]:
            assert interval_overlap_ratio(candidate, previous) < 0.85


def test_preview_pipeline_returns_four_anchors_and_one_supplement() -> None:
    payload = json.loads(Path(SAMPLE_SCENES).read_text(encoding="utf-8"))
    result = run_highlight_pipeline(
        {
            "video_url": "https://www.youtube.com/watch?v=BETA_DEMO01",
            "scene_payload": payload,
            "mode": "preview",
        }
    )

    assert result["mode"] == "offline_preview"
    assert len(result["final_candidates"]) == 5
    assert sum(
        candidate["selection_role"] == "adaptive_coverage_anchor"
        for candidate in result["final_candidates"]
    ) == 4
    assert result["final_candidates"][-1]["selection_role"] == "judge_supplement"
