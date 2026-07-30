from __future__ import annotations

import json
from pathlib import Path

import webapp.server as server
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


def test_env_file_loads_keys_without_overriding_process_env(
    monkeypatch, tmp_path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=from-file\n"
        "OPENAI_API_KEY='from-file-openai'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")

    server.load_env_file(env_file)

    assert server.gemini_api_key() == "from-file"
    assert server.os.getenv("OPENAI_API_KEY") == "already-set"


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


def test_judge_router_uses_gemini_when_openai_key_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    def fake_gemini(**kwargs):
        assert kwargs["text_input"] == '{"candidate":"demo"}'
        return {"score": 4}, "Gemini · test-model"

    monkeypatch.setattr(server, "call_gemini_json", fake_gemini)
    result, model = server.call_judge_json(
        instructions="judge",
        text_input='{"candidate":"demo"}',
        schema_name="demo",
        schema={"type": "object"},
    )

    assert result == {"score": 4}
    assert model == "Gemini · test-model"


def test_judge_router_falls_back_to_gemini_after_openai_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    def failed_openai(**kwargs):
        del kwargs
        raise server.AppError("OpenAI unavailable", status=502)

    monkeypatch.setattr(server, "call_openai_json", failed_openai)
    monkeypatch.setattr(
        server,
        "call_gemini_json",
        lambda **kwargs: ({"score": 3}, "Gemini · fallback-model"),
    )
    result, model = server.call_judge_json(
        instructions="judge",
        text_input="demo",
        schema_name="demo",
        schema={"type": "object"},
    )

    assert result == {"score": 3}
    assert model == "Gemini · fallback-model"


def test_gemini_judge_uses_structured_json_and_inline_thumbnail(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-test")
    monkeypatch.setattr(
        server,
        "gemini_inline_image",
        lambda image_url: {
            "inlineData": {"mimeType": "image/jpeg", "data": "encoded"}
        },
    )

    def fake_generate(payload, *, model, timeout=180):
        assert model == "gemini-test"
        assert timeout == 180
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        assert payload["generationConfig"]["responseSchema"]["required"] == ["score"]
        assert payload["contents"][0]["parts"][1]["inlineData"]["data"] == "encoded"
        return {
            "candidates": [
                {"content": {"parts": [{"text": '{"score": 4}'}]}}
            ]
        }

    monkeypatch.setattr(server, "gemini_generate_content", fake_generate)
    result, model = server.call_gemini_json(
        instructions="judge",
        text_input="demo",
        schema_name="demo",
        schema={
            "type": "object",
            "required": ["score"],
            "properties": {"score": {"type": "integer"}},
        },
        image_url="https://example.com/thumb.jpg",
    )

    assert result == {"score": 4}
    assert model == "Gemini · gemini-test"


def test_transcription_router_uses_gemini_without_openai(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(
        server,
        "transcribe_youtube_audio_gemini",
        lambda video_url: "[0:00-0:04] 자동 전사",
    )

    transcript, source = server.transcribe_youtube_audio(
        "https://youtube.com/shorts/demo"
    )

    assert transcript == "[0:00-0:04] 자동 전사"
    assert source == "gemini_youtube_asr"


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
