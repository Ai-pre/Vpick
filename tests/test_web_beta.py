from __future__ import annotations

import json
from pathlib import Path

from webapp.server import (
    SAMPLE_SCENES,
    extract_scene_list,
    generate_candidate_pool,
    interval_overlap_ratio,
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
