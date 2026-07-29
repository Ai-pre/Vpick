from __future__ import annotations

from source_evidence import source_outline


def make_longform(scene_count: int = 80) -> dict:
    return {
        "title": "긴 원본",
        "duration_ms": scene_count * 30_000,
        "scenes": [
            {
                "start_ms": index * 30_000,
                "end_ms": (index + 1) * 30_000,
                "scene_name": f"장면 {index:02d}",
                "description": ("설명 " + str(index) + " ") * 20,
            }
            for index in range(scene_count)
        ],
    }


def test_outline_preserves_late_candidate_context_and_global_timeline() -> None:
    outline = source_outline(
        make_longform(),
        3000,
        candidate_start_sec=70 * 30,
        candidate_end_sec=71 * 30,
    )

    assert len(outline) <= 3000
    assert "[후보 주변]" in outline
    assert "장면 70" in outline
    assert "장면 00" in outline
    assert "장면 79" in outline


def test_complete_outline_is_kept_when_it_fits() -> None:
    outline = source_outline(
        make_longform(scene_count=4),
        5000,
        candidate_start_sec=60,
        candidate_end_sec=75,
    )

    for index in range(4):
        assert f"장면 {index:02d}" in outline
