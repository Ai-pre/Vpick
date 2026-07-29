from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_highlight_quality_eval_v1 import (
    assert_blind,
    candidate_blind_row,
    longform_overview,
    normalized_scenes,
    read_csv,
    stable_id,
    to_float,
    write_csv,
    write_jsonl,
)
from build_vpick_candidate_features import load_asset_status, load_scene_payload
from highlight_quality_judge_v1 import validate_longform


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = (
    ROOT
    / "deliverables"
    / "2026-07-24"
    / "performance_ranker"
    / "candidate_features_60_PRIVATE.csv"
)
DEFAULT_RAW = ROOT / "data" / "raw" / "vpick"
DEFAULT_FALLBACK = ROOT / "data" / "raw" / "subtitle_fallback_scenes"
DEFAULT_OUTPUT = (
    ROOT / "deliverables" / "2026-07-24" / "performance_judge_v1"
)


def build(
    feature_rows: list[dict[str, str]],
    raw_dir: Path,
    fallback_dir: Path,
) -> dict[str, Any]:
    blind_candidates: list[dict[str, Any]] = []
    private_targets: list[dict[str, Any]] = []
    longforms: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    for source in sorted(feature_rows, key=lambda row: row["candidate_id"]):
        longform_id = source["long_video_id"]
        scenes, _summary, provider, evidence_path = load_scene_payload(
            raw_dir,
            longform_id,
            fallback_dir,
        )
        if not scenes:
            failures.append(
                {
                    "candidate_id": source["candidate_id"],
                    "reason": "scene_evidence_unavailable",
                }
            )
            continue
        asset_status = (
            load_asset_status(raw_dir, longform_id)
            if provider == "vpick_scene_api"
            else {}
        )
        overview = longform_overview(scenes)
        start = to_float(source["start_sec"])
        end = to_float(source["end_sec"])
        judge_candidate_id = stable_id(
            "PJ",
            source["candidate_id"],
            longform_id,
            round(start, 3),
            round(end, 3),
        )
        blind_candidates.append(
            candidate_blind_row(
                judge_candidate_id,
                longform_id,
                start,
                end,
                scenes,
                asset_status,
                overview,
                provider,
                provider == "vpick_scene_api",
            )
        )
        private_targets.append(
            {
                "candidate_id": judge_candidate_id,
                "source_candidate_id": source["candidate_id"],
                "pair_id": source["pair_id"],
                "longform_id": longform_id,
                "short_video_id": source["short_video_id"],
                "channel_name": source["channel_name"],
                "performance_label": source["performance_label"],
                "channel_performance_percentile": source[
                    "channel_performance_percentile"
                ],
                "start_sec": start,
                "end_sec": end,
                "evidence_provider": provider,
                "evidence_path": str(evidence_path or ""),
            }
        )
        if longform_id not in longforms:
            normalized = normalized_scenes(scenes)
            longform = {
                "longform_id": longform_id,
                "channel_id": "",
                "title": source.get("long_title", ""),
                "duration_ms": round(
                    max(float(scene["end_sec"]) for scene in scenes) * 1000
                ),
                "upload_date": source.get("upload_date", ""),
                "view_count": 0,
                "scenes": normalized,
                "evidence_provider": provider,
                "visual_evidence_available": provider == "vpick_scene_api",
            }
            validate_longform(longform)
            longforms[longform_id] = longform

    assert_blind(blind_candidates)
    return {
        "blind_candidates": blind_candidates,
        "private_targets": private_targets,
        "longforms": list(longforms.values()),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build blind pointwise inputs for the 60-short performance Judge."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--fallback-dir", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    built = build(
        read_csv(args.features),
        args.raw_dir,
        args.fallback_dir,
    )
    write_jsonl(
        args.output_dir / "candidates_blind.jsonl",
        built["blind_candidates"],
    )
    write_jsonl(
        args.output_dir / "longforms_PRIVATE.jsonl",
        built["longforms"],
    )
    write_csv(
        args.output_dir / "candidate_targets_PRIVATE.csv",
        built["private_targets"],
    )
    labels: dict[str, int] = {}
    providers: dict[str, int] = {}
    for row in built["private_targets"]:
        label = str(row["performance_label"])
        provider = str(row["evidence_provider"])
        labels[label] = labels.get(label, 0) + 1
        providers[provider] = providers.get(provider, 0) + 1
    summary = {
        "design": "blind_single_short_performance_judge_v1",
        "candidate_count": len(built["blind_candidates"]),
        "unique_longform_count": len(built["longforms"]),
        "label_counts_PRIVATE": labels,
        "evidence_provider_counts_PRIVATE": providers,
        "failure_count": len(built["failures"]),
        "failures": built["failures"],
        "target_policy": (
            "Train on actual published Shorts only. Pos/Neg and channel percentile "
            "remain private until post-Judge calibration."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
