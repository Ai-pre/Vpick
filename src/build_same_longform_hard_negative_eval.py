from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from build_vpick_candidate_features import (
    load_asset_status,
    load_scene_payload,
    vpick_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = (
    ROOT
    / "deliverables"
    / "2026-07-24"
    / "performance_ranker"
    / "candidate_features_60_PRIVATE.csv"
)
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "vpick"
DEFAULT_OUTPUT_DIR = ROOT / "deliverables" / "2026-07-24" / "hard_negative_eval"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlaps_known_short(
    start_sec: float,
    end_sec: float,
    known_intervals: list[tuple[float, float]],
    guard_sec: float = 0.0,
) -> bool:
    return any(
        interval_overlap(
            start_sec,
            end_sec,
            max(0.0, known_start - guard_sec),
            known_end + guard_sec,
        )
        > 0
        for known_start, known_end in known_intervals
    )


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:14]}"


def unique_speeches(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    speeches: dict[tuple[float, float, str], dict[str, Any]] = {}
    for scene in scenes:
        for speech in scene.get("speeches") or []:
            key = (
                float(speech["start_sec"]),
                float(speech["end_sec"]),
                str(speech.get("text") or ""),
            )
            speeches[key] = speech
    return sorted(
        speeches.values(),
        key=lambda speech: (float(speech["start_sec"]), float(speech["end_sec"])),
    )


def generate_scene_windows(
    scenes: list[dict[str, Any]],
    target_duration_sec: float,
    min_ratio: float,
    max_ratio: float,
) -> list[dict[str, Any]]:
    minimum = max(8.0, target_duration_sec * min_ratio)
    maximum = min(120.0, target_duration_sec * max_ratio)
    windows: list[dict[str, Any]] = []
    for start_index in range(len(scenes)):
        for end_index in range(start_index, len(scenes)):
            start_sec = float(scenes[start_index]["start_sec"])
            end_sec = float(scenes[end_index]["end_sec"])
            duration_sec = end_sec - start_sec
            if duration_sec > maximum:
                break
            if duration_sec >= minimum:
                windows.append(
                    {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "duration_sec": duration_sec,
                        "boundary_type": "scene_sequence",
                        "boundary_strength": 1.0,
                    }
                )
    return windows


def generate_speech_windows(
    scenes: list[dict[str, Any]],
    target_duration_sec: float,
    min_ratio: float,
    max_ratio: float,
) -> list[dict[str, Any]]:
    speeches = unique_speeches(scenes)
    minimum = max(8.0, target_duration_sec * min_ratio)
    maximum = min(120.0, target_duration_sec * max_ratio)
    windows: list[dict[str, Any]] = []
    for start_index in range(len(speeches)):
        for end_index in range(start_index, len(speeches)):
            start_sec = float(speeches[start_index]["start_sec"])
            end_sec = float(speeches[end_index]["end_sec"])
            duration_sec = end_sec - start_sec
            if duration_sec > maximum:
                break
            if duration_sec >= minimum:
                windows.append(
                    {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "duration_sec": duration_sec,
                        "boundary_type": "speech_sequence",
                        "boundary_strength": 0.85,
                    }
                )
    return windows


def generate_fallback_windows(
    scenes: list[dict[str, Any]],
    target_duration_sec: float,
) -> list[dict[str, Any]]:
    duration_sec = min(120.0, max(8.0, target_duration_sec))
    windows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_start = float(scene["start_sec"])
        scene_end = float(scene["end_sec"])
        if scene_end - scene_start < duration_sec:
            continue
        for start_sec in (scene_start, max(scene_start, scene_end - duration_sec)):
            windows.append(
                {
                    "start_sec": start_sec,
                    "end_sec": start_sec + duration_sec,
                    "duration_sec": duration_sec,
                    "boundary_type": "scene_edge_duration_trim",
                    "boundary_strength": 0.65,
                }
            )
    return windows


def deduplicate_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_interval: dict[tuple[float, float], dict[str, Any]] = {}
    for window in windows:
        key = (round(float(window["start_sec"]), 3), round(float(window["end_sec"]), 3))
        previous = by_interval.get(key)
        if previous is None or float(window["boundary_strength"]) > float(
            previous["boundary_strength"]
        ):
            by_interval[key] = window
    return list(by_interval.values())


def hardness_score(
    target_duration_sec: float,
    candidate: dict[str, Any],
    features: dict[str, Any],
) -> float:
    duration_sec = max(0.001, float(candidate["duration_sec"]))
    duration_similarity = math.exp(-abs(math.log(duration_sec / target_duration_sec)))
    speech_density = min(1.0, to_float(features.get("vpick_speech_count")) / 8.0)
    speech_coverage = min(1.0, to_float(features.get("vpick_speech_coverage_ratio")))
    speakers = min(1.0, to_float(features.get("vpick_unique_speakers")) / 2.0)
    description = min(
        1.0,
        to_float(features.get("vpick_description_char_count")) / 240.0,
    )
    evidence_usable = max(
        to_float(features.get("vpick_text_usable")),
        to_float(features.get("vpick_transcript_usable")),
    )
    return (
        0.25 * duration_similarity
        + 0.20 * speech_density
        + 0.15 * speech_coverage
        + 0.10 * speakers
        + 0.10 * description
        + 0.10 * float(candidate["boundary_strength"])
        + 0.10 * evidence_usable
    )


def context_text(
    scenes: list[dict[str, Any]],
    asset_status: dict[str, Any],
    start_sec: float,
    end_sec: float,
) -> str:
    _features, _description, transcript = vpick_features(
        scenes,
        asset_status,
        start_sec,
        end_sec,
        preserve_raw_transcript=True,
    )
    return transcript


def judge_evidence(
    scenes: list[dict[str, Any]],
    asset_status: dict[str, Any],
    start_sec: float,
    end_sec: float,
    context_pad_sec: float,
) -> dict[str, Any]:
    features, description, transcript = vpick_features(
        scenes,
        asset_status,
        start_sec,
        end_sec,
        preserve_raw_transcript=True,
    )
    before_context = context_text(
        scenes,
        asset_status,
        max(0.0, start_sec - context_pad_sec),
        start_sec,
    )
    asset_duration = to_float(features.get("vpick_asset_duration_sec"))
    after_end = min(asset_duration, end_sec + context_pad_sec) if asset_duration else end_sec + context_pad_sec
    after_context = context_text(
        scenes,
        asset_status,
        end_sec,
        after_end,
    )
    return {
        "description": description,
        "transcript": transcript,
        "before_context": before_context,
        "after_context": after_context,
        "features": features,
    }


def choose_hard_negative(
    scenes: list[dict[str, Any]],
    asset_status: dict[str, Any],
    target_duration_sec: float,
    known_intervals: list[tuple[float, float]],
    *,
    min_ratio: float = 0.75,
    max_ratio: float = 1.25,
    guard_sec: float = 2.0,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    windows = generate_scene_windows(scenes, target_duration_sec, min_ratio, max_ratio)
    windows += generate_speech_windows(scenes, target_duration_sec, min_ratio, max_ratio)
    windows = deduplicate_windows(windows)
    if not windows:
        windows = deduplicate_windows(generate_fallback_windows(scenes, target_duration_sec))

    valid: list[dict[str, Any]] = []
    for candidate in windows:
        start_sec = float(candidate["start_sec"])
        end_sec = float(candidate["end_sec"])
        if overlaps_known_short(start_sec, end_sec, known_intervals, guard_sec):
            continue
        features, description, transcript = vpick_features(
            scenes,
            asset_status,
            start_sec,
            end_sec,
        )
        if not description and not transcript:
            continue
        enriched = {
            **candidate,
            **features,
            "hardness_score": hardness_score(target_duration_sec, candidate, features),
        }
        valid.append(enriched)
    valid.sort(
        key=lambda row: (
            -float(row["hardness_score"]),
            abs(float(row["duration_sec"]) - target_duration_sec),
            float(row["start_sec"]),
        )
    )
    return (valid[0] if valid else None), valid


def build_dataset(
    feature_rows: list[dict[str, str]],
    raw_dir: Path,
    *,
    context_pad_sec: float = 15.0,
    seed: int = 20260724,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    positives = [row for row in feature_rows if row.get("performance_label") == "pos"]
    intervals_by_longform: dict[str, list[tuple[float, float]]] = {}
    for row in feature_rows:
        long_video_id = row.get("long_video_id", "")
        start_sec = to_float(row.get("start_sec"))
        end_sec = to_float(row.get("end_sec"))
        if long_video_id and end_sec > start_sec:
            intervals_by_longform.setdefault(long_video_id, []).append((start_sec, end_sec))

    private_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    pair_counter = 0

    for positive in sorted(positives, key=lambda row: row["candidate_id"]):
        long_video_id = positive["long_video_id"]
        scenes, _summary, _provider, _scene_path = load_scene_payload(
            raw_dir,
            long_video_id,
        )
        if not scenes:
            failures.append(
                {
                    "source_candidate_id": positive["candidate_id"],
                    "long_video_id": long_video_id,
                    "reason": "vpick_scenes_unavailable",
                }
            )
            continue
        asset_status = load_asset_status(raw_dir, long_video_id)
        positive_start = to_float(positive.get("start_sec"))
        positive_end = to_float(positive.get("end_sec"))
        target_duration = positive_end - positive_start
        hard_negative, valid_candidates = choose_hard_negative(
            scenes,
            asset_status,
            target_duration,
            intervals_by_longform.get(long_video_id, []),
        )
        if hard_negative is None:
            failures.append(
                {
                    "source_candidate_id": positive["candidate_id"],
                    "long_video_id": long_video_id,
                    "reason": "no_nonoverlapping_evidence_candidate",
                }
            )
            continue

        pair_counter += 1
        pair_id = f"HN{pair_counter:03d}"
        pair_members = [
            (
                "positive",
                positive_start,
                positive_end,
                "published_high_performance_short_source_interval",
                "",
                "",
            ),
            (
                "hard_negative",
                float(hard_negative["start_sec"]),
                float(hard_negative["end_sec"]),
                "same_longform_known_short_nonoverlap_vpick_candidate",
                hard_negative["boundary_type"],
                hard_negative["hardness_score"],
            ),
        ]
        for role, start_sec, end_sec, origin, boundary_type, hard_score in pair_members:
            candidate_id = stable_id(
                "HNC",
                long_video_id,
                round(start_sec, 3),
                round(end_sec, 3),
                seed,
            )
            evidence = judge_evidence(
                scenes,
                asset_status,
                start_sec,
                end_sec,
                context_pad_sec,
            )
            blind_rows.append(
                {
                    "candidate_id": candidate_id,
                    "duration_sec": round(end_sec - start_sec, 3),
                    "description": evidence["description"],
                    "transcript": evidence["transcript"],
                    "before_context": evidence["before_context"],
                    "after_context": evidence["after_context"],
                }
            )
            private_rows.append(
                {
                    "eval_pair_id": pair_id,
                    "candidate_id": candidate_id,
                    "reference_role": role,
                    "source_candidate_id": positive["candidate_id"] if role == "positive" else "",
                    "source_pair_id": positive.get("pair_id", ""),
                    "channel_name": positive.get("channel_name", ""),
                    "long_video_id": long_video_id,
                    "short_video_id": positive.get("short_video_id", "") if role == "positive" else "",
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(end_sec - start_sec, 3),
                    "candidate_origin": origin,
                    "boundary_type": boundary_type,
                    "hardness_score": hard_score,
                    "known_short_interval_count": len(
                        intervals_by_longform.get(long_video_id, [])
                    ),
                    "eligible_negative_pool_size": len(valid_candidates),
                    "vpick_scene_count": evidence["features"].get("vpick_scene_count", 0),
                    "vpick_speech_count": evidence["features"].get("vpick_speech_count", 0),
                    "vpick_unique_speakers": evidence["features"].get(
                        "vpick_unique_speakers", 0
                    ),
                    "vpick_description_usable": evidence["features"].get(
                        "vpick_text_usable", 0
                    ),
                    "vpick_transcript_usable": evidence["features"].get(
                        "vpick_transcript_usable", 0
                    ),
                }
            )

    rng = random.Random(seed)
    rng.shuffle(blind_rows)
    summary = {
        "design": "same_longform_positive_vs_known_short_nonoverlap_vpick_hard_negative",
        "positive_definition": "channel-relative top-performing published Short source interval",
        "hard_negative_definition": (
            "same-longform, similar-duration, Vpick-evidence-rich candidate that does not "
            "overlap any known Short interval in the approved 60-item dataset; this does "
            "not claim that the interval was never published outside the dataset"
        ),
        "selection_uses_performance": False,
        "judge_input_uses_performance_or_channel": False,
        "evidence_mode": "vpick_description_and_vpick_asr_for_both_roles",
        "positive_source_count": len(positives),
        "generated_pair_count": pair_counter,
        "blind_candidate_count": len(blind_rows),
        "failure_count": len(failures),
        "failure_reasons": {
            reason: sum(row["reason"] == reason for row in failures)
            for reason in sorted({row["reason"] for row in failures})
        },
        "failures": failures,
        "seed": seed,
    }
    return blind_rows, private_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build same-longform hard-negative evaluation candidates."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-pad-sec", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blind_rows, private_rows, summary = build_dataset(
        read_csv(args.features),
        args.raw_dir,
        context_pad_sec=args.context_pad_sec,
        seed=args.seed,
    )
    write_csv(args.output_dir / "candidates_blind_vpick_only.csv", blind_rows)
    write_csv(args.output_dir / "hard_negative_pairs_PRIVATE.csv", private_rows)
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
