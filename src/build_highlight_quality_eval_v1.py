from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from build_same_longform_hard_negative_eval import (
    choose_hard_negative,
    judge_evidence,
)
from build_vpick_candidate_features import (
    load_asset_status,
    load_scene_payload,
)
from highlight_quality_judge_v1 import (
    validate_candidate,
    validate_longform,
)


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
DEFAULT_VPICK_BASELINE = ROOT / "results" / "vpick_baseline" / "metrics.csv"
DEFAULT_EXISTING = ROOT / "results" / "ours_adaptive_coverage" / "metrics.csv"
DEFAULT_OUTPUT = ROOT / "deliverables" / "2026-07-24" / "highlight_quality_v1"
FORBIDDEN_BLIND_KEYS = {
    "channel_name",
    "view_count",
    "short_views",
    "performance_label",
    "candidate_source",
    "is_published",
    "source_system",
    "vpick_selected",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:14]}"


def overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def scene_ids_for_interval(
    scenes: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[str]:
    return [
        str(scene["scene_id"])
        for scene in scenes
        if overlap(
            start_sec,
            end_sec,
            float(scene["start_sec"]),
            float(scene["end_sec"]),
        )
        > 0
    ]


def longform_overview(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scene in scenes:
        output.append(
            {
                "scene_id": str(scene["scene_id"]),
                "start_ms": round(float(scene["start_sec"]) * 1000),
                "end_ms": round(float(scene["end_sec"]) * 1000),
                "scene_name": str(scene.get("name") or ""),
                "description": str(scene.get("description") or "")[:300],
                "transcript_excerpt": str(scene.get("transcript") or "")[:450],
            }
        )
    return output


def normalized_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scene in scenes:
        speakers = sorted(
            {
                str(speech.get("speaker_id") or "")
                for speech in scene.get("speeches", [])
                if str(speech.get("speaker_id") or "")
            }
        )
        persons = [
            str(person.get("person_id") or "")
            for person in (scene.get("raw", {}).get("persons") or [])
            if str(person.get("person_id") or "")
        ]
        output.append(
            {
                "scene_id": str(scene["scene_id"]),
                "start_ms": round(float(scene["start_sec"]) * 1000),
                "end_ms": round(float(scene["end_sec"]) * 1000),
                "scene_name": str(scene.get("name") or ""),
                "description": str(scene.get("description") or ""),
                "transcript": str(scene.get("transcript") or ""),
                "speaker": "|".join(speakers),
                "person_ids": persons,
            }
        )
    return output


def prediction_intervals(path: Path, model_filter: str = "") -> dict[str, list[dict[str, Any]]]:
    rows = read_csv(path)
    selected: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, float, float]] = set()
    for row in rows:
        if str(row.get("rank") or "") not in {"1", "1.0"}:
            continue
        if model_filter and row.get("model_name") != model_filter:
            continue
        longform_id = row.get("long_video_id", "")
        start = to_float(row.get("pred_start_sec"))
        end = to_float(row.get("pred_end_sec"))
        key = (longform_id, round(start, 3), round(end, 3))
        if not longform_id or end <= start or key in seen:
            continue
        seen.add(key)
        selected.setdefault(longform_id, []).append(
            {
                "start_sec": start,
                "end_sec": end,
                "run_id": row.get("run_id", ""),
                "model_name": row.get("model_name", ""),
            }
        )
    return selected


def candidate_blind_row(
    candidate_id: str,
    longform_id: str,
    start_sec: float,
    end_sec: float,
    scenes: list[dict[str, Any]],
    asset_status: dict[str, Any],
    overview: list[dict[str, Any]],
    evidence_provider: str,
    visual_available: bool,
) -> dict[str, Any]:
    evidence = judge_evidence(
        scenes,
        asset_status if visual_available else {},
        start_sec,
        end_sec,
        15.0,
    )
    return {
        "candidate_id": candidate_id,
        "longform_id": longform_id,
        "start_ms": round(start_sec * 1000),
        "end_ms": round(end_sec * 1000),
        "duration_ms": round((end_sec - start_sec) * 1000),
        "scene_ids": scene_ids_for_interval(scenes, start_sec, end_sec),
        "longform_overview": overview,
        "description": evidence["description"],
        "transcript": evidence["transcript"],
        "before_context": evidence["before_context"],
        "after_context": evidence["after_context"],
        "evidence_provider": evidence_provider,
        "visual_evidence_available": visual_available,
    }


def add_candidate(
    blind_by_id: dict[str, dict[str, Any]],
    private_by_id: dict[str, dict[str, Any]],
    *,
    longform_id: str,
    start_sec: float,
    end_sec: float,
    candidate_source: str,
    source_variant: str,
    is_published: bool,
    scenes: list[dict[str, Any]],
    asset_status: dict[str, Any],
    overview: list[dict[str, Any]],
    evidence_provider: str,
    source_row: dict[str, Any],
    seed: int,
) -> str:
    candidate_id = stable_id(
        "HQ",
        longform_id,
        round(start_sec, 3),
        round(end_sec, 3),
        candidate_source,
        source_variant,
        seed,
    )
    if candidate_id in blind_by_id:
        return candidate_id
    blind_by_id[candidate_id] = candidate_blind_row(
        candidate_id,
        longform_id,
        start_sec,
        end_sec,
        scenes,
        asset_status,
        overview,
        evidence_provider,
        evidence_provider == "vpick_scene_api",
    )
    private = {
        "candidate_id": candidate_id,
        "longform_id": longform_id,
        "start_ms": round(start_sec * 1000),
        "end_ms": round(end_sec * 1000),
        "scene_ids": blind_by_id[candidate_id]["scene_ids"],
        "candidate_source": candidate_source,
        "source_variant": source_variant,
        "is_published": is_published,
        "source_candidate_id": source_row.get("candidate_id", ""),
        "short_video_id": source_row.get("short_video_id", "") if is_published else "",
        "performance_label": source_row.get("performance_label", "") if is_published else "",
        "channel_name": source_row.get("channel_name", ""),
    }
    validate_candidate(private)
    private_by_id[candidate_id] = private
    return candidate_id


def pair_rows(
    anchor_id: str,
    alternative_ids: list[str],
    blind_by_id: dict[str, dict[str, Any]],
    private_by_id: dict[str, dict[str, Any]],
    rng: random.Random,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blind_pairs: list[dict[str, Any]] = []
    private_pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for alternative_id in alternative_ids:
        if alternative_id == anchor_id:
            continue
        unordered = tuple(sorted((anchor_id, alternative_id)))
        if unordered in seen:
            continue
        seen.add(unordered)
        pair_id = stable_id("HQP", unordered[0], unordered[1], seed)
        a_id, b_id = (
            (anchor_id, alternative_id)
            if rng.random() < 0.5
            else (alternative_id, anchor_id)
        )
        left = blind_by_id[a_id]
        right = blind_by_id[b_id]
        if left["longform_id"] != right["longform_id"]:
            raise ValueError("Pair candidates must share a longform")
        blind_pairs.append(
            {
                "pair_id": pair_id,
                "longform_id": left["longform_id"],
                "longform_overview": left["longform_overview"],
                "candidate_a": {
                    key: value
                    for key, value in left.items()
                    if key != "longform_overview"
                },
                "candidate_b": {
                    key: value
                    for key, value in right.items()
                    if key != "longform_overview"
                },
            }
        )
        private_pairs.append(
            {
                "pair_id": pair_id,
                "longform_id": left["longform_id"],
                "candidate_a_id": a_id,
                "candidate_b_id": b_id,
                "display_order": "A-B",
                "published_anchor_id": anchor_id,
                "candidate_a_source": private_by_id[a_id]["candidate_source"],
                "candidate_b_source": private_by_id[b_id]["candidate_source"],
                "reference_signal_only": "published_short_is_noisy_reference_not_ground_truth",
            }
        )
    return blind_pairs, private_pairs


def assert_blind(rows: list[dict[str, Any]]) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            leaked = FORBIDDEN_BLIND_KEYS.intersection(value)
            if leaked:
                raise ValueError(f"Blind data leaked private keys: {sorted(leaked)}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for row in rows:
        walk(row)


def build(
    feature_rows: list[dict[str, str]],
    raw_dir: Path,
    fallback_dir: Path,
    vpick_predictions: dict[str, list[dict[str, Any]]],
    existing_predictions: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    positives = [
        row for row in feature_rows if row.get("performance_label") == "pos"
    ]
    intervals_by_longform: dict[str, list[tuple[float, float]]] = {}
    for row in feature_rows:
        intervals_by_longform.setdefault(row["long_video_id"], []).append(
            (to_float(row["start_sec"]), to_float(row["end_sec"]))
        )

    longforms: dict[str, dict[str, Any]] = {}
    blind_by_id: dict[str, dict[str, Any]] = {}
    private_by_id: dict[str, dict[str, Any]] = {}
    blind_pairs: list[dict[str, Any]] = []
    private_pairs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for source in sorted(positives, key=lambda row: row["candidate_id"]):
        longform_id = source["long_video_id"]
        scenes, _summary, provider, _path = load_scene_payload(
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
        if longform_id not in longforms:
            normalized = normalized_scenes(scenes)
            longform = {
                "longform_id": longform_id,
                "channel_id": "",
                "title": source.get("long_title", ""),
                "duration_ms": round(max(scene["end_sec"] for scene in scenes) * 1000),
                "upload_date": source.get("upload_date", ""),
                "view_count": 0,
                "scenes": normalized,
                "evidence_provider": provider,
                "visual_evidence_available": provider == "vpick_scene_api",
            }
            validate_longform(longform)
            longforms[longform_id] = longform

        start = to_float(source["start_sec"])
        end = to_float(source["end_sec"])
        duration = end - start
        anchor_id = add_candidate(
            blind_by_id,
            private_by_id,
            longform_id=longform_id,
            start_sec=start,
            end_sec=end,
            candidate_source="published_short",
            source_variant=source.get("short_video_id", ""),
            is_published=True,
            scenes=scenes,
            asset_status=asset_status,
            overview=overview,
            evidence_provider=provider,
            source_row=source,
            seed=seed,
        )
        alternatives: list[str] = []
        shift = min(5.0, max(2.0, duration * 0.12))
        if duration - shift >= 10:
            alternatives.append(
                add_candidate(
                    blind_by_id,
                    private_by_id,
                    longform_id=longform_id,
                    start_sec=start + shift,
                    end_sec=end,
                    candidate_source="boundary_shift",
                    source_variant="abrupt_start",
                    is_published=False,
                    scenes=scenes,
                    asset_status=asset_status,
                    overview=overview,
                    evidence_provider=provider,
                    source_row=source,
                    seed=seed,
                )
            )
            alternatives.append(
                add_candidate(
                    blind_by_id,
                    private_by_id,
                    longform_id=longform_id,
                    start_sec=start,
                    end_sec=end - shift,
                    candidate_source="boundary_shift",
                    source_variant="abrupt_end",
                    is_published=False,
                    scenes=scenes,
                    asset_status=asset_status,
                    overview=overview,
                    evidence_provider=provider,
                    source_row=source,
                    seed=seed,
                )
            )

        hard, pool = choose_hard_negative(
            scenes,
            asset_status,
            duration,
            intervals_by_longform.get(longform_id, []),
        )
        if hard:
            alternatives.append(
                add_candidate(
                    blind_by_id,
                    private_by_id,
                    longform_id=longform_id,
                    start_sec=float(hard["start_sec"]),
                    end_sec=float(hard["end_sec"]),
                    candidate_source="hard_negative",
                    source_variant="evidence_rich_nonoverlap",
                    is_published=False,
                    scenes=scenes,
                    asset_status=asset_status,
                    overview=overview,
                    evidence_provider=provider,
                    source_row=source,
                    seed=seed,
                )
            )
        random_pool = pool[1:] if len(pool) > 1 else []
        if random_pool:
            random_candidate = rng.choice(random_pool)
            alternatives.append(
                add_candidate(
                    blind_by_id,
                    private_by_id,
                    longform_id=longform_id,
                    start_sec=float(random_candidate["start_sec"]),
                    end_sec=float(random_candidate["end_sec"]),
                    candidate_source="random",
                    source_variant="similar_duration_nonoverlap",
                    is_published=False,
                    scenes=scenes,
                    asset_status=asset_status,
                    overview=overview,
                    evidence_provider=provider,
                    source_row=source,
                    seed=seed,
                )
            )

        for prediction in vpick_predictions.get(longform_id, [])[:1]:
            alternatives.append(
                add_candidate(
                    blind_by_id,
                    private_by_id,
                    longform_id=longform_id,
                    start_sec=float(prediction["start_sec"]),
                    end_sec=float(prediction["end_sec"]),
                    candidate_source="vpick",
                    source_variant=str(prediction["run_id"]),
                    is_published=False,
                    scenes=scenes,
                    asset_status=asset_status,
                    overview=overview,
                    evidence_provider=provider,
                    source_row=source,
                    seed=seed,
                )
            )
        for prediction in existing_predictions.get(longform_id, [])[:1]:
            alternatives.append(
                add_candidate(
                    blind_by_id,
                    private_by_id,
                    longform_id=longform_id,
                    start_sec=float(prediction["start_sec"]),
                    end_sec=float(prediction["end_sec"]),
                    candidate_source="existing_model",
                    source_variant=str(prediction["run_id"]),
                    is_published=False,
                    scenes=scenes,
                    asset_status=asset_status,
                    overview=overview,
                    evidence_provider=provider,
                    source_row=source,
                    seed=seed,
                )
            )
        new_blind, new_private = pair_rows(
            anchor_id,
            alternatives,
            blind_by_id,
            private_by_id,
            rng,
            seed,
        )
        blind_pairs.extend(new_blind)
        private_pairs.extend(new_private)

    assert_blind(list(blind_by_id.values()))
    assert_blind(blind_pairs)
    return {
        "longforms": list(longforms.values()),
        "blind_candidates": list(blind_by_id.values()),
        "private_candidates": list(private_by_id.values()),
        "blind_pairs": blind_pairs,
        "private_pairs": private_pairs,
        "failures": failures,
    }


def human_rows(
    blind_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for pair in blind_pairs:
        a = pair["candidate_a"]
        b = pair["candidate_b"]
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "longform_id": pair["longform_id"],
                "longform_overview_json": json.dumps(
                    pair["longform_overview"],
                    ensure_ascii=False,
                ),
                "candidate_a_id": a["candidate_id"],
                "a_start_ms": a["start_ms"],
                "a_end_ms": a["end_ms"],
                "a_description": a["description"],
                "a_transcript": a["transcript"],
                "candidate_b_id": b["candidate_id"],
                "b_start_ms": b["start_ms"],
                "b_end_ms": b["end_ms"],
                "b_description": b["description"],
                "b_transcript": b["transcript"],
                "display_order": "A-B",
                "annotator_id": "",
                "winner": "",
                "confidence_1_5": "",
                "reason": "",
                "invalid_reason": "",
                "created_at": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the blind same-longform Highlight Quality v1 evaluation set."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--fallback-dir", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--vpick-baseline", type=Path, default=DEFAULT_VPICK_BASELINE)
    parser.add_argument("--existing-model", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    built = build(
        read_csv(args.features),
        args.raw_dir,
        args.fallback_dir,
        prediction_intervals(args.vpick_baseline),
        prediction_intervals(args.existing_model, model_filter="gpt-4o-mini"),
        seed=args.seed,
    )
    write_jsonl(args.output_dir / "longforms_PRIVATE.jsonl", built["longforms"])
    write_jsonl(
        args.output_dir / "candidates_blind.jsonl",
        built["blind_candidates"],
    )
    write_jsonl(args.output_dir / "pairs_blind.jsonl", built["blind_pairs"])
    write_csv(
        args.output_dir / "candidate_sources_PRIVATE.csv",
        built["private_candidates"],
    )
    write_csv(
        args.output_dir / "pair_sources_PRIVATE.csv",
        built["private_pairs"],
    )
    write_csv(
        args.output_dir / "human_pairwise_annotations.csv",
        human_rows(built["blind_pairs"]),
    )
    source_counts: dict[str, int] = {}
    for row in built["private_candidates"]:
        source = row["candidate_source"]
        source_counts[source] = source_counts.get(source, 0) + 1
    summary = {
        "design": "blind_same_longform_highlight_quality_v1",
        "longform_count": len(built["longforms"]),
        "candidate_count": len(built["blind_candidates"]),
        "pair_count": len(built["blind_pairs"]),
        "candidate_source_counts": source_counts,
        "failure_count": len(built["failures"]),
        "failures": built["failures"],
        "blind_forbidden_fields": sorted(FORBIDDEN_BLIND_KEYS),
        "reference_policy": (
            "Published high-performance Shorts are noisy reference signals, "
            "not forced pairwise ground truth."
        ),
        "seed": args.seed,
    }
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
