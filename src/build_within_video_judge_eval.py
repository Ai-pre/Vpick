from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_same_longform_hard_negative_eval import judge_evidence
from build_vpick_candidate_features import load_asset_status, load_scene_payload


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(prefix: str, *parts: Any) -> str:
    value = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def number(value: Any) -> float:
    return float(str(value).strip())


def interval_iou(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    overlap = max(
        0.0,
        min(left_end, right_end) - max(left_start, right_start),
    )
    union = max(left_end, right_end) - min(left_start, right_start)
    return overlap / union if union > 0 else 0.0


def compact_overview(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": scene.get("scene_id", ""),
            "start_ms": scene.get("start_ms", ""),
            "end_ms": scene.get("end_ms", ""),
            "scene_name": str(scene.get("scene_name", ""))[:160],
            "description": str(scene.get("description", ""))[:500],
        }
        for scene in scenes
    ]


def auto_interval(item: dict[str, Any]) -> tuple[float, float] | None:
    metadata = item.get("generation_metadata") or {}
    source_scenes = metadata.get("scenes") or []
    if not isinstance(source_scenes, list) or not source_scenes:
        return None
    intervals: list[tuple[float, float]] = []
    for source in source_scenes:
        try:
            start = number(source["source_start_ms"]) / 1000.0
            end = number(source["source_end_ms"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            return None
        if end <= start:
            return None
        intervals.append((start, end))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] - previous[1] > 1.0:
            return None
    return intervals[0][0], max(end for _, end in intervals)


def evidence_payload(
    *,
    pool_id: str,
    candidate_id: str,
    longform_id: str,
    start_sec: float,
    end_sec: float,
    scenes: list[dict[str, Any]],
    asset_status: dict[str, Any],
    overview: list[dict[str, Any]],
    context_pad_sec: float,
) -> dict[str, Any]:
    evidence = judge_evidence(
        scenes,
        asset_status,
        start_sec,
        end_sec,
        context_pad_sec,
    )
    provider = str(
        (asset_status.get("summary") or {}).get("evidence_provider")
        or asset_status.get("evidence_provider")
        or "vpick_scene_api"
    )
    return {
        "candidate_id": candidate_id,
        "pool_id": pool_id,
        "longform_id": longform_id,
        "start_ms": round(start_sec * 1000),
        "end_ms": round(end_sec * 1000),
        "duration_sec": round(end_sec - start_sec, 3),
        "longform_overview": overview,
        "scene_ids": [],
        "description": evidence["description"],
        "transcript": evidence["transcript"],
        "before_context": evidence["before_context"],
        "after_context": evidence["after_context"],
        "visual_evidence_available": provider == "vpick_scene_api",
        "evidence_policy": "uniform_longform_scene_assembly_v1",
    }


def build(
    labels: list[dict[str, str]],
    raw_dir: Path,
    *,
    max_auto_candidates: int,
    min_auto_candidates: int,
    gold_iou_threshold: float,
    context_pad_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    blind_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    pool_sizes: list[int] = []

    for label in sorted(labels, key=lambda row: row["candidate_id"]):
        longform_id = label["longform_id"]
        detail_path = raw_dir / f"{longform_id}_shortforms_details.json"
        if not detail_path.exists():
            exclusions.append(
                {
                    "candidate_id": label["candidate_id"],
                    "longform_id": longform_id,
                    "reason": "vpick_auto_details_missing",
                }
            )
            continue
        try:
            auto_items = json.loads(detail_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            auto_items = []
        if not isinstance(auto_items, list):
            auto_items = []

        parsed_auto: list[tuple[int, dict[str, Any], float, float]] = []
        for rank, item in enumerate(auto_items, start=1):
            if not isinstance(item, dict):
                continue
            interval = auto_interval(item)
            if interval is None:
                continue
            parsed_auto.append((rank, item, interval[0], interval[1]))
        parsed_auto = parsed_auto[:max_auto_candidates]
        if len(parsed_auto) < min_auto_candidates:
            exclusions.append(
                {
                    "candidate_id": label["candidate_id"],
                    "longform_id": longform_id,
                    "reason": (
                        f"insufficient_contiguous_vpick_auto_candidates:"
                        f"{len(parsed_auto)}"
                    ),
                }
            )
            continue

        scenes, _summary, provider, _path = load_scene_payload(
            raw_dir,
            longform_id,
        )
        if not scenes:
            exclusions.append(
                {
                    "candidate_id": label["candidate_id"],
                    "longform_id": longform_id,
                    "reason": "longform_scene_evidence_missing",
                }
            )
            continue
        asset_status = load_asset_status(raw_dir, longform_id)
        overview = compact_overview(scenes)
        gold_start = number(label["start_sec"])
        gold_end = number(label["end_sec"])
        pool_id = stable_id("WV", label["candidate_id"], longform_id)

        pool_private: list[dict[str, Any]] = []
        pool_blind: list[dict[str, Any]] = []
        overlaps: list[float] = []
        for original_rank, item, start_sec, end_sec in parsed_auto:
            overlap = interval_iou(
                start_sec,
                end_sec,
                gold_start,
                gold_end,
            )
            overlaps.append(overlap)
            candidate_id = stable_id(
                "WVC",
                pool_id,
                "vpick",
                item.get("id", original_rank),
            )
            pool_blind.append(
                evidence_payload(
                    pool_id=pool_id,
                    candidate_id=candidate_id,
                    longform_id=longform_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    scenes=scenes,
                    asset_status=asset_status,
                    overview=overview,
                    context_pad_sec=context_pad_sec,
                )
            )
            pool_private.append(
                {
                    "pool_id": pool_id,
                    "candidate_id": candidate_id,
                    "source_candidate_id_PRIVATE": label["candidate_id"],
                    "channel_name_PRIVATE": label["channel_name"],
                    "performance_label_PRIVATE": label[
                        "performance_label_PRIVATE"
                    ],
                    "longform_id": longform_id,
                    "candidate_role_PRIVATE": "vpick_auto",
                    "is_gold_equivalent_PRIVATE": int(
                        overlap >= gold_iou_threshold
                    ),
                    "baseline_rank": original_rank,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": round(end_sec - start_sec, 3),
                    "gold_start_sec_PRIVATE": gold_start,
                    "gold_end_sec_PRIVATE": gold_end,
                    "gold_iou_PRIVATE": round(overlap, 6),
                    "vpick_shortform_id_PRIVATE": item.get("id", ""),
                    "evidence_provider": provider,
                }
            )

        if not any(value >= gold_iou_threshold for value in overlaps):
            candidate_id = stable_id(
                "WVC",
                pool_id,
                "injected_gold",
                label["candidate_id"],
            )
            pool_blind.append(
                evidence_payload(
                    pool_id=pool_id,
                    candidate_id=candidate_id,
                    longform_id=longform_id,
                    start_sec=gold_start,
                    end_sec=gold_end,
                    scenes=scenes,
                    asset_status=asset_status,
                    overview=overview,
                    context_pad_sec=context_pad_sec,
                )
            )
            pool_private.append(
                {
                    "pool_id": pool_id,
                    "candidate_id": candidate_id,
                    "source_candidate_id_PRIVATE": label["candidate_id"],
                    "channel_name_PRIVATE": label["channel_name"],
                    "performance_label_PRIVATE": label[
                        "performance_label_PRIVATE"
                    ],
                    "longform_id": longform_id,
                    "candidate_role_PRIVATE": "injected_gold",
                    "is_gold_equivalent_PRIVATE": 1,
                    "baseline_rank": len(parsed_auto) + 1,
                    "start_sec": gold_start,
                    "end_sec": gold_end,
                    "duration_sec": round(gold_end - gold_start, 3),
                    "gold_start_sec_PRIVATE": gold_start,
                    "gold_end_sec_PRIVATE": gold_end,
                    "gold_iou_PRIVATE": 1.0,
                    "vpick_shortform_id_PRIVATE": "",
                    "evidence_provider": provider,
                }
            )

        if len({row["candidate_id"] for row in pool_private}) != len(pool_private):
            raise ValueError(f"Duplicate candidate IDs in {pool_id}")
        blind_rows.extend(pool_blind)
        private_rows.extend(pool_private)
        pool_sizes.append(len(pool_private))

    private_ids = {row["candidate_id"] for row in private_rows}
    blind_ids = {row["candidate_id"] for row in blind_rows}
    if private_ids != blind_ids:
        raise ValueError("Blind/private within-video candidate IDs differ")
    summary = {
        "experiment": "exp2_within_video_segment_alignment",
        "input_policy": "uniform_longform_scene_assembly_v1",
        "pool_count": len(pool_sizes),
        "blind_candidate_count": len(blind_rows),
        "pool_size_counts": dict(sorted(Counter(pool_sizes).items())),
        "mean_pool_size": (
            round(sum(pool_sizes) / len(pool_sizes), 3) if pool_sizes else None
        ),
        "min_auto_candidates": min_auto_candidates,
        "max_auto_candidates": max_auto_candidates,
        "gold_iou_threshold": gold_iou_threshold,
        "exclusion_count": len(exclusions),
        "exclusion_reasons": dict(
            sorted(Counter(row["reason"] for row in exclusions).items())
        ),
        "exclusions_PRIVATE": exclusions,
        "note": (
            "The handover assumed eight Vpick candidates per video. Stored API "
            "artifacts vary by video, so each pool keeps up to eight and the "
            "evaluator computes an exact chance baseline from its actual size."
        ),
    }
    return blind_rows, private_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the primary within-video Judge experiment from stored Vpick "
            "automatic shortform candidates and a published gold interval."
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "vpick",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-auto-candidates", type=int, default=8)
    parser.add_argument("--min-auto-candidates", type=int, default=3)
    parser.add_argument("--gold-iou-threshold", type=float, default=0.5)
    parser.add_argument("--context-pad-sec", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blind, private, summary = build(
        read_csv(args.labels),
        args.raw_dir,
        max_auto_candidates=args.max_auto_candidates,
        min_auto_candidates=args.min_auto_candidates,
        gold_iou_threshold=args.gold_iou_threshold,
        context_pad_sec=args.context_pad_sec,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "within_video_candidates_blind.jsonl", blind)
    write_csv(
        args.output_dir / "within_video_pool_targets_PRIVATE.csv",
        private,
    )
    (args.output_dir / "within_video_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
