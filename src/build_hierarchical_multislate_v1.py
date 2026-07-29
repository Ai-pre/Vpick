from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_longform_slate import (
    candidate_center,
    candidate_text,
    load_scenes,
    select_adaptive_coverage,
)


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def normalized(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def stable_candidate_id(longform_id: str, start: float, end: float) -> str:
    raw = f"{longform_id}|{start:.3f}|{end:.3f}".encode("utf-8")
    return "HC_" + hashlib.sha1(raw).hexdigest()[:14]


def interval_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        round(number(row.get("pred_start_sec")), 3),
        round(number(row.get("pred_end_sec")), 3),
    )


def overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start, left_end = interval_key(left)
    right_start, right_end = interval_key(right)
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    shortest = min(left_end - left_start, right_end - right_start)
    return overlap / shortest if shortest > 0 else 0.0


def parse_components(row: dict[str, Any]) -> dict[str, float]:
    try:
        value = json.loads(str(row.get("rerank_components", "") or "{}"))
    except json.JSONDecodeError:
        value = {}
    return {
        str(key): number(item)
        for key, item in value.items()
        if isinstance(item, (int, float))
    }


def component(components: dict[str, float], name: str) -> float:
    return min(1.0, max(0.0, components.get(name, 0.0)))


def enrich_candidates(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    pointwise_scores: dict[str, tuple[float, str]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[float, float], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: number(item.get("rank"), 9999)):
        unique.setdefault(interval_key(row), dict(row))
    candidates = list(unique.values())
    score_norm = normalized(
        [number(row.get("rerank_score")) for row in candidates]
    )
    completeness_weights = config["completeness"]
    proxy_weights = config["structured_proxy"]
    thumbnail_weights = proxy_weights["thumbnailability_proxy_formula"]
    hierarchy_weights = config["hierarchical_score"]

    for row, deterministic_norm in zip(candidates, score_norm):
        start, end = interval_key(row)
        longform_id = str(row["long_video_id"])
        candidate_id = stable_candidate_id(longform_id, start, end)
        components = parse_components(row)
        change = component(components, "signal")
        titleability = component(components, "titleability")
        thumbnailability = (
            thumbnail_weights["titleability"] * titleability
            + thumbnail_weights["content_volume"]
            * component(components, "content_volume")
            + thumbnail_weights["speech_density"]
            * component(components, "speech_density")
        )
        structured_proxy = (
            proxy_weights["change_or_surprise"] * change
            + proxy_weights["titleability"] * titleability
            + proxy_weights["thumbnailability_proxy"] * thumbnailability
        )
        completeness = (
            completeness_weights["speech_boundary"]
            * component(components, "speech_boundary")
            + completeness_weights["duration"] * component(components, "duration")
            + completeness_weights["speech_density"]
            * component(components, "speech_density")
            + completeness_weights["content_volume"]
            * component(components, "content_volume")
            + completeness_weights["inverse_filler_penalty"]
            * (1.0 - component(components, "filler_penalty"))
        )
        pointwise_record = pointwise_scores.get(candidate_id)
        pointwise = pointwise_record[0] if pointwise_record is not None else None
        success_score = pointwise if pointwise is not None else structured_proxy
        hierarchical = (
            hierarchy_weights["success_score"] * success_score
            + hierarchy_weights["completeness_score"] * completeness
        )
        row.update(
            {
                "candidate_id": candidate_id,
                "pred_start_sec": start,
                "pred_end_sec": end,
                "duration_sec": end - start,
                "deterministic_score_norm": deterministic_norm,
                "change_proxy": change,
                "titleability_proxy": titleability,
                "thumbnailability_proxy": thumbnailability,
                "structured_success_proxy": structured_proxy,
                "completeness_score": completeness,
                "pointwise_score": "" if pointwise is None else pointwise,
                "success_score": success_score,
                "hierarchical_score": hierarchical,
                "score_source": (
                    pointwise_record[1]
                    if pointwise_record is not None
                    else "structured_no_api_proxy"
                ),
            }
        )
    return candidates


def infer_bin_count(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> int:
    timeline = config["timeline"]
    duration_min = (
        max(number(row["pred_end_sec"]) for row in candidates) / 60.0
        if candidates
        else 0.0
    )
    inferred = math.ceil(duration_min / timeline["minutes_per_bin"])
    return max(
        timeline["minimum_bins"],
        min(timeline["maximum_bins"], inferred),
    )


def assign_bins(
    candidates: list[dict[str, Any]], bin_count: int
) -> dict[int, list[dict[str, Any]]]:
    video_end = max(number(row["pred_end_sec"]) for row in candidates)
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        position = candidate_center(row) / max(video_end, 1.0)
        index = min(bin_count - 1, max(0, int(position * bin_count)))
        row["timeline_bin"] = index + 1
        row["timeline_bin_count"] = bin_count
        output[index].append(row)
    return output


def multislate_union(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    bin_count = infer_bin_count(candidates, config)
    grouped = assign_bins(candidates, bin_count)
    per_bin = int(config["timeline"]["candidates_per_bin"])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bin_index in range(bin_count):
        bucket = grouped.get(bin_index, [])
        picks: list[dict[str, Any]] = []
        for field in (
            "success_score",
            "completeness_score",
            "deterministic_score_norm",
        ):
            for candidate in sorted(
                bucket,
                key=lambda item: (
                    number(item[field]),
                    number(item["hierarchical_score"]),
                ),
                reverse=True,
            ):
                if candidate["candidate_id"] not in {
                    item["candidate_id"] for item in picks
                }:
                    picks.append(candidate)
                    break
        for candidate in sorted(
            bucket,
            key=lambda item: number(item["hierarchical_score"]),
            reverse=True,
        ):
            if len(picks) >= per_bin:
                break
            if candidate["candidate_id"] not in {
                item["candidate_id"] for item in picks
            }:
                picks.append(candidate)
        for candidate in picks[:per_bin]:
            if candidate["candidate_id"] not in seen:
                selected.append(candidate)
                seen.add(candidate["candidate_id"])
    return selected


def local_shortlist(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[int(row["timeline_bin"])].append(row)
    keep = int(config["timeline"]["local_keep_per_bin"])
    return [
        candidate
        for bin_index in sorted(grouped)
        for candidate in sorted(
            grouped[bin_index],
            key=lambda item: number(item["hierarchical_score"]),
            reverse=True,
        )[:keep]
    ]


def mmr_select(
    candidates: list[dict[str, Any]],
    top_k: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    mmr = config["mmr"]
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    video_end = max(
        [number(row["pred_end_sec"]) for row in candidates] or [1.0]
    )
    while remaining and len(selected) < top_k:
        best: dict[str, Any] | None = None
        best_value = -math.inf
        for candidate in remaining:
            if selected:
                maximum_overlap = max(
                    overlap_ratio(candidate, chosen) for chosen in selected
                )
                minimum_distance = min(
                    abs(candidate_center(candidate) - candidate_center(chosen))
                    for chosen in selected
                )
                diversity = min(1.0, minimum_distance / (video_end * 0.2))
            else:
                maximum_overlap = 0.0
                diversity = 1.0
            if maximum_overlap > mmr["maximum_overlap"]:
                continue
            value = (
                number(candidate["hierarchical_score"])
                - mmr["overlap_penalty"] * maximum_overlap
                + mmr["timeline_diversity_bonus"] * diversity
            )
            if value > best_value:
                best = candidate
                best_value = value
        if best is None:
            break
        selected.append(best)
        remaining = [
            candidate
            for candidate in remaining
            if candidate["candidate_id"] != best["candidate_id"]
        ]
    if len(selected) < top_k:
        for candidate in sorted(
            remaining,
            key=lambda item: number(item["hierarchical_score"]),
            reverse=True,
        ):
            selected.append(candidate)
            if len(selected) >= top_k:
                break
    return selected


def prediction_rows(
    selected: list[dict[str, Any]],
    gold_rows: list[dict[str, str]],
    variant: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pair in gold_rows:
        for rank, candidate in enumerate(selected, start=1):
            output.append(
                {
                    "pair_id": pair["pair_id"],
                    "long_video_id": pair["long_video_id"],
                    "short_video_id": pair.get("short_video_id", ""),
                    "run_id": variant,
                    "selector_type": variant,
                    "prompt_id": "hierarchical_multislate_v1",
                    "model_name": candidate["score_source"],
                    "rank": rank,
                    "pred_start_sec": candidate["pred_start_sec"],
                    "pred_end_sec": candidate["pred_end_sec"],
                    "selected_scene_ids": candidate.get(
                        "selected_scene_ids", ""
                    ),
                    "confidence": round(
                        number(candidate["hierarchical_score"]), 6
                    ),
                    "notes": (
                        f"candidate_id={candidate['candidate_id']};"
                        f"bin={candidate['timeline_bin']}/"
                        f"{candidate['timeline_bin_count']}"
                    ),
                }
            )
    return output


def parse_pointwise_scores(
    path: Path | None, field: str
) -> dict[str, tuple[float, str]]:
    if path is None:
        return {}
    output: dict[str, tuple[float, str]] = {}
    for row in read_csv(path):
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        score = number(row.get(field), math.nan)
        if math.isfinite(score):
            output[candidate_id] = (
                score,
                str(row.get("score_source") or "external_pointwise_judge"),
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build hierarchical multi-slate reranking variants."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "hierarchical_multislate_v1.json",
    )
    parser.add_argument("--pointwise-scores", type=Path, default=None)
    parser.add_argument("--pointwise-score-field", default="judge_score_0_1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-pad-sec", type=float, default=20.0)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = read_csv(args.dataset)
    predictions = read_csv(args.predictions)
    gold_by_longform: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in dataset:
        gold_by_longform[row["long_video_id"]].append(row)
    predictions_by_longform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        predictions_by_longform[row["long_video_id"]].append(row)
    pointwise = parse_pointwise_scores(
        args.pointwise_scores, args.pointwise_score_field
    )

    variants: dict[str, list[dict[str, Any]]] = {
        "b1_deterministic_top5": [],
        "b2_adaptive_coverage": [],
        "b3_hierarchical_multislate": [],
        "b4_hierarchical_mmr": [],
    }
    candidate_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    longform_summaries: list[dict[str, Any]] = []

    for longform_id, long_rows in sorted(predictions_by_longform.items()):
        if longform_id not in gold_by_longform:
            continue
        enriched = enrich_candidates(long_rows, config, pointwise)
        union = multislate_union(enriched, config)
        shortlist = local_shortlist(union, config)
        b1 = sorted(
            enriched,
            key=lambda item: number(item.get("rank"), 9999),
        )[: args.top_k]
        b2 = select_adaptive_coverage(
            enriched, args.top_k, coverage_per_bin=1
        )
        b3 = sorted(
            shortlist,
            key=lambda item: number(item["hierarchical_score"]),
            reverse=True,
        )[: args.top_k]
        b4 = mmr_select(shortlist, args.top_k, config)
        selections = {
            "b1_deterministic_top5": b1,
            "b2_adaptive_coverage": b2,
            "b3_hierarchical_multislate": b3,
            "b4_hierarchical_mmr": b4,
        }
        for variant, selected in selections.items():
            variants[variant].extend(
                prediction_rows(
                    selected, gold_by_longform[longform_id], variant
                )
            )

        scenes = load_scenes(longform_id, args.raw_dir)
        union_ids = {candidate["candidate_id"] for candidate in union}
        for candidate in enriched:
            description, transcript, context, context_start, context_end = (
                candidate_text(candidate, scenes, args.context_pad_sec)
            )
            candidate_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "longform_id": longform_id,
                    "start_sec": candidate["pred_start_sec"],
                    "end_sec": candidate["pred_end_sec"],
                    "duration_sec": candidate["duration_sec"],
                    "timeline_bin": candidate["timeline_bin"],
                    "timeline_bin_count": candidate["timeline_bin_count"],
                    "structured_success_proxy": round(
                        number(candidate["structured_success_proxy"]), 6
                    ),
                    "completeness_score": round(
                        number(candidate["completeness_score"]), 6
                    ),
                    "hierarchical_score": round(
                        number(candidate["hierarchical_score"]), 6
                    ),
                    "score_source": candidate["score_source"],
                    "in_multislate_union": (
                        candidate["candidate_id"] in union_ids
                    ),
                    "description": description,
                    "transcript": transcript,
                    "context_transcript": context,
                }
            )
            if candidate["candidate_id"] in union_ids:
                packet_rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "longform_id": longform_id,
                        "start_ms": int(
                            number(candidate["pred_start_sec"]) * 1000
                        ),
                        "end_ms": int(
                            number(candidate["pred_end_sec"]) * 1000
                        ),
                        "description": description,
                        "transcript": transcript,
                        "context_before_after": context,
                        "requested_output": {
                            "change_or_surprise_0_4": "integer",
                            "generated_title": "string",
                            "title_packaging_0_4": "integer",
                            "thumbnailability_proxy_0_4": "integer",
                            "judge_score_0_1": (
                                "(0.40*change + 0.15*title + "
                                "0.45*thumbnailability_proxy)/4"
                            ),
                        },
                    }
                )
        longform_summaries.append(
            {
                "longform_id": longform_id,
                "source_candidate_count": len(enriched),
                "bin_count": infer_bin_count(enriched, config),
                "multislate_union_count": len(union),
                "local_shortlist_count": len(shortlist),
                "score_source": enriched[0]["score_source"] if enriched else "",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, rows in variants.items():
        write_csv(args.output_dir / f"{variant}.csv", rows)
    write_csv(args.output_dir / "candidate_pool.csv", candidate_rows)
    write_jsonl(args.output_dir / "pointwise_judge_packet.jsonl", packet_rows)
    write_csv(args.output_dir / "longform_summary.csv", longform_summaries)
    summary = {
        "pipeline_id": config["pipeline_id"],
        "longform_count": len(longform_summaries),
        "candidate_pool_count": len(candidate_rows),
        "pointwise_packet_count": len(packet_rows),
        "pointwise_score_count": len(pointwise),
        "score_source": (
            sorted({row["score_source"] for row in candidate_rows})
            if candidate_rows
            else []
        ),
        "variants": {
            name: len(rows) for name, rows in variants.items()
        },
        "warning": config["interpretation"]["structured_proxy"],
    }
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
