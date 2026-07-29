from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def to_int(value: Any, default: int = 1) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return int(float(text))


def first_float(
    row: dict[str, Any],
    fields: tuple[str, ...],
) -> float | None:
    for field in fields:
        value = to_float(row.get(field))
        if value is not None:
            return value
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_ranked_predictions(
    predictions: list[dict[str, str]],
    interval_decimals: int = 3,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Keep the first occurrence of each ranked interval and refill from later ranks."""
    group_fields = (
        "run_id",
        "selector_type",
        "prompt_id",
        "model_name",
        "pair_id",
    )
    grouped: dict[
        tuple[str, ...],
        list[tuple[int, dict[str, str]]],
    ] = defaultdict(list)
    for row_index, row in enumerate(predictions):
        key = tuple(str(row.get(field, "")) for field in group_fields)
        grouped[key].append((row_index, row))

    deduplicated: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    for key, indexed_rows in grouped.items():
        ordered = sorted(
            indexed_rows,
            key=lambda item: (to_int(item[1].get("rank"), 1), item[0]),
        )
        seen: set[tuple[float, float]] = set()
        retained: list[dict[str, str]] = []
        original_top5: set[tuple[float, float]] = set()
        for _, row in ordered:
            start = to_float(row.get("pred_start_sec"))
            end = to_float(row.get("pred_end_sec"))
            if start is None or end is None:
                raise ValueError(
                    f"Missing predicted segment for pair_id={row.get('pair_id')}"
                )
            interval = (
                round(start, interval_decimals),
                round(end, interval_decimals),
            )
            if to_int(row.get("rank"), 1) <= 5:
                original_top5.add(interval)
            if interval in seen:
                continue
            seen.add(interval)
            retained.append(dict(row))

        for new_rank, row in enumerate(retained, start=1):
            row["rank"] = str(new_rank)
            deduplicated.append(row)

        audit_rows.append(
            {
                **dict(zip(group_fields, key)),
                "original_candidate_count": len(ordered),
                "unique_candidate_count": len(retained),
                "duplicate_rows_removed": len(ordered) - len(retained),
                "original_top5_unique_count": len(original_top5),
                "final_top5_unique_count": min(5, len(retained)),
                "top5_underfilled": len(retained) < 5,
            }
        )
    return deduplicated, audit_rows


def overlap_seconds(pred_start: float, pred_end: float, gold_start: float, gold_end: float) -> float:
    return max(0.0, min(pred_end, gold_end) - max(pred_start, gold_start))


def score_prediction(
    gold: dict[str, str],
    pred: dict[str, str],
    core_coverage_threshold: float,
    tight_iou_threshold: float,
    start_tolerance_sec: float,
) -> dict[str, Any]:
    gold_start = first_float(
        gold,
        ("gold_start_sec", "start_sec", "gold_span_start_sec"),
    )
    gold_end = first_float(
        gold,
        ("gold_end_sec", "end_sec", "gold_span_end_sec"),
    )
    pred_start = to_float(pred.get("pred_start_sec"))
    pred_end = to_float(pred.get("pred_end_sec"))
    if gold_start is None or gold_end is None:
        raise ValueError(f"Missing gold segment for pair_id={gold.get('pair_id')}")
    if pred_start is None or pred_end is None:
        raise ValueError(f"Missing predicted segment for pair_id={pred.get('pair_id')}")

    gold_duration = max(0.0, gold_end - gold_start)
    pred_duration = max(0.0, pred_end - pred_start)
    overlap = overlap_seconds(pred_start, pred_end, gold_start, gold_end)
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    coverage = overlap / gold_duration if gold_duration > 0 else 0.0
    iou = overlap / union if union > 0 else 0.0
    start_error = abs(pred_start - gold_start)
    end_error = abs(pred_end - gold_end)
    center_error = abs(((pred_start + pred_end) / 2) - ((gold_start + gold_end) / 2))
    length_score = min(pred_duration, gold_duration) / max(pred_duration, gold_duration) if max(pred_duration, gold_duration) > 0 else 0.0
    start_score = max(0.0, 1.0 - (start_error / start_tolerance_sec)) if start_tolerance_sec > 0 else 0.0
    min_overlap_for_core = min(5.0, gold_duration * 0.5)
    core_hit = coverage >= core_coverage_threshold and overlap >= min_overlap_for_core
    tight_hit = core_hit and iou >= tight_iou_threshold
    final_score = 100.0 * ((0.45 * coverage) + (0.30 * iou) + (0.15 * start_score) + (0.10 * length_score))

    return {
        "pair_id": pred["pair_id"],
        "long_video_id": pred.get("long_video_id") or gold.get("long_video_id", ""),
        "short_video_id": pred.get("short_video_id") or gold.get("short_video_id", ""),
        "run_id": pred.get("run_id", ""),
        "selector_type": pred.get("selector_type", ""),
        "prompt_id": pred.get("prompt_id", ""),
        "model_name": pred.get("model_name", ""),
        "rank": to_int(pred.get("rank"), 1),
        "pred_start_sec": pred_start,
        "pred_end_sec": pred_end,
        "pred_duration_sec": pred_duration,
        "gold_start_sec": gold_start,
        "gold_end_sec": gold_end,
        "gold_duration_sec": gold_duration,
        "overlap_sec": overlap,
        "gold_coverage": coverage,
        "temporal_iou": iou,
        "start_error_sec": start_error,
        "end_error_sec": end_error,
        "center_error_sec": center_error,
        "length_score": length_score,
        "start_score": start_score,
        "core_hit": core_hit,
        "tight_hit": tight_hit,
        "final_score": final_score,
        "selected_scene_ids": pred.get("selected_scene_ids", ""),
        "confidence": pred.get("confidence", ""),
        "notes": pred.get("notes", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["run_id"]),
            str(row["selector_type"]),
            str(row["prompt_id"]),
            str(row["model_name"]),
        )
        groups[key].append(row)

    summaries = []
    for (run_id, selector_type, prompt_id, model_name), group in sorted(groups.items()):
        top1 = [r for r in group if int(r["rank"]) == 1]
        if not top1:
            top1 = group
        top3_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            if int(row["rank"]) <= 3:
                top3_by_pair[str(row["pair_id"])].append(row)
        hit_at_3_core = []
        hit_at_3_tight = []
        hit_at_5_core = []
        hit_at_5_tight = []
        best_iou_at_3 = []
        best_iou_at_5 = []
        for pair_rows in top3_by_pair.values():
            hit_at_3_core.append(any(bool(r["core_hit"]) for r in pair_rows))
            hit_at_3_tight.append(any(bool(r["tight_hit"]) for r in pair_rows))
            best_iou_at_3.append(max(float(r["temporal_iou"]) for r in pair_rows))
        top5_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            if int(row["rank"]) <= 5:
                top5_by_pair[str(row["pair_id"])].append(row)
        for pair_rows in top5_by_pair.values():
            hit_at_5_core.append(any(bool(r["core_hit"]) for r in pair_rows))
            hit_at_5_tight.append(any(bool(r["tight_hit"]) for r in pair_rows))
            best_iou_at_5.append(max(float(r["temporal_iou"]) for r in pair_rows))

        summaries.append(
            {
                "run_id": run_id,
                "selector_type": selector_type,
                "prompt_id": prompt_id,
                "model_name": model_name,
                "pair_count": len({str(r["pair_id"]) for r in group}),
                "prediction_count": len(group),
                "top1_core_hit_rate": mean([bool(r["core_hit"]) for r in top1]),
                "top1_tight_hit_rate": mean([bool(r["tight_hit"]) for r in top1]),
                "top1_mean_coverage": mean([float(r["gold_coverage"]) for r in top1]),
                "top1_mean_iou": mean([float(r["temporal_iou"]) for r in top1]),
                "top1_mean_final_score": mean([float(r["final_score"]) for r in top1]),
                "hit_at_3_core_rate": mean(hit_at_3_core),
                "hit_at_3_tight_rate": mean(hit_at_3_tight),
                "best_iou_at_3_mean": mean(best_iou_at_3),
                "hit_at_5_core_rate": mean(hit_at_5_core),
                "hit_at_5_tight_rate": mean(hit_at_5_tight),
                "best_iou_at_5_mean": mean(best_iou_at_5),
            }
        )

    return {"runs": summaries, "long_videos": summarize_long_level(rows)}


def summarize_long_level(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("long_video_id") or "unknown"),
            str(row["run_id"]),
            str(row["selector_type"]),
            str(row["prompt_id"]),
            str(row["model_name"]),
        )
        groups[key].append(row)

    summaries = []
    for (long_video_id, run_id, selector_type, prompt_id, model_name), group in sorted(groups.items()):
        pair_ids = sorted({str(row["pair_id"]) for row in group})
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_pair[str(row["pair_id"])].append(row)

        def recall_at(k: int, field: str) -> float:
            hits = []
            for pair_id in pair_ids:
                ranked = [row for row in by_pair[pair_id] if int(row["rank"]) <= k]
                hits.append(any(bool(row[field]) for row in ranked))
            return mean(hits)

        def mean_best_iou_at(k: int) -> float:
            values = []
            for pair_id in pair_ids:
                ranked = [row for row in by_pair[pair_id] if int(row["rank"]) <= k]
                values.append(max([float(row["temporal_iou"]) for row in ranked] or [0.0]))
            return mean(values)

        summaries.append(
            {
                "long_video_id": long_video_id,
                "run_id": run_id,
                "selector_type": selector_type,
                "prompt_id": prompt_id,
                "model_name": model_name,
                "gold_pair_count": len(pair_ids),
                "core_recall_at_1": recall_at(1, "core_hit"),
                "core_recall_at_3": recall_at(3, "core_hit"),
                "core_recall_at_5": recall_at(5, "core_hit"),
                "core_recall_at_10": recall_at(10, "core_hit"),
                "core_recall_at_12": recall_at(12, "core_hit"),
                "core_recall_at_20": recall_at(20, "core_hit"),
                "core_recall_at_30": recall_at(30, "core_hit"),
                "core_recall_at_50": recall_at(50, "core_hit"),
                "core_recall_at_60": recall_at(60, "core_hit"),
                "core_recall_at_120": recall_at(120, "core_hit"),
                "tight_recall_at_3": recall_at(3, "tight_hit"),
                "tight_recall_at_5": recall_at(5, "tight_hit"),
                "tight_recall_at_10": recall_at(10, "tight_hit"),
                "tight_recall_at_12": recall_at(12, "tight_hit"),
                "tight_recall_at_20": recall_at(20, "tight_hit"),
                "tight_recall_at_30": recall_at(30, "tight_hit"),
                "tight_recall_at_50": recall_at(50, "tight_hit"),
                "tight_recall_at_60": recall_at(60, "tight_hit"),
                "tight_recall_at_120": recall_at(120, "tight_hit"),
                "mean_best_iou_at_3": mean_best_iou_at(3),
                "mean_best_iou_at_5": mean_best_iou_at(5),
                "mean_best_iou_at_10": mean_best_iou_at(10),
                "mean_best_iou_at_12": mean_best_iou_at(12),
                "mean_best_iou_at_20": mean_best_iou_at(20),
                "mean_best_iou_at_30": mean_best_iou_at(30),
                "mean_best_iou_at_50": mean_best_iou_at(50),
                "mean_best_iou_at_60": mean_best_iou_at(60),
                "mean_best_iou_at_120": mean_best_iou_at(120),
            }
        )
    return summaries


def mean(values: list[Any]) -> float:
    if not values:
        return 0.0
    return sum(1.0 if v is True else 0.0 if v is False else float(v) for v in values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate highlight predictions against long-short gold segments.")
    parser.add_argument("--dataset", required=True, help="CSV with gold long-short pairs.")
    parser.add_argument("--predictions", required=True, help="CSV with predicted segments.")
    parser.add_argument("--out-dir", default="data/processed/evaluation")
    parser.add_argument("--core-coverage-threshold", type=float, default=0.70)
    parser.add_argument("--tight-iou-threshold", type=float, default=0.30)
    parser.add_argument("--start-tolerance-sec", type=float, default=10.0)
    parser.add_argument(
        "--keep-duplicate-intervals",
        action="store_true",
        help=(
            "Preserve duplicate time intervals for historical reproduction. "
            "By default, exact duplicate intervals are removed and ranks are refilled."
        ),
    )
    args = parser.parse_args()

    dataset = {row["pair_id"]: row for row in read_csv(Path(args.dataset))}
    predictions = read_csv(Path(args.predictions))
    if args.keep_duplicate_intervals:
        duplicate_audit: list[dict[str, Any]] = []
    else:
        predictions, duplicate_audit = deduplicate_ranked_predictions(predictions)
    metric_rows = []
    for pred in predictions:
        pair_id = pred["pair_id"]
        if pair_id not in dataset:
            raise KeyError(f"Prediction references unknown pair_id={pair_id}")
        metric_rows.append(
            score_prediction(
                dataset[pair_id],
                pred,
                core_coverage_threshold=args.core_coverage_threshold,
                tight_iou_threshold=args.tight_iou_threshold,
                start_tolerance_sec=args.start_tolerance_sec,
            )
        )

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "metrics.csv", metric_rows)
    summary = summarize(metric_rows)
    if duplicate_audit:
        write_csv(out_dir / "duplicate_interval_audit.csv", duplicate_audit)
        summary["prediction_audit"] = {
            "duplicate_policy": "deduplicate_exact_interval_and_refill_rank",
            "duplicate_rows_removed": sum(
                int(row["duplicate_rows_removed"]) for row in duplicate_audit
            ),
            "groups_with_duplicates": sum(
                int(row["duplicate_rows_removed"]) > 0 for row in duplicate_audit
            ),
            "groups_with_fewer_than_5_unique_candidates": sum(
                bool(row["top5_underfilled"]) for row in duplicate_audit
            ),
            "audit_csv": str(out_dir / "duplicate_interval_audit.csv"),
        }
    else:
        summary["prediction_audit"] = {
            "duplicate_policy": "preserve_input_rows",
            "duplicate_rows_removed": 0,
        }
    write_csv(out_dir / "long_recall.csv", summary["long_videos"])
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"metrics_csv": str(out_dir / "metrics.csv"), "summary_json": str(out_dir / "summary.json"), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
