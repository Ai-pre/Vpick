from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "source_importance",
    "standalone_completeness",
    "boundary_quality",
    "engagement",
)
EQUAL_WEIGHTS = tuple(0.25 for _ in DIMENSIONS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weight_grid(step: float, minimum: float) -> list[tuple[float, ...]]:
    units = round(1.0 / step)
    min_units = math.ceil(minimum / step - 1e-9)
    combinations: list[tuple[float, ...]] = []
    for values in itertools.product(
        range(min_units, units + 1),
        repeat=len(DIMENSIONS),
    ):
        if sum(values) != units:
            continue
        combinations.append(tuple(round(value * step, 10) for value in values))
    if not combinations:
        raise ValueError("The step and minimum produce no valid weight vectors")
    return combinations


def composite(row: dict[str, Any], weights: tuple[float, ...]) -> float:
    return sum(
        float(row[f"{dimension}_100"]) * weight
        for dimension, weight in zip(DIMENSIONS, weights)
    )


def pair_metrics(
    rows: list[dict[str, Any]],
    weights: tuple[float, ...],
) -> dict[str, float | int]:
    by_channel: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_channel[row["channel_name"]][row["performance_label"]].append(
            composite(row, weights)
        )
    correct = tied = wrong = 0
    for channel, labels in by_channel.items():
        if not labels["pos"] or not labels["neg"]:
            raise ValueError(f"Channel lacks a pos/neg comparison: {channel}")
        for positive in labels["pos"]:
            for negative in labels["neg"]:
                if positive > negative:
                    correct += 1
                elif positive == negative:
                    tied += 1
                else:
                    wrong += 1
    count = correct + tied + wrong
    return {
        "pair_count": count,
        "correct": correct,
        "tied": tied,
        "wrong": wrong,
        "strict_accuracy": correct / count if count else 0.0,
        "half_credit": (correct + 0.5 * tied) / count if count else 0.0,
    }


def equal_distance(weights: tuple[float, ...]) -> float:
    return sum((weight - 0.25) ** 2 for weight in weights)


def weight_key(weights: tuple[float, ...]) -> str:
    return "|".join(f"{weight:.4f}" for weight in weights)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--minimum-weight", type=float, default=0.1)
    args = parser.parse_args()

    score_rows = read_csv(Path(args.scores))
    manifest_rows = read_csv(Path(args.manifest))
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Manifest contains duplicate candidate IDs")
    if len({row["candidate_id"] for row in score_rows}) != len(score_rows):
        raise ValueError("Score file contains duplicate candidate IDs")
    if set(row["candidate_id"] for row in score_rows) != set(manifest):
        raise ValueError("Score and manifest candidate ID sets must match exactly")

    rows: list[dict[str, Any]] = []
    for score_row in score_rows:
        metadata = manifest[score_row["candidate_id"]]
        rows.append(
            {
                **score_row,
                "channel_name": metadata["channel_name"],
                "performance_label": metadata["performance_label"],
            }
        )
    channels = sorted({row["channel_name"] for row in rows})
    if len(channels) < 3:
        raise ValueError("At least three channels are required for grouped CV")

    grid = weight_grid(args.step, args.minimum_weight)
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    selected_lookup: dict[str, tuple[float, ...]] = {}

    for held_out_channel in channels:
        train = [
            row for row in rows if row["channel_name"] != held_out_channel
        ]
        held_out = [
            row for row in rows if row["channel_name"] == held_out_channel
        ]
        candidates: list[
            tuple[float, float, float, tuple[float, ...], dict[str, Any]]
        ] = []
        for weights in grid:
            metrics = pair_metrics(train, weights)
            candidates.append(
                (
                    float(metrics["half_credit"]),
                    float(metrics["strict_accuracy"]),
                    -equal_distance(weights),
                    weights,
                    metrics,
                )
            )
        candidates.sort(
            key=lambda item: (item[0], item[1], item[2], item[3]),
            reverse=True,
        )
        _, _, _, selected_weights, train_metrics = candidates[0]
        held_out_metrics = pair_metrics(held_out, selected_weights)
        key = weight_key(selected_weights)
        selected_counts[key] += 1
        selected_lookup[key] = selected_weights
        fold_rows.append(
            {
                "held_out_channel": held_out_channel,
                **{
                    f"weight_{dimension}": weight
                    for dimension, weight in zip(
                        DIMENSIONS,
                        selected_weights,
                    )
                },
                "train_pair_count": train_metrics["pair_count"],
                "train_half_credit": round(
                    float(train_metrics["half_credit"]),
                    4,
                ),
                "held_out_pair_count": held_out_metrics["pair_count"],
                "held_out_correct": held_out_metrics["correct"],
                "held_out_tied": held_out_metrics["tied"],
                "held_out_wrong": held_out_metrics["wrong"],
                "held_out_half_credit": round(
                    float(held_out_metrics["half_credit"]),
                    4,
                ),
            }
        )
        for row in held_out:
            oof_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "channel_name": held_out_channel,
                    "performance_label_PRIVATE": row["performance_label"],
                    "oof_quality_score_100": round(
                        composite(row, selected_weights),
                        4,
                    ),
                    "fold_weight_key": key,
                }
            )

    most_frequent = max(selected_counts.values())
    finalists = [
        selected_lookup[key]
        for key, count in selected_counts.items()
        if count == most_frequent
    ]
    locked_weights = min(finalists, key=lambda weights: (equal_distance(weights), weights))
    equal_metrics = pair_metrics(rows, EQUAL_WEIGHTS)
    locked_metrics = pair_metrics(rows, locked_weights)
    oof_correct = sum(int(row["held_out_correct"]) for row in fold_rows)
    oof_tied = sum(int(row["held_out_tied"]) for row in fold_rows)
    oof_wrong = sum(int(row["held_out_wrong"]) for row in fold_rows)
    oof_count = oof_correct + oof_tied + oof_wrong

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "grouped_cv_fold_weights_PRIVATE.csv", fold_rows)
    write_csv(out_dir / "grouped_cv_oof_scores_PRIVATE.csv", oof_rows)
    summary = {
        "candidate_count": len(rows),
        "channel_count": len(channels),
        "grid_size": len(grid),
        "step": args.step,
        "minimum_weight": args.minimum_weight,
        "weight_selection_rule": (
            "maximize training half-credit, then strict accuracy, then prefer "
            "the vector closest to equal weights"
        ),
        "fold_weight_frequency": dict(selected_counts),
        "locked_weights": {
            dimension: weight
            for dimension, weight in zip(DIMENSIONS, locked_weights)
        },
        "equal_weight_dev_metrics": equal_metrics,
        "locked_weight_apparent_dev_metrics": locked_metrics,
        "grouped_oof_metrics": {
            "pair_count": oof_count,
            "correct": oof_correct,
            "tied": oof_tied,
            "wrong": oof_wrong,
            "strict_accuracy": round(oof_correct / max(1, oof_count), 4),
            "half_credit": round(
                (oof_correct + 0.5 * oof_tied) / max(1, oof_count),
                4,
            ),
        },
        "warning": (
            "The development set contains only one pos/neg pair per channel. "
            "Weights remain provisional and must be frozen before locked-test "
            "evaluation."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weight_tuning_summary_PRIVATE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
