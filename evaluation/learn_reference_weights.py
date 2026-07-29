from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import ROOT, as_float, read_csv, rounded, spearman, write_csv, write_json


DEFAULT_SCORES = (
    ROOT / "results" / "evaluation_system_v1" / "reference_v7_codex_vpick_direct_scores.csv"
)
DEFAULT_TARGETS = (
    ROOT / "results" / "evaluation_system_v1" / "prepared" / "targets_private.csv"
)
DEFAULT_BEHAVIOR = (
    ROOT / "results" / "evaluation_system_v1" / "behavior_labels_private.csv"
)
DEFAULT_OUTPUT = ROOT / "results" / "evaluation_system_v1" / "weight_learning"

CHECK_FEATURES = (
    "hook",
    "surprise",
    "emotion",
    "quotable",
    "payoff",
    "natural_start",
    "natural_end",
)
SOURCE_FIELDS = {
    "hook": "check_hook_within_3s",
    "surprise": "check_surprise_or_twist",
    "emotion": "check_emotional_peak",
    "quotable": "check_quotable_moment",
    "payoff": "check_payoff_or_conclusion",
    "natural_start": "check_natural_start",
    "natural_end": "check_natural_end",
    "saliency": "saliency_market_1_5",
}
LAMBDAS = (0.0, 0.02, 0.1)


def _compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def weight_grid(parts: int) -> np.ndarray:
    return np.asarray(list(_compositions(10, parts)), dtype=float) / 10.0


def _column_spearman(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction_ranks = (
        pd.DataFrame(predictions).rank(axis=0, method="average").to_numpy(copy=True)
    )
    target_ranks = pd.Series(target).rank(method="average").to_numpy(copy=True)
    prediction_ranks -= prediction_ranks.mean(axis=0, keepdims=True)
    target_ranks -= target_ranks.mean()
    numerator = target_ranks @ prediction_ranks
    denominator = np.sqrt(
        np.sum(target_ranks**2) * np.sum(prediction_ranks**2, axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def _macro_train_objective(
    x: np.ndarray,
    target: np.ndarray,
    channels: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    predictions = x @ weights.T
    channel_values: list[np.ndarray] = []
    for channel in sorted(set(channels.tolist())):
        mask = channels == channel
        if int(mask.sum()) < 2:
            continue
        channel_values.append(_column_spearman(predictions[mask], target[mask]))
    return np.nanmean(np.vstack(channel_values), axis=0)


def _metrics(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    scores = [float(row[score_field]) for row in rows]
    target = [float(row["channel_view_percentile"]) for row in rows]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["channel_name"])].append(row)
    channel_rows: list[dict[str, Any]] = []
    centered_scores: list[float] = []
    centered_target: list[float] = []
    for channel, group in sorted(grouped.items()):
        group_scores = [float(row[score_field]) for row in group]
        group_target = [float(row["channel_view_percentile"]) for row in group]
        channel_rows.append(
            {
                "channel_name": channel,
                "n": len(group),
                "spearman": rounded(spearman(group_scores, group_target)),
            }
        )
        score_mean = statistics.mean(group_scores)
        target_mean = statistics.mean(group_target)
        centered_scores.extend(value - score_mean for value in group_scores)
        centered_target.extend(value - target_mean for value in group_target)
    valid = [
        float(row["spearman"])
        for row in channel_rows
        if row["spearman"] is not None
    ]
    return {
        "n": len(rows),
        "overall_spearman": rounded(spearman(scores, target)),
        "within_channel_centered_spearman": rounded(
            spearman(centered_scores, centered_target)
        ),
        "channel_macro_spearman": rounded(statistics.mean(valid) if valid else None),
        "positive_channel_count": sum(
            float(row["spearman"]) > 0
            for row in channel_rows
            if row["spearman"] is not None
        ),
        "valid_channel_count": len(valid),
        "channel_metrics": channel_rows,
    }


def _load_records(
    scores_path: Path,
    targets_path: Path,
    behavior_path: Path,
) -> list[dict[str, Any]]:
    targets = read_csv(targets_path)
    behavior_by_current = {
        row["candidate_id"]: row for row in read_csv(behavior_path)
    }
    current_by_source = {
        row["source_candidate_id"]: row["candidate_id"] for row in targets
    }
    records: list[dict[str, Any]] = []
    for row in read_csv(scores_path):
        if row.get("verdict") != "score":
            continue
        current_id = current_by_source.get(str(row.get("candidate_id", "")))
        behavior = behavior_by_current.get(str(current_id or ""))
        if not behavior:
            continue
        values = {
            feature: as_float(row.get(field))
            for feature, field in SOURCE_FIELDS.items()
        }
        percentile = as_float(behavior.get("channel_view_percentile"))
        if percentile is None or any(value is None for value in values.values()):
            continue
        records.append(
            {
                "candidate_id": current_id,
                "source_candidate_id": row["candidate_id"],
                "longform_id": behavior["longform_id"],
                "channel_name": behavior["channel_name"],
                "channel_view_percentile": percentile,
                **{
                    feature: (
                        (float(value) - 1.0) / 4.0
                        if feature == "saliency"
                        else float(value) / 2.0
                    )
                    for feature, value in values.items()
                },
            }
        )
    return records


def learn(
    scores_path: Path,
    targets_path: Path,
    behavior_path: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    records = _load_records(scores_path, targets_path, behavior_path)
    channels = np.asarray([row["channel_name"] for row in records], dtype=object)
    target = np.asarray(
        [float(row["channel_view_percentile"]) for row in records], dtype=float
    )
    feature_sets = {
        "checklist7": CHECK_FEATURES,
        "saliency_plus_checklist7": ("saliency", *CHECK_FEATURES),
    }
    fold_weight_rows: list[dict[str, Any]] = []
    config_predictions: dict[str, np.ndarray] = {}
    final_weights_by_config: dict[str, dict[str, float]] = {}

    for feature_set_name, features in feature_sets.items():
        x = np.asarray(
            [[float(row[feature]) for feature in features] for row in records],
            dtype=float,
        )
        grid = weight_grid(len(features))
        penalties = np.sum((grid - (1.0 / len(features))) ** 2, axis=1)
        fold_predictions = {
            regularization: np.full(len(records), np.nan) for regularization in LAMBDAS
        }
        for held_channel in sorted(set(channels.tolist())):
            train_mask = channels != held_channel
            test_mask = ~train_mask
            base_objective = _macro_train_objective(
                x[train_mask], target[train_mask], channels[train_mask], grid
            )
            for regularization in LAMBDAS:
                objective = base_objective - regularization * penalties
                best_index = int(np.nanargmax(objective))
                best = grid[best_index]
                fold_predictions[regularization][test_mask] = x[test_mask] @ best
                fold_weight_rows.append(
                    {
                        "run_name": run_name,
                        "feature_set": feature_set_name,
                        "regularization": regularization,
                        "held_channel": held_channel,
                        "training_objective": rounded(base_objective[best_index]),
                        **{
                            f"weight_{feature}": rounded(weight)
                            for feature, weight in zip(features, best)
                        },
                    }
                )

        full_objective = _macro_train_objective(x, target, channels, grid)
        for regularization in LAMBDAS:
            config_name = f"{feature_set_name}_lambda_{regularization}"
            config_predictions[config_name] = fold_predictions[regularization]
            objective = full_objective - regularization * penalties
            best = grid[int(np.nanargmax(objective))]
            final_weights_by_config[config_name] = {
                feature: rounded(weight) for feature, weight in zip(features, best)
            }

    equal_x = np.asarray(
        [[float(row[feature]) for feature in CHECK_FEATURES] for row in records],
        dtype=float,
    )
    config_predictions["baseline_equal_checklist7"] = equal_x.mean(axis=1)
    config_predictions["baseline_saliency_only"] = np.asarray(
        [float(row["saliency"]) for row in records], dtype=float
    )

    oof_rows: list[dict[str, Any]] = []
    config_summaries: list[dict[str, Any]] = []
    for config_name, predictions in config_predictions.items():
        rows = [
            {
                **record,
                "config": config_name,
                "oof_score": rounded(prediction * 100.0),
            }
            for record, prediction in zip(records, predictions)
        ]
        metrics = _metrics(rows, "oof_score")
        config_summaries.append(
            {
                "config": config_name,
                "feature_weights_fit": final_weights_by_config.get(config_name),
                **{key: value for key, value in metrics.items() if key != "channel_metrics"},
                "channel_metrics": metrics["channel_metrics"],
            }
        )
        oof_rows.extend(rows)

    learned = [
        row for row in config_summaries if not row["config"].startswith("baseline_")
    ]
    selected = max(
        learned,
        key=lambda row: (
            float(row["channel_macro_spearman"] or -2),
            float(row["within_channel_centered_spearman"] or -2),
            -len([v for v in (row.get("feature_weights_fit") or {}).values() if v == 0]),
        ),
    )
    selected_folds = [
        row
        for row in fold_weight_rows
        if f"{row['feature_set']}_lambda_{row['regularization']}"
        == selected["config"]
    ]
    selected_features = list((selected.get("feature_weights_fit") or {}).keys())
    stability = [
        {
            "feature": feature,
            "final_weight": selected["feature_weights_fit"][feature],
            "fold_mean": rounded(
                statistics.mean(float(row.get(f"weight_{feature}", 0)) for row in selected_folds)
            ),
            "fold_std": rounded(
                statistics.pstdev(
                    float(row.get(f"weight_{feature}", 0)) for row in selected_folds
                )
            ),
        }
        for feature in selected_features
    ]
    equal_baseline = next(
        row for row in config_summaries if row["config"] == "baseline_equal_checklist7"
    )
    saliency_baseline = next(
        row for row in config_summaries if row["config"] == "baseline_saliency_only"
    )
    summary = {
        "run_name": run_name,
        "source_scores": str(scores_path),
        "candidate_count": len(records),
        "target": "continuous channel_view_percentile",
        "labels_used": "none; pos/neg and thresholds excluded",
        "validation": "leave-one-channel-out out-of-fold prediction",
        "weight_constraints": "nonnegative, sum to 1, grid step 0.1",
        "tested_weight_count": {
            "checklist7": len(weight_grid(7)),
            "saliency_plus_checklist7": len(weight_grid(8)),
        },
        "selected_config": selected,
        "selected_weight_stability": stability,
        "equal_weight_baseline": equal_baseline,
        "saliency_only_baseline": saliency_baseline,
        "all_configs": config_summaries,
        "warning": (
            "The final all-data weights are deployable candidates, but only OOF metrics should "
            "be used as evidence. Large fold-to-fold weight variance indicates instability."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"{run_name}_oof_predictions_PRIVATE.csv", oof_rows)
    write_csv(output_dir / f"{run_name}_fold_weights.csv", fold_weight_rows)
    write_csv(output_dir / f"{run_name}_selected_weight_stability.csv", stability)
    write_json(output_dir / f"{run_name}_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Learn nonnegative v7 criterion weights with leave-one-channel-out CV."
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--behavior", type=Path, default=DEFAULT_BEHAVIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="codex_vpick_v7")
    args = parser.parse_args()
    summary = learn(
        args.scores,
        args.targets,
        args.behavior,
        args.output_dir,
        args.run_name,
    )
    print(json.dumps(summary["selected_config"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
