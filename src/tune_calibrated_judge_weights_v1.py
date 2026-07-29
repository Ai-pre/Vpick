from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_package_and_context_v1 import (
    ALPHAS,
    fit_ridge,
    group_folds,
    metrics,
    nested_group_oof,
    predict_ridge,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_WEIGHTS = np.array([0.50, 0.25, 0.25], dtype=float)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def weight_grid(step: float, minimum: float) -> list[np.ndarray]:
    units = round(1.0 / step)
    minimum_units = math.ceil(minimum / step - 1e-9)
    output: list[np.ndarray] = []
    for content_units in range(minimum_units, units + 1):
        for title_units in range(minimum_units, units - content_units + 1):
            thumbnail_units = units - content_units - title_units
            if thumbnail_units < minimum_units:
                continue
            output.append(
                np.array(
                    [content_units, title_units, thumbnail_units], dtype=float
                )
                / units
            )
    return output


def build_features(
    dimensions: np.ndarray, context: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    judge_score = dimensions @ weights
    return np.column_stack([judge_score, context])


def select_weights_and_alpha(
    dimensions: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    groups: list[str],
    candidates: list[np.ndarray],
    seed: str,
) -> tuple[np.ndarray, float, float]:
    folds = group_folds(groups, min(4, len(set(groups))), seed)
    all_indices = np.arange(len(target))
    best: tuple[float, float, float, np.ndarray, float] | None = None
    for weights in candidates:
        x = build_features(dimensions, context, weights)
        for alpha in ALPHAS:
            prediction = np.full(len(target), np.nan)
            valid = True
            for validation in folds:
                train = np.setdiff1d(all_indices, validation)
                if len(train) < 5 or len(validation) < 2:
                    valid = False
                    break
                model = fit_ridge(x[train], target[train], alpha)
                prediction[validation] = predict_ridge(model, x[validation])
            if not valid or np.isnan(prediction).any():
                continue
            score = spearman(prediction, target)
            distance = float(np.abs(weights - BASELINE_WEIGHTS).sum())
            # Prefer the baseline-nearest weights and stronger regularization on ties.
            key = (score, -distance, alpha)
            if best is None or key > best[:3]:
                best = (score, -distance, alpha, weights.copy(), alpha)
    if best is None:
        raise RuntimeError("Could not select weights and alpha")
    return best[3], best[4], best[0]


def nested_weight_oof(
    dimensions: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    groups: list[str],
    candidates: list[np.ndarray],
    seed: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = group_folds(groups, min(5, len(set(groups))), f"{seed}-outer")
    all_indices = np.arange(len(target))
    prediction = np.full(len(target), np.nan)
    selections: list[dict[str, Any]] = []
    for fold_index, validation in enumerate(folds):
        train = np.setdiff1d(all_indices, validation)
        train_groups = [groups[index] for index in train]
        weights, alpha, inner_score = select_weights_and_alpha(
            dimensions[train],
            context[train],
            target[train],
            train_groups,
            candidates,
            f"{seed}-inner-{fold_index}",
        )
        x_train = build_features(dimensions[train], context[train], weights)
        x_validation = build_features(
            dimensions[validation], context[validation], weights
        )
        model = fit_ridge(x_train, target[train], alpha)
        prediction[validation] = predict_ridge(model, x_validation)
        selections.append(
            {
                "fold": fold_index + 1,
                "validation_n": len(validation),
                "content_weight": float(weights[0]),
                "title_weight": float(weights[1]),
                "thumbnail_weight": float(weights[2]),
                "alpha": alpha,
                "inner_spearman": inner_score,
            }
        )
    return prediction, selections


def pairwise_accuracy(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    concordant = 0
    discordant = 0
    ties = 0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            predicted_direction = np.sign(prediction[left] - prediction[right])
            target_direction = np.sign(target[left] - target[right])
            if predicted_direction == 0 or target_direction == 0:
                ties += 1
            elif predicted_direction == target_direction:
                concordant += 1
            else:
                discordant += 1
    return {
        "concordant": concordant,
        "discordant": discordant,
        "ties": ties,
        "accuracy": round(concordant / (concordant + discordant), 6),
    }


def summarize_selections(selections: list[dict[str, Any]]) -> dict[str, Any]:
    weight_counts = Counter(
        (
            row["content_weight"],
            row["title_weight"],
            row["thumbnail_weight"],
        )
        for row in selections
    )
    return {
        "fold_selections": selections,
        "mean_weights": {
            "content": round(
                float(np.mean([row["content_weight"] for row in selections])), 4
            ),
            "title": round(
                float(np.mean([row["title_weight"] for row in selections])), 4
            ),
            "thumbnail": round(
                float(np.mean([row["thumbnail_weight"] for row in selections])), 4
            ),
        },
        "unique_weight_sets": len(weight_counts),
        "weight_frequency": [
            {
                "content": weights[0],
                "title": weights[1],
                "thumbnail": weights[2],
                "count": count,
            }
            for weights, count in weight_counts.most_common()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "results/package_context_performance_v1_2026-07-29/candidate_predictions_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/calibrated_weight_search_v1_2026-07-29",
    )
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_csv(args.input)
    dimensions = np.array(
        [
            [
                float(row["content_success_0_1"]),
                float(row["title_success_0_1"]),
                float(row["thumbnail_success_0_1"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    context = np.array(
        [
            [
                math.log1p(float(row["channel_prior_median_views"])),
                math.log1p(float(row["upload_age_days"])),
            ]
            for row in rows
        ],
        dtype=float,
    )
    raw_views = np.array([float(row["raw_views"]) for row in rows])
    percentile = np.array([float(row["channel_percentile"]) for row in rows])
    target = np.log1p(raw_views)
    groups = [row["longform_id"] for row in rows]
    split = np.array([row["split"] for row in rows])
    dev = np.flatnonzero(split == "dev")
    locked = np.flatnonzero(split == "locked_test")

    policies = {
        "free_simplex_5pct": weight_grid(args.step, 0.0),
        "min10_each_5pct": weight_grid(args.step, 0.10),
    }
    summary: dict[str, Any] = {
        "candidate_count": len(rows),
        "step": args.step,
        "policies": {},
        "warnings": [
            "Weights are selected only inside training folds.",
            "The locked75 weights are selected from dev19 only.",
            "The all94 result uses fold-specific weights and is not one deployment formula.",
        ],
    }
    prediction_rows = [
        {
            "candidate_id": row["candidate_id"],
            "longform_id": row["longform_id"],
            "split": row["split"],
            "raw_views_PRIVATE": row["raw_views"],
        }
        for row in rows
    ]

    for policy_name, candidates in policies.items():
        oof_prediction, fold_selections = nested_weight_oof(
            dimensions,
            context,
            target,
            groups,
            candidates,
            "calibrated-weight-v1-common-folds",
        )
        dev_groups = [groups[index] for index in dev]
        locked_weights, locked_alpha, dev_inner_score = select_weights_and_alpha(
            dimensions[dev],
            context[dev],
            target[dev],
            dev_groups,
            candidates,
            "calibrated-weight-v1-common-locked-inner-folds",
        )
        locked_train = build_features(
            dimensions[dev], context[dev], locked_weights
        )
        locked_test = build_features(
            dimensions[locked], context[locked], locked_weights
        )
        locked_model = fit_ridge(
            locked_train, target[dev], locked_alpha
        )
        locked_prediction = predict_ridge(locked_model, locked_test)

        summary["policies"][policy_name] = {
            "grid_size": len(candidates),
            "all94_nested_group_oof": {
                **metrics(oof_prediction, raw_views, percentile),
                "pairwise_ranking": pairwise_accuracy(
                    oof_prediction, raw_views
                ),
                **summarize_selections(fold_selections),
            },
            "locked75_dev19_selected": {
                **metrics(
                    locked_prediction,
                    raw_views[locked],
                    percentile[locked],
                ),
                "pairwise_ranking": pairwise_accuracy(
                    locked_prediction, raw_views[locked]
                ),
                "selected_weights": {
                    "content": float(locked_weights[0]),
                    "title": float(locked_weights[1]),
                    "thumbnail": float(locked_weights[2]),
                },
                "alpha": locked_alpha,
                "dev_inner_spearman": dev_inner_score,
            },
        }
        for index, prediction in enumerate(oof_prediction):
            prediction_rows[index][f"oof_prediction_{policy_name}"] = round(
                float(prediction), 8
            )
        locked_lookup = {
            record_index: position for position, record_index in enumerate(locked)
        }
        for index in range(len(rows)):
            prediction_rows[index][f"locked_prediction_{policy_name}"] = (
                round(float(locked_prediction[locked_lookup[index]]), 8)
                if index in locked_lookup
                else ""
            )

    posthoc_best: tuple[float, np.ndarray, list[float]] | None = None
    for weights in policies["free_simplex_5pct"]:
        prediction, alphas = nested_group_oof(
            build_features(dimensions, context, weights),
            target,
            groups,
            "package-context-v1-common-folds",
        )
        score = spearman(prediction, target)
        if posthoc_best is None or score > posthoc_best[0]:
            posthoc_best = (score, weights.copy(), alphas)
    assert posthoc_best is not None
    summary["posthoc_common_fold_diagnostic"] = {
        "warning": (
            "Exploratory upper bound only. The weights were selected after seeing "
            "all 94 out-of-fold outcomes and are not an unbiased validation result."
        ),
        "weights": {
            "content": float(posthoc_best[1][0]),
            "title": float(posthoc_best[1][1]),
            "thumbnail": float(posthoc_best[1][2]),
        },
        "spearman_raw_views": round(posthoc_best[0], 6),
        "fold_alphas": posthoc_best[2],
        "comparison": {
            "fixed_50_25_25_common_fold_spearman": 0.409531,
            "posthoc_30_20_50_common_fold_spearman": 0.412941,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "weight_search_summary_PRIVATE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (
        args.output_dir / "weight_search_predictions_94_PRIVATE.csv"
    ).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
