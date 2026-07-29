from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_package_and_context_v1 import (
    choose_alpha,
    fit_ridge,
    metrics,
    nested_group_oof,
    predict_ridge,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "results/package_context_performance_v1_2026-07-29"
    / "candidate_predictions_94_PRIVATE.csv"
)
SALIENCE = (
    ROOT
    / "data/private/judge_validation_94"
    / "codex_direct_source_salience_94_PRIVATE.csv"
)
OUTPUT_DIR = ROOT / "results/salience_augmented_weight_search_v1_2026-07-29"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def weight_grid(step_units: int = 20) -> list[np.ndarray]:
    weights: list[np.ndarray] = []
    for salience in range(step_units + 1):
        for change in range(step_units - salience + 1):
            for title in range(step_units - salience - change + 1):
                thumbnail = step_units - salience - change - title
                weights.append(
                    np.array(
                        [salience, change, title, thumbnail], dtype=float
                    )
                    / step_units
                )
    return weights


def pairwise_accuracy(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    concordant = 0
    discordant = 0
    ties = 0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            predicted = np.sign(prediction[left] - prediction[right])
            actual = np.sign(target[left] - target[right])
            if predicted == 0 or actual == 0:
                ties += 1
            elif predicted == actual:
                concordant += 1
            else:
                discordant += 1
    denominator = concordant + discordant
    return {
        "concordant": concordant,
        "discordant": discordant,
        "ties": ties,
        "accuracy": round(concordant / denominator, 6),
    }


def main() -> int:
    rows = read_csv(INPUT)
    salience_by_id = {
        row["candidate_id"]: row for row in read_csv(SALIENCE)
    }
    dimensions = np.array(
        [
            [
                float(salience_by_id[row["candidate_id"]][
                    "source_salience_0_4"
                ])
                / 4.0,
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
    percentile = np.array(
        [float(row["channel_percentile"]) for row in rows]
    )
    target = np.log1p(raw_views)
    groups = [row["longform_id"] for row in rows]
    split = np.array([row["split"] for row in rows])
    dev = np.flatnonzero(split == "dev")
    locked = np.flatnonzero(split == "locked_test")

    candidates = weight_grid()
    records: list[dict[str, Any]] = []
    best_pointwise: tuple[float, np.ndarray] | None = None
    best_oof: tuple[float, np.ndarray, list[float], np.ndarray] | None = None
    best_locked: tuple[float, np.ndarray, float, np.ndarray] | None = None

    for weights in candidates:
        judge_score = dimensions @ weights
        pointwise = spearman(judge_score, raw_views)
        if best_pointwise is None or pointwise > best_pointwise[0]:
            best_pointwise = (pointwise, weights.copy())

        features = np.column_stack([judge_score, context])
        oof_prediction, alphas = nested_group_oof(
            features,
            target,
            groups,
            "package-context-v1-common-folds",
        )
        oof = spearman(oof_prediction, raw_views)
        if best_oof is None or oof > best_oof[0]:
            best_oof = (
                oof,
                weights.copy(),
                alphas,
                oof_prediction.copy(),
            )

        dev_groups = [groups[index] for index in dev]
        alpha = choose_alpha(
            features[dev],
            target[dev],
            dev_groups,
            "package-context-v1-common-locked-inner-folds",
        )
        model = fit_ridge(features[dev], target[dev], alpha)
        locked_prediction = predict_ridge(model, features[locked])
        locked_score = spearman(locked_prediction, raw_views[locked])
        if best_locked is None or locked_score > best_locked[0]:
            best_locked = (
                locked_score,
                weights.copy(),
                alpha,
                locked_prediction.copy(),
            )
        records.append(
            {
                "salience_weight": weights[0],
                "change_weight": weights[1],
                "title_weight": weights[2],
                "thumbnail_weight": weights[3],
                "pointwise_spearman": round(pointwise, 6),
                "oof_calibrated_spearman": round(oof, 6),
                "locked_calibrated_spearman": round(locked_score, 6),
            }
        )

    assert best_pointwise is not None
    assert best_oof is not None
    assert best_locked is not None

    def weight_dict(weights: np.ndarray) -> dict[str, float]:
        return {
            "salience": float(weights[0]),
            "change": float(weights[1]),
            "title": float(weights[2]),
            "thumbnail": float(weights[3]),
        }

    baseline_weights = np.array([0.0, 0.5, 0.25, 0.25])
    baseline_score = dimensions @ baseline_weights
    baseline_features = np.column_stack([baseline_score, context])
    baseline_oof, baseline_alphas = nested_group_oof(
        baseline_features,
        target,
        groups,
        "package-context-v1-common-folds",
    )
    baseline_alpha = choose_alpha(
        baseline_features[dev],
        target[dev],
        [groups[index] for index in dev],
        "package-context-v1-common-locked-inner-folds",
    )
    baseline_model = fit_ridge(
        baseline_features[dev], target[dev], baseline_alpha
    )
    baseline_locked = predict_ridge(
        baseline_model, baseline_features[locked]
    )
    summary = {
        "candidate_count": len(rows),
        "grid_size": len(candidates),
        "step": 0.05,
        "selection_warning": (
            "All best formulas are post-hoc exploratory selections on the "
            "same 94 candidates and must not be described as unbiased validation."
        ),
        "baseline_fixed_0_50_25_25": {
            "weights": weight_dict(baseline_weights),
            "pointwise": metrics(baseline_score, raw_views, percentile),
            "all94_group_oof": {
                **metrics(baseline_oof, raw_views, percentile),
                "pairwise_ranking": pairwise_accuracy(
                    baseline_oof, raw_views
                ),
                "alphas": baseline_alphas,
            },
            "locked75": {
                **metrics(
                    baseline_locked, raw_views[locked], percentile[locked]
                ),
                "pairwise_ranking": pairwise_accuracy(
                    baseline_locked, raw_views[locked]
                ),
                "alpha": baseline_alpha,
            },
        },
        "best_pointwise": {
            "weights": weight_dict(best_pointwise[1]),
            "spearman_raw_views": round(best_pointwise[0], 6),
        },
        "best_all94_group_oof": {
            "weights": weight_dict(best_oof[1]),
            **metrics(best_oof[3], raw_views, percentile),
            "pairwise_ranking": pairwise_accuracy(
                best_oof[3], raw_views
            ),
            "alphas": best_oof[2],
        },
        "best_locked75": {
            "weights": weight_dict(best_locked[1]),
            **metrics(
                best_locked[3], raw_views[locked], percentile[locked]
            ),
            "pairwise_ranking": pairwise_accuracy(
                best_locked[3], raw_views[locked]
            ),
            "alpha": best_locked[2],
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "all_weight_results_PRIVATE.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT_DIR / "weight_search_summary_PRIVATE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
