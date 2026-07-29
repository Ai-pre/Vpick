from __future__ import annotations

import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_package_and_context_v1 import (
    by_id,
    choose_alpha,
    metrics,
    nested_group_oof,
    normalized_channel,
    parse_datetime,
    predict_ridge,
    read_csv,
    spearman,
    fit_ridge,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/seven_axis_package_judge_v1_2026-07-30"

SEVEN_AXES = (
    "self_contained_clarity",
    "progression_payoff",
    "boundary_integrity",
    "opening_pull",
    "change_or_surprise",
    "emotional_or_information_gain",
    "memorable_specificity",
)

# The improvement reranker gives 15/25/20/15/15/10 to opening,
# event/change, progression, self-contained, boundary, and titleability.
# Here event/change is split into change (15) and gain (10), while
# memorable specificity takes the titleability role. This keeps the original
# completion emphasis without adding a second title score.
SEVEN_AXIS_WEIGHTS = {
    "opening_pull": 0.15,
    "change_or_surprise": 0.15,
    "emotional_or_information_gain": 0.10,
    "progression_payoff": 0.20,
    "self_contained_clarity": 0.15,
    "boundary_integrity": 0.15,
    "memorable_specificity": 0.10,
}


def value(row: dict[str, str], field: str) -> float:
    return float(row[field])


def completion_gate(row: dict[str, str]) -> float:
    completion = [
        value(row, "progression_payoff"),
        value(row, "self_contained_clarity"),
        value(row, "boundary_integrity"),
    ]
    if any(score == 0 for score in completion):
        return 0.50
    low_count = sum(score <= 1 for score in completion)
    if low_count >= 2:
        return 0.65
    if low_count == 1:
        return 0.80
    return 1.00


def auc(binary_target: np.ndarray, prediction: np.ndarray) -> float:
    positive = prediction[binary_target == 1]
    negative = prediction[binary_target == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    wins = 0.0
    for pos in positive:
        wins += float(np.sum(pos > negative))
        wins += 0.5 * float(np.sum(pos == negative))
    return wins / (len(positive) * len(negative))


def pairwise_agreement(prediction: np.ndarray, target: np.ndarray) -> float:
    agree = 0.0
    eligible = 0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            target_delta = target[left] - target[right]
            prediction_delta = prediction[left] - prediction[right]
            if target_delta == 0 or prediction_delta == 0:
                continue
            eligible += 1
            agree += float(target_delta * prediction_delta > 0)
    return agree / eligible if eligible else float("nan")


def extended_metrics(
    prediction: np.ndarray,
    raw_views: np.ndarray,
    channel_percentile: np.ndarray,
) -> dict[str, float]:
    result = metrics(prediction, raw_views, channel_percentile)
    raw_high_cut = np.quantile(raw_views, 0.75)
    channel_high = (channel_percentile >= 75.0).astype(int)
    raw_high = (raw_views >= raw_high_cut).astype(int)
    result.update(
        {
            "raw_top25_auc": round(auc(raw_high, prediction), 6),
            "channel_high_auc": round(auc(channel_high, prediction), 6),
            "raw_pairwise_agreement": round(
                pairwise_agreement(prediction, raw_views), 6
            ),
            "unique_score_count": int(len(np.unique(np.round(prediction, 10)))),
        }
    )
    return result


def main() -> int:
    targets_path = (
        ROOT / "data/private/judge_validation_94/validation_targets_94_PRIVATE.csv"
    )
    performance_path = (
        ROOT
        / "data/private/judge_validation_94/raw_short_performance_2026-07-29_PRIVATE.csv"
    )
    dimensions_path = (
        ROOT / "data/private/judge_validation_94/codex_direct_v10_dimensions.csv"
    )
    success_dimensions_path = (
        ROOT
        / "results/judge_success_v1_codex_direct_94_2026-07-29/"
        "codex_direct_success_dimensions_94.csv"
    )
    title_path = (
        ROOT
        / "results/judge_success_v1_codex_direct_94_2026-07-29/"
        "codex_direct_title_packaging_94_PRIVATE.csv"
    )
    thumbnail_path = (
        ROOT
        / "results/judge_success_v1_codex_direct_94_2026-07-29/"
        "codex_direct_thumbnail_packaging_94_PRIVATE.csv"
    )
    dates_path = (
        ROOT
        / "data/private/judge_validation_94/"
        "short_publish_dates_2026-07-29_PRIVATE.csv"
    )
    cohorts_path = ROOT / "data/processed/channel_short_cohorts_2026-07-23.csv"

    targets = read_csv(targets_path)
    performance = by_id(read_csv(performance_path))
    dimensions = by_id(read_csv(dimensions_path))
    success_dimensions = by_id(read_csv(success_dimensions_path))
    titles = by_id(read_csv(title_path))
    thumbnails = by_id(read_csv(thumbnail_path))
    dates = by_id(read_csv(dates_path))
    target_video_ids = {row["short_video_id"] for row in targets}

    reference_views: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(cohorts_path):
        if row.get("video_id", "") in target_video_ids:
            continue
        if not row.get("view_count", ""):
            continue
        reference_views[normalized_channel(row["channel_name"])].append(
            float(row["view_count"])
        )
    channel_prior = {
        channel: float(np.median(values)) for channel, values in reference_views.items()
    }

    records: list[dict[str, Any]] = []
    for target in sorted(targets, key=lambda row: row["candidate_id"]):
        candidate_id = target["candidate_id"]
        dimension = dimensions[candidate_id]
        success_dimension = success_dimensions[candidate_id]
        perf = performance[candidate_id]
        channel = normalized_channel(target["channel_name"])
        published = parse_datetime(dates[candidate_id]["published_at"])
        snapshot = parse_datetime(perf["snapshot_utc"])
        age_days = max(1.0, (snapshot - published).total_seconds() / 86400.0)

        axis_values = {axis: value(dimension, axis) / 4.0 for axis in SEVEN_AXES}
        # Preserve the final performance-oriented change/surprise re-score used
        # by the selected Judge instead of falling back to the older v10 value.
        axis_values["change_or_surprise"] = (
            value(success_dimension, "change_or_surprise_0_4") / 4.0
        )
        seven_equal = sum(axis_values.values()) / len(axis_values)
        seven_weighted = sum(
            axis_values[axis] * weight for axis, weight in SEVEN_AXIS_WEIGHTS.items()
        )
        gate = completion_gate(dimension)
        seven_gated = seven_weighted * gate
        change = axis_values["change_or_surprise"]
        title = value(titles[candidate_id], "title_packaging_0_4") / 4.0
        thumbnail = (
            value(thumbnails[candidate_id], "thumbnail_packaging_0_4") / 4.0
        )

        records.append(
            {
                "candidate_id": candidate_id,
                "longform_id": target["longform_id"],
                "split": target["dataset_role_v3"],
                "channel_name": channel,
                "raw_views": float(perf["view_count"]),
                "channel_percentile": float(
                    target["channel_performance_percentile_PRIVATE"]
                ),
                "change_only": change,
                "seven_axis_equal": seven_equal,
                "seven_axis_weighted": seven_weighted,
                "seven_axis_weighted_gated": seven_gated,
                "completion_gate": gate,
                "title": title,
                "thumbnail": thumbnail,
                "change_title_thumbnail_40_15_45": (
                    0.40 * change + 0.15 * title + 0.45 * thumbnail
                ),
                "seven_title_thumbnail_40_15_45": (
                    0.40 * seven_weighted + 0.15 * title + 0.45 * thumbnail
                ),
                "seven_gated_title_thumbnail_40_15_45": (
                    0.40 * seven_gated + 0.15 * title + 0.45 * thumbnail
                ),
                "channel_prior_median_views": channel_prior[channel],
                "upload_age_days": age_days,
            }
        )

    raw_views = np.array([row["raw_views"] for row in records], dtype=float)
    channel_percentile = np.array(
        [row["channel_percentile"] for row in records], dtype=float
    )
    y = np.log1p(raw_views)
    groups = [row["longform_id"] for row in records]
    split = np.array([row["split"] for row in records])
    dev = np.flatnonzero(split == "dev")
    locked = np.flatnonzero(split == "locked_test")

    score_fields = (
        "change_only",
        "seven_axis_equal",
        "seven_axis_weighted",
        "seven_axis_weighted_gated",
        "change_title_thumbnail_40_15_45",
        "seven_title_thumbnail_40_15_45",
        "seven_gated_title_thumbnail_40_15_45",
    )
    summary: dict[str, Any] = {
        "candidate_count": len(records),
        "dev_count": int(len(dev)),
        "locked_count": int(len(locked)),
        "seven_axes": list(SEVEN_AXES),
        "seven_axis_weights": SEVEN_AXIS_WEIGHTS,
        "note": (
            "Post-hoc comparison on the available 94-candidate dataset. "
            "The locked split is no longer an untouched external holdout."
        ),
        "component_diagnostics": {},
        "direct": {},
        "performance_calibrated": {},
    }
    prediction_rows = [dict(row) for row in records]

    component_fields = (*SEVEN_AXES, "title", "thumbnail")
    for component in component_fields:
        if component in {"title", "thumbnail"}:
            component_prediction = np.array(
                [row[component] for row in records], dtype=float
            )
        else:
            component_prediction = np.array(
                [
                    (
                        value(
                            success_dimensions[row["candidate_id"]],
                            "change_or_surprise_0_4",
                        )
                        / 4.0
                        if component == "change_or_surprise"
                        else value(dimensions[row["candidate_id"]], component)
                        / 4.0
                    )
                    for row in records
                ],
                dtype=float,
            )
        summary["component_diagnostics"][component] = extended_metrics(
            component_prediction, raw_views, channel_percentile
        )

    for score_field in score_fields:
        direct_prediction = np.array(
            [row[score_field] for row in records], dtype=float
        )
        summary["direct"][score_field] = {
            "all94": extended_metrics(
                direct_prediction, raw_views, channel_percentile
            ),
            "locked75": extended_metrics(
                direct_prediction[locked],
                raw_views[locked],
                channel_percentile[locked],
            ),
        }

        x = np.array(
            [
                [
                    row[score_field],
                    math.log1p(row["channel_prior_median_views"]),
                    math.log1p(row["upload_age_days"]),
                ]
                for row in records
            ],
            dtype=float,
        )
        oof_prediction, oof_alphas = nested_group_oof(
            x, y, groups, "package-context-v1-common-folds"
        )
        dev_groups = [groups[index] for index in dev]
        locked_alpha = choose_alpha(
            x[dev],
            y[dev],
            dev_groups,
            "package-context-v1-common-locked-inner-folds",
        )
        locked_model = fit_ridge(x[dev], y[dev], locked_alpha)
        locked_prediction = predict_ridge(locked_model, x[locked])
        summary["performance_calibrated"][score_field] = {
            "all94_group_oof": extended_metrics(
                oof_prediction, raw_views, channel_percentile
            ),
            "locked75_dev19_fit": extended_metrics(
                locked_prediction,
                raw_views[locked],
                channel_percentile[locked],
            ),
            "oof_alphas": oof_alphas,
            "locked_alpha": locked_alpha,
        }
        for index, row in enumerate(prediction_rows):
            row[f"{score_field}_oof_expected_log_views"] = round(
                float(oof_prediction[index]), 8
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "candidate_scores_and_oof_predictions_PRIVATE.csv",
        prediction_rows,
        list(prediction_rows[0]),
    )
    (OUTPUT_DIR / "comparison_summary_PRIVATE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
