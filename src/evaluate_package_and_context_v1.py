from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.0, 0.1, 1.0, 10.0, 100.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def by_id(rows: list[dict[str, str]], field: str = "candidate_id") -> dict[str, dict[str, str]]:
    return {row[field]: row for row in rows}


def normalized_channel(value: str) -> str:
    cleaned = str(value or "").strip()
    aliases = {
        "OOTB_Studio": "OOTB",
        "ootb STUDIO": "OOTB",
        "빠더너스": "BDNS",
        "빠더너스 BDNS": "BDNS",
        "안녕하세요원이입니다잘부탁드립니다": "안원잘부",
    }
    return aliases.get(cleaned, cleaned)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        ranks[order[index:end]] = average_rank
        index = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def top_quartile_precision(prediction: np.ndarray, target: np.ndarray) -> float:
    count = max(1, math.ceil(len(target) * 0.25))
    predicted_top = set(np.argsort(prediction)[-count:])
    actual_top = set(np.argsort(target)[-count:])
    return len(predicted_top & actual_top) / count


def metrics(prediction: np.ndarray, raw_views: np.ndarray, percentile: np.ndarray) -> dict[str, float]:
    return {
        "spearman_raw_views": round(spearman(prediction, raw_views), 6),
        "spearman_channel_percentile": round(spearman(prediction, percentile), 6),
        "top25_precision_raw_views": round(
            top_quartile_precision(prediction, raw_views), 6
        ),
    }


def group_folds(groups: list[str], fold_count: int, seed: str) -> list[np.ndarray]:
    unique = sorted(
        set(groups),
        key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(),
    )
    bins: list[list[str]] = [[] for _ in range(min(fold_count, len(unique)))]
    sizes = [0 for _ in bins]
    counts = {group: groups.count(group) for group in unique}
    for group in sorted(unique, key=lambda item: (-counts[item], unique.index(item))):
        target_bin = min(range(len(bins)), key=lambda index: sizes[index])
        bins[target_bin].append(group)
        sizes[target_bin] += counts[group]
    return [
        np.array([index for index, group in enumerate(groups) if group in fold], dtype=int)
        for fold in bins
    ]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    x_scaled = (x - mean) / scale
    y_mean = float(y.mean())
    centered = y - y_mean
    identity = np.eye(x.shape[1])
    coefficient = np.linalg.pinv(x_scaled.T @ x_scaled + alpha * identity) @ x_scaled.T @ centered
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": y_mean,
    }


def predict_ridge(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    return (
        (x - model["mean"]) / model["scale"]
    ) @ model["coefficient"] + float(model["intercept"])


def choose_alpha(x: np.ndarray, y: np.ndarray, groups: list[str], seed: str) -> float:
    folds = group_folds(groups, min(4, len(set(groups))), seed)
    if len(folds) < 2:
        return 1.0
    alpha_scores: dict[float, list[float]] = defaultdict(list)
    all_indices = np.arange(len(y))
    for validation in folds:
        train = np.setdiff1d(all_indices, validation)
        if len(train) < x.shape[1] + 2 or len(validation) < 2:
            continue
        for alpha in ALPHAS:
            model = fit_ridge(x[train], y[train], alpha)
            prediction = predict_ridge(model, x[validation])
            alpha_scores[alpha].append(spearman(prediction, y[validation]))
    if not alpha_scores:
        return 1.0
    return max(
        ALPHAS,
        key=lambda alpha: (
            float(np.mean(alpha_scores.get(alpha, [-999.0]))),
            -alpha,
        ),
    )


def nested_group_oof(
    x: np.ndarray, y: np.ndarray, groups: list[str], seed: str
) -> tuple[np.ndarray, list[float]]:
    prediction = np.full(len(y), np.nan)
    chosen: list[float] = []
    outer_folds = group_folds(groups, min(5, len(set(groups))), f"{seed}-outer")
    all_indices = np.arange(len(y))
    for fold_index, validation in enumerate(outer_folds):
        train = np.setdiff1d(all_indices, validation)
        train_groups = [groups[index] for index in train]
        alpha = choose_alpha(
            x[train], y[train], train_groups, f"{seed}-inner-{fold_index}"
        )
        model = fit_ridge(x[train], y[train], alpha)
        prediction[validation] = predict_ridge(model, x[validation])
        chosen.append(alpha)
    return prediction, chosen


def parse_datetime(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "data/private/judge_validation_94/validation_targets_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=ROOT
        / "data/private/judge_validation_94/raw_short_performance_2026-07-29_PRIVATE.csv",
    )
    parser.add_argument(
        "--package-scores",
        type=Path,
        default=ROOT
        / "results/package_success_judge_v1_codex_direct_2026-07-29/codex_package_judge_aggregate_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--content-scores",
        type=Path,
        default=ROOT
        / "results/judge_success_v1_codex_direct_94_2026-07-29/codex_direct_success_dimensions_94.csv",
    )
    parser.add_argument(
        "--title-scores",
        type=Path,
        default=ROOT
        / "results/judge_success_v1_codex_direct_94_2026-07-29/codex_direct_title_packaging_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--thumbnail-scores",
        type=Path,
        default=ROOT
        / "results/judge_success_v1_codex_direct_94_2026-07-29/codex_direct_thumbnail_packaging_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--dates",
        type=Path,
        default=ROOT
        / "data/private/judge_validation_94/short_publish_dates_2026-07-29_PRIVATE.csv",
    )
    parser.add_argument(
        "--cohorts",
        type=Path,
        default=ROOT / "data/processed/channel_short_cohorts_2026-07-23.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/package_context_performance_v1_2026-07-29",
    )
    args = parser.parse_args()

    targets = read_csv(args.targets)
    performance = by_id(read_csv(args.performance))
    package = by_id(read_csv(args.package_scores))
    content = by_id(read_csv(args.content_scores))
    title_scores = by_id(read_csv(args.title_scores))
    thumbnail_scores = by_id(read_csv(args.thumbnail_scores))
    dates = by_id(read_csv(args.dates))
    target_video_ids = {row["short_video_id"] for row in targets}

    reference_views: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(args.cohorts):
        video_id = row.get("video_id", "")
        if video_id in target_video_ids:
            continue
        view_text = row.get("view_count", "")
        if not view_text:
            continue
        reference_views[normalized_channel(row["channel_name"])].append(float(view_text))
    channel_prior = {
        channel: float(np.median(values)) for channel, values in reference_views.items()
    }

    records: list[dict[str, Any]] = []
    for row in sorted(targets, key=lambda item: item["candidate_id"]):
        candidate_id = row["candidate_id"]
        perf = performance[candidate_id]
        published = parse_datetime(dates[candidate_id]["published_at"])
        snapshot = parse_datetime(perf["snapshot_utc"])
        age_days = max(1.0, (snapshot - published).total_seconds() / 86400.0)
        channel = normalized_channel(row["channel_name"])
        if channel not in channel_prior:
            raise RuntimeError(f"No independent channel prior for {channel}")
        package_score = float(
            package[candidate_id]["joint_package_score_1_10_mean"]
        )
        change_score = float(content[candidate_id]["change_or_surprise_0_4"])
        title_score = float(title_scores[candidate_id]["title_packaging_0_4"])
        thumbnail_score = float(
            thumbnail_scores[candidate_id]["thumbnail_packaging_0_4"]
        )
        legacy_balanced = (
            0.50 * change_score + 0.25 * title_score + 0.25 * thumbnail_score
        ) / 4.0
        optimized_success = (
            0.40 * change_score + 0.15 * title_score + 0.45 * thumbnail_score
        ) / 4.0
        records.append(
            {
                "candidate_id": candidate_id,
                "longform_id": row["longform_id"],
                "split": row["dataset_role_v3"],
                "channel_name": channel,
                "raw_views": float(perf["view_count"]),
                "channel_percentile": float(
                    row["channel_performance_percentile_PRIVATE"]
                ),
                "content_success_0_1": change_score / 4.0,
                "title_success_0_1": title_score / 4.0,
                "thumbnail_success_0_1": thumbnail_score / 4.0,
                "package_success_0_1": (package_score - 1.0) / 9.0,
                "content_package_fixed_0_1": (
                    change_score / 4.0 + (package_score - 1.0) / 9.0
                )
                / 2.0,
                "legacy_balanced_50_25_25_0_1": legacy_balanced,
                "optimized_success_40_15_45_0_1": optimized_success,
                "channel_prior_median_views": channel_prior[channel],
                "upload_age_days": age_days,
            }
        )

    raw_views = np.array([record["raw_views"] for record in records], dtype=float)
    percentile = np.array(
        [record["channel_percentile"] for record in records], dtype=float
    )
    y = np.log1p(raw_views)
    groups = [record["longform_id"] for record in records]
    split = np.array([record["split"] for record in records])
    dev = np.flatnonzero(split == "dev")
    locked = np.flatnonzero(split == "locked_test")

    feature_sets = {
        "content_only": ["content_success_0_1"],
        "package_only_3pass": ["package_success_0_1"],
        "content_package_fixed_50_50": ["content_package_fixed_0_1"],
        "legacy_balanced_50_25_25": ["legacy_balanced_50_25_25_0_1"],
        "optimized_success_40_15_45": [
            "optimized_success_40_15_45_0_1"
        ],
        "channel_prior_only": ["channel_prior_median_views"],
        "upload_age_only": ["upload_age_days"],
        "channel_plus_age": [
            "channel_prior_median_views",
            "upload_age_days",
        ],
        "content_package_plus_age": [
            "content_package_fixed_0_1",
            "upload_age_days",
        ],
        "content_package_plus_channel": [
            "content_package_fixed_0_1",
            "channel_prior_median_views",
        ],
        "content_package_plus_channel_age": [
            "content_package_fixed_0_1",
            "channel_prior_median_views",
            "upload_age_days",
        ],
        "legacy_balanced_plus_channel_age": [
            "legacy_balanced_50_25_25_0_1",
            "channel_prior_median_views",
            "upload_age_days",
        ],
        "optimized_success_40_15_45_plus_channel_age": [
            "optimized_success_40_15_45_0_1",
            "channel_prior_median_views",
            "upload_age_days",
        ],
        "separate_content_title_thumbnail_channel_age": [
            "content_success_0_1",
            "title_success_0_1",
            "thumbnail_success_0_1",
            "channel_prior_median_views",
            "upload_age_days",
        ],
    }
    direct_models = {
        "content_only",
        "package_only_3pass",
        "content_package_fixed_50_50",
        "legacy_balanced_50_25_25",
        "optimized_success_40_15_45",
        "channel_prior_only",
        "upload_age_only",
    }
    predictions: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, Any]] = []

    for model_name, fields in feature_sets.items():
        raw_x = np.array(
            [
                [
                    (
                        math.log1p(float(record[field]))
                        if field in {"channel_prior_median_views", "upload_age_days"}
                        else float(record[field])
                    )
                    for field in fields
                ]
                for record in records
            ],
            dtype=float,
        )
        x = raw_x
        if model_name in direct_models:
            overall_prediction = x[:, 0]
            locked_prediction = overall_prediction[locked]
            alpha_text = "not_applicable"
        else:
            overall_prediction, alphas = nested_group_oof(
                x, y, groups, "package-context-v1-common-folds"
            )
            dev_groups = [groups[index] for index in dev]
            alpha = choose_alpha(
                x[dev],
                y[dev],
                dev_groups,
                "package-context-v1-common-locked-inner-folds",
            )
            locked_model = fit_ridge(x[dev], y[dev], alpha)
            locked_prediction = predict_ridge(locked_model, x[locked])
            alpha_text = json.dumps({"oof": alphas, "locked_fit": alpha})
        predictions[model_name] = {
            "oof_or_direct": overall_prediction,
            "locked": locked_prediction,
        }
        for evaluation_name, indices, prediction in (
            ("all94_group_oof_or_direct", np.arange(len(records)), overall_prediction),
            ("locked75", locked, locked_prediction),
        ):
            result = metrics(prediction, raw_views[indices], percentile[indices])
            summary_rows.append(
                {
                    "model": model_name,
                    "evaluation": evaluation_name,
                    "n": len(indices),
                    "features": "|".join(fields),
                    **result,
                    "chosen_alphas": alpha_text,
                }
            )

    dev_sorted = sorted(dev, key=lambda index: raw_views[index])
    anchor_positions = np.linspace(0, len(dev_sorted) - 1, 9).round().astype(int)
    anchor_indices = [dev_sorted[position] for position in anchor_positions]
    package_prediction = predictions["package_only_3pass"]["oof_or_direct"]
    anchor_scores = package_prediction[anchor_indices]
    anchors: list[dict[str, Any]] = []
    for order, index in enumerate(anchor_indices, start=1):
        anchors.append(
            {
                "anchor_order_low_to_high": order,
                "candidate_id": records[index]["candidate_id"],
                "package_score_0_1": round(float(package_prediction[index]), 6),
                "raw_views_PRIVATE": int(raw_views[index]),
            }
        )
    anchor_wins = np.array(
        [sum(score > anchor for anchor in anchor_scores) for score in package_prediction],
        dtype=float,
    )
    anchor_metric = metrics(anchor_wins, raw_views, percentile)
    summary_rows.append(
        {
            "model": "package_anchor_calibrated_0_9",
            "evaluation": "all94_direct_monotonic_calibration",
            "n": len(records),
            "features": "package_success_0_1 compared_with_9_dev_anchors",
            **anchor_metric,
            "chosen_alphas": "not_applicable",
        }
    )

    output_rows: list[dict[str, Any]] = []
    locked_position = {record_index: position for position, record_index in enumerate(locked)}
    for index, record in enumerate(records):
        output = dict(record)
        output["anchor_wins_0_9"] = int(anchor_wins[index])
        for model_name, model_prediction in predictions.items():
            output[f"prediction_{model_name}"] = round(
                float(model_prediction["oof_or_direct"][index]), 8
            )
            output[f"lockedfit_prediction_{model_name}"] = (
                round(
                    float(
                        model_prediction["locked"][locked_position[index]]
                    ),
                    8,
                )
                if index in locked_position
                else ""
            )
        output_rows.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "model_comparison_PRIVATE.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    write_csv(
        args.output_dir / "candidate_predictions_94_PRIVATE.csv",
        output_rows,
        list(output_rows[0]),
    )
    write_csv(
        args.output_dir / "dev_anchor_set_9_PRIVATE.csv",
        anchors,
        list(anchors[0]),
    )
    best_locked = max(
        (row for row in summary_rows if row["evaluation"] == "locked75"),
        key=lambda row: row["spearman_raw_views"],
    )
    best_oof = max(
        (
            row
            for row in summary_rows
            if row["evaluation"] == "all94_group_oof_or_direct"
        ),
        key=lambda row: row["spearman_raw_views"],
    )
    summary = {
        "candidate_count": len(records),
        "dev_count": len(dev),
        "locked_test_count": len(locked),
        "independent_channel_reference_counts": {
            channel: len(values) for channel, values in sorted(reference_views.items())
        },
        "target_video_ids_excluded_from_channel_prior": len(target_video_ids),
        "representative_frames_used": False,
        "performance_features_visible_to_codex_judge": False,
        "best_locked75": best_locked,
        "best_all94_group_oof_or_direct": best_oof,
        "notes": [
            "Channel prior is computed only from non-target cohort Shorts.",
            "Learned models use grouped nested cross-validation by longform.",
            "The locked75 model is fit only on the 19-item dev split.",
            "Anchor calibration is monotonic and is for interpretation, not rank improvement.",
        ],
    }
    (args.output_dir / "experiment_summary_PRIVATE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
