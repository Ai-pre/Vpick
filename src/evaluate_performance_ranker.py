from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = (
    ROOT
    / "deliverables"
    / "2026-07-24"
    / "performance_ranker"
    / "candidate_features_60_PRIVATE.csv"
)
DEFAULT_JUDGE = (
    ROOT
    / "results"
    / "gold_reference_judge_v9_v7"
    / "direct_codex"
    / "reference_judge_v7_scores.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "deliverables" / "2026-07-24" / "performance_ranker"

JUDGE_FEATURES = [
    "evidence_description_support",
    "evidence_transcript_intelligibility",
    "evidence_boundary_observability",
    "saliency_market_1_5",
    "check_hook_within_3s",
    "check_surprise_or_twist",
    "check_emotional_peak",
    "check_quotable_moment",
    "check_payoff_or_conclusion",
    "check_natural_start",
    "check_natural_end",
    "checklist_score_100",
    "overall_shortform_suitable",
    "confidence_1_5",
]

METADATA_FEATURES = [
    "duration_sec",
    "transcript_char_count",
    "transcript_line_count",
    "transcript_timed_line_count",
    "transcript_unique_speakers",
    "transcript_speaker_turns",
    "transcript_question_count",
    "transcript_exclamation_count",
    "transcript_speech_coverage_ratio",
    "short_title_char_count",
    "short_title_question_count",
    "short_title_exclamation_count",
    "short_title_has_number",
    "long_title_char_count",
]

VPICK_FEATURES = [
    "vpick_scene_count",
    "vpick_scene_change_rate_per_min",
    "vpick_scene_coverage_ratio",
    "vpick_fallback_scene_ratio",
    "vpick_person_count",
    "vpick_speech_count",
    "vpick_speech_rate_per_min",
    "vpick_unique_speakers",
    "vpick_speech_coverage_ratio",
    "vpick_description_char_count",
    "vpick_description_hangul_ratio",
    "vpick_description_mojibake_ratio",
    "vpick_start_boundary_distance_sec",
    "vpick_end_boundary_distance_sec",
    "vpick_start_aligned_2s",
    "vpick_end_aligned_2s",
    "vpick_asset_duration_sec",
    "vpick_asset_resolution",
    "vpick_asset_person_count",
    "vpick_asset_fallback_count",
]

FEATURE_SETS = {
    "judge_only": JUDGE_FEATURES,
    "metadata_only": METADATA_FEATURES,
    "vpick_only": VPICK_FEATURES,
    "judge_plus_metadata": JUDGE_FEATURES + METADATA_FEATURES,
    "combined_with_vpick": JUDGE_FEATURES + METADATA_FEATURES + VPICK_FEATURES,
}

LOG_FEATURES = {
    "transcript_char_count",
    "transcript_line_count",
    "transcript_timed_line_count",
    "transcript_speaker_turns",
    "vpick_scene_count",
    "vpick_scene_change_rate_per_min",
    "vpick_person_count",
    "vpick_speech_count",
    "vpick_speech_rate_per_min",
    "vpick_description_char_count",
    "vpick_asset_duration_sec",
    "vpick_asset_person_count",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def join_judge(
    feature_rows: list[dict[str, str]],
    judge_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    judge_by_id = {row["candidate_id"]: row for row in judge_rows}
    missing = [row["candidate_id"] for row in feature_rows if row["candidate_id"] not in judge_by_id]
    if missing:
        raise ValueError(f"Missing Judge scores for {len(missing)} candidates: {missing[:5]}")
    joined = []
    for row in feature_rows:
        merged = dict(row)
        judge = judge_by_id[row["candidate_id"]]
        for feature in JUDGE_FEATURES:
            merged[feature] = judge.get(feature, "")
        joined.append(merged)
    return joined


def feature_matrix(rows: list[dict[str, str]], feature_names: list[str]) -> np.ndarray:
    matrix = np.array(
        [
            [
                (
                    math.nan
                    if feature in VPICK_FEATURES and to_float(row.get("vpick_available")) != 1.0
                    else to_float(row.get(feature))
                )
                for feature in feature_names
            ]
            for row in rows
        ],
        dtype=float,
    )
    for index, feature in enumerate(feature_names):
        if feature in LOG_FEATURES:
            values = matrix[:, index]
            matrix[:, index] = np.where(np.isfinite(values), np.log1p(np.maximum(values, 0.0)), values)
    return matrix


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(train_x), train_x, medians)
    test = np.where(np.isfinite(test_x), test_x, medians)
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    train = (train - means) / stds
    test = (test - means) / stds
    train_design = np.column_stack([np.ones(len(train)), train])
    test_design = np.column_stack([np.ones(len(test)), test])
    penalty = np.eye(train_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(train_design.T @ train_design + penalty) @ train_design.T @ train_y
    return test_design @ weights, weights, means, stds


def lolo_predictions(
    rows: list[dict[str, str]],
    feature_names: list[str],
    alpha: float,
) -> np.ndarray:
    x = feature_matrix(rows, feature_names)
    y = np.array([to_float(row["channel_performance_percentile"]) / 100.0 for row in rows])
    groups = np.array([row["long_video_id"] for row in rows])
    predictions = np.full(len(rows), np.nan)
    for group in sorted(set(groups)):
        test_mask = groups == group
        train_mask = ~test_mask
        fold_predictions, _, _, _ = fit_ridge(
            x[train_mask],
            y[train_mask],
            x[test_mask],
            alpha,
        )
        predictions[test_mask] = fold_predictions
    return predictions


def fit_zero_intercept_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(train_x), train_x, medians)
    test = np.where(np.isfinite(test_x), test_x, medians)
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    train = (train - means) / stds
    test = (test - means) / stds
    penalty = np.eye(train.shape[1]) * alpha
    weights = np.linalg.pinv(train.T @ train + penalty) @ train.T @ train_y
    return test @ weights


def stacked_vpick_predictions(
    rows: list[dict[str, str]],
    alpha: float,
) -> np.ndarray:
    base_names = FEATURE_SETS["judge_plus_metadata"]
    base_x = feature_matrix(rows, base_names)
    vpick_x = feature_matrix(rows, VPICK_FEATURES)
    y = np.array([to_float(row["channel_performance_percentile"]) / 100.0 for row in rows])
    groups = np.array([row["long_video_id"] for row in rows])
    covered = np.array([to_float(row.get("vpick_available")) == 1.0 for row in rows])
    predictions = np.full(len(rows), np.nan)

    for group in sorted(set(groups)):
        test_mask = groups == group
        train_mask = ~test_mask
        combined_base, _, _, _ = fit_ridge(
            base_x[train_mask],
            y[train_mask],
            np.vstack([base_x[train_mask], base_x[test_mask]]),
            alpha,
        )
        train_count = int(train_mask.sum())
        train_base = combined_base[:train_count]
        test_base = combined_base[train_count:]
        fold_predictions = test_base.copy()
        covered_train = train_mask & covered
        covered_test_local = covered[test_mask]
        if covered_train.sum() >= 8 and covered_test_local.any():
            residual = y[train_mask] - train_base
            residual = residual - float(np.mean(residual[covered[train_mask]]))
            correction = fit_zero_intercept_ridge(
                vpick_x[covered_train],
                residual[covered[train_mask]],
                vpick_x[test_mask][covered_test_local],
                alpha * 2.0,
            )
            correction = np.clip(correction, -0.25, 0.25)
            fold_predictions[covered_test_local] += correction
        predictions[test_mask] = fold_predictions
    return predictions


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + end - 1) / 2.0 + 1.0
        ranks[order[position:end]] = average
        position = end
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return math.nan
    wins = 0.0
    for pos_score in positive:
        wins += float(np.sum(pos_score > negative))
        wins += 0.5 * float(np.sum(pos_score == negative))
    return wins / (len(positive) * len(negative))


def pairwise_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    return binary_auc(labels, scores)


def mean_valid(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    return float(np.mean(valid)) if valid else math.nan


def evaluate_predictions(
    rows: list[dict[str, str]],
    predictions: np.ndarray,
    model_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    percentiles = np.array([to_float(row["channel_performance_percentile"]) for row in rows])
    labels = np.array([1 if row["performance_label"] == "pos" else 0 for row in rows])
    channels = np.array([row["channel_name"] for row in rows])
    channel_rows: list[dict[str, Any]] = []
    for channel in sorted(set(channels)):
        mask = channels == channel
        channel_auc = binary_auc(labels[mask], predictions[mask])
        channel_spearman = spearman(percentiles[mask], predictions[mask])
        channel_rows.append(
            {
                "model": model_name,
                "channel_name": channel,
                "candidate_count": int(mask.sum()),
                "pos_count": int(labels[mask].sum()),
                "neg_count": int(mask.sum() - labels[mask].sum()),
                "percentile_span": float(np.ptp(percentiles[mask])),
                "auc_pos_vs_neg": channel_auc,
                "spearman_percentile": channel_spearman,
            }
        )
    stable = [row for row in channel_rows if row["percentile_span"] >= 50.0]
    covered_mask = np.array([to_float(row.get("vpick_available")) == 1.0 for row in rows])
    summary = {
        "model": model_name,
        "feature_count": len(FEATURE_SETS[model_name]),
        "candidate_count": len(rows),
        "macro_channel_auc": mean_valid([row["auc_pos_vs_neg"] for row in channel_rows]),
        "stable_channel_macro_auc": mean_valid([row["auc_pos_vs_neg"] for row in stable]),
        "macro_channel_spearman": mean_valid(
            [row["spearman_percentile"] for row in channel_rows]
        ),
        "micro_pairwise_accuracy": pairwise_accuracy(labels, predictions),
        "pooled_spearman_percentile": spearman(percentiles, predictions),
        "vpick_covered_count": int(covered_mask.sum()),
        "vpick_covered_auc": (
            binary_auc(labels[covered_mask], predictions[covered_mask])
            if covered_mask.sum() >= 2
            else math.nan
        ),
        "vpick_covered_spearman": (
            spearman(percentiles[covered_mask], predictions[covered_mask])
            if covered_mask.sum() >= 2
            else math.nan
        ),
    }
    return summary, channel_rows


def final_coefficients(
    rows: list[dict[str, str]],
    feature_names: list[str],
    alpha: float,
) -> list[dict[str, Any]]:
    x = feature_matrix(rows, feature_names)
    y = np.array([to_float(row["channel_performance_percentile"]) / 100.0 for row in rows])
    _, weights, _, stds = fit_ridge(x, y, x[:1], alpha)
    coefficients = [
        {
            "feature": feature,
            "standardized_coefficient": float(weight),
            "training_std": float(std),
        }
        for feature, weight, std in zip(feature_names, weights[1:], stds)
    ]
    return sorted(coefficients, key=lambda row: abs(row["standardized_coefficient"]), reverse=True)


def format_metric(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.3f}" if math.isfinite(number) else "NA"


def write_report(
    path: Path,
    summaries: list[dict[str, Any]],
    best_model: str,
    alpha: float,
) -> None:
    lines = [
        "# Vpick Judge + Channel-relative Performance Ranker",
        "",
        "## 설계",
        "",
        "- 1단계 Judge는 영상 구간 자체의 완결성·흥미도·경계 자연스러움을 평가한다.",
        "- 2단계 Ranker는 Judge 점수, 제목/길이/대사 구조, Vpick 장면 분석 특성으로 채널 내 성과 백분위를 예측한다.",
        "- 조회수, 좋아요, 성과 백분위, pos/neg 라벨, 채널명은 Ranker 입력 특성에서 제외했다.",
        "- 같은 롱폼의 다른 숏폼이 학습·검증에 동시에 들어가지 않도록 롱폼 단위 leave-one-out 교차검증을 사용했다.",
        f"- 표본이 60개뿐이므로 CPU 기반 Ridge(alpha={alpha:g})를 사용했다.",
        "",
        "## 절제실험",
        "",
        "| 특성 조합 | 채널 Macro AUC | 안정 채널 Macro AUC | 채널 Macro Spearman | 전체 pairwise |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {auc} | {stable} | {rho} | {pairwise} |".format(
                model=row["model"],
                auc=format_metric(row["macro_channel_auc"]),
                stable=format_metric(row["stable_channel_macro_auc"]),
                rho=format_metric(row["macro_channel_spearman"]),
                pairwise=format_metric(row["micro_pairwise_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            f"교차검증 기준 최종 후보는 **{best_model}**이다.",
            "",
            "## 해석 제한",
            "",
            "- 이 Ranker는 60개 표본에서의 탐색적 결과이며 독립 외부 테스트셋 성능이 아니다.",
            "- Vpick 장면 JSON은 현재 60개 전부를 덮지 않으므로 `vpick_available`을 명시하고 결측 자체를 숨기지 않았다.",
            "- 모델 선택 뒤 같은 60개로 보고한 수치는 낙관적일 수 있다. 다음 데이터 수집분은 프롬프트와 가중치를 고정한 최종 테스트셋으로 남겨야 한다.",
            "- 성과 예측과 콘텐츠 품질 평가는 분리한다. Ranker의 낮은 점수가 곧 영상 자체가 나쁘다는 뜻은 아니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()

    rows = join_judge(read_csv(args.features), read_csv(args.judge))
    summaries: list[dict[str, Any]] = []
    channel_metrics: list[dict[str, Any]] = []
    prediction_rows = [
        {
            "candidate_id": row["candidate_id"],
            "pair_id": row["pair_id"],
            "long_video_id": row["long_video_id"],
            "short_video_id": row["short_video_id"],
            "channel_name": row["channel_name"],
            "performance_label": row["performance_label"],
            "channel_performance_percentile": row["channel_performance_percentile"],
            "vpick_available": row["vpick_available"],
        }
        for row in rows
    ]
    model_predictions: dict[str, np.ndarray] = {}

    for model_name, feature_names in FEATURE_SETS.items():
        if model_name == "combined_with_vpick":
            predictions = stacked_vpick_predictions(rows, args.alpha)
        else:
            predictions = lolo_predictions(rows, feature_names, args.alpha)
        model_predictions[model_name] = predictions
        summary, per_channel = evaluate_predictions(rows, predictions, model_name)
        summaries.append(summary)
        channel_metrics.extend(per_channel)
        for output_row, prediction in zip(prediction_rows, predictions):
            output_row[f"pred_{model_name}"] = float(prediction)

    best = max(
        summaries,
        key=lambda row: (
            -math.inf
            if not math.isfinite(row["macro_channel_auc"])
            else row["macro_channel_auc"],
            -math.inf
            if not math.isfinite(row["macro_channel_spearman"])
            else row["macro_channel_spearman"],
        ),
    )
    coefficients = final_coefficients(rows, FEATURE_SETS[best["model"]], args.alpha)
    sensitivity_rows: list[dict[str, Any]] = []
    for sensitivity_alpha in (1.0, 3.0, 10.0, 30.0, 100.0):
        for model_name, feature_names in FEATURE_SETS.items():
            if model_name == "combined_with_vpick":
                predictions = stacked_vpick_predictions(rows, sensitivity_alpha)
            else:
                predictions = lolo_predictions(rows, feature_names, sensitivity_alpha)
            summary, _ = evaluate_predictions(rows, predictions, model_name)
            sensitivity_rows.append({"alpha": sensitivity_alpha, **summary})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ranker_ablation_summary.csv", summaries)
    write_csv(args.output_dir / "ranker_channel_metrics.csv", channel_metrics)
    write_csv(args.output_dir / "ranker_cross_validated_predictions_PRIVATE.csv", prediction_rows)
    write_csv(args.output_dir / "ranker_best_model_coefficients.csv", coefficients)
    write_csv(args.output_dir / "ranker_alpha_sensitivity.csv", sensitivity_rows)
    result = {
        "cross_validation": "leave_one_longform_out",
        "target": "channel_performance_percentile",
        "label_usage": "evaluation_only",
        "forbidden_input_columns": [
            "short_views",
            "short_likes",
            "short_like_rate",
            "channel_performance_percentile",
            "performance_label",
            "channel_name",
        ],
        "alpha": args.alpha,
        "best_model": best["model"],
        "models": summaries,
    }
    (args.output_dir / "ranker_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "README_ranker.md",
        summaries,
        best["model"],
        args.alpha,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
