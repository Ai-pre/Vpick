from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from performance_judge_v1 import (
    CODEX_FEATURES,
    GEMINI_FEATURES,
    RUBRIC_FEATURES,
    STRUCTURE_FEATURES,
    bootstrap_group_auc,
    binary_auc,
    evaluate_scores,
    extract_structure_features,
    feature_matrix,
    fit_logistic,
    grouped_cv_predictions,
    read_csv,
    read_jsonl,
    serializable_model,
    write_csv,
    write_json,
    write_jsonl,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    ROOT / "deliverables" / "2026-07-24" / "performance_judge_v1"
)
DEFAULT_CODEX = (
    ROOT
    / "deliverables"
    / "2026-07-23"
    / "vpick_llm_judge_v7_codex_scores_60.csv"
)
DEFAULT_GEMINI = (
    ROOT
    / "results"
    / "performance_judge_v1_gemini"
    / "pointwise_gemini-3.1-flash-lite_judgments.jsonl"
)
DEFAULT_CLAUDE = (
    ROOT
    / "results"
    / "performance_judge_v1_claude"
    / "pointwise_claude-opus-4-8_judgments.jsonl"
)
DEFAULT_CLAUDE_V7 = (
    ROOT
    / "results"
    / "performance_judge_v1_claude_v7"
    / "reference_judge_v7_scores.csv"
)


def build_training_rows(
    candidates: list[dict[str, Any]],
    targets: list[dict[str, str]],
    codex_rows: list[dict[str, str]],
    gemini_rows: list[dict[str, Any]],
    claude_rows: list[dict[str, Any]],
    claude_v7_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    def unique_index(
        source_rows: list[dict[str, Any]],
        source_name: str,
    ) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for source_row in source_rows:
            candidate_id = str(source_row["candidate_id"])
            if candidate_id in index:
                raise ValueError(
                    f"Duplicate candidate_id in {source_name}: {candidate_id}"
                )
            index[candidate_id] = source_row
        return index

    targets_by_id = unique_index(targets, "targets")
    codex_by_source_id = unique_index(codex_rows, "Codex scores")
    gemini_by_id = unique_index(gemini_rows, "Gemini scores")
    claude_by_id = unique_index(claude_rows, "Claude scores")
    claude_v7_by_source_id = unique_index(
        claude_v7_rows,
        "Claude v7 scores",
    )
    rows: list[dict[str, Any]] = []
    missing_codex = 0
    missing_gemini = 0
    missing_claude = 0
    missing_claude_v7 = 0

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in targets_by_id:
            raise ValueError(f"Missing private target for {candidate_id}")
        target = targets_by_id[candidate_id]
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "source_candidate_id": target["source_candidate_id"],
            "pair_id": target["pair_id"],
            "longform_id": target["longform_id"],
            "short_video_id": target["short_video_id"],
            "channel_name": target["channel_name"],
            "performance_label": target["performance_label"],
            "target": 1 if target["performance_label"] == "pos" else 0,
            "channel_performance_percentile": target[
                "channel_performance_percentile"
            ],
            "evidence_provider": target["evidence_provider"],
        }
        row.update(extract_structure_features(candidate))

        codex = codex_by_source_id.get(target["source_candidate_id"])
        if codex and codex.get("verdict") == "score":
            for feature in CODEX_FEATURES:
                row[feature] = codex.get(feature, "")
        else:
            missing_codex += 1

        gemini = gemini_by_id.get(candidate_id)
        if gemini and gemini.get("verdict") == "score":
            for feature in GEMINI_FEATURES:
                row[feature] = gemini.get(feature, "")
            row["gemini_highlight_quality_score_100"] = gemini.get(
                "highlight_quality_score_100",
                "",
            )
        else:
            missing_gemini += 1

        claude = claude_by_id.get(candidate_id)
        if claude and claude.get("verdict") == "score":
            for feature in RUBRIC_FEATURES:
                row[f"claude_{feature}"] = claude.get(feature, "")
            row["claude_highlight_quality_score_100"] = claude.get(
                "highlight_quality_score_100",
                "",
            )
        else:
            missing_claude += 1

        claude_v7 = claude_v7_by_source_id.get(target["source_candidate_id"])
        if claude_v7 and claude_v7.get("verdict") == "score":
            for feature in CODEX_FEATURES:
                row[f"claude_v7_{feature}"] = claude_v7.get(feature, "")
            row["claude_v7_checklist_score_100"] = claude_v7.get(
                "checklist_score_100",
                "",
            )
        else:
            missing_claude_v7 += 1
        rows.append(row)

    if len(rows) != 60:
        raise ValueError(f"Expected 60 actual Shorts, got {len(rows)}")
    label_counts = {
        "pos": sum(row["target"] == 1 for row in rows),
        "neg": sum(row["target"] == 0 for row in rows),
    }
    if label_counts != {"pos": 30, "neg": 30}:
        raise ValueError(f"Expected balanced 30/30 labels, got {label_counts}")
    return rows, {
        "missing_codex": missing_codex,
        "missing_gemini": missing_gemini,
        "missing_claude": missing_claude,
        "missing_claude_v7": missing_claude_v7,
    }


def available_feature_sets(
    rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    def coverage(features: list[str]) -> float:
        expected = len(rows) * len(features)
        present = sum(
            row.get(feature, "") not in ("", None)
            for row in rows
            for feature in features
        )
        return present / expected if expected else 0.0

    feature_sets: dict[str, list[str]] = {
        "structure_only": STRUCTURE_FEATURES,
    }
    # A small number of abstentions are imputed from the training fold only.
    if coverage(CODEX_FEATURES) >= 0.9:
        feature_sets.update(
            {
                "codex_only": CODEX_FEATURES,
                "codex_plus_structure": CODEX_FEATURES + STRUCTURE_FEATURES,
            }
        )
    if coverage(GEMINI_FEATURES) >= 0.9:
        feature_sets.update(
            {
                "gemini_only": GEMINI_FEATURES,
                "gemini_plus_structure": GEMINI_FEATURES + STRUCTURE_FEATURES,
            }
        )
    claude_features = [f"claude_{feature}" for feature in RUBRIC_FEATURES]
    if coverage(claude_features) >= 0.9:
        feature_sets.update(
            {
                "claude_only": claude_features,
                "claude_plus_structure": claude_features + STRUCTURE_FEATURES,
            }
        )
        if coverage(CODEX_FEATURES) >= 0.9:
            feature_sets.update(
                {
                    "codex_plus_claude": CODEX_FEATURES + claude_features,
                    "codex_plus_claude_plus_structure": (
                        CODEX_FEATURES + claude_features + STRUCTURE_FEATURES
                    ),
                }
            )
    claude_v7_features = [f"claude_v7_{feature}" for feature in CODEX_FEATURES]
    if coverage(claude_v7_features) >= 0.9:
        feature_sets.update(
            {
                "claude_v7_only": claude_v7_features,
                "claude_v7_plus_structure": (
                    claude_v7_features + STRUCTURE_FEATURES
                ),
            }
        )
        if coverage(CODEX_FEATURES) >= 0.9:
            feature_sets["codex_plus_claude_v7"] = (
                CODEX_FEATURES + claude_v7_features
            )
    return feature_sets


def compare_models(
    rows: list[dict[str, Any]],
    feature_sets: dict[str, list[str]],
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    comparison: list[dict[str, Any]] = []
    predictions_by_model: dict[str, dict[str, np.ndarray]] = {}
    for model_name, feature_names in feature_sets.items():
        lolo = grouped_cv_predictions(
            rows,
            feature_names,
            group_key="longform_id",
            alpha=alpha,
        )
        loco = grouped_cv_predictions(
            rows,
            feature_names,
            group_key="channel_name",
            alpha=alpha,
        )
        lolo_metrics = evaluate_scores(rows, lolo)
        loco_metrics = evaluate_scores(rows, loco)
        ci_low, ci_high = bootstrap_group_auc(rows, lolo)
        comparison.append(
            {
                "model": model_name,
                "evaluation_provenance": (
                    "non_isolated_codex_session"
                    if "codex" in model_name
                    else "isolated_api_or_deterministic"
                ),
                "deployment_eligible": "codex" not in model_name,
                "feature_count": len(feature_names),
                "lolo_pooled_auc": lolo_metrics["pooled_auc"],
                "lolo_pooled_auc_ci95_low": ci_low,
                "lolo_pooled_auc_ci95_high": ci_high,
                "lolo_macro_channel_auc": lolo_metrics["macro_channel_auc"],
                "lolo_balanced_accuracy_at_0_5": lolo_metrics[
                    "balanced_accuracy_at_0_5"
                ],
                "lolo_pooled_percentile_spearman": lolo_metrics[
                    "pooled_percentile_spearman"
                ],
                "lolo_macro_channel_percentile_spearman": lolo_metrics[
                    "macro_channel_percentile_spearman"
                ],
                "loco_pooled_auc": loco_metrics["pooled_auc"],
                "loco_balanced_accuracy_at_0_5": loco_metrics[
                    "balanced_accuracy_at_0_5"
                ],
                "loco_pooled_percentile_spearman": loco_metrics[
                    "pooled_percentile_spearman"
                ],
            }
        )
        predictions_by_model[model_name] = {"lolo": lolo, "loco": loco}
    comparison.sort(
        key=lambda row: (
            -(
                row["lolo_macro_channel_auc"]
                if math.isfinite(row["lolo_macro_channel_auc"])
                else -1.0
            ),
            -row["lolo_pooled_auc"],
            row["feature_count"],
        )
    )
    return comparison, predictions_by_model


def choose_deploy_model(
    comparison: list[dict[str, Any]],
) -> tuple[str, str, str]:
    single_provider_models = {
        "structure_only",
        "gemini_only",
        "claude_only",
        "claude_v7_only",
    }
    singles = [
        row
        for row in comparison
        if row["model"] in single_provider_models
        and bool(row.get("deployment_eligible"))
    ]
    if not singles:
        return (
            comparison[0]["model"],
            "rejected",
            "격리된 단일 제공자 모델 결과가 없어 배포할 수 없다.",
        )
    best_single = singles[0]
    passed = (
        best_single["lolo_pooled_auc"] >= 0.60
        and best_single["lolo_macro_channel_auc"] >= 0.55
        and best_single["lolo_pooled_auc_ci95_low"] >= 0.50
    )
    if passed:
        return (
            best_single["model"],
            "validated",
            "격리 API 단일 모델이 pooled AUC 0.60 이상, 채널별 평균 AUC "
            "0.55 이상, bootstrap 신뢰구간 하한 0.50 이상을 모두 통과했다.",
        )
    return (
        best_single["model"],
        "rejected",
        "최고 격리 API 단일 모델이 pooled AUC 0.60 이상, 채널별 평균 AUC "
        "0.55 이상, bootstrap 신뢰구간 하한 0.50 이상을 통과하지 못했다. "
        "Codex 세션 결과는 비공개 라벨로부터 실행 맥락이 격리되지 않아 "
        "배포 후보에서 제외했다.",
    )


def direct_judge_diagnostics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for judge_name, score_field in (
        ("gemini_raw_weighted_total", "gemini_highlight_quality_score_100"),
        ("claude_raw_weighted_total", "claude_highlight_quality_score_100"),
        ("claude_v7_raw_checklist", "claude_v7_checklist_score_100"),
    ):
        covered_rows = [
            row for row in rows if row.get(score_field, "") not in ("", None)
        ]
        if len(covered_rows) < 10:
            continue
        scores = np.array(
            [float(row[score_field]) / 100.0 for row in covered_rows]
        )
        metrics = evaluate_scores(covered_rows, scores)
        ci_low, ci_high = bootstrap_group_auc(covered_rows, scores)
        output.append(
            {
                "judge": judge_name,
                "candidate_count": len(covered_rows),
                "pooled_auc": metrics["pooled_auc"],
                "pooled_auc_ci95_low": ci_low,
                "pooled_auc_ci95_high": ci_high,
                "macro_channel_auc": metrics["macro_channel_auc"],
                "balanced_accuracy_at_0_5": metrics[
                    "balanced_accuracy_at_0_5"
                ],
                "pooled_percentile_spearman": metrics[
                    "pooled_percentile_spearman"
                ],
                "macro_channel_percentile_spearman": metrics[
                    "macro_channel_percentile_spearman"
                ],
            }
        )
    return output


def prediction_rows(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        result = {
            "candidate_id": row["candidate_id"],
            "source_candidate_id": row["source_candidate_id"],
            "pair_id": row["pair_id"],
            "longform_id": row["longform_id"],
            "short_video_id": row["short_video_id"],
            "channel_name": row["channel_name"],
            "performance_label_PRIVATE": row["performance_label"],
            "channel_performance_percentile_PRIVATE": row[
                "channel_performance_percentile"
            ],
        }
        for model_name, values in predictions.items():
            result[f"{model_name}_lolo_score"] = round(
                float(values["lolo"][index]) * 100.0,
                4,
            )
            result[f"{model_name}_loco_score"] = round(
                float(values["loco"][index]) * 100.0,
                4,
            )
        output.append(result)
    return output


def channel_metric_rows(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    channels = sorted({str(row["channel_name"]) for row in rows})
    for model_name, model_predictions in predictions.items():
        for validation_scheme, scores in model_predictions.items():
            for channel in channels:
                indices = [
                    index
                    for index, row in enumerate(rows)
                    if row["channel_name"] == channel
                ]
                labels = np.array(
                    [int(rows[index]["target"]) for index in indices]
                )
                percentiles = np.array(
                    [
                        float(rows[index]["channel_performance_percentile"])
                        for index in indices
                    ]
                )
                channel_scores = scores[indices]
                output.append(
                    {
                        "model": model_name,
                        "validation_scheme": validation_scheme,
                        "channel_name": channel,
                        "candidate_count": len(indices),
                        "pos_count": int(labels.sum()),
                        "neg_count": int(len(labels) - labels.sum()),
                        "auc_pos_vs_neg": binary_auc(labels, channel_scores),
                        "percentile_spearman": spearman(
                            percentiles,
                            channel_scores,
                        ),
                    }
                )
    return output


def choose_decision_thresholds(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    min_precision: float = 0.75,
    min_count: int = 5,
) -> dict[str, float | int | str]:
    labels = np.array([int(row["target"]) for row in rows])
    high_threshold = 0.65
    low_threshold = 0.35
    high_count = 0
    low_count = 0
    high_precision = math.nan
    low_precision = math.nan

    for threshold in sorted(set(float(score) for score in scores)):
        mask = scores >= threshold
        if int(mask.sum()) < min_count:
            continue
        precision = float(np.mean(labels[mask] == 1))
        if precision >= min_precision:
            high_threshold = threshold
            high_count = int(mask.sum())
            high_precision = precision
            break

    for threshold in sorted(
        set(float(score) for score in scores),
        reverse=True,
    ):
        mask = scores <= threshold
        if int(mask.sum()) < min_count:
            continue
        precision = float(np.mean(labels[mask] == 0))
        if precision >= min_precision:
            low_threshold = threshold
            low_count = int(mask.sum())
            low_precision = precision
            break

    if low_threshold >= high_threshold:
        low_threshold, high_threshold = 0.35, 0.65
        low_count = int(np.sum(scores <= low_threshold))
        high_count = int(np.sum(scores >= high_threshold))
        low_precision = (
            float(np.mean(labels[scores <= low_threshold] == 0))
            if low_count
            else math.nan
        )
        high_precision = (
            float(np.mean(labels[scores >= high_threshold] == 1))
            if high_count
            else math.nan
        )

    return {
        "source": "LOLO cross-validated predictions",
        "target_precision": min_precision,
        "minimum_tier_count": min_count,
        "low_max_score_0_100": round(low_threshold * 100.0, 4),
        "high_min_score_0_100": round(high_threshold * 100.0, 4),
        "observed_low_precision": (
            round(low_precision, 4) if math.isfinite(low_precision) else ""
        ),
        "observed_high_precision": (
            round(high_precision, 4) if math.isfinite(high_precision) else ""
        ),
        "observed_low_count": low_count,
        "observed_high_count": high_count,
    }


def report_text(
    comparison: list[dict[str, Any]],
    best_model: str,
    feature_names: list[str],
    coverage: dict[str, int],
    decision_policy: dict[str, Any],
    direct_diagnostics: list[dict[str, Any]],
    channel_metrics: list[dict[str, Any]],
    selection_reason: str,
    deployment_status: str,
) -> str:
    lines = [
        "# Performance Judge v1",
        "",
        "## 목적",
        "",
        "새로 생성된 단일 숏폼 후보가 같은 채널 안에서 고성과군에 가까운지를 "
        "0~100 점수로 추정한다. 실제 조회수나 정확한 백분위를 예측하는 모델이 아니다.",
        "",
        "## 데이터와 누수 방지",
        "",
        "- 실제 공개 롱폼-숏폼 60개만 사용: Pos 30, Neg 30",
        "- LLM 입력에는 채널명, 조회수, 백분위, Pos/Neg 라벨을 넣지 않음",
        "- 같은 롱폼의 다른 숏폼이 학습과 검증에 동시에 들어가지 않도록 "
        "Leave-One-Longform-Out 교차검증 사용",
        "- Leave-One-Channel-Out은 처음 보는 채널에 대한 별도 스트레스 테스트",
        "",
        "## 모델 역할",
        "",
        "LLM은 완결성·훅·payoff·경계 등 내용 특징을 추출한다. 최종 성과 점수는 "
        "그 특징과 자막/Vpick 구조 특징을 결합한 L2 정규화 로지스틱 모델이 계산한다.",
        "",
        "## 비교 결과",
        "",
        "| 모델 | 격리·배포 적격 | 특징 수 | LOLO AUC | 채널별 평균 AUC | LOCO AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | "
            f"{'예' if row['deployment_eligible'] else '아니오'} | "
            f"{row['feature_count']} | "
            f"{row['lolo_pooled_auc']:.3f} | "
            f"{row['lolo_macro_channel_auc']:.3f} | "
            f"{row['loco_pooled_auc']:.3f} |"
        )
    if direct_diagnostics:
        lines.extend(
            [
                "",
                "## LLM 고정 총점 진단",
                "",
                "| Judge | 평가 수 | raw AUC | 채널별 평균 AUC |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in direct_diagnostics:
            lines.append(
                f"| {row['judge']} | {row['candidate_count']} | "
                f"{row['pooled_auc']:.3f} | "
                f"{row['macro_channel_auc']:.3f} |"
            )
    best = next(row for row in comparison if row["model"] == best_model)
    best_channel_metrics = [
        row
        for row in channel_metrics
        if row["model"] == best_model
        and row["validation_scheme"] == "lolo"
    ]
    lines.extend(
        [
            "",
            "## 검증 판정",
            "",
            f"- 격리 API 후보: `{best_model}`",
            f"- 배포 상태: `{deployment_status}`",
            f"- 선택 규칙: {selection_reason}",
            f"- 특징: {', '.join(feature_names)}",
            f"- LOLO pooled AUC: {best['lolo_pooled_auc']:.3f} "
            f"(longform bootstrap 95% CI "
            f"{best['lolo_pooled_auc_ci95_low']:.3f}~"
            f"{best['lolo_pooled_auc_ci95_high']:.3f})",
            f"- 채널별 평균 AUC: {best['lolo_macro_channel_auc']:.3f}",
            f"- 처음 보는 채널 LOCO AUC: {best['loco_pooled_auc']:.3f}",
            "",
            "### 채널별 LOLO",
            "",
            "| 채널 | N | Pos/Neg | AUC | 백분위 Spearman |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in best_channel_metrics:
        lines.append(
            f"| {row['channel_name']} | {row['candidate_count']} | "
            f"{row['pos_count']}/{row['neg_count']} | "
            f"{row['auc_pos_vs_neg']:.3f} | "
            f"{row['percentile_spearman']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "30/30 극단 표본으로 학습했으므로 출력값은 운영 환경의 실제 확률이 아니다. "
            f"{decision_policy['high_min_score_0_100']}점 이상은 고성과 신호, "
            f"{decision_policy['low_max_score_0_100']}점 이하는 저성과 신호, "
            "그 사이는 불확실로 사용한다. 두 경계는 LOLO 예측에서 각 극단군 "
            f"정밀도 {decision_policy['target_precision']:.0%}를 목표로 정했다. "
            "데이터가 60개뿐이므로 신뢰구간과 모델 간 차이를 함께 보고한다. "
            "동일한 60개 교차검증으로 특징 세트까지 선택했으므로 현재 수치는 탐색적이며, "
            "향후 새로 수집한 미사용 테스트셋에서 한 번 더 확정해야 한다.",
            (
                "현재 artifact는 검증 실패 상태이므로 위 경계는 진단용일 뿐이다. "
                "추론 CLI는 `--allow-unvalidated`를 명시하지 않으면 실행을 거부한다."
                if deployment_status != "validated"
                else ""
            ),
            "",
            "## 입력 커버리지",
            "",
            f"- Codex/GPT 계열 점수 누락: {coverage['missing_codex']}개",
            f"- Gemini 점수 누락: {coverage['missing_gemini']}개",
            f"- Claude 점수 누락: {coverage['missing_claude']}개",
            f"- Claude v7 점수 누락: {coverage['missing_claude_v7']}개",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and validate the single-Short performance Judge v1."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--codex-scores", type=Path, default=DEFAULT_CODEX)
    parser.add_argument(
        "--gemini-scores",
        type=Path,
        nargs="+",
        default=[DEFAULT_GEMINI],
    )
    parser.add_argument(
        "--claude-scores",
        type=Path,
        nargs="+",
        default=[DEFAULT_CLAUDE],
    )
    parser.add_argument(
        "--claude-v7-scores",
        type=Path,
        nargs="+",
        default=[DEFAULT_CLAUDE_V7],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--alpha", type=float, default=2.0)
    args = parser.parse_args()

    candidates = read_jsonl(args.dataset_dir / "candidates_blind.jsonl")
    targets = read_csv(args.dataset_dir / "candidate_targets_PRIVATE.csv")
    codex_rows = read_csv(args.codex_scores) if args.codex_scores.exists() else []
    gemini_rows = [
        row
        for path in args.gemini_scores
        if path.exists()
        for row in read_jsonl(path)
    ]
    claude_rows = [
        row
        for path in args.claude_scores
        if path.exists()
        for row in read_jsonl(path)
    ]
    claude_v7_rows = [
        row
        for path in args.claude_v7_scores
        if path.exists()
        for row in read_csv(path)
    ]
    rows, coverage = build_training_rows(
        candidates,
        targets,
        codex_rows,
        gemini_rows,
        claude_rows,
        claude_v7_rows,
    )
    if len(claude_rows) == len(candidates):
        candidate_order = {
            str(candidate["candidate_id"]): index
            for index, candidate in enumerate(candidates)
        }
        write_jsonl(
            args.output_dir / "claude_opus_4_8_pointwise_scores_60.jsonl",
            sorted(
                claude_rows,
                key=lambda row: candidate_order[str(row["candidate_id"])],
            ),
        )
    if len(claude_v7_rows) == len(candidates):
        write_csv(
            args.output_dir / "claude_opus_4_8_v7_scores_60.csv",
            claude_v7_rows,
        )
    feature_sets = available_feature_sets(rows)
    comparison, predictions = compare_models(rows, feature_sets, args.alpha)
    direct_diagnostics = direct_judge_diagnostics(rows)
    channel_metrics = channel_metric_rows(rows, predictions)
    best_model, deployment_status, selection_reason = choose_deploy_model(
        comparison
    )
    best_features = feature_sets[best_model]
    final_model = fit_logistic(
        feature_matrix(rows, best_features),
        np.array([row["target"] for row in rows], dtype=float),
        alpha=args.alpha,
    )
    best_validation = dict(
        next(row for row in comparison if row["model"] == best_model)
    )
    artifact = serializable_model(
        final_model,
        best_features,
        best_model,
        best_validation,
    )
    decision_policy = choose_decision_thresholds(
        rows,
        predictions[best_model]["lolo"],
    )
    artifact.update(
        {
            "deployment_status": deployment_status,
            "deployment_block_reason": (
                "" if deployment_status == "validated" else selection_reason
            ),
            "training_candidate_count": len(rows),
            "training_label_counts": {"pos": 30, "neg": 30},
            "training_unique_longforms": len(
                {row["longform_id"] for row in rows}
            ),
            "training_unique_channels": len(
                {row["channel_name"] for row in rows}
            ),
            "decision_policy": decision_policy,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "model_comparison.csv", comparison)
    write_csv(
        args.output_dir / "raw_judge_diagnostics.csv",
        direct_diagnostics,
    )
    write_csv(args.output_dir / "channel_metrics.csv", channel_metrics)
    write_csv(
        args.output_dir / "cross_validated_predictions_PRIVATE.csv",
        prediction_rows(rows, predictions),
    )
    write_csv(
        args.output_dir / "training_features_PRIVATE.csv",
        rows,
    )
    write_json(args.output_dir / "model_artifact.json", artifact)
    write_json(
        args.output_dir / "training_summary.json",
        {
            "best_model": best_model,
            "observed_leader": comparison[0]["model"],
            "deployment_status": deployment_status,
            "selection_reason": selection_reason,
            "best_features": best_features,
            "coverage": coverage,
            "comparison": comparison,
            "raw_judge_diagnostics": direct_diagnostics,
        },
    )
    (args.output_dir / "PERFORMANCE_JUDGE_REPORT.md").write_text(
        report_text(
            comparison,
            best_model,
            best_features,
            coverage,
            decision_policy,
            direct_diagnostics,
            channel_metrics,
            selection_reason,
            deployment_status,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "best_model": best_model,
                "observed_leader": comparison[0]["model"],
                "deployment_status": deployment_status,
                "selection_reason": selection_reason,
                "validation": best_validation,
                "coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
