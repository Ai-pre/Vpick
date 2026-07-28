from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train_performance_calibrator_v11 import (
    DEFAULT_PRIVATE_DIR,
    QUALITY_COLUMNS,
    ROOT,
    STRUCTURE_COLUMNS,
    finite_spearman,
    grouped_splits,
    json_safe,
    load_bundle,
    pairwise_accuracy,
    residualize,
    write_csv,
)
from train_performance_calibrator_v12 import (
    RankerSpec,
    candidate_reliability,
    fit_prepared,
    prepare_fold,
)


DEFAULT_CONFIG = ROOT / "config" / "performance_calibrator_v14_dev.json"
DEFAULT_PUBLIC_DIR = ROOT / "results" / "performance_calibrator_v14_dev"
DEFAULT_PRIVATE_OUTPUT = (
    DEFAULT_PRIVATE_DIR / "performance_calibrator_v14_dev"
)
DEFAULT_REPORT = ROOT / "reports" / "performance_calibrator_v14_dev_2026-07-28.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the nested, mid-sensitive v14 development ranker."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--private-output", type=Path, default=DEFAULT_PRIVATE_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def candidate_specs(config: dict[str, Any]) -> dict[str, RankerSpec]:
    weighting = config["pair_weighting"]
    shared = {
        "score_calibration": "train_ecdf",
        "engineered_numeric": False,
        "channel_balanced_pairs": bool(
            weighting["channel_balanced_pairs"]
        ),
        "min_gap": float(weighting["minimum_gap"]),
        "local_boost": float(weighting["local_pair_boost"]),
        "reliability_weighting": False,
        "local_gap_min": float(weighting["local_gap_min"]),
        "local_gap_max": float(weighting["local_gap_max"]),
        "mid_percentile_min": float(weighting["mid_percentile_min"]),
        "mid_percentile_max": float(weighting["mid_percentile_max"]),
        "mid_pair_boost": float(weighting["mid_pair_boost"]),
        "extreme_pair_weight": float(
            weighting["opposite_extreme_pair_weight"]
        ),
    }
    return {
        "normalized_char_clean_numeric025": RankerSpec(
            name="normalized_char_clean_numeric025",
            representation="concat_normalized_char",
            numeric_scale=0.25,
            **shared,
        ),
        "field_aware_clean_numeric025": RankerSpec(
            name="field_aware_clean_numeric025",
            representation="field_aware_char_word",
            numeric_scale=0.25,
            **shared,
        ),
        "field_aware_clean_text_only": RankerSpec(
            name="field_aware_clean_text_only",
            representation="field_aware_char_word",
            numeric_scale=0.0,
            **shared,
        ),
    }


def remove_proxy_features(bundle: Any, excluded: list[str]) -> tuple[Any, list[str]]:
    unknown = sorted(set(excluded) - set(STRUCTURE_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown excluded structure features: {unknown}")
    kept_structure = [
        column for column in STRUCTURE_COLUMNS if column not in set(excluded)
    ]
    all_columns = QUALITY_COLUMNS + STRUCTURE_COLUMNS
    kept_columns = QUALITY_COLUMNS + kept_structure
    indices = [all_columns.index(column) for column in kept_columns]
    cleaned = replace(
        bundle,
        quality_structure=bundle.quality_structure[:, indices],
    )
    return cleaned, kept_structure


def labels_from_targets(
    y: np.ndarray,
    mid_min: float,
    mid_max: float,
) -> np.ndarray:
    return np.where(
        y <= mid_min,
        "neg",
        np.where(y >= mid_max, "pos", "mid"),
    )


def within_label_pairwise_accuracy(
    y: np.ndarray,
    scores: np.ndarray,
    channels: np.ndarray,
    labels: np.ndarray,
    min_gap: float,
) -> tuple[float, int]:
    weighted_correct = 0.0
    pair_count = 0
    for label in ("neg", "mid", "pos"):
        mask = labels == label
        accuracy, count = pairwise_accuracy(
            y[mask],
            scores[mask],
            channels[mask],
            min_gap=min_gap,
        )
        if count and math.isfinite(accuracy):
            weighted_correct += accuracy * count
            pair_count += count
    return (
        weighted_correct / pair_count if pair_count else math.nan,
        pair_count,
    )


def development_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    channels: np.ndarray,
    mid_min: float,
    mid_max: float,
    min_gap: float,
    selection_weights: dict[str, float],
) -> dict[str, float | int]:
    labels = labels_from_targets(y, mid_min, mid_max)
    mid = labels == "mid"
    extremes = np.isin(labels, ["neg", "pos"])
    centered = finite_spearman(
        residualize(y, channels),
        residualize(scores, channels),
    )
    mid_centered = finite_spearman(
        residualize(y[mid], channels[mid]),
        residualize(scores[mid], channels[mid]),
    )
    mid_pooled = finite_spearman(y[mid], scores[mid])
    pair_accuracy, pair_count = pairwise_accuracy(
        y,
        scores,
        channels,
        min_gap=min_gap,
    )
    local_accuracy, local_count = pairwise_accuracy(
        y,
        scores,
        channels,
        min_gap=0.10,
        max_gap=0.40,
    )
    mid_accuracy, mid_pair_count = pairwise_accuracy(
        y[mid],
        scores[mid],
        channels[mid],
        min_gap=min_gap,
    )
    within_label, within_label_count = within_label_pairwise_accuracy(
        y,
        scores,
        channels,
        labels,
        min_gap,
    )
    extreme_labels = (labels[extremes] == "pos").astype(int)
    extreme_auc = (
        float(roc_auc_score(extreme_labels, scores[extremes]))
        if len(set(extreme_labels.tolist())) == 2
        else math.nan
    )
    mid_skill = (
        2.0 * mid_accuracy - 1.0
        if math.isfinite(mid_accuracy)
        else math.nan
    )
    local_skill = (
        2.0 * local_accuracy - 1.0
        if math.isfinite(local_accuracy)
        else math.nan
    )
    components = {
        "mid_channel_centered_spearman": mid_centered,
        "mid_pairwise_skill": mid_skill,
        "same_channel_local_pairwise_skill": local_skill,
    }
    selection_score = (
        sum(
            float(selection_weights[name]) * float(value)
            for name, value in components.items()
        )
        if all(math.isfinite(value) for value in components.values())
        else math.nan
    )
    return {
        "pooled_spearman": finite_spearman(y, scores),
        "channel_centered_spearman": centered,
        "mid_only_pooled_spearman": mid_pooled,
        "mid_only_channel_centered_spearman": mid_centered,
        "same_channel_pairwise_accuracy": pair_accuracy,
        "same_channel_pair_count": pair_count,
        "same_channel_local_pairwise_accuracy": local_accuracy,
        "same_channel_local_pair_count": local_count,
        "mid_only_pairwise_accuracy": mid_accuracy,
        "mid_only_pair_count": mid_pair_count,
        "within_label_pairwise_accuracy": within_label,
        "within_label_pair_count": within_label_count,
        "extremes_pos_neg_auc": extreme_auc,
        "selection_score": selection_score,
    }


def select_spec_and_c(
    bundle: Any,
    outer_train_indices: np.ndarray,
    specs: dict[str, RankerSpec],
    reliability: np.ndarray,
    c_values: list[float],
    inner_splits: int,
    seed: int,
    config: dict[str, Any],
) -> tuple[str, float, list[dict[str, Any]]]:
    local_count = len(outer_train_indices)
    splits = grouped_splits(
        bundle.groups[outer_train_indices],
        inner_splits,
        seed,
    )
    candidate_rows: list[dict[str, Any]] = []
    spec_order = list(specs)
    for spec_name, spec in specs.items():
        predictions = {
            c_value: np.full(local_count, np.nan, dtype=float)
            for c_value in c_values
        }
        for fold_index, (train_local, test_local) in enumerate(splits):
            train_indices = outer_train_indices[train_local]
            test_indices = outer_train_indices[test_local]
            prepared = prepare_fold(
                bundle,
                train_indices,
                test_indices,
                spec,
                reliability,
            )
            for c_value in c_values:
                predictions[c_value][test_local] = fit_prepared(
                    prepared,
                    c_value,
                    spec.score_calibration,
                    seed + fold_index,
                )
        for c_value, scores in predictions.items():
            metrics = development_metrics(
                bundle.y[outer_train_indices],
                scores,
                bundle.channels[outer_train_indices],
                spec.mid_percentile_min,
                spec.mid_percentile_max,
                spec.min_gap,
                config["inner_selection_weights"],
            )
            candidate_rows.append(
                {
                    "spec": spec_name,
                    "c_value": c_value,
                    **metrics,
                }
            )
    selected = max(
        candidate_rows,
        key=lambda row: (
            float(row["selection_score"])
            if math.isfinite(float(row["selection_score"]))
            else -math.inf,
            -spec_order.index(str(row["spec"])),
            -abs(math.log10(float(row["c_value"]))),
        ),
    )
    return str(selected["spec"]), float(selected["c_value"]), candidate_rows


def repeated_nested_oof(
    bundle: Any,
    specs: dict[str, RankerSpec],
    reliability: np.ndarray,
    seeds: list[int],
    c_values: list[float],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_sum = np.zeros(len(bundle.y), dtype=float)
    prediction_count = np.zeros(len(bundle.y), dtype=int)
    tuning_log = []
    repeat_metrics = []
    for repeat_index, seed in enumerate(seeds):
        print(f"[v14] repeat {repeat_index + 1}/{len(seeds)}", flush=True)
        repeat_predictions = np.full(len(bundle.y), np.nan, dtype=float)
        splits = grouped_splits(
            bundle.groups,
            int(config["outer_splits"]),
            seed,
        )
        for fold_index, (train_indices, test_indices) in enumerate(splits):
            selected_spec, selected_c, inner_rows = select_spec_and_c(
                bundle,
                train_indices,
                specs,
                reliability,
                c_values,
                int(config["inner_splits"]),
                seed + 1000 + fold_index,
                config,
            )
            spec = specs[selected_spec]
            prepared = prepare_fold(
                bundle,
                train_indices,
                test_indices,
                spec,
                reliability,
            )
            fold_predictions = fit_prepared(
                prepared,
                selected_c,
                spec.score_calibration,
                seed + fold_index,
            )
            repeat_predictions[test_indices] = fold_predictions
            prediction_sum[test_indices] += fold_predictions
            prediction_count[test_indices] += 1
            tuning_log.append(
                {
                    "repeat_index": repeat_index,
                    "outer_fold": fold_index,
                    "selected_spec": selected_spec,
                    "selected_c": selected_c,
                    "train_count": int(len(train_indices)),
                    "test_count": int(len(test_indices)),
                    "training_pair_count": prepared.base_pair_count,
                    "inner_candidates": inner_rows,
                }
            )
        repeat_metrics.append(
            {
                "repeat_index": repeat_index,
                **development_metrics(
                    bundle.y,
                    repeat_predictions,
                    bundle.channels,
                    float(config["pair_weighting"]["mid_percentile_min"]),
                    float(config["pair_weighting"]["mid_percentile_max"]),
                    float(config["pair_weighting"]["minimum_gap"]),
                    config["inner_selection_weights"],
                ),
            }
        )
    if np.any(prediction_count != len(seeds)):
        raise RuntimeError(
            f"Invalid OOF coverage: {Counter(prediction_count.tolist())}"
        )
    return prediction_sum / prediction_count, tuning_log, repeat_metrics


def clustered_bootstrap(
    bundle: Any,
    scores: np.ndarray,
    config: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(int(config["random_seeds"][0]))
    unique_groups = np.array(sorted(set(bundle.groups)), dtype=object)
    group_indices = {
        group: np.flatnonzero(bundle.groups == group)
        for group in unique_groups
    }
    metric_names = [
        "mid_only_channel_centered_spearman",
        "same_channel_local_pairwise_accuracy",
        "mid_only_pairwise_accuracy",
        "extremes_pos_neg_auc",
    ]
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(config["bootstrap_repetitions"])):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )
        metrics = development_metrics(
            bundle.y[indices],
            scores[indices],
            bundle.channels[indices],
            float(config["pair_weighting"]["mid_percentile_min"]),
            float(config["pair_weighting"]["mid_percentile_max"]),
            float(config["pair_weighting"]["minimum_gap"]),
            config["inner_selection_weights"],
        )
        for name in metric_names:
            value = float(metrics[name])
            if math.isfinite(value):
                samples[name].append(value)
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
            "valid_repetitions": len(values),
        }
        for name, values in samples.items()
        if values
    }


def stable_noise(candidate_id: str) -> float:
    digest = hashlib.sha256(
        f"v14-bucket-null:{candidate_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def baseline_metrics(
    bundle: Any,
    v14_scores: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    weighting = config["pair_weighting"]
    labels = labels_from_targets(
        bundle.y,
        float(weighting["mid_percentile_min"]),
        float(weighting["mid_percentile_max"]),
    )
    ids = bundle.frame["candidate_id"].astype(str).tolist()
    bucket = np.array(
        [
            {"neg": 0.0, "mid": 0.5, "pos": 1.0}[label]
            + stable_noise(candidate_id) * 1e-3
            for candidate_id, label in zip(ids, labels)
        ],
        dtype=float,
    )
    comparisons: list[tuple[str, np.ndarray]] = [
        ("v14_nested_procedure", v14_scores),
        (
            "duration_only",
            pd.to_numeric(
                bundle.frame["duration_sec_feature"],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(dtype=float),
        ),
        ("label_bucket_oracle_invalid", bucket),
    ]
    v13_path = (
        DEFAULT_PRIVATE_DIR
        / "performance_calibrator_v13"
        / "oof_predictions_PRIVATE.csv"
    )
    if v13_path.exists():
        v13 = pd.read_csv(v13_path, encoding="utf-8-sig")
        mapping = dict(
            zip(
                v13["candidate_id"].astype(str),
                pd.to_numeric(v13["oof_frozen_ensemble"], errors="raise"),
            )
        )
        comparisons.insert(
            1,
            (
                "v13_repeated_grouped_oof",
                np.array([mapping[candidate_id] for candidate_id in ids]),
            ),
        )
    rows = []
    for name, scores in comparisons:
        rows.append(
            {
                "model": name,
                **development_metrics(
                    bundle.y,
                    scores,
                    bundle.channels,
                    float(weighting["mid_percentile_min"]),
                    float(weighting["mid_percentile_max"]),
                    float(weighting["minimum_gap"]),
                    config["inner_selection_weights"],
                ),
            }
        )
    return rows


def write_report(
    path: Path,
    summary: dict[str, Any],
    comparison: list[dict[str, Any]],
) -> None:
    def value(row: dict[str, Any], key: str) -> str:
        item = row.get(key)
        return f"{float(item):.4f}" if item is not None else "NA"

    table = [
        "| 모델 | 전체 채널 중심 rho | mid 채널 중심 rho | mid pairwise | local pairwise | 극단 AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        table.append(
            "| {model} | {all_rho} | {mid_rho} | {mid_pair} | {local_pair} | {auc} |".format(
                model=row["model"],
                all_rho=value(row, "channel_centered_spearman"),
                mid_rho=value(row, "mid_only_channel_centered_spearman"),
                mid_pair=value(row, "mid_only_pairwise_accuracy"),
                local_pair=value(
                    row,
                    "same_channel_local_pairwise_accuracy",
                ),
                auc=value(row, "extremes_pos_neg_auc"),
            )
        )
    metrics = summary["metrics"]
    bootstrap = summary["bootstrap"]
    report = f"""# Performance Calibrator v14 개발 진단

## 결론

이 결과는 새 홀드아웃 검증이 아니라 기존 94건 개발 데이터의 nested OOF 진단이다.
모델 구조와 pair 가중치는 v13 실패를 본 뒤 설계했으므로 최종 성능 주장에 사용할 수
없다.

{chr(10).join(table)}

## v14 핵심 수치

- mid 채널 중심 Spearman: {metrics['mid_only_channel_centered_spearman']:.4f}
- mid pairwise: {metrics['mid_only_pairwise_accuracy']:.4f}
- local pairwise: {metrics['same_channel_local_pairwise_accuracy']:.4f}
- POS-vs-NEG AUC: {metrics['extremes_pos_neg_auc']:.4f}
- mid rho 95% CI:
  [{bootstrap['mid_only_channel_centered_spearman']['lower_95']:.4f},
   {bootstrap['mid_only_channel_centered_spearman']['upper_95']:.4f}]

## 설계

- 롱폼 단위 outer 5-fold / inner 4-fold
- outer fold 안에서 표현 구조와 C를 함께 선택
- 모든 자막 형식 프록시 수치 특징 제외
- 정규화된 자막·설명만 사용
- mid-mid pair 3배, local pair 2배, pos-neg 쉬운 pair 0.25배
- 채널별 pair 총가중치 균형화
- 10개 seed 반복 OOF 평균

## 상태

`development_only_not_validated`. 새 mid-enriched holdout에서 실제 배포 artifact를
검증하기 전에는 성과 예측 Judge로 승인하지 않는다.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_bundle = load_bundle(args.private_dir)
    bundle, kept_structure = remove_proxy_features(
        raw_bundle,
        [str(value) for value in config["excluded_proxy_features"]],
    )
    all_specs = candidate_specs(config)
    requested = [str(value) for value in config["candidate_specs"]]
    missing = sorted(set(requested) - set(all_specs))
    if missing:
        raise ValueError(f"Unknown candidate specs: {missing}")
    specs = {name: all_specs[name] for name in requested}
    reliability = candidate_reliability(bundle)
    scores, tuning_log, repeat_metrics = repeated_nested_oof(
        bundle,
        specs,
        reliability,
        [int(value) for value in config["random_seeds"]],
        [float(value) for value in config["c_values"]],
        config,
    )
    args.private_output.mkdir(parents=True, exist_ok=True)
    oof = bundle.frame[
        [
            "candidate_id",
            "longform_id",
            "channel_name",
            "channel_performance_percentile_PRIVATE",
        ]
    ].copy()
    oof["oof_v14_nested"] = scores
    oof.to_csv(
        args.private_output / "oof_predictions_PRIVATE.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        args.private_output / "nested_tuning_log_PRIVATE.json"
    ).write_text(
        json.dumps(
            json_safe(
                {
                    "tuning_log": tuning_log,
                    "repeat_metrics": repeat_metrics,
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    metrics = development_metrics(
        bundle.y,
        scores,
        bundle.channels,
        float(config["pair_weighting"]["mid_percentile_min"]),
        float(config["pair_weighting"]["mid_percentile_max"]),
        float(config["pair_weighting"]["minimum_gap"]),
        config["inner_selection_weights"],
    )
    bootstrap = clustered_bootstrap(bundle, scores, config)
    comparison = baseline_metrics(bundle, scores, config)
    selections = Counter(
        str(row["selected_spec"]) for row in tuning_log
    )

    args.public_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.public_dir / "model_comparison_PUBLIC.csv", comparison)
    summary = {
        "protocol_id": config["protocol_id"],
        "status": config["status"],
        "accepted_as_performance_judge": False,
        "accepted_false_reason": (
            "Architecture and pair weighting were designed after the v13 audit "
            "on the same 94 development candidates. A fresh mid-enriched "
            "holdout scored by the packaged artifact is required."
        ),
        "metrics": metrics,
        "bootstrap": bootstrap,
        "candidate_specs": {
            name: asdict(spec) for name, spec in specs.items()
        },
        "selection_counts_across_outer_folds": dict(sorted(selections.items())),
        "repeat_metrics": repeat_metrics,
        "kept_structure_features": kept_structure,
        "excluded_proxy_features": config["excluded_proxy_features"],
        "comparison": comparison,
        "claim_policy": config["claim_policy"],
    }
    (
        args.public_dir / "summary_PUBLIC.json"
    ).write_text(
        json.dumps(
            json_safe(summary),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    write_report(args.report, summary, comparison)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "metrics": metrics,
                "selection_counts": dict(sorted(selections.items())),
                "summary": str(args.public_dir / "summary_PUBLIC.json"),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
