from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from train_performance_calibrator_v11 import (
    DEFAULT_PRIVATE_DIR,
    ROOT,
    json_safe,
    load_bundle,
    markdown_table,
    write_csv,
)
from train_performance_calibrator_v12 import (
    RankerSpec,
    bootstrap_intervals,
    candidate_reliability,
    public_metrics,
    repeated_nested_oof,
)


DEFAULT_CONFIG = ROOT / "config" / "performance_calibrator_v12.json"
DEFAULT_PUBLIC_DIR = ROOT / "results" / "performance_calibrator_v12_phase2"
DEFAULT_PRIVATE_OUTPUT = (
    DEFAULT_PRIVATE_DIR / "performance_calibrator_v12_phase2"
)
DEFAULT_REPORT = (
    ROOT / "reports" / "performance_calibrator_v12_phase2_2026-07-28.md"
)


PHASE2_SPECS = [
    RankerSpec(
        "baseline_extended_c",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.0,
        False,
    ),
    RankerSpec(
        "baseline_train_zscore",
        "concat_raw_char",
        "train_zscore",
        False,
        False,
        0.05,
        1.0,
        False,
    ),
    RankerSpec(
        "baseline_gap03",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.03,
        1.0,
        False,
    ),
    RankerSpec(
        "baseline_gap08",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.08,
        1.0,
        False,
    ),
    RankerSpec(
        "baseline_channel_balanced",
        "concat_raw_char",
        "raw",
        False,
        True,
        0.05,
        1.0,
        False,
    ),
    RankerSpec(
        "baseline_local_boost",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.5,
        False,
    ),
    RankerSpec(
        "baseline_reliability",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.0,
        True,
    ),
    RankerSpec(
        "baseline_numeric_025",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.0,
        False,
        numeric_scale=0.25,
    ),
    RankerSpec(
        "baseline_numeric_050",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.0,
        False,
        numeric_scale=0.50,
    ),
    RankerSpec(
        "baseline_numeric_200",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.0,
        False,
        numeric_scale=2.00,
    ),
    RankerSpec(
        "baseline_char_3_5",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.0,
        False,
        char_ngram_min=3,
        char_ngram_max=5,
    ),
    RankerSpec(
        "baseline_char_2_6",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.0,
        False,
        char_ngram_min=2,
        char_ngram_max=6,
        char_max_features=8000,
    ),
    RankerSpec(
        "baseline_concat_char_word",
        "concat_raw_char_word",
        "raw",
        False,
        False,
        0.05,
        1.0,
        False,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused ablations around the best v11 pairwise ranker."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--private-output", type=Path, default=DEFAULT_PRIVATE_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle = load_bundle(args.private_dir)
    reliability = candidate_reliability(bundle)
    seeds = [int(value) for value in config["random_seeds"]]
    c_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    args.public_dir.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)

    predictions = {}
    comparison = []
    tuning_logs = {}
    repeat_metrics = {}
    for spec in PHASE2_SPECS:
        print(f"[v12-phase2] evaluating {spec.name}", flush=True)
        scores, tuning, repeats = repeated_nested_oof(
            bundle,
            spec,
            reliability,
            seeds,
            int(config["outer_splits"]),
            int(config["inner_splits"]),
            c_values,
        )
        predictions[spec.name] = scores
        comparison.append({"spec": spec.name, **public_metrics(bundle, scores)})
        tuning_logs[spec.name] = tuning
        repeat_metrics[spec.name] = repeats

    comparison.sort(
        key=lambda row: float(row["selection_score"]),
        reverse=True,
    )
    best = comparison[0]
    best_name = str(best["spec"])
    bootstrap = bootstrap_intervals(
        bundle,
        predictions[best_name],
        int(config["bootstrap_repetitions"]),
        seeds[0],
    )

    oof = bundle.frame[
        [
            "candidate_id",
            "longform_id",
            "channel_name",
            "channel_performance_percentile_PRIVATE",
        ]
    ].copy()
    for name, values in predictions.items():
        oof[f"oof_{name}"] = values
    oof.to_csv(
        args.private_output / "oof_predictions_PRIVATE.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.private_output / "tuning_log_PRIVATE.json").write_text(
        json.dumps(
            json_safe(
                {
                    "per_spec": tuning_logs,
                    "repeat_metrics": repeat_metrics,
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    write_csv(args.public_dir / "model_comparison_PUBLIC.csv", comparison)
    summary = {
        "protocol_id": "performance_calibrator_v12_phase2_development",
        "best_development_spec": best_name,
        "best_development_metrics": best,
        "bootstrap_best_development": bootstrap,
        "specifications": [
            {
                **spec.__dict__,
                "c_values": c_values,
            }
            for spec in PHASE2_SPECS
        ],
        "warning": (
            "All variants were compared on the same exploratory OOF dataset. "
            "The selected best is development-only and must be frozen before "
            "evaluation on a fresh holdout."
        ),
    }
    (args.public_dir / "summary_PUBLIC.json").write_text(
        json.dumps(
            json_safe(summary),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    table = markdown_table(
        comparison,
        [
            ("spec", "실험"),
            ("channel_centered_spearman", "채널 중심 rho"),
            ("channel_macro_spearman", "채널 Macro rho"),
            ("same_channel_pairwise_accuracy", "Pairwise"),
            ("same_channel_local_pairwise_accuracy", "Local Pairwise"),
            ("selection_score", "선택 점수"),
        ],
    )
    ci = bootstrap["channel_centered_spearman"]
    report = f"""# Vpick 성과 보정기 v12 2차 집중 실험

## 목적

1차 ablation에서 가장 강했던 기존 Pairwise 문자 TF-IDF + 수치 특징 구조
주변의 단일 변경만 비교했다. 모든 실험은 동일한 롱폼 GroupKFold와 내부 C
선택을 사용한다.

## 결과

{table}

사후 최고 개발 실험은 `{best_name}`이다.

- 채널 중심 Spearman: {best['channel_centered_spearman']:.4f}
- 채널 Macro Spearman: {best['channel_macro_spearman']:.4f}
- Pairwise 정확도: {best['same_channel_pairwise_accuracy']:.4f}
- Local Pairwise 정확도:
  {best['same_channel_local_pairwise_accuracy']:.4f}
- 채널 중심 Spearman 2,000회 bootstrap 95% CI:
  [{ci['lower_95']:.4f}, {ci['upper_95']:.4f}]

## 해석

이 표는 개선 방향을 고르기 위한 개발 결과다. 같은 94개에서 최고안을 선택했기
때문에 최종 검증값으로 사용하지 않는다. 선택한 구조를 고정한 뒤 새 미공개
holdout에서 한 번만 평가해야 한다.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            json_safe(
                {
                    "best_development_spec": best_name,
                    "best_development_metrics": best,
                    "bootstrap_channel_centered_spearman": (
                        bootstrap["channel_centered_spearman"]
                    ),
                    "report": str(args.report),
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
