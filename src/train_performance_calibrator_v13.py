from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from train_performance_calibrator_v11 import (
    DEFAULT_PRIVATE_DIR,
    ROOT,
    acceptance_result,
    json_safe,
    load_bundle,
    markdown_table,
    write_csv,
)
from train_performance_calibrator_v12 import (
    RankerSpec,
    bootstrap_intervals,
    candidate_input_audit,
    candidate_reliability,
    leave_one_channel_out,
    public_metrics,
    repeated_nested_oof,
)


DEFAULT_CONFIG = ROOT / "config" / "performance_calibrator_v13.json"
DEFAULT_PUBLIC_DIR = ROOT / "results" / "performance_calibrator_v13"
DEFAULT_PRIVATE_OUTPUT = (
    DEFAULT_PRIVATE_DIR / "performance_calibrator_v13"
)
DEFAULT_REPORT = ROOT / "reports" / "performance_calibrator_v13_2026-07-28.md"

MEMBER_SPECS = {
    "numeric_050": RankerSpec(
        "numeric_050",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.0,
        False,
        numeric_scale=0.50,
    ),
    "channel_balanced": RankerSpec(
        "channel_balanced",
        "concat_raw_char",
        "raw",
        False,
        True,
        0.05,
        1.0,
        False,
    ),
    "numeric_025": RankerSpec(
        "numeric_025",
        "concat_raw_char",
        "raw",
        False,
        False,
        0.05,
        1.0,
        False,
        numeric_scale=0.25,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen v13 shortform performance ensemble."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--private-output", type=Path, default=DEFAULT_PRIVATE_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def weighted_average(
    values: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    if len(values) != len(weights):
        raise ValueError("Value and weight counts differ.")
    normalized = np.asarray(weights, dtype=float)
    normalized /= float(np.sum(normalized))
    return np.average(np.vstack(values), axis=0, weights=normalized)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle = load_bundle(args.private_dir)
    reliability = candidate_reliability(bundle)
    member_names = [str(value) for value in config["ensemble_members"]]
    weights = [float(value) for value in config["ensemble_weights"]]
    seeds = [int(value) for value in config["random_seeds"]]
    c_values = [float(value) for value in config["c_values"]]
    args.public_dir.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)

    missing = sorted(set(member_names) - set(MEMBER_SPECS))
    if missing:
        raise ValueError(f"Unknown ensemble members: {missing}")

    member_predictions = {}
    member_rows = []
    tuning_logs = {}
    repeat_metrics = {}
    for member_name in member_names:
        spec = MEMBER_SPECS[member_name]
        print(f"[v13] evaluating {member_name}", flush=True)
        scores, tuning, repeats = repeated_nested_oof(
            bundle,
            spec,
            reliability,
            seeds,
            int(config["outer_splits"]),
            int(config["inner_splits"]),
            c_values,
        )
        member_predictions[member_name] = scores
        member_rows.append(
            {
                "model": member_name,
                **public_metrics(bundle, scores),
            }
        )
        tuning_logs[member_name] = tuning
        repeat_metrics[member_name] = repeats

    ensemble_scores = weighted_average(
        [member_predictions[name] for name in member_names],
        weights,
    )
    ensemble_metrics = {
        "model": "frozen_equal_weight_ensemble",
        **public_metrics(bundle, ensemble_scores),
    }
    comparison = [ensemble_metrics, *member_rows]
    bootstrap = bootstrap_intervals(
        bundle,
        ensemble_scores,
        int(config["bootstrap_repetitions"]),
        seeds[0],
    )
    internal_pass, gates = acceptance_result(
        ensemble_metrics,
        bootstrap,
        config["acceptance_gates"],
    )

    print("[v13] evaluating leave-one-channel-out", flush=True)
    loco_member_scores = []
    for offset, member_name in enumerate(member_names):
        loco_member_scores.append(
            leave_one_channel_out(
                bundle,
                MEMBER_SPECS[member_name],
                reliability,
                c_values,
                int(config["inner_splits"]),
                seeds[0] + offset * 100,
            )
        )
    loco_scores = weighted_average(loco_member_scores, weights)
    loco_metrics = public_metrics(bundle, loco_scores)
    audit = candidate_input_audit(bundle)

    oof = bundle.frame[
        [
            "candidate_id",
            "longform_id",
            "channel_name",
            "channel_performance_percentile_PRIVATE",
        ]
    ].copy()
    for name, values in member_predictions.items():
        oof[f"oof_{name}"] = values
    oof["oof_frozen_ensemble"] = ensemble_scores
    oof["loco_frozen_ensemble"] = loco_scores
    oof.to_csv(
        args.private_output / "oof_predictions_PRIVATE.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.private_output / "tuning_log_PRIVATE.json").write_text(
        json.dumps(
            json_safe(
                {
                    "per_member": tuning_logs,
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
        "protocol_id": config["protocol_id"],
        "status": "development_frozen_pending_holdout",
        "internal_acceptance_gates_passed": internal_pass,
        "accepted_as_performance_judge": False,
        "accepted_false_reason": (
            "The architecture was selected using the same 94-candidate development "
            "dataset. A fresh untouched holdout is required."
        ),
        "ensemble_members": member_names,
        "ensemble_weights": weights,
        "ensemble_metrics": ensemble_metrics,
        "member_metrics": member_rows,
        "leave_one_channel_out_metrics": loco_metrics,
        "acceptance_gates": gates,
        "bootstrap_ensemble": bootstrap,
        "candidate_input_audit": audit,
        "member_specifications": {
            name: asdict(MEMBER_SPECS[name]) for name in member_names
        },
        "claim_policy": config["claim_policy"],
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
            ("model", "모델"),
            ("channel_centered_spearman", "채널 중심 rho"),
            ("channel_macro_spearman", "채널 Macro rho"),
            ("same_channel_pairwise_accuracy", "Pairwise"),
            ("same_channel_local_pairwise_accuracy", "Local Pairwise"),
            ("selection_score", "선택 점수"),
        ],
    )
    gate_table = markdown_table(
        gates,
        [
            ("gate", "게이트"),
            ("observed", "관측값"),
            ("required_minimum", "최소 기준"),
            ("passed", "통과"),
        ],
    )
    ci = bootstrap["channel_centered_spearman"]
    report = f"""# Vpick 성과 보정기 v13 동결 후보 검증

## 1. 고정 구조

2차 개발 실험이 끝난 뒤 다음 세 Pairwise 모델을 동일 가중치로 고정했다.

1. Codex·구조 수치 특징 비중 0.50
2. 채널별 학습 쌍 총 가중치 균등
3. Codex·구조 수치 특징 비중 0.25

세 모델 모두 익명 설명·자막·경계 문맥의 문자 TF-IDF와 Codex 7개 특징을
사용한다. 채널명, 조회수, 좋아요, 성과 라벨·백분위, URL, 데이터 역할,
자막 출처는 입력에서 제외했다.

## 2. 검증

- 롱폼 ID 기반 외부 5-fold·내부 4-fold GroupKFold
- 10개 seed 반복 후 OOF 평균
- C는 각 외부 학습 fold 내부에서만 선택
- 2,000회 롱폼 bootstrap
- leave-one-channel-out 별도 진단

## 3. 결과

{table}

- 채널 중심 Spearman 95% CI:
  [{ci['lower_95']:.4f}, {ci['upper_95']:.4f}]
- Leave-one-channel-out 채널 중심 Spearman:
  {loco_metrics['channel_centered_spearman']:.4f}

## 4. 내부 게이트

{gate_table}

내부 게이트 통과: **{internal_pass}**

## 5. 판정

현재 상태는 `development_frozen_pending_holdout`이다. 내부 게이트를 모두
통과하더라도 같은 94개에서 구조를 선택했으므로 최종 검증 통과로 주장하지
않는다. 다음 신규 미공개 holdout에서는 구조·가중치·C 후보를 바꾸지 않고
한 번만 평가한다.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            json_safe(
                {
                    "status": "development_frozen_pending_holdout",
                    "internal_acceptance_gates_passed": internal_pass,
                    "ensemble_metrics": ensemble_metrics,
                    "leave_one_channel_out_metrics": loco_metrics,
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
