from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import ROOT, load_config, resolve_path, rounded, write_csv, write_json
from .run_case import run_case


DEFAULT_CONFIG = ROOT / "configs" / "evaluation.yaml"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def compare(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = resolve_path(config["output_dir"])
    for case in (
        "channel_baseline",
        "standalone_pointwise",
        "source_pointwise",
        "source_pairwise",
        "hybrid",
    ):
        run_case(case, config)

    case_1 = _load(output_dir / "case_1_summary.json")
    case_2 = _load(output_dir / "case_2_summary.json")
    case_3 = _load(output_dir / "case_3_summary.json")
    case_4 = _load(output_dir / "case_4_summary.json")
    case_5 = _load(output_dir / "case_5_summary.json")

    selected_alpha = case_5.get("selected_alpha")
    selected_hybrid = next(
        (
            row
            for row in case_5.get("alpha_results", [])
            if row.get("alpha") == selected_alpha
        ),
        None,
    )
    selected_hybrid_metrics = selected_hybrid.get("metrics", {}) if selected_hybrid else {}

    rows = [
        {
            "case": "Case 1",
            "name": "channel_baseline",
            "validation_scope": "60 current published shorts",
            "coverage_n": case_1["valid_candidate_count"],
            "performance_spearman": case_1["relative_log_vs_percentile_spearman"],
            "top25_auc": "N/A",
            "human_agreement": "N/A",
            "source_context": "no",
            "repeat_reliability": "N/A",
            "order_consistency": "N/A",
            "channel_holdout": "N/A",
            "cost": "none",
            "status": "reference signal usable; exploratory only",
        },
        {
            "case": "Case 2",
            "name": "standalone_pointwise",
            "validation_scope": "29/60 current overlap from actual historical run",
            "coverage_n": case_2["metrics"]["evaluated_candidate_count"],
            "performance_spearman": case_2["metrics"]["score_vs_channel_percentile_spearman"],
            "top25_auc": case_2["metrics"]["top25_roc_auc"],
            "human_agreement": "N/A",
            "source_context": "no",
            "repeat_reliability": case_2["repeat_reliability"]["repeat_spearman"],
            "order_consistency": "N/A",
            "channel_holdout": case_2["metrics"]["channel_holdout_macro_f1"],
            "cost": "N/A: historical usage has no price/time log",
            "status": "not validated as performance predictor",
        },
        {
            "case": "Case 3",
            "name": "source_pointwise",
            "validation_scope": "57/60 scored; repeat overlap 11",
            "coverage_n": case_3["metrics"]["evaluated_candidate_count"],
            "performance_spearman": case_3["metrics"]["score_vs_channel_percentile_spearman"],
            "top25_auc": case_3["metrics"]["top25_roc_auc"],
            "human_agreement": "N/A",
            "source_context": "yes",
            "repeat_reliability": case_3["repeat_reliability"]["repeat_spearman"],
            "order_consistency": "N/A",
            "channel_holdout": case_3["metrics"]["channel_holdout_macro_f1"],
            "cost": "$5.787 estimated for 71 requests",
            "status": "editorial diagnostic only; performance validity failed",
        },
        {
            "case": "Case 4",
            "name": "source_pairwise",
            "validation_scope": "124 synthetic-alternative pairs; 8 current pairs unscored",
            "coverage_n": case_4["historical_pair_count"],
            "performance_spearman": "N/A",
            "top25_auc": "N/A",
            "human_agreement": "N/A",
            "source_context": "yes",
            "repeat_reliability": "N/A",
            "order_consistency": case_4["historical_order_consistency_rate"],
            "channel_holdout": "N/A",
            "cost": "N/A: provider usage has no price/time log",
            "status": "not validated; order sensitivity is too high",
        },
        {
            "case": "Case 5",
            "name": f"hybrid_alpha_{selected_alpha}" if selected_alpha is not None else "hybrid",
            "validation_scope": f"{case_5['overlap_candidate_count']}/60 historical overlap",
            "coverage_n": selected_hybrid_metrics.get("evaluated_candidate_count", 0),
            "performance_spearman": selected_hybrid_metrics.get(
                "score_vs_channel_percentile_spearman"
            ),
            "top25_auc": selected_hybrid_metrics.get("top25_roc_auc"),
            "human_agreement": "N/A",
            "source_context": "yes",
            "repeat_reliability": "N/A",
            "order_consistency": "N/A",
            "channel_holdout": selected_hybrid_metrics.get("channel_holdout_macro_f1"),
            "cost": "sum of Case 2 and Case 3 inference",
            "status": "not validated; no consistent gain over component judges",
        },
    ]
    write_csv(output_dir / "case_comparison.csv", rows)

    status_rows = [
        {
            "requirement": "channel-relative performance signal",
            "status": "completed_exploratory",
            "evidence": (
                f"60/60; relative-log vs percentile rho="
                f"{case_1['relative_log_vs_percentile_spearman']}"
            ),
            "blocker": "No fixed 7-day/30-day view snapshots; upload age missing for 33/60",
        },
        {
            "requirement": "standalone pointwise actual validation",
            "status": "partial_historical_proxy",
            "evidence": f"{case_2['current_overlap_count']}/60 current overlap",
            "blocker": "New rubric has not been run on all 60 current candidates",
        },
        {
            "requirement": "source pointwise actual validation",
            "status": "partial_repeat",
            "evidence": (
                f"{case_3['scored_candidate_count']}/60 scored; "
                f"repeat overlap={case_3['repeat_reliability']['repeat_common_candidate_count']}"
            ),
            "blocker": "Second pass incomplete and current human sheet blank",
        },
        {
            "requirement": "source pairwise human/performance validation",
            "status": "not_completed",
            "evidence": (
                f"Historical synthetic pair order consistency="
                f"{case_4['historical_order_consistency_rate']}"
            ),
            "blocker": "8 published-published pairs and human labels are unscored",
        },
        {
            "requirement": "hybrid independent validation",
            "status": "exploratory_proxy",
            "evidence": f"Overlap={case_5['overlap_candidate_count']}; alpha={selected_alpha}",
            "blocker": "Standalone component is historical proxy; no human labels",
        },
    ]
    write_csv(output_dir / "validation_status.csv", status_rows)

    report_lines = [
        "# Vpick 5개 평가체계 비교 보고서",
        "",
        "## 결론",
        "",
        "현재 데이터로 **공식 성과 예측 Judge로 승인할 수 있는 Case는 없다.** "
        "채널 상대 성과 라벨은 안정적이지만 고정 관측 기간이 없어서 탐색적 신호이고, "
        "LLM 점수들은 그 신호와 유의미한 관계를 보이지 않았다.",
        "",
        "운영에 가장 가까운 것은 **Case 3 원본 조건부 Pointwise Judge**다. "
        "원본 적합성과 독립 숏폼 품질을 한 후보 단위로 설명할 수 있고 57개를 실제 채점했으며, "
        "11개 중복 표본에서는 반복 Spearman이 0.8611이었다. 다만 성과 백분위 상관은 "
        f"{_fmt(case_3['metrics']['score_vs_channel_percentile_spearman'])}, "
        f"Top-25 AUC는 {_fmt(case_3['metrics']['top25_roc_auc'])}이므로 "
        "현재는 편집 품질 진단용일 뿐 성과 예측기로 부르면 안 된다.",
        "",
        "## 공통 성과 신호",
        "",
        f"- 현재 60개 모두 채널 코호트와 결합되었다. 채널은 6개, 코호트는 총 302개다.",
        f"- `relative_log_view_score`와 `channel_view_percentile`의 Spearman은 "
        f"{_fmt(case_1['relative_log_vs_percentile_spearman'])}이다.",
        f"- 한 표본을 제외했을 때 백분위 평균 변화는 "
        f"{_fmt(case_1['leave_one_out_percentile_mean_absolute_delta'])}p다.",
        "- 7일·30일 조회수는 없으며 업로드 경과일도 27/60에서만 계산된다. 따라서 모든 성과 검증은 `exploratory`다.",
        "",
        "## 비교표",
        "",
        "| Case | 실제 범위 | 성과 백분위 rho | Top-25 AUC | 반복/순서 안정성 | 판정 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        stability = (
            row["order_consistency"]
            if row["order_consistency"] != "N/A"
            else row["repeat_reliability"]
        )
        report_lines.append(
            f"| {row['case']} {row['name']} | {row['coverage_n']} | "
            f"{_fmt(row['performance_spearman'])} | {_fmt(row['top25_auc'])} | "
            f"{_fmt(stability)} | {row['status']} |"
        )
    report_lines.extend(
        [
            "",
            "## 실제로 확인된 내용",
            "",
            f"- Case 2는 현재 데이터와 겹치는 {case_2['current_overlap_count']}개에서 "
            f"성과 백분위 rho={_fmt(case_2['metrics']['score_vs_channel_percentile_spearman'])}, "
            f"AUC={_fmt(case_2['metrics']['top25_roc_auc'])}였다.",
            f"- Case 3은 57개에서 rho={_fmt(case_3['metrics']['score_vs_channel_percentile_spearman'])}, "
            f"AUC={_fmt(case_3['metrics']['top25_roc_auc'])}였다. "
            f"그룹 bootstrap AUC 95% CI는 "
            f"[{_fmt(case_3['metrics']['top25_auc_group_bootstrap']['ci_lower'])}, "
            f"{_fmt(case_3['metrics']['top25_auc_group_bootstrap']['ci_upper'])}]다.",
            f"- Case 4의 A/B 순서 일관성은 {_fmt(case_4['historical_order_consistency_rate'])}로, "
            "공식 Judge로 사용하기에는 순서 민감도가 크다.",
            f"- Case 5는 검증 split에서 고른 alpha={_fmt(selected_alpha)}도 "
            f"겹치는 {case_5['overlap_candidate_count']}개에서 성과 상관이나 AUC를 일관되게 개선하지 못했다.",
            "",
            "## 아직 검증하지 못한 내용",
            "",
            "- 현재 루브릭의 인간 점수와 pairwise 선호는 템플릿만 있고 값이 비어 있어 모든 인간 일치도는 N/A다.",
            "- Case 2 신규 프롬프트는 60개 전체에서 아직 실행하지 않았다.",
            "- Case 3 두 번째 반복은 API 한도로 11/60만 완료됐다.",
            "- 현재 8개 published-published 동일 롱폼 pair는 생성했지만 LLM·인간 비교값이 없다.",
            "- 고정 7일·30일 조회수와 동일 업로드 기간 코호트가 없어 인과적 성과 예측을 주장할 수 없다.",
            "",
            "## 추천",
            "",
            "1. 성과 정답은 Case 1의 두 연속값을 별도로 유지한다. Pos/Neg는 보고용 파생값일 뿐 학습·평가의 유일한 정답으로 쓰지 않는다.",
            "2. 편집·하이라이트 품질의 운영 점수는 Case 3을 잠정 사용하되, 명칭을 `성과 예측 점수`가 아니라 `원본 조건부 품질 점수`로 제한한다.",
            "3. Case 4는 동일 롱폼 후보 간 결승 비교 보조도구로만 남기고, 순서 일관성 0.8 이상과 인간 검증 전에는 주 Judge로 쓰지 않는다.",
            "4. Case 5는 현재 채택하지 않는다. 구성 요소 중 하나가 과거 29개 proxy라 결합 점수의 우월성을 검증할 수 없다.",
            "",
            "공식 평가체계 승인을 위해 필요한 최소 다음 데이터는 고정 관측 기간 조회수, "
            "현재 루브릭의 2인 인간 점수, Case 2 전체 60개 반복 채점, Case 3 나머지 반복 49개다.",
            "",
        ]
    )
    report_path = output_dir / "EVALUATION_SYSTEM_COMPARISON_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "evaluation_id": config["evaluation_id"],
        "result": "no_case_passed_full_validation",
        "provisional_operational_case": "source_pointwise",
        "provisional_use": "source-conditioned editorial/highlight quality diagnosis only",
        "not_validated_for": "future short performance prediction",
        "selected_hybrid_alpha_on_validation": selected_alpha,
        "outputs": {
            "comparison_csv": str(output_dir / "case_comparison.csv"),
            "validation_status_csv": str(output_dir / "validation_status.csv"),
            "report": str(report_path),
        },
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare all five Vpick evaluation systems.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = compare(load_config(args.config))
    print(summary["result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
