from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def normalized_choice(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    normalized = {"l": "left", "r": "right", "t": "tie", "왼쪽": "left", "오른쪽": "right", "동점": "tie"}.get(
        normalized, normalized
    )
    return normalized if normalized in {"left", "right", "tie"} else None


def is_true(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "예", "네"}


def complete_fleiss_kappa(labels_by_item: dict[str, list[str]]) -> float | None:
    categories = ("left", "right", "tie", "abstain")
    if not labels_by_item:
        return None
    total_labels = sum(len(labels) for labels in labels_by_item.values())
    if total_labels == 0:
        return None
    category_totals = Counter(label for labels in labels_by_item.values() for label in labels)
    expected = sum((category_totals[category] / total_labels) ** 2 for category in categories)
    observed_values: list[float] = []
    for labels in labels_by_item.values():
        if len(labels) < 2:
            return None
        counts = Counter(labels)
        n = len(labels)
        observed_values.append(
            sum(counts[category] * (counts[category] - 1) for category in categories) / (n * (n - 1))
        )
    observed = mean(observed_values)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def consensus(values: list[str]) -> str:
    if not values:
        return "abstain"
    counts = Counter(values)
    top = max(counts.values())
    winners = [value for value, count in counts.items() if count == top]
    return winners[0] if len(winners) == 1 else "unstable"


def aggregate_scores(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["judge_run_id"], row["comparison_id"])].append(row)

    output: list[dict[str, Any]] = []
    for (run_id, comparison_id), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(float(row.get("repeat_index") or 0)))
        scored = [row for row in ordered if row.get("verdict") == "score"]
        editorial_values = [value for row in scored if (value := normalized_choice(row.get("editorial_preference")))]
        performance_values = [value for row in scored if (value := normalized_choice(row.get("performance_preference")))]

        def mean_field(field: str) -> float | None:
            values = [value for row in scored if (value := as_float(row.get(field))) is not None]
            return round(mean(values), 4) if values else None

        output.append(
            {
                "judge_run_id": run_id,
                "provider": ordered[0].get("provider", ""),
                "model": ordered[0].get("model", ""),
                "comparison_id": comparison_id,
                "repeat_count": len(ordered),
                "scored_repeat_count": len(scored),
                "abstain_repeat_count": len(ordered) - len(scored),
                "aggregate_status": "scored" if scored else "abstain",
                "editorial_consensus": consensus(editorial_values),
                "performance_consensus": consensus(performance_values),
                "editorial_repeat_agreement": len(editorial_values) >= 2 and len(set(editorial_values)) == 1,
                "performance_repeat_agreement": len(performance_values) >= 2 and len(set(performance_values)) == 1,
                "left_editorial_score_mean": mean_field("left_editorial_score"),
                "right_editorial_score_mean": mean_field("right_editorial_score"),
                "left_performance_score_mean": mean_field("left_performance_score"),
                "right_performance_score_mean": mean_field("right_performance_score"),
                "confidence_mean": mean_field("confidence"),
            }
        )
    return output


def model_metrics(
    aggregates: list[dict[str, Any]], manifest: list[dict[str, str]], total_comparisons: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    private = {row["comparison_id"]: row for row in manifest}
    runs = sorted({row["judge_run_id"] for row in aggregates})
    metric_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []

    for run_id in runs:
        run_rows = [row for row in aggregates if row["judge_run_id"] == run_id]
        scored = [row for row in run_rows if row["aggregate_status"] == "scored"]
        repeat_eligible = [row for row in run_rows if row["scored_repeat_count"] >= 2]
        correct = ties = wrong = unstable = 0
        for row in scored:
            expected = private[row["comparison_id"]]["positive_side"]
            choice = row["performance_consensus"]
            if choice == expected:
                correct += 1
            elif choice == "tie":
                ties += 1
            elif choice == "unstable":
                unstable += 1
            else:
                wrong += 1
        denominator = len(scored)
        metric_rows.append(
            {
                "judge_run_id": run_id,
                "model": run_rows[0]["model"] if run_rows else "",
                "comparison_count": total_comparisons,
                "scored_comparison_count": len(scored),
                "scoring_coverage": round(len(scored) / total_comparisons, 4) if total_comparisons else None,
                "repeat_eligible_count": len(repeat_eligible),
                "editorial_repeat_agreement": round(
                    sum(bool(row["editorial_repeat_agreement"]) for row in repeat_eligible) / len(repeat_eligible), 4
                ) if repeat_eligible else None,
                "performance_repeat_agreement": round(
                    sum(bool(row["performance_repeat_agreement"]) for row in repeat_eligible) / len(repeat_eligible), 4
                ) if repeat_eligible else None,
                "pos_preferred_count": correct,
                "tie_count": ties,
                "neg_preferred_count": wrong,
                "unstable_count": unstable,
                "pos_preference_accuracy": round(correct / denominator, 4) if denominator else None,
                "pos_preference_half_credit": round((correct + 0.5 * ties) / denominator, 4) if denominator else None,
                "decisive_pos_preference_accuracy": round(correct / (correct + wrong), 4) if correct + wrong else None,
                "mean_confidence": round(mean(row["confidence_mean"] for row in scored if row["confidence_mean"] is not None), 4)
                if any(row["confidence_mean"] is not None for row in scored) else None,
            }
        )

        by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            by_channel[private[row["comparison_id"]]["channel_name"]].append(row)
        for channel_name, group in sorted(by_channel.items()):
            channel_correct = channel_ties = channel_wrong = channel_unstable = 0
            for row in group:
                expected = private[row["comparison_id"]]["positive_side"]
                choice = row["performance_consensus"]
                if choice == expected:
                    channel_correct += 1
                elif choice == "tie":
                    channel_ties += 1
                elif choice == "unstable":
                    channel_unstable += 1
                else:
                    channel_wrong += 1
            channel_rows.append(
                {
                    "judge_run_id": run_id,
                    "channel_name": channel_name,
                    "comparison_count": len(group),
                    "pos_preferred_count": channel_correct,
                    "tie_count": channel_ties,
                    "neg_preferred_count": channel_wrong,
                    "unstable_count": channel_unstable,
                    "pos_preference_accuracy": round(channel_correct / len(group), 4),
                    "pos_preference_half_credit": round((channel_correct + 0.5 * channel_ties) / len(group), 4),
                }
            )
    return metric_rows, channel_rows


def human_metrics(
    aggregates: list[dict[str, Any]], human_rows: list[dict[str, str]], required_annotators: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_comparisons = {row["comparison_id"] for row in aggregates}
    valid: list[dict[str, str]] = []
    validation_errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in human_rows:
        fields = (
            row.get("editorial_preference"), row.get("performance_preference"),
            row.get("confidence_1_to_5"), row.get("insufficient_evidence"), row.get("notes"),
        )
        if not any(str(value or "").strip() for value in fields):
            continue
        comparison_id = str(row.get("comparison_id") or "").strip()
        annotator_id = str(row.get("annotator_id") or "").strip()
        key = (comparison_id, annotator_id)
        if comparison_id not in expected_comparisons:
            validation_errors.append(f"unknown comparison_id: {comparison_id or '<blank>'}")
            continue
        if not annotator_id:
            validation_errors.append(f"missing annotator_id: {comparison_id}")
            continue
        if key in seen:
            validation_errors.append(f"duplicate annotator row: {comparison_id}/{annotator_id}")
            continue
        seen.add(key)

        insufficient = is_true(row.get("insufficient_evidence"))
        editorial = "abstain" if insufficient else normalized_choice(row.get("editorial_preference"))
        performance = "abstain" if insufficient else normalized_choice(row.get("performance_preference"))
        confidence = as_float(row.get("confidence_1_to_5"))
        if editorial is None or performance is None:
            validation_errors.append(f"incomplete preference: {comparison_id}/{annotator_id}")
            continue
        if confidence is None or not 1 <= confidence <= 5:
            validation_errors.append(f"invalid confidence: {comparison_id}/{annotator_id}")
            continue
        valid.append(
            {
                **row,
                "normalized_editorial": editorial,
                "normalized_performance": performance,
            }
        )

    by_comparison: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in valid:
        by_comparison[row["comparison_id"]].append(row)

    complete = {
        comparison_id: rows
        for comparison_id, rows in by_comparison.items()
        if len({row["annotator_id"] for row in rows}) == required_annotators
        and len(rows) == required_annotators
    }

    human_consensus: dict[tuple[str, str], str] = {}
    kappas: dict[str, float | None] = {}
    for dimension in ("editorial", "performance"):
        labels_by_item: dict[str, list[str]] = {}
        for comparison_id, rows in complete.items():
            labels = [row[f"normalized_{dimension}"] for row in rows]
            labels_by_item[comparison_id] = labels
            human_consensus[(dimension, comparison_id)] = consensus(labels)
        kappa = complete_fleiss_kappa(labels_by_item)
        kappas[dimension] = round(kappa, 4) if kappa is not None else None

    output: list[dict[str, Any]] = []
    for run_id in sorted({row["judge_run_id"] for row in aggregates}):
        run_rows = {row["comparison_id"]: row for row in aggregates if row["judge_run_id"] == run_id}
        result: dict[str, Any] = {"judge_run_id": run_id}
        for dimension in ("editorial", "performance"):
            total = correct = 0
            for comparison_id in complete:
                human_choice = human_consensus.get((dimension, comparison_id))
                model_row = run_rows.get(comparison_id)
                model_choice = (
                    "abstain" if model_row and model_row.get("aggregate_status") == "abstain"
                    else model_row.get(f"{dimension}_consensus") if model_row else None
                )
                if human_choice not in {"left", "right", "tie", "abstain"} or model_choice not in {
                    "left", "right", "tie", "abstain"
                }:
                    continue
                total += 1
                correct += int(human_choice == model_choice)
            result[f"{dimension}_comparison_count"] = total
            result[f"{dimension}_human_agreement"] = round(correct / total, 4) if total else None
        output.append(result)
    summary = {
        "completed_label_row_count": len(valid),
        "expected_label_row_count": len(expected_comparisons) * required_annotators,
        "partially_labeled_comparison_count": len(by_comparison),
        "comparison_count": len(complete),
        "expected_comparison_count": len(expected_comparisons),
        "comparison_coverage": round(len(complete) / len(expected_comparisons), 4) if expected_comparisons else 0.0,
        "required_annotators": required_annotators,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors[:50],
        "editorial_fleiss_kappa": kappas["editorial"],
        "performance_fleiss_kappa": kappas["performance"],
    }
    return output, summary


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Gold Pairwise Judge 검증 보고서",
        "",
        f"- validation status: `{summary['validation_status']}`",
        f"- matched comparisons: {summary['counts']['comparison_count']}",
        f"- unique negative cases: {summary['counts']['negative_case_count']}",
        f"- channels: {summary['counts']['channel_count']}",
        "",
        "## 모델 결과",
        "",
    ]
    for row in summary["model_metrics"]:
        lines.append(
            f"- {row['judge_run_id']}: coverage {row['scoring_coverage']}, performance repeat agreement "
            f"{row['performance_repeat_agreement']}, Pos strict accuracy {row['pos_preference_accuracy']}, "
            f"half-credit accuracy {row['pos_preference_half_credit']}"
        )
    lines.extend(["", "## 검증 게이트", ""])
    for row in summary["model_gates"]:
        lines.append(
            f"- {row['judge_run_id']}: reliability={row['reliability_gate_pass']}, "
            f"performance alignment={row['performance_alignment_gate_pass']}, "
            f"human agreement={row['human_agreement_gate_pass']}, status=`{row['content_judge_status']}`"
        )
    lines.extend(["", "## 인간 평가", ""])
    if summary["human"]["comparison_count"] == 0:
        lines.append("- 인간 블라인드 라벨이 비어 있어 최종 타당성 검증은 대기 상태다.")
    else:
        lines.append(
            f"- complete comparison coverage: {summary['human']['comparison_coverage']}; "
            f"editorial Fleiss kappa: {summary['human']['editorial_fleiss_kappa']}; "
            f"performance Fleiss kappa: {summary['human']['performance_fleiss_kappa']}"
        )
    if summary["human"]["validation_error_count"]:
        lines.append(f"- human label validation errors: {summary['human']['validation_error_count']}")
    lines.extend(
        [
            "",
            "## 해석 원칙",
            "",
            "- Pos/Neg, 조회수, 좋아요, 채널명은 모델 입력에서 제외했다.",
            "- 2회차에는 좌우 후보를 뒤집고 결과를 원래 방향으로 복원해 위치 편향을 검사했다.",
            "- Pos 선호 정확도는 외부 성과 정합성 진단이며 콘텐츠 품질의 절대적 정답이 아니다.",
            "- 인간 블라인드 평가가 완료되기 전에는 Judge를 validated로 선언하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate matched Gold pairwise LLM judgments.")
    parser.add_argument("--scores", action="append", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--build-summary", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--human-labels")
    args = parser.parse_args()

    all_scores = [row for score_path in args.scores for row in read_csv(Path(score_path))]
    scores = [row for row in all_scores if not is_true(row.get("dry_run"))]
    sources = read_csv(Path(args.sources))
    build_summary = json.loads(Path(args.build_summary).read_text(encoding="utf-8"))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    human_rows = read_csv(Path(args.human_labels)) if args.human_labels else []
    out_dir = Path(args.out_dir)

    aggregates = aggregate_scores(scores)
    metric_rows, channel_rows = model_metrics(aggregates, sources, int(build_summary["comparison_count"]))
    gates = dict(config["gates"])
    required_annotators = int(gates.get("required_human_annotators", 3))
    human_rows_out, human_summary = human_metrics(aggregates, human_rows, required_annotators)

    human_available = human_summary["completed_label_row_count"] > 0
    human_complete = (
        human_summary["comparison_coverage"] >= float(gates.get("min_human_coverage", 1.0))
        and human_summary["validation_error_count"] == 0
    )
    human_by_run = {row["judge_run_id"]: row for row in human_rows_out}
    model_gate_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        reliability_pass = (
            (row["scoring_coverage"] or 0) >= float(gates["min_scoring_coverage"])
            and (row["editorial_repeat_agreement"] or 0) >= float(gates["min_repeat_agreement"])
            and (row["performance_repeat_agreement"] or 0) >= float(gates["min_repeat_agreement"])
        )
        performance_alignment_pass = (
            (row["pos_preference_accuracy"] or 0) >= float(gates["min_pos_preference_accuracy"])
        )
        human_row = human_by_run.get(row["judge_run_id"], {})
        human_agreement_pass = bool(human_complete) and (
            (human_row.get("editorial_human_agreement") or 0) >= float(gates["min_human_agreement"])
            and (human_row.get("performance_human_agreement") or 0) >= float(gates["min_human_agreement"])
            and (human_summary.get("editorial_fleiss_kappa") or 0) >= float(gates["min_human_fleiss_kappa"])
            and (human_summary.get("performance_fleiss_kappa") or 0) >= float(gates["min_human_fleiss_kappa"])
        )
        status = (
            "validated_content_judge"
            if reliability_pass and human_agreement_pass
            else "pending_human_labels"
            if not human_complete
            else "needs_revision"
        )
        model_gate_rows.append(
            {
                "judge_run_id": row["judge_run_id"],
                "reliability_gate_pass": reliability_pass,
                "performance_alignment_gate_pass": performance_alignment_pass,
                "human_agreement_gate_pass": human_agreement_pass,
                "content_judge_status": status,
            }
        )
    validation_status = (
        "validated_content_judge"
        if any(row["content_judge_status"] == "validated_content_judge" for row in model_gate_rows)
        else "pending_human_labels"
        if not human_complete
        else "needs_revision"
    )

    summary = {
        "validation_status": validation_status,
        "gates": gates,
        "counts": build_summary,
        "excluded_dry_run_score_row_count": len(all_scores) - len(scores),
        "model_metrics": metric_rows,
        "channel_metrics": channel_rows,
        "model_gates": model_gate_rows,
        "human_alignment": human_rows_out,
        "human": human_summary,
        "interpretation": {
            "content_judge_validation": "Reliability and blind human agreement are the primary validity gates.",
            "performance_alignment": "Pos/Neg accuracy is a separate external-performance diagnostic and does not define editorial quality.",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "pairwise_scores_aggregated.csv",
        aggregates,
        [
            "judge_run_id", "provider", "model", "comparison_id", "repeat_count", "scored_repeat_count",
            "abstain_repeat_count", "aggregate_status", "editorial_consensus", "performance_consensus",
            "editorial_repeat_agreement", "performance_repeat_agreement", "left_editorial_score_mean",
            "right_editorial_score_mean", "left_performance_score_mean", "right_performance_score_mean",
            "confidence_mean",
        ],
    )
    write_csv(
        out_dir / "pairwise_model_metrics.csv",
        metric_rows,
        [
            "judge_run_id", "model", "comparison_count", "scored_comparison_count", "scoring_coverage",
            "repeat_eligible_count", "editorial_repeat_agreement", "performance_repeat_agreement",
            "pos_preferred_count", "tie_count", "neg_preferred_count", "unstable_count",
            "pos_preference_accuracy", "pos_preference_half_credit", "decisive_pos_preference_accuracy", "mean_confidence",
        ],
    )
    write_csv(
        out_dir / "pairwise_model_gates.csv",
        model_gate_rows,
        [
            "judge_run_id", "reliability_gate_pass", "performance_alignment_gate_pass",
            "human_agreement_gate_pass", "content_judge_status",
        ],
    )
    write_csv(
        out_dir / "pairwise_channel_metrics.csv",
        channel_rows,
        [
            "judge_run_id", "channel_name", "comparison_count", "pos_preferred_count", "tie_count",
            "neg_preferred_count", "unstable_count", "pos_preference_accuracy", "pos_preference_half_credit",
        ],
    )
    write_csv(
        out_dir / "pairwise_human_alignment.csv",
        human_rows_out,
        [
            "judge_run_id", "editorial_comparison_count", "editorial_human_agreement",
            "performance_comparison_count", "performance_human_agreement",
        ],
    )
    (out_dir / "pairwise_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "pairwise_validation_report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
