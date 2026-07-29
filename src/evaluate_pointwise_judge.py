from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from evaluate_llm_judge import as_float, round_or_none, spearman
from run_pairwise_judge import EDITORIAL_DIMENSIONS, PERFORMANCE_DIMENSIONS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_scores(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["judge_run_id"], row["candidate_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (run_id, candidate_id), group in sorted(grouped.items()):
        scored = [row for row in group if as_float(row.get("editorial_score")) is not None and as_float(row.get("performance_score")) is not None]
        editorial_scores = [float(row["editorial_score"]) for row in scored]
        performance_scores = [float(row["performance_score"]) for row in scored]
        item: dict[str, Any] = {
            "judge_run_id": run_id,
            "provider": group[0].get("provider", ""),
            "model": group[0].get("model", ""),
            "input_modality": group[0].get("input_modality", ""),
            "candidate_id": candidate_id,
            "long_video_id": group[0].get("long_video_id", ""),
            "repeat_count": len(group),
            "scored_repeat_count": len(scored),
            "abstain_repeat_count": len(group) - len(scored),
            "aggregate_status": "scored" if scored else "abstain",
            "editorial_score_mean": round_or_none(mean(editorial_scores) if editorial_scores else None),
            "editorial_score_std": round(pstdev(editorial_scores), 4) if len(editorial_scores) > 1 else (0.0 if editorial_scores else None),
            "performance_score_mean": round_or_none(mean(performance_scores) if performance_scores else None),
            "performance_score_std": round(pstdev(performance_scores), 4) if len(performance_scores) > 1 else (0.0 if performance_scores else None),
            "confidence_mean": round_or_none(mean([float(row["confidence"]) for row in group])),
        }
        for prefix, dimensions in (("editorial", EDITORIAL_DIMENSIONS), ("performance", PERFORMANCE_DIMENSIONS)):
            for dimension in dimensions:
                values = [as_float(row.get(f"{prefix}_{dimension}")) for row in scored]
                clean = [value for value in values if value is not None]
                item[f"{prefix}_{dimension}_mean"] = round_or_none(mean(clean) if clean else None)
        output.append(item)
    return output


def repeat_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["judge_run_id"]].append(row)
    for run_id, group in sorted(by_run.items()):
        by_repeat: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
        for row in group:
            editorial = as_float(row.get("editorial_score"))
            performance = as_float(row.get("performance_score"))
            if editorial is not None and performance is not None:
                by_repeat[int(row["repeat_index"])][row["candidate_id"]] = {
                    "editorial": editorial,
                    "performance": performance,
                }
        editorial_corrs: list[float] = []
        performance_corrs: list[float] = []
        editorial_errors: list[float] = []
        performance_errors: list[float] = []
        common_counts: list[int] = []
        for left_repeat, right_repeat in itertools.combinations(sorted(by_repeat), 2):
            common = sorted(set(by_repeat[left_repeat]) & set(by_repeat[right_repeat]))
            common_counts.append(len(common))
            for axis, correlations, errors in (
                ("editorial", editorial_corrs, editorial_errors),
                ("performance", performance_corrs, performance_errors),
            ):
                left_values = [by_repeat[left_repeat][candidate_id][axis] for candidate_id in common]
                right_values = [by_repeat[right_repeat][candidate_id][axis] for candidate_id in common]
                correlation = spearman(left_values, right_values)
                if correlation is not None:
                    correlations.append(correlation)
                errors.extend(abs(left - right) for left, right in zip(left_values, right_values))
        candidate_ids = {row["candidate_id"] for row in group}
        scored_ids = {row["candidate_id"] for row in group if as_float(row.get("editorial_score")) is not None}
        output.append(
            {
                "judge_run_id": run_id,
                "model": group[0].get("model", ""),
                "input_modality": group[0].get("input_modality", ""),
                "candidate_count": len(candidate_ids),
                "candidate_scoring_coverage": round_or_none(len(scored_ids) / len(candidate_ids) if candidate_ids else None),
                "repeat_count": len(by_repeat),
                "repeat_common_candidate_count": min(common_counts) if common_counts else 0,
                "editorial_repeat_spearman": round_or_none(mean(editorial_corrs) if editorial_corrs else None),
                "performance_repeat_spearman": round_or_none(mean(performance_corrs) if performance_corrs else None),
                "editorial_repeat_mae": round_or_none(mean(editorial_errors) if editorial_errors else None),
                "performance_repeat_mae": round_or_none(mean(performance_errors) if performance_errors else None),
                "mean_confidence": round_or_none(mean([float(row["confidence"]) for row in group])),
            }
        )
    return output


def score_distributions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["judge_run_id"]].append(row)
    output: list[dict[str, Any]] = []
    for run_id, group in sorted(by_run.items()):
        scored = [row for row in group if as_float(row.get("editorial_score")) is not None]
        editorial_values = [
            float(row[f"editorial_{dimension}"])
            for row in scored for dimension in EDITORIAL_DIMENSIONS
        ]
        performance_values = [
            float(row[f"performance_{dimension}"])
            for row in scored for dimension in PERFORMANCE_DIMENSIONS
        ]
        editorial_scores = [float(row["editorial_score"]) for row in scored]
        performance_scores = [float(row["performance_score"]) for row in scored]
        output.append(
            {
                "judge_run_id": run_id,
                "scored_row_count": len(scored),
                "editorial_score_mean": round_or_none(mean(editorial_scores) if editorial_scores else None),
                "performance_score_mean": round_or_none(mean(performance_scores) if performance_scores else None),
                "editorial_dimension_4plus_rate": round_or_none(
                    sum(value >= 4 for value in editorial_values) / len(editorial_values) if editorial_values else None
                ),
                "performance_dimension_4plus_rate": round_or_none(
                    sum(value >= 4 for value in performance_values) / len(performance_values) if performance_values else None
                ),
            }
        )
    return output


def inter_model_correlations(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in aggregates:
        editorial = as_float(row.get("editorial_score_mean"))
        performance = as_float(row.get("performance_score_mean"))
        if editorial is not None and performance is not None:
            by_run[row["judge_run_id"]][row["candidate_id"]] = {
                "editorial": editorial,
                "performance": performance,
            }
    output: list[dict[str, Any]] = []
    for left_run, right_run in itertools.combinations(sorted(by_run), 2):
        common = sorted(set(by_run[left_run]) & set(by_run[right_run]))
        output.append(
            {
                "left_judge_run_id": left_run,
                "right_judge_run_id": right_run,
                "candidate_count": len(common),
                "editorial_spearman": round_or_none(spearman(
                    [by_run[left_run][candidate_id]["editorial"] for candidate_id in common],
                    [by_run[right_run][candidate_id]["editorial"] for candidate_id in common],
                )),
                "performance_spearman": round_or_none(spearman(
                    [by_run[left_run][candidate_id]["performance"] for candidate_id in common],
                    [by_run[right_run][candidate_id]["performance"] for candidate_id in common],
                )),
            }
        )
    return output


def auc(high_scores: list[float], low_scores: list[float]) -> float | None:
    comparisons = len(high_scores) * len(low_scores)
    if not comparisons:
        return None
    wins = sum(high > low for high in high_scores for low in low_scores)
    ties = sum(high == low for high in high_scores for low in low_scores)
    return (wins + (0.5 * ties)) / comparisons


def percentile(row: dict[str, str]) -> float | None:
    direct = as_float(row.get("channel_performance_percentile"))
    if direct is not None:
        return direct
    match = re.search(r"(?:채널내백분위|channel_percentile)\s*=\s*([0-9]+(?:\.[0-9]+)?)", row.get("source_notes", ""))
    return float(match.group(1)) if match else None


def performance_metrics(aggregates: list[dict[str, Any]], sources: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_id = {row["candidate_id"]: row for row in sources if row.get("source_system") == "gold"}
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        source = source_by_id.get(row["candidate_id"])
        score = as_float(row.get("performance_score_mean"))
        if source and score is not None:
            by_run[row["judge_run_id"]].append({"score": score, "source": source})

    group_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    for run_id, records in sorted(by_run.items()):
        groups = {
            label: [record["score"] for record in records if record["source"].get("performance_label", "").lower() == label]
            for label in ("pos", "neg")
        }
        group_rows.append(
            {
                "judge_run_id": run_id,
                "pos_count": len(groups["pos"]),
                "neg_count": len(groups["neg"]),
                "pos_mean_performance_score": round_or_none(mean(groups["pos"]) if groups["pos"] else None),
                "neg_mean_performance_score": round_or_none(mean(groups["neg"]) if groups["neg"] else None),
                "pos_median_performance_score": round_or_none(median(groups["pos"]) if groups["pos"] else None),
                "neg_median_performance_score": round_or_none(median(groups["neg"]) if groups["neg"] else None),
                "mean_score_gap": round_or_none(mean(groups["pos"]) - mean(groups["neg"]) if groups["pos"] and groups["neg"] else None),
                "pos_over_neg_auc": round_or_none(auc(groups["pos"], groups["neg"])),
            }
        )
        metric_extractors = {
            "views": lambda source: as_float(source.get("short_views")),
            "likes": lambda source: as_float(source.get("short_likes")),
            "like_rate": lambda source: (
                (as_float(source.get("short_likes")) or 0) / (as_float(source.get("short_views")) or 1)
                if (as_float(source.get("short_views")) or 0) > 0 else None
            ),
            "channel_percentile": percentile,
        }
        for metric, extractor in metric_extractors.items():
            pairs = [(record["score"], extractor(record["source"])) for record in records]
            filtered = [(score, value) for score, value in pairs if value is not None]
            correlation_rows.append(
                {
                    "judge_run_id": run_id,
                    "performance_metric": metric,
                    "candidate_count": len(filtered),
                    "spearman": round_or_none(spearman([item[0] for item in filtered], [float(item[1]) for item in filtered])),
                }
            )
        channels = sorted({record["source"].get("channel_name", "") for record in records if record["source"].get("channel_name", "")})
        for channel in channels:
            channel_records = [record for record in records if record["source"].get("channel_name") == channel]
            pos = [record["score"] for record in channel_records if record["source"].get("performance_label", "").lower() == "pos"]
            neg = [record["score"] for record in channel_records if record["source"].get("performance_label", "").lower() == "neg"]
            channel_rows.append(
                {
                    "judge_run_id": run_id,
                    "channel_name": channel,
                    "candidate_count": len(channel_records),
                    "pos_count": len(pos),
                    "neg_count": len(neg),
                    "pos_mean_performance_score": round_or_none(mean(pos) if pos else None),
                    "neg_mean_performance_score": round_or_none(mean(neg) if neg else None),
                    "pos_over_neg_auc": round_or_none(auc(pos, neg)),
                }
            )
    return group_rows, correlation_rows, channel_rows


def human_metrics(
    aggregates: list[dict[str, Any]], human_rows: list[dict[str, str]], expected_candidates: int, required_annotators: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in human_rows:
        insufficient = row.get("insufficient_evidence", "").strip().lower() in {"1", "true", "yes", "y", "예"}
        editorial = as_float(row.get("editorial_quality_1_5"))
        performance = as_float(row.get("performance_potential_1_5"))
        if not insufficient and editorial is not None and performance is not None and 1 <= editorial <= 5 and 1 <= performance <= 5:
            valid.append({**row, "editorial": editorial, "performance": performance})
    expected_rows = expected_candidates * required_annotators
    annotators = sorted({row["annotator_id"] for row in valid})
    by_annotator = {
        annotator: {row["candidate_id"]: row for row in valid if row["annotator_id"] == annotator}
        for annotator in annotators
    }
    inter_rater: dict[str, list[float]] = {"editorial": [], "performance": []}
    for left, right in itertools.combinations(annotators, 2):
        common = sorted(set(by_annotator[left]) & set(by_annotator[right]))
        for axis in ("editorial", "performance"):
            corr = spearman(
                [by_annotator[left][candidate_id][axis] for candidate_id in common],
                [by_annotator[right][candidate_id][axis] for candidate_id in common],
            )
            if corr is not None:
                inter_rater[axis].append(corr)
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_candidate[row["candidate_id"]].append(row)
    human_means = {
        candidate_id: {
            "editorial": mean([row["editorial"] for row in rows]),
            "performance": mean([row["performance"] for row in rows]),
        }
        for candidate_id, rows in by_candidate.items()
    }
    model_rows: list[dict[str, Any]] = []
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        if row["candidate_id"] in human_means:
            by_run[row["judge_run_id"]].append(row)
    for run_id, rows in sorted(by_run.items()):
        model_rows.append(
            {
                "judge_run_id": run_id,
                "candidate_count": len(rows),
                "editorial_human_spearman": round_or_none(spearman(
                    [float(row["editorial_score_mean"]) for row in rows],
                    [human_means[row["candidate_id"]]["editorial"] for row in rows],
                )),
                "performance_human_spearman": round_or_none(spearman(
                    [float(row["performance_score_mean"]) for row in rows],
                    [human_means[row["candidate_id"]]["performance"] for row in rows],
                )),
            }
        )
    summary = {
        "completed_score_row_count": len(valid),
        "expected_score_row_count": expected_rows,
        "score_coverage": round_or_none(len(valid) / expected_rows if expected_rows else None),
        "rated_candidate_count": len(human_means),
        "annotator_count": len(annotators),
        "editorial_inter_rater_spearman": round_or_none(mean(inter_rater["editorial"]) if inter_rater["editorial"] else None),
        "performance_inter_rater_spearman": round_or_none(mean(inter_rater["performance"]) if inter_rater["performance"] else None),
    }
    return model_rows, summary


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Gold Pointwise Judge 검증 보고서",
        "",
        f"- validation status: `{summary['validation_status']}`",
        f"- Gold candidates: {summary['counts']['candidate_count']}",
        f"- Pos/Neg/Unlabeled: {summary['counts']['pos_count']}/{summary['counts']['neg_count']}/{summary['counts']['unlabeled_count']}",
        "",
        "## 모델 반복 안정성",
        "",
    ]
    for row in summary["model_metrics"]:
        lines.append(
            f"- {row['judge_run_id']}: coverage={row['candidate_scoring_coverage']}, "
            f"editorial rho={row['editorial_repeat_spearman']}, performance rho={row['performance_repeat_spearman']}"
        )
    lines.extend(["", "## Pos vs Neg 성과 정합성", ""])
    for row in summary["performance_group_metrics"]:
        lines.append(
            f"- {row['judge_run_id']}: Pos={row['pos_mean_performance_score']}, "
            f"Neg={row['neg_mean_performance_score']}, gap={row['mean_score_gap']}, AUC={row['pos_over_neg_auc']}"
        )
    lines.extend(["", "## 인간 평가", ""])
    if summary["human"]["score_coverage"] == 1.0:
        lines.append("- 3인 pointwise 인간 평가가 완료되어 모델-인간 상관을 계산했다.")
    else:
        lines.append("- 3인 pointwise 인간 평가가 비어 있어 최종 Judge 타당성 검증은 대기 상태다.")
    lines.extend(
        [
            "",
            "## 해석 원칙",
            "",
            "- Pos/Neg와 조회 성과는 모델 입력에서 제외하고 평가 후에만 결합한다.",
            "- Pos>Neg AUC는 외부 성과 정합성이지 콘텐츠 품질의 절대 정답이 아니다.",
            "- 반복 안정성과 인간 점수 상관을 통과하기 전에는 validated Judge로 선언하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pointwise Gold Judge validity.")
    parser.add_argument("--scores", action="append", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--human-scores")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    all_rows = [row for path in args.scores for row in read_csv(Path(path))]
    rows = [row for row in all_rows if row.get("dry_run", "").strip().lower() not in {"true", "1", "yes"}]
    sources = read_csv(Path(args.sources))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    gates = dict(config["gates"])
    aggregates = aggregate_scores(rows)
    model_metrics = repeat_metrics(rows)
    distribution_metrics = score_distributions(rows)
    inter_model_metrics = inter_model_correlations(aggregates)
    group_metrics, correlations, channel_metrics = performance_metrics(aggregates, sources)
    required_annotators = int(gates.get("required_human_annotators", 3))
    human_rows = read_csv(Path(args.human_scores)) if args.human_scores and Path(args.human_scores).exists() else []
    human_candidate_count = len({row.get("candidate_id", "") for row in human_rows if row.get("candidate_id", "")})
    human_model_metrics, human_summary = human_metrics(
        aggregates,
        human_rows,
        human_candidate_count or len({row["candidate_id"] for row in sources}),
        required_annotators,
    )
    human_lookup = {row["judge_run_id"]: row for row in human_model_metrics}
    group_lookup = {row["judge_run_id"]: row for row in group_metrics}
    human_complete = human_summary.get("score_coverage") == 1.0
    model_gates = []
    for row in model_metrics:
        run_id = row["judge_run_id"]
        reliability = (
            (row.get("candidate_scoring_coverage") or 0) >= float(gates["min_candidate_coverage"])
            and (row.get("editorial_repeat_spearman") or 0) >= float(gates["min_repeat_spearman"])
            and (row.get("performance_repeat_spearman") or 0) >= float(gates["min_repeat_spearman"])
        )
        human_row = human_lookup.get(run_id, {})
        human_pass = (
            human_complete
            and (human_row.get("editorial_human_spearman") or 0) >= float(gates["min_human_spearman"])
            and (human_row.get("performance_human_spearman") or 0) >= float(gates["min_human_spearman"])
        )
        performance_pass = (group_lookup.get(run_id, {}).get("pos_over_neg_auc") or 0) >= float(gates["min_pos_over_neg_auc"])
        status = "validated" if reliability and human_pass else "pending_human_scores" if not human_complete else "needs_revision"
        model_gates.append(
            {
                "judge_run_id": run_id,
                "reliability_gate_pass": reliability,
                "human_alignment_gate_pass": human_pass,
                "performance_alignment_diagnostic_pass": performance_pass,
                "content_judge_status": status,
            }
        )
    validation_status = (
        "validated" if any(row["content_judge_status"] == "validated" for row in model_gates)
        else "pending_human_scores" if not human_complete else "needs_revision"
    )
    gold_sources = [row for row in sources if row.get("source_system") == "gold"]
    label_counts = {label: sum(row.get("performance_label", "").lower() == label for row in gold_sources) for label in ("pos", "neg", "unlabeled")}
    summary = {
        "validation_status": validation_status,
        "gates": gates,
        "counts": {
            "candidate_count": len({row["candidate_id"] for row in gold_sources}),
            "pos_count": label_counts["pos"],
            "neg_count": label_counts["neg"],
            "unlabeled_count": label_counts["unlabeled"],
            "score_row_count": len(rows),
            "excluded_dry_run_row_count": len(all_rows) - len(rows),
        },
        "model_metrics": model_metrics,
        "score_distributions": distribution_metrics,
        "inter_model_correlations": inter_model_metrics,
        "performance_group_metrics": group_metrics,
        "performance_correlations": correlations,
        "channel_metrics": channel_metrics,
        "human_model_metrics": human_model_metrics,
        "human": human_summary,
        "model_gates": model_gates,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_fields = list(aggregates[0].keys()) if aggregates else ["judge_run_id", "candidate_id"]
    write_csv(out_dir / "pointwise_scores_aggregated.csv", aggregates, aggregate_fields)
    write_csv(out_dir / "pointwise_model_metrics.csv", model_metrics, list(model_metrics[0].keys()) if model_metrics else ["judge_run_id"])
    write_csv(out_dir / "pointwise_score_distributions.csv", distribution_metrics, list(distribution_metrics[0].keys()) if distribution_metrics else ["judge_run_id"])
    write_csv(out_dir / "pointwise_inter_model_correlations.csv", inter_model_metrics, list(inter_model_metrics[0].keys()) if inter_model_metrics else ["left_judge_run_id"])
    write_csv(out_dir / "pointwise_performance_groups.csv", group_metrics, list(group_metrics[0].keys()) if group_metrics else ["judge_run_id"])
    write_csv(out_dir / "pointwise_performance_correlations.csv", correlations, list(correlations[0].keys()) if correlations else ["judge_run_id"])
    write_csv(out_dir / "pointwise_channel_metrics.csv", channel_metrics, list(channel_metrics[0].keys()) if channel_metrics else ["judge_run_id"])
    write_csv(out_dir / "pointwise_human_alignment.csv", human_model_metrics, list(human_model_metrics[0].keys()) if human_model_metrics else ["judge_run_id"])
    write_csv(out_dir / "pointwise_model_gates.csv", model_gates, list(model_gates[0].keys()) if model_gates else ["judge_run_id"])
    (out_dir / "pointwise_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "pointwise_validation_report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
