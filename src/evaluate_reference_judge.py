from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from evaluate_llm_judge import as_float, round_or_none, spearman
from reference_judge import CHECKLIST_DIMENSIONS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["judge_run_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "예", "ㅇ"}:
        return 1
    if normalized in {"0", "false", "no", "n", "아니오", "ㄴ"}:
        return 0
    return None


def aggregate_scores(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["judge_run_id"], row["candidate_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (run_id, candidate_id), group in sorted(grouped.items()):
        scored = [row for row in group if as_float(row.get("reference_score_100")) is not None]
        item: dict[str, Any] = {
            "judge_run_id": run_id,
            "provider": group[0].get("provider", ""),
            "model": group[0].get("model", ""),
            "candidate_id": candidate_id,
            "long_video_id": group[0].get("long_video_id", ""),
            "repeat_count": len(group),
            "scored_repeat_count": len(scored),
            "abstain_repeat_count": len(group) - len(scored),
            "aggregate_status": "scored" if scored else "abstain",
        }
        for field in (
            "highlight_saliency_1_5",
            "saliency_score_100",
            "checklist_score_100",
            "reference_score_100",
            "overall_shortform_suitable",
            "confidence",
        ):
            values = [value for row in scored if (value := as_float(row.get(field))) is not None]
            item[f"{field}_mean"] = round_or_none(mean(values) if values else None)
            if field == "reference_score_100":
                item[f"{field}_std"] = round(pstdev(values), 4) if len(values) > 1 else (0.0 if values else None)
        for name in CHECKLIST_DIMENSIONS:
            values = [value for row in scored if (value := as_float(row.get(f"check_{name}"))) is not None]
            item[f"check_{name}_rate"] = round_or_none(mean(values) if values else None)
        output.append(item)
    return output


def repeat_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["judge_run_id"]].append(row)
    output: list[dict[str, Any]] = []
    for run_id, group in sorted(by_run.items()):
        by_repeat: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
        for row in group:
            reference_score = as_float(row.get("reference_score_100"))
            if reference_score is None:
                continue
            by_repeat[int(row["repeat_index"])][row["candidate_id"]] = {
                "saliency": float(row["highlight_saliency_1_5"]),
                "checklist": float(row["checklist_score_100"]),
                "reference": reference_score,
                "suitable": float(row["overall_shortform_suitable"]),
            }
        correlations: dict[str, list[float]] = {key: [] for key in ("saliency", "checklist", "reference")}
        suitable_agreement: list[float] = []
        for left, right in itertools.combinations(sorted(by_repeat), 2):
            common = sorted(set(by_repeat[left]) & set(by_repeat[right]))
            for field in correlations:
                correlation = spearman(
                    [by_repeat[left][candidate_id][field] for candidate_id in common],
                    [by_repeat[right][candidate_id][field] for candidate_id in common],
                )
                if correlation is not None:
                    correlations[field].append(correlation)
            suitable_agreement.extend(
                float(by_repeat[left][candidate_id]["suitable"] == by_repeat[right][candidate_id]["suitable"])
                for candidate_id in common
            )
        candidate_ids = {row["candidate_id"] for row in group}
        scored_ids = {row["candidate_id"] for row in group if as_float(row.get("reference_score_100")) is not None}
        output.append({
            "judge_run_id": run_id,
            "candidate_count": len(candidate_ids),
            "candidate_scoring_coverage": round_or_none(len(scored_ids) / len(candidate_ids) if candidate_ids else None),
            "repeat_count": len(by_repeat),
            "saliency_repeat_spearman": round_or_none(mean(correlations["saliency"]) if correlations["saliency"] else None),
            "checklist_repeat_spearman": round_or_none(mean(correlations["checklist"]) if correlations["checklist"] else None),
            "reference_repeat_spearman": round_or_none(mean(correlations["reference"]) if correlations["reference"] else None),
            "suitable_repeat_agreement": round_or_none(mean(suitable_agreement) if suitable_agreement else None),
        })
    return output


def auc(high_scores: list[float], low_scores: list[float]) -> float | None:
    if not high_scores or not low_scores:
        return None
    wins = sum(high > low for high in high_scores for low in low_scores)
    ties = sum(high == low for high in high_scores for low in low_scores)
    return (wins + 0.5 * ties) / (len(high_scores) * len(low_scores))


def performance_metrics(aggregates: list[dict[str, Any]], sources: list[dict[str, str]]) -> list[dict[str, Any]]:
    source_by_id = {row["candidate_id"]: row for row in sources if row.get("source_system") == "gold"}
    by_run: dict[str, list[tuple[float, dict[str, str]]]] = defaultdict(list)
    for row in aggregates:
        source = source_by_id.get(row["candidate_id"])
        score = as_float(row.get("reference_score_100_mean"))
        if source and score is not None:
            by_run[row["judge_run_id"]].append((score, source))
    output: list[dict[str, Any]] = []
    for run_id, records in sorted(by_run.items()):
        pos = [score for score, source in records if source.get("performance_label", "").lower() == "pos"]
        neg = [score for score, source in records if source.get("performance_label", "").lower() == "neg"]
        percentile_pairs = [
            (score, value)
            for score, source in records
            if (value := as_float(source.get("channel_performance_percentile"))) is not None
        ]
        like_rate_pairs: list[tuple[float, float]] = []
        for score, source in records:
            views = as_float(source.get("short_views"))
            likes = as_float(source.get("short_likes"))
            if views and likes is not None:
                like_rate_pairs.append((score, likes / views))
        output.append({
            "judge_run_id": run_id,
            "pos_count": len(pos),
            "neg_count": len(neg),
            "pos_mean_reference_score": round_or_none(mean(pos) if pos else None),
            "neg_mean_reference_score": round_or_none(mean(neg) if neg else None),
            "mean_score_gap": round_or_none(mean(pos) - mean(neg) if pos and neg else None),
            "pos_over_neg_auc": round_or_none(auc(pos, neg)),
            "channel_percentile_spearman": round_or_none(spearman(
                [pair[0] for pair in percentile_pairs], [pair[1] for pair in percentile_pairs]
            )),
            "like_rate_spearman": round_or_none(spearman(
                [pair[0] for pair in like_rate_pairs], [pair[1] for pair in like_rate_pairs]
            )),
        })
    return output


def fleiss_kappa(labels_by_item: dict[str, list[str]], categories: tuple[str, ...]) -> float | None:
    usable = [labels for labels in labels_by_item.values() if len(labels) >= 2]
    if not usable:
        return None
    total_labels = sum(len(labels) for labels in usable)
    totals = Counter(label for labels in usable for label in labels)
    expected = sum((totals[category] / total_labels) ** 2 for category in categories)
    observed_values = []
    for labels in usable:
        counts = Counter(labels)
        n = len(labels)
        observed_values.append(sum(counts[category] * (counts[category] - 1) for category in categories) / (n * (n - 1)))
    if expected >= 1.0:
        return None
    return (mean(observed_values) - expected) / (1.0 - expected)


def cohen_kappa(
    labels_by_item: dict[str, list[tuple[str, str]]], categories: tuple[str, ...]
) -> float | None:
    pairs = [
        (ordered[0][1], ordered[1][1])
        for labels in labels_by_item.values()
        if len(ordered := sorted(labels)) >= 2
    ]
    if not pairs:
        return None
    observed = mean(float(left == right) for left, right in pairs)
    left_totals = Counter(left for left, _right in pairs)
    right_totals = Counter(right for _left, right in pairs)
    expected = sum(
        (left_totals[category] / len(pairs)) * (right_totals[category] / len(pairs))
        for category in categories
    )
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def inter_rater_kappa(
    labels_by_item: dict[str, list[tuple[str, str]]], categories: tuple[str, ...], annotator_count: int
) -> float | None:
    if annotator_count == 2:
        return cohen_kappa(labels_by_item, categories)
    return fleiss_kappa(
        {
            item_id: [label for _annotator, label in labels]
            for item_id, labels in labels_by_item.items()
        },
        categories,
    )


def mean_pairwise_agreement(labels_by_item: dict[str, list[tuple[str, str]]]) -> float | None:
    agreements: list[float] = []
    for labels in labels_by_item.values():
        values = [label for _annotator, label in sorted(labels)]
        agreements.extend(
            float(left == right)
            for left, right in itertools.combinations(values, 2)
        )
    return mean(agreements) if agreements else None


def binary_f1(predicted: list[int], actual: list[int]) -> float | None:
    if len(predicted) != len(actual) or not predicted:
        return None
    tp = sum(pred == truth == 1 for pred, truth in zip(predicted, actual))
    fp = sum(pred == 1 and truth == 0 for pred, truth in zip(predicted, actual))
    fn = sum(pred == 0 and truth == 1 for pred, truth in zip(predicted, actual))
    denominator = (2 * tp) + fp + fn
    return (2 * tp / denominator) if denominator else None


def human_metrics(
    aggregates: list[dict[str, Any]], human_rows: list[dict[str, str]], required_annotators: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in human_rows:
        insufficient = parse_bool(row.get("insufficient_evidence")) == 1
        saliency = as_float(row.get("highlight_saliency_1_5"))
        checklist = {name: parse_bool(row.get(f"check_{name}")) for name in CHECKLIST_DIMENSIONS}
        suitable = parse_bool(row.get("overall_shortform_suitable"))
        if (
            not insufficient
            and saliency is not None
            and 1 <= saliency <= 5
            and all(value is not None for value in checklist.values())
            and suitable is not None
        ):
            checklist_values = [int(value) for value in checklist.values() if value is not None]
            checklist_score = 100.0 * mean(checklist_values)
            valid.append({
                **row,
                "saliency": saliency,
                "checklist": checklist_values,
                "checklist_score": checklist_score,
                "reference_score": (((saliency - 1) * 25.0) + checklist_score) / 2.0,
                "suitable": suitable,
            })

    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_candidate[row["candidate_id"]].append(row)
    complete = {candidate_id: rows for candidate_id, rows in by_candidate.items() if len(rows) >= required_annotators}
    human_consensus: dict[str, dict[str, Any]] = {}
    for candidate_id, rows in complete.items():
        suitable_mean = mean(row["suitable"] for row in rows)
        human_consensus[candidate_id] = {
            "saliency": mean(row["saliency"] for row in rows),
            "checklist": mean(row["checklist_score"] for row in rows),
            "reference": mean(row["reference_score"] for row in rows),
            "suitable": 1 if suitable_mean > 0.5 else 0 if suitable_mean < 0.5 else None,
        }

    annotators = sorted({row["annotator_id"] for row in valid})
    by_annotator = {
        annotator: {row["candidate_id"]: row for row in valid if row["annotator_id"] == annotator}
        for annotator in annotators
    }
    saliency_correlations: list[float] = []
    for left, right in itertools.combinations(annotators, 2):
        common = sorted(set(by_annotator[left]) & set(by_annotator[right]))
        correlation = spearman(
            [by_annotator[left][candidate_id]["saliency"] for candidate_id in common],
            [by_annotator[right][candidate_id]["saliency"] for candidate_id in common],
        )
        if correlation is not None:
            saliency_correlations.append(correlation)

    checklist_kappas: dict[str, float | None] = {}
    checklist_agreements: dict[str, float | None] = {}
    for index, name in enumerate(CHECKLIST_DIMENSIONS):
        labels = {
                candidate_id: [
                    (row["annotator_id"], str(row["checklist"][index]))
                    for row in sorted(rows, key=lambda item: item["annotator_id"])
                ]
                for candidate_id, rows in complete.items()
        }
        checklist_kappas[name] = inter_rater_kappa(
            labels,
            ("0", "1"),
            len(annotators),
        )
        checklist_agreements[name] = mean_pairwise_agreement(labels)
    suitable_labels = {
            candidate_id: [
                (row["annotator_id"], str(row["suitable"]))
                for row in sorted(rows, key=lambda item: item["annotator_id"])
            ]
            for candidate_id, rows in complete.items()
    }
    suitable_kappa = inter_rater_kappa(
        suitable_labels,
        ("0", "1"),
        len(annotators),
    )
    suitable_agreement = mean_pairwise_agreement(suitable_labels)

    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        if row["candidate_id"] in human_consensus and as_float(row.get("reference_score_100_mean")) is not None:
            by_run[row["judge_run_id"]].append(row)
    alignment: list[dict[str, Any]] = []
    for run_id, rows in sorted(by_run.items()):
        suitable_rows = [
            row for row in rows
            if human_consensus[row["candidate_id"]]["suitable"] is not None
        ]
        predicted_suitable = [
            int(float(row["overall_shortform_suitable_mean"]) >= 0.5)
            for row in suitable_rows
        ]
        human_suitable = [
            int(human_consensus[row["candidate_id"]]["suitable"])
            for row in suitable_rows
        ]
        alignment.append({
            "judge_run_id": run_id,
            "candidate_count": len(rows),
            "suitable_consensus_candidate_count": len(suitable_rows),
            "saliency_human_spearman": round_or_none(spearman(
                [float(row["highlight_saliency_1_5_mean"]) for row in rows],
                [human_consensus[row["candidate_id"]]["saliency"] for row in rows],
            )),
            "checklist_human_spearman": round_or_none(spearman(
                [float(row["checklist_score_100_mean"]) for row in rows],
                [human_consensus[row["candidate_id"]]["checklist"] for row in rows],
            )),
            "reference_human_spearman": round_or_none(spearman(
                [float(row["reference_score_100_mean"]) for row in rows],
                [human_consensus[row["candidate_id"]]["reference"] for row in rows],
            )),
            "suitable_human_accuracy": round_or_none(mean(
                float(pred == truth) for pred, truth in zip(predicted_suitable, human_suitable)
            ) if human_suitable else None),
            "suitable_human_f1": round_or_none(binary_f1(predicted_suitable, human_suitable)),
        })

    expected_rows = len({row.get("candidate_id", "") for row in human_rows}) * required_annotators
    completed_rows = sum(len(rows) for rows in complete.values())
    summary: dict[str, Any] = {
        "expected_score_row_count": expected_rows,
        "completed_score_row_count": completed_rows,
        "score_coverage": round_or_none(completed_rows / expected_rows if expected_rows else None),
        "rated_candidate_count": len(complete),
        "annotator_count": len(annotators),
        "inter_rater_kappa_method": "cohen_kappa" if len(annotators) == 2 else "fleiss_kappa",
        "saliency_mean_pairwise_spearman": round_or_none(mean(saliency_correlations) if saliency_correlations else None),
        "suitable_inter_rater_kappa": round_or_none(suitable_kappa),
        "suitable_raw_agreement": round_or_none(suitable_agreement),
        "checklist_inter_rater_kappa": {
            name: round_or_none(value) for name, value in checklist_kappas.items()
        },
        "checklist_raw_agreement": {
            name: round_or_none(value) for name, value in checklist_agreements.items()
        },
    }
    return alignment, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the reference-centered LLM Judge.")
    parser.add_argument("--scores", action="append", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--human-scores")
    parser.add_argument("--required-human-annotators", type=int)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    all_rows = [row for path in args.scores for row in read_csv(Path(path))]
    rows = [row for row in all_rows if row.get("dry_run", "").strip().lower() not in {"true", "1", "yes"}]
    sources = read_csv(Path(args.sources))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    gates = config["gates"]
    aggregates = aggregate_scores(rows)
    reliability = repeat_metrics(rows)
    performance = performance_metrics(aggregates, sources)
    human_rows = read_csv(Path(args.human_scores)) if args.human_scores and Path(args.human_scores).exists() else []
    configured_annotators = int(gates.get("required_human_annotators", 3))
    required_annotators = args.required_human_annotators or configured_annotators
    human_alignment, human_summary = human_metrics(aggregates, human_rows, required_annotators)
    preliminary_human_validation = required_annotators < configured_annotators

    human_by_run = {row["judge_run_id"]: row for row in human_alignment}
    gate_rows: list[dict[str, Any]] = []
    human_complete = human_summary.get("score_coverage") == 1.0
    for row in reliability:
        stable = (
            (row.get("candidate_scoring_coverage") or 0) >= float(gates["min_candidate_coverage"])
            and (row.get("reference_repeat_spearman") or 0) >= float(gates["min_repeat_spearman"])
            and (row.get("suitable_repeat_agreement") or 0) >= float(gates["min_repeat_binary_agreement"])
        )
        human = human_by_run.get(row["judge_run_id"], {})
        aligned = (
            human_complete
            and (human.get("reference_human_spearman") or 0) >= float(gates["min_human_spearman"])
            and (human.get("suitable_human_accuracy") or 0) >= float(gates["min_human_binary_accuracy"])
        )
        if stable and aligned:
            status = "validated_preliminary" if preliminary_human_validation else "validated"
        elif not human_complete:
            status = "pending_human_scores"
        else:
            status = "needs_revision_preliminary" if preliminary_human_validation else "needs_revision"
        gate_rows.append({
            "judge_run_id": row["judge_run_id"],
            "reliability_gate_pass": stable,
            "human_alignment_gate_pass": aligned,
            "judge_status": status,
        })

    summary = {
        "protocol": "reference_centered_llm_judge_v6",
        "validation_status": (
            "validated_preliminary_2_rater"
            if any(row["judge_status"] == "validated_preliminary" for row in gate_rows)
            else "validated"
            if any(row["judge_status"] == "validated" for row in gate_rows)
            else "pending_human_scores"
            if not human_complete
            else "needs_revision_preliminary_2_rater"
            if preliminary_human_validation
            else "needs_revision"
        ),
        "human_validation_mode": {
            "required_annotators": required_annotators,
            "configured_final_annotators": configured_annotators,
            "preliminary": preliminary_human_validation,
        },
        "counts": {
            "candidate_count": len({row["candidate_id"] for row in aggregates}),
            "score_row_count": len(rows),
            "pos_count": sum(row.get("performance_label", "").lower() == "pos" for row in sources),
            "neg_count": sum(row.get("performance_label", "").lower() == "neg" for row in sources),
            "unlabeled_count": sum(row.get("performance_label", "").lower() == "unlabeled" for row in sources),
        },
        "gates": gates,
        "reliability": reliability,
        "human": human_summary,
        "human_alignment": human_alignment,
        "performance_external_validity": performance,
        "model_gates": gate_rows,
    }
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "reference_scores_aggregated.csv", aggregates)
    write_csv(out_dir / "reference_repeat_metrics.csv", reliability)
    write_csv(out_dir / "reference_human_alignment.csv", human_alignment)
    write_csv(out_dir / "reference_performance_validity.csv", performance)
    write_csv(out_dir / "reference_model_gates.csv", gate_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
