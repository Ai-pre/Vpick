from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from evaluate_llm_judge import spearman
from evaluate_reference_judge import auc
from reference_judge import CHECKLIST_DIMENSIONS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def round4(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def metric_row(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [record for record in records if record["label"] == "pos"]
    neg = [record for record in records if record["label"] == "neg"]
    pos_scores = [record["score"] for record in pos]
    neg_scores = [record["score"] for record in neg]
    tp = sum(record["suitable"] for record in pos)
    fn = len(pos) - tp
    tn = sum(not record["suitable"] for record in neg)
    fp = len(neg) - tn
    tpr = tp / len(pos) if pos else None
    tnr = tn / len(neg) if neg else None
    balanced_accuracy = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
    return {
        "subset": name,
        "scored_count": len(records),
        "pos_count": len(pos),
        "neg_count": len(neg),
        "pos_mean_reference_score": round4(mean(pos_scores) if pos_scores else None),
        "neg_mean_reference_score": round4(mean(neg_scores) if neg_scores else None),
        "mean_score_gap": round4(mean(pos_scores) - mean(neg_scores) if pos_scores and neg_scores else None),
        "pos_over_neg_auc": round4(auc(pos_scores, neg_scores)),
        "suitable_balanced_accuracy": round4(balanced_accuracy),
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
    }


def human_reference_scores(rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, bool], list[str]]:
    scores: dict[str, list[float]] = defaultdict(list)
    suitable: dict[str, list[bool]] = defaultdict(list)
    annotators: set[str] = set()
    for row in rows:
        saliency = as_float(row.get("highlight_saliency_1_5"))
        checklist = [as_float(row.get(f"check_{name}")) for name in CHECKLIST_DIMENSIONS]
        suitable_value = as_float(row.get("overall_shortform_suitable"))
        if saliency is None or suitable_value is None or any(value is None for value in checklist):
            continue
        annotators.add(row.get("annotator_id", ""))
        saliency_score = (saliency - 1.0) * 25.0
        checklist_score = mean(float(value) for value in checklist) * 100.0
        scores[row["candidate_id"]].append(0.5 * saliency_score + 0.5 * checklist_score)
        suitable[row["candidate_id"]].append(bool(suitable_value))
    return (
        {candidate_id: mean(values) for candidate_id, values in scores.items()},
        {candidate_id: mean(values) >= 0.5 for candidate_id, values in suitable.items()},
        sorted(annotator for annotator in annotators if annotator),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze balanced Reference Judge results by evidence and channel.")
    parser.add_argument("--aggregates", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--human-scores", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    aggregates = read_csv(args.aggregates)
    sources = read_csv(args.sources)
    source_by_id = {row["candidate_id"]: row for row in sources}
    records: list[dict[str, Any]] = []
    for row in aggregates:
        source = source_by_id.get(row["candidate_id"])
        score = as_float(row.get("reference_score_100_mean"))
        suitable = as_float(row.get("overall_shortform_suitable_mean"))
        if not source or score is None or suitable is None:
            continue
        records.append(
            {
                "candidate_id": row["candidate_id"],
                "pair_id": source.get("pair_id", ""),
                "channel_name": source.get("channel_name", ""),
                "label": source.get("performance_label", "").lower(),
                "score": score,
                "suitable": suitable >= 0.5,
                "evidence_status": source.get("performance_evidence_status", ""),
                "alignment_status": source.get("alignment_status", ""),
            }
        )

    clean_statuses = {"continuous", "light_edit"}
    subsets: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("all_scored", lambda record: True),
        ("verified_performance_snapshot", lambda record: record["evidence_status"] == "verified_channel_percentile_snapshot"),
        ("pending_stats_snapshot", lambda record: record["evidence_status"] == "teammate_selected_pending_stats_snapshot"),
        ("clean_alignment", lambda record: record["alignment_status"] in clean_statuses),
        (
            "verified_and_clean_alignment",
            lambda record: record["evidence_status"] == "verified_channel_percentile_snapshot"
            and record["alignment_status"] in clean_statuses,
        ),
    ]
    subset_metrics = [metric_row(name, [record for record in records if predicate(record)]) for name, predicate in subsets]
    write_csv(args.out_dir / "balanced_subset_metrics.csv", subset_metrics)

    channel_metrics = [
        metric_row(channel, [record for record in records if record["channel_name"] == channel])
        for channel in sorted({record["channel_name"] for record in records})
    ]
    write_csv(args.out_dir / "balanced_channel_metrics.csv", channel_metrics)

    errors = sorted(
        records,
        key=lambda record: (
            0 if (record["label"] == "neg" and record["suitable"]) or (record["label"] == "pos" and not record["suitable"]) else 1,
            -record["score"] if record["label"] == "neg" else record["score"],
        ),
    )
    write_csv(args.out_dir / "balanced_candidate_diagnostics.csv", errors)

    human_summary: dict[str, Any] = {}
    if args.human_scores and args.human_scores.exists():
        human_scores, human_suitable, annotators = human_reference_scores(read_csv(args.human_scores))
        overlap = [record for record in records if record["candidate_id"] in human_scores]
        human_summary = {
            "completed_annotators": annotators,
            "overlap_candidate_count": len(overlap),
            "reference_score_spearman": round4(
                spearman(
                    [record["score"] for record in overlap],
                    [human_scores[record["candidate_id"]] for record in overlap],
                )
            ),
            "suitable_accuracy": round4(
                mean(
                    record["suitable"] == human_suitable[record["candidate_id"]]
                    for record in overlap
                )
                if overlap
                else None
            ),
        }

    summary = {
        "candidate_count": len(sources),
        "scored_count": len(records),
        "abstain_count": len(sources) - len(records),
        "subset_metrics": subset_metrics,
        "human_two_rater_diagnostic": human_summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "balanced_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
