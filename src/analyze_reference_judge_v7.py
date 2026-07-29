from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from reference_judge_v7 import CHECK_DIMENSIONS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: str | None) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = statistics.mean(left_ranks)
    right_mean = statistics.mean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    )
    return numerator / denominator if denominator else None


def rounded(value: float | None, digits: int = 4) -> float | str:
    return "" if value is None else round(value, digits)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze v7 reference Judge validity.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--v6-scores")
    args = parser.parse_args()

    scores = read_csv(Path(args.scores))
    labels = {row["candidate_id"]: row for row in read_csv(Path(args.labels))}
    score_by_id = {row["candidate_id"]: row for row in scores}
    if set(score_by_id) != set(labels):
        raise ValueError("v7 scores and private labels must contain identical candidate IDs")

    joined: list[dict[str, Any]] = []
    for candidate_id, score in score_by_id.items():
        label = labels[candidate_id]
        joined.append(
            {
                "pair_id": label.get("pair_id", ""),
                "candidate_id": candidate_id,
                "performance_label": label.get("performance_label", ""),
                "channel_name": label.get("channel_name", ""),
                "short_video_id": label.get("short_video_id", ""),
                "channel_performance_percentile": label.get("channel_performance_percentile", ""),
                **score,
            }
        )

    scored = [row for row in joined if row["verdict"] == "score"]
    pos = [row for row in scored if row["performance_label"] == "pos"]
    neg = [row for row in scored if row["performance_label"] == "neg"]
    labels_binary = [int(row["performance_label"] == "pos") for row in scored]
    checklist_scores = [float(row["checklist_score_100"]) for row in scored]
    saliency_scores = [float(row["saliency_market_1_5"]) for row in scored]

    summary: dict[str, Any] = {
        "prompt_id": "shortform_reference_judge_v7_ko",
        "primary_score": "checklist_score_100",
        "candidate_count": len(joined),
        "scored_count": len(scored),
        "abstain_count": len(joined) - len(scored),
        "description_nonempty_count": 0,
        "description_empty_count": len(joined),
        "pos_scored_count": len(pos),
        "neg_scored_count": len(neg),
        "checklist_pos_mean_100": rounded(statistics.mean(float(row["checklist_score_100"]) for row in pos)),
        "checklist_neg_mean_100": rounded(statistics.mean(float(row["checklist_score_100"]) for row in neg)),
        "checklist_pos_minus_neg_100": rounded(
            statistics.mean(float(row["checklist_score_100"]) for row in pos)
            - statistics.mean(float(row["checklist_score_100"]) for row in neg)
        ),
        "checklist_pos_neg_auc": rounded(roc_auc(labels_binary, checklist_scores)),
        "saliency_pos_neg_auc": rounded(roc_auc(labels_binary, saliency_scores)),
        "overall_suitable_pos_rate": rounded(
            statistics.mean(float(row["overall_shortform_suitable"]) for row in pos)
        ),
        "overall_suitable_neg_rate": rounded(
            statistics.mean(float(row["overall_shortform_suitable"]) for row in neg)
        ),
        "saliency_5_count": sum(float(row["saliency_market_1_5"]) == 5 for row in scored),
        "saliency_4_or_5_count": sum(float(row["saliency_market_1_5"]) >= 4 for row in scored),
        "saliency_5_rate": rounded(
            sum(float(row["saliency_market_1_5"]) == 5 for row in scored) / len(scored)
        ),
        "saliency_4_or_5_rate": rounded(
            sum(float(row["saliency_market_1_5"]) >= 4 for row in scored) / len(scored)
        ),
    }

    dimension_rows: list[dict[str, Any]] = []
    for dimension in CHECK_DIMENSIONS:
        field = f"check_{dimension}"
        pos_mean = statistics.mean(float(row[field]) for row in pos)
        neg_mean = statistics.mean(float(row[field]) for row in neg)
        dimension_rows.append(
            {
                "dimension": dimension,
                "pos_mean_0_2": rounded(pos_mean),
                "neg_mean_0_2": rounded(neg_mean),
                "pos_minus_neg_0_2": rounded(pos_mean - neg_mean),
            }
        )

    channel_rows: list[dict[str, Any]] = []
    for channel in sorted({row["channel_name"] for row in scored}):
        channel_rows_raw = [row for row in scored if row["channel_name"] == channel]
        channel_pos = [row for row in channel_rows_raw if row["performance_label"] == "pos"]
        channel_neg = [row for row in channel_rows_raw if row["performance_label"] == "neg"]
        pos_mean = (
            statistics.mean(float(row["checklist_score_100"]) for row in channel_pos)
            if channel_pos
            else None
        )
        neg_mean = (
            statistics.mean(float(row["checklist_score_100"]) for row in channel_neg)
            if channel_neg
            else None
        )
        channel_rows.append(
            {
                "channel_name": channel,
                "pos_count": len(channel_pos),
                "neg_count": len(channel_neg),
                "pos_mean_100": rounded(pos_mean),
                "neg_mean_100": rounded(neg_mean),
                "pos_minus_neg_100": (
                    rounded(pos_mean - neg_mean)
                    if pos_mean is not None and neg_mean is not None
                    else ""
                ),
            }
        )

    if args.v6_scores:
        v6 = {
            row["candidate_id"]: row
            for row in read_csv(Path(args.v6_scores))
        }
        common = [
            row
            for row in scored
            if row["candidate_id"] in v6
            and number(v6[row["candidate_id"]].get("reference_score_mean_100")) is not None
        ]
        v7_values = [float(row["checklist_score_100"]) for row in common]
        v6_values = [
            float(v6[row["candidate_id"]]["reference_score_mean_100"])
            for row in common
        ]
        summary["v6_common_scored_count"] = len(common)
        summary["v6_v7_score_spearman"] = rounded(spearman(v6_values, v7_values))
        summary["v6_reported_pos_neg_auc"] = 0.5661
        summary["v7_auc_minus_v6_auc"] = rounded(
            float(summary["checklist_pos_neg_auc"]) - 0.5661
        )

    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "reference_judge_v7_labeled_diagnostics.csv",
        joined,
        list(joined[0]),
    )
    write_csv(
        out_dir / "reference_judge_v7_dimension_gaps.csv",
        dimension_rows,
        ["dimension", "pos_mean_0_2", "neg_mean_0_2", "pos_minus_neg_0_2"],
    )
    write_csv(
        out_dir / "reference_judge_v7_channel_metrics.csv",
        channel_rows,
        [
            "channel_name", "pos_count", "neg_count",
            "pos_mean_100", "neg_mean_100", "pos_minus_neg_100",
        ],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_judge_v7_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
