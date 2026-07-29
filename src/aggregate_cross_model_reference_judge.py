from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: str | None) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def binary(value: str | None) -> int | None:
    parsed = numeric(value)
    if parsed is None:
        return None
    return int(parsed >= 0.5)


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


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def load_model_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    by_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id", "").strip()
        if candidate_id:
            by_id.setdefault(candidate_id, []).append(row)

    output: dict[str, dict[str, Any]] = {}
    for candidate_id, items in by_id.items():
        scores = [
            value
            for row in items
            if (value := numeric(row.get("reference_score_100") or row.get("reference_score_mean_100"))) is not None
        ]
        suitable = [
            value
            for row in items
            if (value := numeric(
                row.get("overall_shortform_suitable")
                or row.get("overall_shortform_suitable_vote_rate")
            )) is not None
        ]
        verdicts = [row.get("verdict", "") for row in items]
        if not any(verdicts):
            verdicts = [
                row.get("verdict_run1", "") or row.get("verdict_run2", "")
                for row in items
            ]
        output[candidate_id] = {
            "score": statistics.mean(scores) if scores else None,
            "suitable_rate": statistics.mean(suitable) if suitable else None,
            "verdict": "score" if scores else ("abstain" if "abstain" in verdicts else ""),
            "confidence": numeric(
                items[0].get("confidence")
                or items[0].get("confidence_mean_1_5")
            ),
            "reason": items[0].get("reason") or items[0].get("reason_run1", ""),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate blind cross-model reference Judge outputs.")
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Model source in name=path form. Repeat for every model.",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    labels = read_csv(Path(args.labels))
    label_by_id = {row["candidate_id"]: row for row in labels}
    sources: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.source:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --source {spec!r}; expected name=path")
        sources[name.strip()] = load_model_rows(Path(raw_path.strip()))

    candidate_rows: list[dict[str, Any]] = []
    for candidate_id, label_row in sorted(
        label_by_id.items(),
        key=lambda item: (item[1].get("pair_id", ""), item[0]),
    ):
        row: dict[str, Any] = {
            "pair_id": label_row.get("pair_id", ""),
            "candidate_id": candidate_id,
            "performance_label": label_row.get("performance_label", ""),
            "channel_name": label_row.get("channel_name", ""),
            "short_video_id": label_row.get("short_video_id", ""),
            "channel_performance_percentile": label_row.get("channel_performance_percentile", ""),
        }
        scores: list[float] = []
        suitable_rates: list[float] = []
        for name, model_rows in sources.items():
            model = model_rows.get(candidate_id, {})
            score = model.get("score")
            suitable_rate = model.get("suitable_rate")
            row[f"{name}_verdict"] = model.get("verdict", "")
            row[f"{name}_reference_score_100"] = "" if score is None else round(score, 3)
            row[f"{name}_overall_suitable_rate"] = (
                "" if suitable_rate is None else round(suitable_rate, 3)
            )
            row[f"{name}_confidence_1_5"] = (
                "" if model.get("confidence") is None else model["confidence"]
            )
            if score is not None:
                scores.append(score)
            if suitable_rate is not None:
                suitable_rates.append(suitable_rate)

        suitable_mean = statistics.mean(suitable_rates) if suitable_rates else None
        row.update(
            {
                "consensus_model_count": len(scores),
                "consensus_reference_score_mean_100": (
                    "" if not scores else round(statistics.mean(scores), 3)
                ),
                "consensus_reference_score_std_100": (
                    "" if len(scores) < 2 else round(statistics.pstdev(scores), 3)
                ),
                "consensus_score_range_100": (
                    "" if len(scores) < 2 else round(max(scores) - min(scores), 3)
                ),
                "consensus_overall_suitable_rate": (
                    "" if suitable_mean is None else round(suitable_mean, 3)
                ),
                "consensus_overall_suitable": (
                    ""
                    if suitable_mean is None or suitable_mean == 0.5
                    else int(suitable_mean > 0.5)
                ),
            }
        )
        candidate_rows.append(row)

    model_summary: list[dict[str, Any]] = []
    for name, model_rows in sources.items():
        scored = [
            (label_by_id[candidate_id].get("performance_label"), data["score"])
            for candidate_id, data in model_rows.items()
            if candidate_id in label_by_id and data["score"] is not None
        ]
        positives = [score for label, score in scored if label == "pos"]
        negatives = [score for label, score in scored if label == "neg"]
        auc_labels = [int(label == "pos") for label, _ in scored if label in {"pos", "neg"}]
        auc_scores = [score for label, score in scored if label in {"pos", "neg"}]
        model_summary.append(
            {
                "model": name,
                "candidate_count": len(label_by_id),
                "scored_count": len(scored),
                "abstain_or_missing_count": len(label_by_id) - len(scored),
                "pos_mean_score_100": round(statistics.mean(positives), 3) if positives else "",
                "neg_mean_score_100": round(statistics.mean(negatives), 3) if negatives else "",
                "pos_minus_neg_100": (
                    round(statistics.mean(positives) - statistics.mean(negatives), 3)
                    if positives and negatives
                    else ""
                ),
                "pos_neg_auc": (
                    round(value, 4)
                    if (value := roc_auc(auc_labels, auc_scores)) is not None
                    else ""
                ),
            }
        )

    agreement_rows: list[dict[str, Any]] = []
    names = list(sources)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            common = sorted(set(sources[left_name]) & set(sources[right_name]))
            paired = [
                (sources[left_name][candidate_id], sources[right_name][candidate_id])
                for candidate_id in common
                if sources[left_name][candidate_id]["score"] is not None
                and sources[right_name][candidate_id]["score"] is not None
            ]
            left_scores = [left["score"] for left, _ in paired]
            right_scores = [right["score"] for _, right in paired]
            suitable_pairs = [
                (binary(str(left["suitable_rate"])), binary(str(right["suitable_rate"])))
                for left, right in paired
                if left["suitable_rate"] is not None and right["suitable_rate"] is not None
            ]
            suitable_agreement = [
                int(left == right)
                for left, right in suitable_pairs
                if left is not None and right is not None
            ]
            agreement_rows.append(
                {
                    "left_model": left_name,
                    "right_model": right_name,
                    "common_scored_count": len(paired),
                    "score_spearman": (
                        round(value, 4)
                        if (value := spearman(left_scores, right_scores)) is not None
                        else ""
                    ),
                    "overall_suitable_agreement": (
                        round(statistics.mean(suitable_agreement), 4)
                        if suitable_agreement
                        else ""
                    ),
                    "mean_absolute_score_difference_100": (
                        round(statistics.mean(
                            abs(left - right)
                            for left, right in zip(left_scores, right_scores)
                        ), 3)
                        if paired
                        else ""
                    ),
                }
            )

    out_dir = Path(args.out_dir)
    candidate_fields = [
        "pair_id", "candidate_id", "performance_label", "channel_name",
        "short_video_id", "channel_performance_percentile",
    ]
    for name in sources:
        candidate_fields.extend(
            [
                f"{name}_verdict",
                f"{name}_reference_score_100",
                f"{name}_overall_suitable_rate",
                f"{name}_confidence_1_5",
            ]
        )
    candidate_fields.extend(
        [
            "consensus_model_count",
            "consensus_reference_score_mean_100",
            "consensus_reference_score_std_100",
            "consensus_score_range_100",
            "consensus_overall_suitable_rate",
            "consensus_overall_suitable",
        ]
    )
    write_csv(out_dir / "vpick_cross_model_judge_scores_60.csv", candidate_rows, candidate_fields)
    write_csv(
        out_dir / "cross_model_summary.csv",
        model_summary,
        [
            "model", "candidate_count", "scored_count", "abstain_or_missing_count",
            "pos_mean_score_100", "neg_mean_score_100", "pos_minus_neg_100", "pos_neg_auc",
        ],
    )
    write_csv(
        out_dir / "cross_model_agreement.csv",
        agreement_rows,
        [
            "left_model", "right_model", "common_scored_count", "score_spearman",
            "overall_suitable_agreement", "mean_absolute_score_difference_100",
        ],
    )
    summary = {
        "candidate_count": len(candidate_rows),
        "models": names,
        "complete_consensus_count": sum(
            row["consensus_model_count"] == len(sources)
            for row in candidate_rows
        ),
    }
    (out_dir / "cross_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
