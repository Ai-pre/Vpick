from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OOF = ROOT / "data" / "oof_predictions_PRIVATE.csv"
OUTPUT = ROOT / "results" / "independent_oof_audit.json"


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or len(left) != len(right):
        return math.nan
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_delta, right_delta))
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    return numerator / denominator if denominator else math.nan


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def center(values: list[float], groups: list[str]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        grouped[group].append(value)
    means = {
        group: sum(items) / len(items)
        for group, items in grouped.items()
    }
    return [
        value - means[group]
        for value, group in zip(values, groups)
    ]


def indices_for(values: Iterable[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        result[value].append(index)
    return dict(result)


def select(values: list[float], indices: list[int]) -> list[float]:
    return [values[index] for index in indices]


def pairwise_accuracy(
    y: list[float],
    scores: list[float],
    channels: list[str],
    labels: list[str] | None = None,
    min_gap: float = 0.03,
    max_gap: float | None = None,
) -> tuple[float, int]:
    correct = 0.0
    count = 0
    for indices in indices_for(channels).values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if labels is not None and labels[left] != labels[right]:
                    continue
                gap = abs(y[left] - y[right])
                if gap < min_gap or (
                    max_gap is not None and gap > max_gap
                ):
                    continue
                target = 1 if y[left] > y[right] else -1
                predicted = (
                    1
                    if scores[left] > scores[right]
                    else -1
                    if scores[left] < scores[right]
                    else 0
                )
                correct += (
                    1.0
                    if target == predicted
                    else 0.5
                    if predicted == 0
                    else 0.0
                )
                count += 1
    return (correct / count if count else math.nan), count


def binary_auc(labels: list[int], scores: list[float]) -> float:
    positive = sum(labels)
    negative = len(labels) - positive
    ranks = average_ranks(scores)
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label == 1
    )
    return (
        positive_rank_sum - positive * (positive + 1) / 2.0
    ) / (positive * negative)


def main() -> None:
    with OOF.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = [
        float(row["channel_performance_percentile_PRIVATE"]) / 100.0
        for row in rows
    ]
    scores = [float(row["oof_v14_nested"]) for row in rows]
    channels = [row["channel_name"] for row in rows]
    labels = [
        "neg" if value <= 0.20 else "pos" if value >= 0.80 else "mid"
        for value in y
    ]
    mid = [index for index, label in enumerate(labels) if label == "mid"]
    extremes = [
        index for index, label in enumerate(labels) if label in {"neg", "pos"}
    ]
    mid_pairwise, mid_pair_count = pairwise_accuracy(
        select(y, mid),
        select(scores, mid),
        select(channels, mid),
    )
    local_pairwise, local_pair_count = pairwise_accuracy(
        y,
        scores,
        channels,
        min_gap=0.10,
        max_gap=0.40,
    )
    within_label, within_label_count = pairwise_accuracy(
        y,
        scores,
        channels,
        labels=labels,
    )
    output = {
        "candidate_count": len(rows),
        "mid_candidate_count": len(mid),
        "pooled_spearman": spearman(y, scores),
        "channel_centered_spearman": spearman(
            center(y, channels),
            center(scores, channels),
        ),
        "mid_only_pooled_spearman": spearman(
            select(y, mid),
            select(scores, mid),
        ),
        "mid_only_channel_centered_spearman": spearman(
            center(select(y, mid), select(channels, mid)),
            center(select(scores, mid), select(channels, mid)),
        ),
        "mid_only_pairwise_accuracy": mid_pairwise,
        "mid_only_pair_count": mid_pair_count,
        "same_channel_local_pairwise_accuracy": local_pairwise,
        "same_channel_local_pair_count": local_pair_count,
        "within_label_pairwise_accuracy": within_label,
        "within_label_pair_count": within_label_count,
        "extremes_pos_neg_auc": binary_auc(
            [1 if labels[index] == "pos" else 0 for index in extremes],
            select(scores, extremes),
        ),
    }
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
