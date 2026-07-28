from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
TARGET_PATH = ROOT / "data" / "reproduction" / "validation_targets_94_PRIVATE.csv"
SCORE_PATH = ROOT / "data" / "reproduction" / "codex_direct_v10_scores_94.csv"
OOF_PATH = ROOT / "data" / "diagnostics" / "oof_predictions_PRIVATE.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        value = values[order[offset]]
        while end < len(order) and values[order[end]] == value:
            end += 1
        average = ((offset + 1) + end) / 2.0
        for position in range(offset, end):
            ranks[order[position]] = average
        offset = end
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
    return numerator / denominator if denominator > 0 else math.nan


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def binary_auc(labels: list[int], scores: list[float]) -> float:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return math.nan
    ranks = average_ranks(scores)
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label == 1
    )
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def center(values: list[float], groups: list[str]) -> list[float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for value, group in zip(values, groups):
        totals[group] += value
        counts[group] += 1
    means = {group: totals[group] / counts[group] for group in totals}
    return [value - means[group] for value, group in zip(values, groups)]


def indices_for(values: Iterable[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        result[value].append(index)
    return dict(result)


def select(values: list[float], indices: list[int]) -> list[float]:
    return [values[index] for index in indices]


def macro_spearman(
    y: list[float],
    score: list[float],
    channels: list[str],
) -> float:
    results = []
    for indices in indices_for(channels).values():
        value = spearman(select(y, indices), select(score, indices))
        if math.isfinite(value):
            results.append(value)
    return sum(results) / len(results) if results else math.nan


def pairwise_accuracy(
    y: list[float],
    score: list[float],
    channels: list[str],
    labels: list[str] | None = None,
    min_gap: float = 0.01,
    max_gap: float | None = None,
) -> tuple[float, int]:
    correct = 0.0
    total = 0
    for indices in indices_for(channels).values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if labels is not None and labels[left] != labels[right]:
                    continue
                gap = abs(y[left] - y[right])
                if gap < min_gap or (max_gap is not None and gap > max_gap):
                    continue
                target_direction = 1 if y[left] > y[right] else -1
                if score[left] == score[right]:
                    correct += 0.5
                else:
                    score_direction = 1 if score[left] > score[right] else -1
                    correct += 1.0 if score_direction == target_direction else 0.0
                total += 1
    return (correct / total if total else math.nan), total


def top_quintile_precision(
    y: list[float],
    score: list[float],
    channels: list[str],
) -> float:
    hits = 0
    selected = 0
    for indices in indices_for(channels).values():
        count = max(1, math.ceil(len(indices) * 0.2))
        predicted = set(sorted(indices, key=lambda index: score[index])[-count:])
        actual = set(sorted(indices, key=lambda index: y[index])[-count:])
        hits += len(predicted & actual)
        selected += count
    return hits / selected


def dcg(relevance: list[float], order: list[int]) -> float:
    return sum(
        relevance[index] / math.log2(rank + 2.0)
        for rank, index in enumerate(order)
    )


def macro_ndcg(
    y: list[float],
    score: list[float],
    channels: list[str],
) -> float:
    results = []
    for indices in indices_for(channels).values():
        predicted = sorted(indices, key=lambda index: score[index], reverse=True)
        ideal = sorted(indices, key=lambda index: y[index], reverse=True)
        denominator = dcg(y, ideal)
        results.append(dcg(y, predicted) / denominator if denominator else 0.0)
    return sum(results) / len(results)


def metrics(
    y: list[float],
    score: list[float],
    channels: list[str],
    labels: list[str],
) -> dict[str, float | int]:
    pair, pair_count = pairwise_accuracy(y, score, channels)
    local, local_count = pairwise_accuracy(
        y,
        score,
        channels,
        min_gap=0.10,
        max_gap=0.40,
    )
    within_bucket, within_bucket_count = pairwise_accuracy(
        y,
        score,
        channels,
        labels=labels,
    )
    mid_indices = [index for index, label in enumerate(labels) if label == "mid"]
    extreme_indices = [
        index for index, label in enumerate(labels) if label in {"pos", "neg"}
    ]
    mid_y = select(y, mid_indices)
    mid_score = select(score, mid_indices)
    mid_channels = select(channels, mid_indices)
    extreme_y = select(y, extreme_indices)
    extreme_score = select(score, extreme_indices)
    extreme_channels = select(channels, extreme_indices)
    return {
        "pooled_spearman": spearman(y, score),
        "channel_centered_spearman": spearman(
            center(y, channels),
            center(score, channels),
        ),
        "channel_macro_spearman": macro_spearman(y, score, channels),
        "same_channel_pairwise_accuracy": pair,
        "same_channel_pair_count": pair_count,
        "same_channel_local_pairwise_accuracy": local,
        "same_channel_local_pair_count": local_count,
        "within_label_bucket_pairwise_accuracy": within_bucket,
        "within_label_bucket_pair_count": within_bucket_count,
        "mid_only_pooled_spearman": spearman(mid_y, mid_score),
        "mid_only_channel_centered_spearman": spearman(
            center(mid_y, mid_channels),
            center(mid_score, mid_channels),
        ),
        "extremes_pos_neg_channel_centered_spearman": spearman(
            center(extreme_y, extreme_channels),
            center(extreme_score, extreme_channels),
        ),
        "extremes_pos_neg_auc": binary_auc(
            [1 if labels[index] == "pos" else 0 for index in extreme_indices],
            extreme_score,
        ),
        "top_quintile_precision": top_quintile_precision(y, score, channels),
        "channel_macro_ndcg": macro_ndcg(y, score, channels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently audit v13 OOF predictions and metric inflation."
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    targets = read_csv(TARGET_PATH)
    scores = {
        row["candidate_id"]: row
        for row in read_csv(SCORE_PATH)
    }
    oof = {
        row["candidate_id"]: row
        for row in read_csv(OOF_PATH)
    }

    candidate_ids = [row["candidate_id"] for row in targets]
    y = [
        float(row["channel_performance_percentile_PRIVATE"]) / 100.0
        for row in targets
    ]
    channels = [row["channel_name"] for row in targets]
    labels = [row["performance_label_PRIVATE"] for row in targets]
    duration = [float(row["duration_sec"]) for row in targets]
    codex_score = [
        float(scores[candidate_id]["judge_score_100"])
        for candidate_id in candidate_ids
    ]
    v13_oof = [
        float(oof[candidate_id]["oof_frozen_ensemble"])
        for candidate_id in candidate_ids
    ]

    bucket_base = {"neg": 0.0, "mid": 0.5, "pos": 1.0}
    random_generator = random.Random(20260728)
    bucket_oracle = [
        bucket_base[label] + random_generator.random() * 1e-3
        for label in labels
    ]

    comparisons = {
        "v13_repeated_grouped_oof": v13_oof,
        "codex_fixed_judge_score": codex_score,
        "duration_only": duration,
        "label_bucket_oracle_invalid_for_deployment": bucket_oracle,
    }
    output = {
        "warning": (
            "The label-bucket oracle deliberately reads diagnostic labels. "
            "It is an invalid model and exists only to expose metric inflation."
        ),
        "metrics": {
            name: metrics(y, values, channels, labels)
            for name, values in comparisons.items()
        },
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
