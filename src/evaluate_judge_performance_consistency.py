from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LABEL_VALUE = {"neg": 0, "mid": 1, "pos": 2}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
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


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    return pearson(average_ranks(left), average_ranks(right))


def qwk(actual: list[int], predicted: list[int], classes: int = 3) -> float | None:
    if not actual or len(actual) != len(predicted):
        return None
    observed = [[0.0] * classes for _ in range(classes)]
    actual_hist = [0.0] * classes
    predicted_hist = [0.0] * classes
    for left, right in zip(actual, predicted):
        observed[left][right] += 1
        actual_hist[left] += 1
        predicted_hist[right] += 1
    total = float(len(actual))
    numerator = 0.0
    denominator = 0.0
    for left in range(classes):
        for right in range(classes):
            weight = ((left - right) / (classes - 1)) ** 2
            expected = actual_hist[left] * predicted_hist[right] / total
            numerator += weight * observed[left][right]
            denominator += weight * expected
    return 1.0 - numerator / denominator if denominator else None


def rank_classes(rows: list[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (float(row["score"]), str(row["candidate_id"])),
    )
    count = len(ordered)
    neg_end = round(count * 30 / 94)
    mid_end = round(count * 64 / 94)
    output: dict[str, int] = {}
    for index, row in enumerate(ordered):
        output[row["candidate_id"]] = 0 if index < neg_end else 1 if index < mid_end else 2
    return output


def score_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [LABEL_VALUE[row["label"]] for row in rows]
    predicted_by_id = rank_classes(rows)
    predicted = [predicted_by_id[row["candidate_id"]] for row in rows]
    scores = [float(row["score"]) for row in rows]
    percentiles = [float(row["percentile"]) for row in rows]

    channel_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        channel_values[row["channel_name"]].append(row)
    channel_spearman = {
        channel: spearman(
            [float(row["score"]) for row in members],
            [float(row["percentile"]) for row in members],
        )
        for channel, members in sorted(channel_values.items())
    }
    valid_channel = [
        value for value in channel_spearman.values() if value is not None
    ]
    centered_scores: list[float] = []
    centered_targets: list[float] = []
    for members in channel_values.values():
        member_scores = [float(row["score"]) for row in members]
        member_targets = [float(row["percentile"]) for row in members]
        score_mean = statistics.mean(member_scores)
        target_mean = statistics.mean(member_targets)
        centered_scores.extend(value - score_mean for value in member_scores)
        centered_targets.extend(value - target_mean for value in member_targets)
    frequencies = {
        value: scores.count(value)
        for value in set(scores)
    }
    return {
        "candidate_count": len(rows),
        "three_class_accuracy": sum(a == b for a, b in zip(actual, predicted))
        / len(actual),
        "quadratic_weighted_kappa": qwk(actual, predicted),
        "pooled_spearman_score_percentile": spearman(scores, percentiles),
        "within_channel_centered_spearman": spearman(
            centered_scores,
            centered_targets,
        ),
        "channel_macro_spearman": (
            statistics.mean(valid_channel) if valid_channel else None
        ),
        "channel_spearman": channel_spearman,
        "unique_score_count": len(set(scores)),
        "largest_tie_group": max(frequencies.values()) if frequencies else 0,
    }


def random_control(
    base_rows: list[dict[str, Any]],
    *,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    metrics: list[dict[str, Any]] = []
    for _ in range(simulations):
        rows = [{**row, "score": rng.random()} for row in base_rows]
        metrics.append(score_metrics(rows))

    def distribution(key: str) -> dict[str, float]:
        values = sorted(
            float(row[key])
            for row in metrics
            if row.get(key) is not None
        )
        return {
            "mean": statistics.mean(values),
            "p2_5": values[max(0, round(0.025 * (len(values) - 1)))],
            "p97_5": values[min(len(values) - 1, round(0.975 * (len(values) - 1)))],
        }

    return {
        "simulations": simulations,
        "seed": seed,
        "three_class_accuracy": distribution("three_class_accuracy"),
        "quadratic_weighted_kappa": distribution(
            "quadratic_weighted_kappa"
        ),
        "within_channel_centered_spearman": distribution(
            "within_channel_centered_spearman"
        ),
        "channel_macro_spearman": distribution("channel_macro_spearman"),
    }


def label_oracle_control(
    base_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    def score_subset(
        rows: list[dict[str, Any]],
        subset_seed: int,
    ) -> dict[str, Any]:
        rng = random.Random(subset_seed)
        scored = [
            {
                **row,
                "score": LABEL_VALUE[row["label"]] + rng.random(),
            }
            for row in rows
        ]
        return score_metrics(scored)

    polar = [row for row in base_rows if row["label"] in {"neg", "pos"}]
    return {
        "warning": (
            "Deliberate label-leakage upper control. It is not a deployable Judge "
            "and exists only to expose how label-aware metrics remain inflated. "
            "Adding mid examples cannot make a label oracle collapse by itself."
        ),
        "polar_60_metrics": score_subset(polar, seed),
        "all_94_metrics": score_subset(base_rows, seed + 1),
    }


def polar_bucket_shortcut_control(
    base_rows: list[dict[str, Any]],
    *,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    metrics: list[dict[str, Any]] = []
    for _ in range(simulations):
        scored: list[dict[str, Any]] = []
        for row in base_rows:
            if row["label"] == "neg":
                score = rng.random()
            elif row["label"] == "pos":
                score = 2.0 + rng.random()
            else:
                score = 3.0 * rng.random()
            scored.append({**row, "score": score})
        metrics.append(score_metrics(scored))

    def distribution(key: str) -> dict[str, float]:
        values = sorted(
            float(row[key])
            for row in metrics
            if row.get(key) is not None
        )
        return {
            "mean": statistics.mean(values),
            "p2_5": values[max(0, round(0.025 * (len(values) - 1)))],
            "p97_5": values[min(len(values) - 1, round(0.975 * (len(values) - 1)))],
        }

    return {
        "simulations": simulations,
        "seed": seed,
        "definition": (
            "POS receives a random score in [2,3), NEG in [0,1), and MID "
            "receives a random score across [0,3). This deliberately shallow "
            "control knows only the polar buckets and has no within-bucket skill."
        ),
        "three_class_accuracy": distribution("three_class_accuracy"),
        "quadratic_weighted_kappa": distribution(
            "quadratic_weighted_kappa"
        ),
        "pooled_spearman_score_percentile": distribution(
            "pooled_spearman_score_percentile"
        ),
        "within_channel_centered_spearman": distribution(
            "within_channel_centered_spearman"
        ),
        "channel_macro_spearman": distribution("channel_macro_spearman"),
    }


def source_shortcut_control(
    targets: list[dict[str, str]],
) -> dict[str, Any]:
    train = [
        row
        for row in targets
        if row.get("dataset_role_v3") == "dev"
        and row.get("performance_label_PRIVATE") in LABEL_VALUE
    ]
    test = [
        row
        for row in targets
        if row.get("dataset_role_v3") == "locked_test"
        and row.get("performance_label_PRIVATE") in LABEL_VALUE
    ]
    if not train or not test:
        return {
            "available": False,
            "reason": "dataset_role_v3 dev/locked_test rows are required",
        }

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train:
        by_source[row.get("transcript_source", "unknown")].append(row)
    global_label = statistics.mode(
        row["performance_label_PRIVATE"] for row in train
    )
    global_percentile = statistics.mean(
        float(row["channel_performance_percentile_PRIVATE"]) for row in train
    )
    source_rule: dict[str, dict[str, Any]] = {}
    for source, members in by_source.items():
        counts = {
            label: sum(
                row["performance_label_PRIVATE"] == label for row in members
            )
            for label in LABEL_VALUE
        }
        predicted_label = max(
            sorted(LABEL_VALUE),
            key=lambda label: counts[label],
        )
        source_rule[source] = {
            "predicted_label_PRIVATE": predicted_label,
            "mean_percentile_PRIVATE": statistics.mean(
                float(row["channel_performance_percentile_PRIVATE"])
                for row in members
            ),
            "dev_count": len(members),
        }

    actual: list[int] = []
    predicted: list[int] = []
    score_rows: list[dict[str, Any]] = []
    for row in test:
        rule = source_rule.get(row.get("transcript_source", "unknown"))
        predicted_label = (
            rule["predicted_label_PRIVATE"] if rule else global_label
        )
        score = (
            float(rule["mean_percentile_PRIVATE"])
            if rule
            else global_percentile
        )
        actual.append(LABEL_VALUE[row["performance_label_PRIVATE"]])
        predicted.append(LABEL_VALUE[predicted_label])
        score_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "channel_name": row["channel_name"],
                "label": row["performance_label_PRIVATE"],
                "percentile": float(
                    row["channel_performance_percentile_PRIVATE"]
                ),
                "score": score,
            }
        )
    return {
        "available": True,
        "warning": (
            "This control uses transcript_source only. The source field is not "
            "passed to the Judge, but source-specific text formatting can act as "
            "a proxy, so above-random performance indicates a confounding risk."
        ),
        "dev_count": len(train),
        "locked_test_count": len(test),
        "source_rules_PRIVATE": source_rule,
        "three_class_accuracy": sum(a == b for a, b in zip(actual, predicted))
        / len(actual),
        "quadratic_weighted_kappa": qwk(actual, predicted),
        "score_metrics": score_metrics(score_rows),
    }


def joined_rows(
    targets: list[dict[str, str]],
    scores: list[dict[str, str]] | None,
    score_field: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    score_by_id: dict[str, list[float]] = defaultdict(list)
    abstained: set[str] = set()
    if scores is not None:
        for row in scores:
            candidate_id = row["candidate_id"]
            if row.get("verdict", "score") == "abstain":
                abstained.add(candidate_id)
                continue
            value = number(row.get(score_field))
            if value is not None:
                score_by_id[candidate_id].append(value)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for target in targets:
        candidate_id = target["candidate_id"]
        label = target["performance_label_PRIVATE"]
        percentile = number(target["channel_performance_percentile_PRIVATE"])
        if label not in LABEL_VALUE or percentile is None:
            continue
        value = None
        if scores is not None and score_by_id.get(candidate_id):
            value = statistics.mean(score_by_id[candidate_id])
        if scores is not None and value is None and candidate_id not in abstained:
            missing.append(candidate_id)
        rows.append(
            {
                "candidate_id": candidate_id,
                "channel_name": target["channel_name"],
                "label": label,
                "percentile": percentile,
                "score": value,
                "dataset_role": target.get("dataset_role_v3", ""),
                "transcript_source": target.get("transcript_source", "unknown"),
            }
        )
    return rows, missing


def rows_for_role(
    rows: list[dict[str, Any]],
    role: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("dataset_role") == role]


def controls_for_rows(
    rows: list[dict[str, Any]],
    *,
    random_simulations: int,
    seed: int,
) -> dict[str, Any]:
    control_rows = [{**row, "score": 0.0} for row in rows]
    return {
        "candidate_count": len(control_rows),
        "random": random_control(
            control_rows,
            simulations=random_simulations,
            seed=seed,
        ),
        "label_oracle_sanity_only": label_oracle_control(
            control_rows,
            seed=seed,
        ),
        "polar_bucket_shortcut_sanity": polar_bucket_shortcut_control(
            control_rows,
            simulations=random_simulations,
            seed=seed + 10_000,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the auxiliary across-video performance-consistency experiment "
            "with random and deliberate label-oracle controls."
        )
    )
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--score-field", default="judge_score_100")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = read_csv(args.targets)
    scores = read_csv(args.scores) if args.scores else None
    base_rows, missing = joined_rows(targets, scores, args.score_field)
    dev_rows = rows_for_role(base_rows, "dev")
    locked_rows = rows_for_role(base_rows, "locked_test")
    summary: dict[str, Any] = {
        "experiment": "exp1_auxiliary_performance_consistency",
        "target_candidate_count": len(base_rows),
        "dev_candidate_count": len(dev_rows),
        "locked_test_candidate_count": len(locked_rows),
        "score_field": args.score_field,
        "interpretation": (
            "This is an auxiliary performance-consistency diagnostic, not the "
            "primary validity test for the segment-selection Judge."
        ),
        "split_policy": (
            "Split by longform_id. Development metrics may guide diagnosis; "
            "locked_test metrics must not be used to tune the prompt or weights."
        ),
        "controls_full_94": controls_for_rows(
            base_rows,
            random_simulations=args.random_simulations,
            seed=args.seed,
        ),
        "controls_locked_test": controls_for_rows(
            locked_rows,
            random_simulations=args.random_simulations,
            seed=args.seed + 100,
        ),
        "transcript_source_shortcut_control_PRIVATE": source_shortcut_control(
            targets
        ),
    }
    if scores is not None:
        scored = [row for row in base_rows if row["score"] is not None]
        scored_dev = rows_for_role(scored, "dev")
        scored_locked = rows_for_role(scored, "locked_test")
        summary["judge_full_94_descriptive_only"] = (
            score_metrics(scored) if scored else None
        )
        summary["judge_dev_diagnostic"] = (
            score_metrics(scored_dev) if scored_dev else None
        )
        summary["judge_locked_test_primary"] = (
            score_metrics(scored_locked) if scored_locked else None
        )
        summary["scored_candidate_count"] = len(scored)
        summary["abstain_or_unscored_count"] = len(base_rows) - len(scored)
        summary["missing_score_candidate_ids"] = missing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
