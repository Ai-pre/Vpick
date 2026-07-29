from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2
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
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(average_ranks(left), average_ranks(right))


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_csv(Path(args.scores))
    parsed = [row for row in rows if row["parse_status"] == "score"]
    expected_keys = {
        (
            row["candidate_id"],
            row["dimension"],
            int(row["repeat_index"]),
        )
        for row in rows
    }
    if len(expected_keys) != len(rows):
        raise ValueError("Specialized Judge scores contain duplicate keys")

    dimension_summary: dict[str, Any] = {}
    for dimension in sorted({row["dimension"] for row in rows}):
        values = [
            int(row["score_1_5"])
            for row in parsed
            if row["dimension"] == dimension
        ]
        counts = Counter(values)
        mode_count = max(counts.values(), default=0)
        dimension_summary[dimension] = {
            "count": len(values),
            "mean_1_5": rounded(statistics.mean(values) if values else None),
            "stddev_1_5": rounded(
                statistics.pstdev(values) if len(values) > 1 else None
            ),
            "distribution": dict(sorted(counts.items())),
            "mode_share": round(mode_count / max(1, len(values)), 4),
            "floor_rate": round(counts[1] / max(1, len(values)), 4),
            "ceiling_rate": round(counts[5] / max(1, len(values)), 4),
        }

    candidate_repeat: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in parsed:
        candidate_repeat[
            (row["candidate_id"], int(row["repeat_index"]))
        ].append(int(row["score_1_5"]))
    aggregate = {
        key: statistics.mean(values)
        for key, values in candidate_repeat.items()
        if len(values) == len(dimension_summary)
    }
    repeats = sorted({repeat for _, repeat in aggregate})
    reliability: dict[str, Any] = {}
    if len(repeats) >= 2:
        first, second = repeats[:2]
        ids = sorted(
            {
                candidate_id
                for candidate_id, repeat in aggregate
                if repeat == first
                and (candidate_id, second) in aggregate
            }
        )
        first_values = [aggregate[(candidate_id, first)] for candidate_id in ids]
        second_values = [
            aggregate[(candidate_id, second)] for candidate_id in ids
        ]
        reliability = {
            "repeat_pair": [first, second],
            "candidate_count": len(ids),
            "aggregate_spearman": rounded(
                spearman(first_values, second_values)
            ),
            "aggregate_exact_agreement_rate": round(
                statistics.mean(
                    left == right
                    for left, right in zip(first_values, second_values)
                ),
                4,
            )
            if ids
            else None,
            "aggregate_mean_absolute_difference_1_5": rounded(
                statistics.mean(
                    abs(left - right)
                    for left, right in zip(first_values, second_values)
                )
            )
            if ids
            else None,
        }

    parse_rate = len(parsed) / max(1, len(rows))
    distribution_gate = all(
        item["mode_share"] < 0.7 for item in dimension_summary.values()
    )
    repeat_gate = (
        reliability.get("aggregate_spearman") is None
        or float(reliability["aggregate_spearman"]) >= 0.6
    )
    summary = {
        "score_row_count": len(rows),
        "parsed_score_count": len(parsed),
        "parse_success_rate": round(parse_rate, 4),
        "candidate_count": len({row["candidate_id"] for row in rows}),
        "repeat_count": len({row["repeat_index"] for row in rows}),
        "dimension_summary": dimension_summary,
        "reliability": reliability,
        "pilot_gates": {
            "parse_success_rate_at_least_0_90": parse_rate >= 0.9,
            "all_dimension_mode_share_below_0_70": distribution_gate,
            "repeat_spearman_at_least_0_60_when_available": repeat_gate,
            "pass": parse_rate >= 0.9 and distribution_gate and repeat_gate,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
