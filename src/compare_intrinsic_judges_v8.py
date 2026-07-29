from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from intrinsic_judge_v8 import CHECK_DIMENSIONS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
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
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(average_ranks(left), average_ranks(right))


def rounded(value: float | None) -> float | str:
    return "" if value is None else round(value, 4)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two independent v8 intrinsic Judge score files."
    )
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    left = {row["candidate_id"]: row for row in read_csv(Path(args.left))}
    right = {row["candidate_id"]: row for row in read_csv(Path(args.right))}
    if set(left) != set(right):
        raise ValueError("Judge files must contain identical candidate IDs")

    common_scored = [
        candidate_id
        for candidate_id in left
        if left[candidate_id].get("verdict") == "score"
        and right[candidate_id].get("verdict") == "score"
        and number(left[candidate_id].get("quality_score_100")) is not None
        and number(right[candidate_id].get("quality_score_100")) is not None
    ]
    left_scores = [
        float(left[candidate_id]["quality_score_100"])
        for candidate_id in common_scored
    ]
    right_scores = [
        float(right[candidate_id]["quality_score_100"])
        for candidate_id in common_scored
    ]

    dimension_rows: list[dict[str, Any]] = []
    total_equal = total_checks = 0
    for dimension in CHECK_DIMENSIONS:
        field = f"check_{dimension}"
        equal = sum(
            left[candidate_id].get(field) == right[candidate_id].get(field)
            for candidate_id in common_scored
        )
        total_equal += equal
        total_checks += len(common_scored)
        dimension_rows.append(
            {
                "dimension": dimension,
                "candidate_count": len(common_scored),
                "exact_agreement": rounded(
                    equal / len(common_scored) if common_scored else None
                ),
            }
        )

    suitable_equal = sum(
        left[candidate_id].get("overall_editorial_suitable")
        == right[candidate_id].get("overall_editorial_suitable")
        for candidate_id in common_scored
    )
    content_mode_equal = sum(
        left[candidate_id].get("content_mode")
        == right[candidate_id].get("content_mode")
        for candidate_id in common_scored
    )
    abstain_equal = sum(
        (left[candidate_id].get("verdict") == "abstain")
        == (right[candidate_id].get("verdict") == "abstain")
        for candidate_id in left
    )
    summary = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "candidate_count": len(left),
        "common_scored_count": len(common_scored),
        "quality_score_spearman": rounded(
            spearman(left_scores, right_scores) if common_scored else None
        ),
        "quality_score_mean_absolute_difference": rounded(
            statistics.mean(
                abs(left_value - right_value)
                for left_value, right_value in zip(left_scores, right_scores)
            )
            if common_scored
            else None
        ),
        "checklist_micro_exact_agreement": rounded(
            total_equal / total_checks if total_checks else None
        ),
        "overall_editorial_suitable_agreement": rounded(
            suitable_equal / len(common_scored) if common_scored else None
        ),
        "content_mode_agreement": rounded(
            content_mode_equal / len(common_scored) if common_scored else None
        ),
        "abstain_decision_agreement": rounded(abstain_equal / len(left)),
        "provisional_reliability_targets": {
            "quality_score_spearman": ">=0.70",
            "checklist_micro_exact_agreement": ">=0.75",
            "overall_editorial_suitable_agreement": ">=0.75",
        },
        "interpretation": (
            "Cross-model agreement is a reliability diagnostic, not proof of validity."
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "cross_model_dimension_agreement.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dimension", "candidate_count", "exact_agreement"],
        )
        writer.writeheader()
        writer.writerows(dimension_rows)
    (out_dir / "cross_model_reliability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
