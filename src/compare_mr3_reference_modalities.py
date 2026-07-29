from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def raw_stats(raw_dir: Path) -> dict[str, Any]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(raw_dir.glob("*.json"))
    ]
    attempt_counts = [len(record.get("attempts", [])) for record in records]
    return {
        "raw_record_count": len(records),
        "format_retry_candidate_count": sum(count > 1 for count in attempt_counts),
        "max_attempt_count": max(attempt_counts, default=0),
        "max_generated_tokens": max(
            (int(record.get("generated_tokens", 0)) for record in records),
            default=0,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript-scores", required=True)
    parser.add_argument("--vpick-scores", required=True)
    parser.add_argument("--transcript-raw-dir", required=True)
    parser.add_argument("--vpick-raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    transcript = {
        row["candidate_id"]: row for row in read_csv(Path(args.transcript_scores))
    }
    vpick = {row["candidate_id"]: row for row in read_csv(Path(args.vpick_scores))}
    if set(transcript) != set(vpick):
        raise ValueError("Transcript and Vpick score files must contain identical IDs")

    paired: list[dict[str, Any]] = []
    for candidate_id in sorted(transcript):
        left = transcript[candidate_id]
        right = vpick[candidate_id]
        transcript_checklist = float(left["checklist_score_100"])
        vpick_checklist = float(right["checklist_score_100"])
        transcript_saliency = float(left["saliency_market_1_5"])
        vpick_saliency = float(right["saliency_market_1_5"])
        paired.append(
            {
                "candidate_id": candidate_id,
                "transcript_checklist_score_100": transcript_checklist,
                "vpick_checklist_score_100": vpick_checklist,
                "checklist_delta_vpick_minus_transcript": (
                    vpick_checklist - transcript_checklist
                ),
                "transcript_saliency_market_1_5": transcript_saliency,
                "vpick_saliency_market_1_5": vpick_saliency,
                "saliency_delta_vpick_minus_transcript": (
                    vpick_saliency - transcript_saliency
                ),
                "transcript_overall_suitable": int(
                    left["overall_shortform_suitable"]
                ),
                "vpick_overall_suitable": int(right["overall_shortform_suitable"]),
            }
        )

    transcript_checklist = [
        float(row["transcript_checklist_score_100"]) for row in paired
    ]
    vpick_checklist = [float(row["vpick_checklist_score_100"]) for row in paired]
    transcript_saliency = [
        float(row["transcript_saliency_market_1_5"]) for row in paired
    ]
    vpick_saliency = [float(row["vpick_saliency_market_1_5"]) for row in paired]

    summary = {
        "candidate_count": len(paired),
        "candidate_id_sets_equal": True,
        "transcript": {
            "checklist_mean_100": round(statistics.mean(transcript_checklist), 4),
            "saliency_mean_1_5": round(statistics.mean(transcript_saliency), 4),
            "saliency_distribution": dict(
                sorted(Counter(transcript_saliency).items())
            ),
            "overall_suitable_rate": round(
                statistics.mean(
                    int(row["transcript_overall_suitable"]) for row in paired
                ),
                4,
            ),
            **raw_stats(Path(args.transcript_raw_dir)),
        },
        "vpick": {
            "checklist_mean_100": round(statistics.mean(vpick_checklist), 4),
            "saliency_mean_1_5": round(statistics.mean(vpick_saliency), 4),
            "saliency_distribution": dict(sorted(Counter(vpick_saliency).items())),
            "overall_suitable_rate": round(
                statistics.mean(
                    int(row["vpick_overall_suitable"]) for row in paired
                ),
                4,
            ),
            **raw_stats(Path(args.vpick_raw_dir)),
        },
        "paired": {
            "checklist_mean_delta_vpick_minus_transcript": round(
                statistics.mean(
                    float(row["checklist_delta_vpick_minus_transcript"])
                    for row in paired
                ),
                4,
            ),
            "checklist_vpick_higher_count": sum(
                float(row["checklist_delta_vpick_minus_transcript"]) > 0
                for row in paired
            ),
            "checklist_equal_count": sum(
                float(row["checklist_delta_vpick_minus_transcript"]) == 0
                for row in paired
            ),
            "checklist_vpick_lower_count": sum(
                float(row["checklist_delta_vpick_minus_transcript"]) < 0
                for row in paired
            ),
            "checklist_spearman": round(
                spearman(transcript_checklist, vpick_checklist) or 0.0,
                4,
            ),
            "saliency_exact_agreement_rate": round(
                statistics.mean(
                    left == right
                    for left, right in zip(transcript_saliency, vpick_saliency)
                ),
                4,
            ),
            "saliency_spearman": round(
                spearman(transcript_saliency, vpick_saliency) or 0.0,
                4,
            ),
        },
    }

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "paired_score_differences.csv", paired)
    (out_dir / "mr3_modality_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
