from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Each candidate JSONL row must be an object")
                rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["candidate_id", "score"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: Any, field: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def aggregate_pointwise(
    rows: list[dict[str, str]],
    score_field: str,
) -> tuple[dict[str, float], set[str]]:
    values: dict[str, list[float]] = defaultdict(list)
    abstained: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError("Pointwise score row is missing candidate_id")
        if str(row.get("verdict", "score")).strip().lower() == "abstain":
            abstained.add(candidate_id)
            continue
        values[candidate_id].append(
            finite_number(row.get(score_field), score_field)
        )
    return (
        {
            candidate_id: statistics.mean(scores)
            for candidate_id, scores in values.items()
        },
        abstained,
    )


def read_v14_predictions(
    path: Path,
    score_field: str,
) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("v14 prediction JSON must contain a results array")
    output: dict[str, float] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id in output:
            raise ValueError("v14 candidate_id must be complete and unique")
        output[candidate_id] = finite_number(row.get(score_field), score_field)
    return output


def average_rank_percentiles(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        only_id = next(iter(values))
        return {only_id: 50.0}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    output: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        percentile = (average_rank - 1.0) / (len(ordered) - 1.0) * 100.0
        for candidate_id, _value in ordered[index:end]:
            output[candidate_id] = percentile
        index = end
    return output


def group_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    group_by_id: dict[str, str] = {}
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for row in candidates:
        candidate_id = str(row.get("candidate_id", "")).strip()
        group_id = str(
            row.get("pool_id") or row.get("longform_id") or ""
        ).strip()
        if not candidate_id or not group_id:
            raise ValueError(
                "Each candidate needs candidate_id and pool_id or longform_id"
            )
        if candidate_id in candidate_by_id:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        group_by_id[candidate_id] = group_id
        candidate_by_id[candidate_id] = row
    return group_by_id, candidate_by_id


def rank_within_groups(
    scores: dict[str, float],
    group_by_id: dict[str, str],
) -> dict[str, float]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for candidate_id, score in scores.items():
        grouped[group_by_id[candidate_id]][candidate_id] = score
    return {
        candidate_id: percentile
        for group in grouped.values()
        for candidate_id, percentile in average_rank_percentiles(group).items()
    }


def output_rows(
    candidate_by_id: dict[str, dict[str, Any]],
    group_by_id: dict[str, str],
    pointwise_raw: dict[str, float],
    v14_raw: dict[str, float],
    pointwise_rank: dict[str, float],
    v14_rank: dict[str, float],
    pointwise_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    combined_weight = 1.0 - pointwise_weight
    diagnostic_rows: list[dict[str, Any]] = []
    variants: dict[str, list[dict[str, Any]]] = {
        "pointwise_only": [],
        "v14_only": [],
        "hybrid_50_50": [],
    }
    for candidate_id in sorted(candidate_by_id):
        candidate = candidate_by_id[candidate_id]
        hybrid = (
            pointwise_weight * pointwise_rank[candidate_id]
            + combined_weight * v14_rank[candidate_id]
        )
        base = {
            "candidate_id": candidate_id,
            "group_id": group_by_id[candidate_id],
            "longform_id": candidate.get("longform_id", ""),
            "verdict": "score",
        }
        diagnostic_rows.append(
            {
                **base,
                "pointwise_raw_score": round(pointwise_raw[candidate_id], 6),
                "v14_raw_score": round(v14_raw[candidate_id], 6),
                "pointwise_group_percentile": round(
                    pointwise_rank[candidate_id], 6
                ),
                "v14_group_percentile": round(v14_rank[candidate_id], 6),
                "hybrid_score_100": round(hybrid, 6),
            }
        )
        variants["pointwise_only"].append(
            {**base, "score": round(pointwise_rank[candidate_id], 6)}
        )
        variants["v14_only"].append(
            {**base, "score": round(v14_rank[candidate_id], 6)}
        )
        variants["hybrid_50_50"].append(
            {**base, "score": round(hybrid, 6)}
        )
    return diagnostic_rows, variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank one fixed candidate pool with Pointwise Judge, v14, "
            "and a preregistered rank-normalized hybrid."
        )
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pointwise-scores", type=Path, required=True)
    parser.add_argument("--v14-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pointwise-score-field", default="judge_score_100")
    parser.add_argument("--v14-score-field", default="raw_ranker_score")
    parser.add_argument("--pointwise-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.pointwise_weight <= 1.0:
        raise ValueError("--pointwise-weight must be in [0, 1]")
    candidates = read_jsonl(args.candidates)
    group_by_id, candidate_by_id = group_candidates(candidates)
    pointwise_raw, abstained = aggregate_pointwise(
        read_csv(args.pointwise_scores),
        args.pointwise_score_field,
    )
    v14_raw = read_v14_predictions(
        args.v14_predictions,
        args.v14_score_field,
    )
    expected = set(candidate_by_id)
    missing_pointwise = sorted(expected - set(pointwise_raw))
    missing_v14 = sorted(expected - set(v14_raw))
    unexpected_pointwise = sorted(set(pointwise_raw) - expected)
    unexpected_v14 = sorted(set(v14_raw) - expected)
    if missing_pointwise or missing_v14 or unexpected_pointwise or unexpected_v14:
        raise ValueError(
            "Candidate score coverage differs: "
            f"missing_pointwise={missing_pointwise[:5]}, "
            f"missing_v14={missing_v14[:5]}, "
            f"unexpected_pointwise={unexpected_pointwise[:5]}, "
            f"unexpected_v14={unexpected_v14[:5]}, "
            f"abstained={sorted(abstained)[:5]}"
        )

    pointwise_rank = rank_within_groups(pointwise_raw, group_by_id)
    v14_rank = rank_within_groups(v14_raw, group_by_id)
    diagnostics, variants = output_rows(
        candidate_by_id,
        group_by_id,
        pointwise_raw,
        v14_raw,
        pointwise_rank,
        v14_rank,
        args.pointwise_weight,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_scores.csv", diagnostics)
    for name, rows in variants.items():
        write_csv(args.output_dir / f"{name}_scores.csv", rows)
    summary = {
        "candidate_count": len(candidates),
        "group_count": len(set(group_by_id.values())),
        "pointwise_score_field": args.pointwise_score_field,
        "v14_score_field": args.v14_score_field,
        "pointwise_weight": args.pointwise_weight,
        "v14_weight": 1.0 - args.pointwise_weight,
        "normalization": "average rank percentile within candidate group",
        "variants": list(variants),
        "abstained_candidate_count": len(abstained),
    }
    (args.output_dir / "rerank_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
