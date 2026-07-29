from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["pair_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Score must be finite")
    return parsed


V2_DIMENSION_WEIGHTS = {
    "opening_clarity_pull_0_4": 0.15,
    "event_reaction_change_0_4": 0.25,
    "progression_payoff_0_4": 0.20,
    "self_contained_0_4": 0.15,
    "boundary_integrity_0_4": 0.15,
    "titleability_0_4": 0.10,
}


def bounded_score(value: Any, *, minimum: float, maximum: float) -> float:
    parsed = number(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"Score must be between {minimum} and {maximum}")
    return parsed


def calculate_v2_score(score: dict[str, Any]) -> tuple[float, float]:
    dimensions = {
        key: bounded_score(score[key], minimum=0.0, maximum=4.0)
        for key in V2_DIMENSION_WEIGHTS
    }
    raw = sum(
        dimensions[key] * weight
        for key, weight in V2_DIMENSION_WEIGHTS.items()
    ) / 4.0
    completion_axes = [
        dimensions["progression_payoff_0_4"],
        dimensions["self_contained_0_4"],
        dimensions["boundary_integrity_0_4"],
    ]
    low_count = sum(value <= 1.0 for value in completion_axes)
    if any(value == 0.0 for value in completion_axes):
        gate = 0.50
    elif low_count >= 2:
        gate = 0.65
    elif low_count == 1:
        gate = 0.80
    else:
        gate = 1.00
    completeness = sum(completion_axes) / (len(completion_axes) * 4.0)
    return raw * gate, completeness


def overlap_ratio(left: dict[str, str], right: dict[str, str]) -> float:
    left_start = float(left["start_sec"])
    left_end = float(left["end_sec"])
    right_start = float(right["start_sec"])
    right_end = float(right["end_sec"])
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    shortest = min(left_end - left_start, right_end - right_start)
    return overlap / shortest if shortest > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate listwise scores, apply MMR, and build predictions."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, nargs="+")
    parser.add_argument(
        "--longform-ids",
        type=Path,
        help="Optional newline-delimited longform IDs for a subset run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-overlap", type=float, default=0.58)
    parser.add_argument("--overlap-penalty", type=float, default=0.30)
    parser.add_argument("--completeness-weight", type=float, default=0.15)
    parser.add_argument(
        "--candidate-prior-weight",
        type=float,
        default=0.0,
        help=(
            "Blend weight for the pre-LLM hierarchical candidate score. "
            "0 uses only the judge; 1 uses only the candidate prior."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.candidate_prior_weight <= 1.0:
        raise ValueError("--candidate-prior-weight must be between 0 and 1")

    allowed_longforms = None
    if args.longform_ids:
        allowed_longforms = {
            line.strip()
            for line in args.longform_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    candidate_by_id = {
        row["candidate_id"]: row
        for row in read_csv(args.candidate_pool)
        if str(row.get("in_multislate_union", "")).lower()
        in {"true", "1", "yes"}
        and (
            allowed_longforms is None
            or row["longform_id"] in allowed_longforms
        )
    }
    expected_by_longform: dict[str, set[str]] = defaultdict(set)
    for candidate_id, row in candidate_by_id.items():
        expected_by_longform[row["longform_id"]].add(candidate_id)

    scores_by_longform: dict[str, list[dict[str, Any]]] = {}
    result_rows = [
        result
        for result_path in args.results
        for result in read_jsonl(result_path)
    ]
    for result in result_rows:
        longform_id = str(result["longform_id"])
        if longform_id in scores_by_longform:
            raise ValueError(f"Duplicate longform result: {longform_id}")
        scored = result.get("candidate_scores")
        if not isinstance(scored, list):
            raise ValueError(f"Missing candidate_scores: {longform_id}")
        ids = [str(row["candidate_id"]) for row in scored]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate candidate score: {longform_id}")
        if set(ids) != expected_by_longform.get(longform_id, set()):
            raise ValueError(f"Candidate coverage mismatch: {longform_id}")
        scores_by_longform[longform_id] = scored
    if set(scores_by_longform) != set(expected_by_longform):
        raise ValueError("Longform result coverage mismatch")

    selected_by_longform: dict[str, list[dict[str, Any]]] = {}
    for longform_id, scored in scores_by_longform.items():
        remaining = []
        for score in scored:
            candidate = candidate_by_id[str(score["candidate_id"])]
            if all(key in score for key in V2_DIMENSION_WEIGHTS):
                base, completeness = calculate_v2_score(score)
                score_schema = "intrinsic_v2"
            else:
                success = bounded_score(
                    score["success_score_0_1"], minimum=0.0, maximum=1.0
                )
                completeness = (
                    bounded_score(
                        score["completeness_0_4"], minimum=0.0, maximum=4.0
                    )
                    / 4.0
                )
                base = (
                    (1.0 - args.completeness_weight) * success
                    + args.completeness_weight * completeness
                )
                score_schema = "legacy_v1"
            intrinsic_base = base
            candidate_prior = bounded_score(
                candidate["hierarchical_score"], minimum=0.0, maximum=1.0
            )
            base = (
                (1.0 - args.candidate_prior_weight) * intrinsic_base
                + args.candidate_prior_weight * candidate_prior
            )
            remaining.append(
                {
                    "candidate": candidate,
                    "score": score,
                    "base": base,
                    "intrinsic_base": intrinsic_base,
                    "candidate_prior": candidate_prior,
                    "completeness": completeness,
                    "score_schema": score_schema,
                }
            )
        selected = []
        while remaining and len(selected) < args.top_k:
            best = None
            best_value = -math.inf
            for record in remaining:
                if selected:
                    maximum_overlap = max(
                        overlap_ratio(
                            record["candidate"], chosen["candidate"]
                        )
                        for chosen in selected
                    )
                else:
                    maximum_overlap = 0.0
                if maximum_overlap > args.max_overlap:
                    continue
                value = record["base"] - args.overlap_penalty * maximum_overlap
                if value > best_value:
                    best = record
                    best_value = value
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)
        selected_by_longform[longform_id] = selected

    gold_by_longform: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.dataset):
        gold_by_longform[row["long_video_id"]].append(row)
    output: list[dict[str, Any]] = []
    for longform_id, selected in selected_by_longform.items():
        for pair in gold_by_longform.get(longform_id, []):
            for rank, record in enumerate(selected, start=1):
                candidate = record["candidate"]
                output.append(
                    {
                        "pair_id": pair["pair_id"],
                        "long_video_id": longform_id,
                        "short_video_id": pair.get("short_video_id", ""),
                        "run_id": "b4_intrinsic_listwise_judge_mmr",
                        "selector_type": "hierarchical_intrinsic_listwise_mmr",
                        "prompt_id": "hierarchical_multislate_listwise_v2_ko",
                        "model_name": "external_or_direct_llm",
                        "rank": rank,
                        "pred_start_sec": candidate["start_sec"],
                        "pred_end_sec": candidate["end_sec"],
                        "selected_scene_ids": "",
                        "confidence": round(record["base"], 6),
                        "notes": (
                            f"candidate_id={candidate['candidate_id']};"
                            f"score_schema={record['score_schema']};"
                            f"intrinsic={record['intrinsic_base']:.6f};"
                            f"candidate_prior={record['candidate_prior']:.6f};"
                            f"candidate_prior_weight={args.candidate_prior_weight:.2f};"
                            f"completeness={record['completeness']:.6f}"
                        ),
                    }
                )
    write_csv(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "longform_count": len(selected_by_longform),
                "prediction_rows": len(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
