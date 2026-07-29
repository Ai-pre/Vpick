from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "source_importance",
    "standalone_completeness",
    "boundary_quality",
    "engagement",
)
DEFAULT_WEIGHTS = {dimension: 0.25 for dimension in DIMENSIONS}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_weights(raw: str | None) -> dict[str, float]:
    if raw is None:
        return dict(DEFAULT_WEIGHTS)
    parsed = json.loads(raw)
    if set(parsed) != set(DIMENSIONS):
        raise ValueError(
            "--weights must define exactly: " + ", ".join(DIMENSIONS)
        )
    weights = {key: float(value) for key, value in parsed.items()}
    if any(value < 0 for value in weights.values()):
        raise ValueError("Weights cannot be negative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return {key: value / total for key, value in weights.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--weights",
        help=(
            "JSON object with one weight per dimension. "
            "Defaults to equal weights."
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    score_rows = read_csv(Path(args.scores))
    manifest_rows = read_csv(Path(args.manifest))
    weights = parse_weights(args.weights)
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Manifest contains duplicate candidate_id values")

    observed_keys: set[tuple[str, str, int]] = set()
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    parse_status_counts: Counter[str] = Counter()
    for row in score_rows:
        candidate_id = row["candidate_id"]
        dimension = row["dimension"]
        repeat_index = int(row["repeat_index"])
        key = (candidate_id, dimension, repeat_index)
        if key in observed_keys:
            raise ValueError(f"Duplicate score row: {key}")
        observed_keys.add(key)
        if candidate_id not in manifest:
            raise ValueError(f"Score candidate missing from manifest: {candidate_id}")
        if dimension not in DIMENSIONS:
            raise ValueError(f"Unexpected dimension: {dimension}")
        parse_status_counts[row["parse_status"]] += 1
        grouped[(candidate_id, repeat_index)][dimension] = row

    incomplete: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for (candidate_id, repeat_index), dimension_rows in sorted(grouped.items()):
        missing = [
            dimension
            for dimension in DIMENSIONS
            if dimension not in dimension_rows
            or dimension_rows[dimension]["parse_status"] != "score"
            or not dimension_rows[dimension]["score_1_5"]
        ]
        if missing:
            incomplete.append(
                {
                    "candidate_id": candidate_id,
                    "repeat_index": repeat_index,
                    "missing_or_unparsed_dimensions": "|".join(missing),
                }
            )
            continue

        scores_100 = {
            dimension: float(dimension_rows[dimension]["score_100"])
            for dimension in DIMENSIONS
        }
        composite = sum(
            scores_100[dimension] * weights[dimension]
            for dimension in DIMENSIONS
        )
        repeat_rows.append(
            {
                "candidate_id": candidate_id,
                "repeat_index": repeat_index,
                **{
                    f"{dimension}_1_5": int(
                        float(dimension_rows[dimension]["score_1_5"])
                    )
                    for dimension in DIMENSIONS
                },
                **{
                    f"{dimension}_100": round(scores_100[dimension], 4)
                    for dimension in DIMENSIONS
                },
                "quality_score_100": round(composite, 4),
            }
        )

    if incomplete and not args.allow_incomplete:
        preview = incomplete[:10]
        raise ValueError(
            f"{len(incomplete)} candidate-repeat groups are incomplete: {preview}"
        )
    if not repeat_rows:
        raise ValueError("No complete candidate-repeat groups were available")

    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in repeat_rows:
        by_candidate[row["candidate_id"]].append(row)

    candidate_rows: list[dict[str, Any]] = []
    for candidate_id, rows in sorted(by_candidate.items()):
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "repeat_count": len(rows),
                **{
                    f"{dimension}_100": round(
                        sum(float(row[f"{dimension}_100"]) for row in rows)
                        / len(rows),
                        4,
                    )
                    for dimension in DIMENSIONS
                },
                "quality_score_100": round(
                    sum(float(row["quality_score_100"]) for row in rows)
                    / len(rows),
                    4,
                ),
            }
        )

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "candidate_repeat_scores.csv", repeat_rows)
    write_csv(out_dir / "candidate_scores.csv", candidate_rows)
    if incomplete:
        write_csv(out_dir / "incomplete_groups.csv", incomplete)

    expected_candidates = sorted(
        {
            row["candidate_id"]
            for row in score_rows
        }
    )
    aggregated_candidates = sorted(by_candidate)
    summary = {
        "source_score_rows": len(score_rows),
        "parse_status_counts": dict(parse_status_counts),
        "candidate_count_in_scores": len(expected_candidates),
        "aggregated_candidate_count": len(aggregated_candidates),
        "candidate_repeat_count": len(repeat_rows),
        "incomplete_candidate_repeat_count": len(incomplete),
        "weights": weights,
        "missing_aggregated_candidate_ids": sorted(
            set(expected_candidates) - set(aggregated_candidates)
        ),
    }
    (out_dir / "aggregation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
