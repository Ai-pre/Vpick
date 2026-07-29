from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def pair_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], []).append(row)
    output: dict[str, dict[str, float]] = {}
    for pair_id, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda row: int(float(row["rank"])))

        def hit_at(k: int, field: str) -> float:
            return float(
                any(
                    boolean(row[field])
                    for row in ordered
                    if int(float(row["rank"])) <= k
                )
            )

        def best_iou(k: int) -> float:
            return max(
                [
                    float(row["temporal_iou"])
                    for row in ordered
                    if int(float(row["rank"])) <= k
                ]
                or [0.0]
            )

        output[pair_id] = {
            "core_at_1": hit_at(1, "core_hit"),
            "core_at_3": hit_at(3, "core_hit"),
            "core_at_5": hit_at(5, "core_hit"),
            "tight_at_5": hit_at(5, "tight_hit"),
            "best_iou_at_5": best_iou(5),
            "oracle_core_at_50": hit_at(50, "core_hit"),
            "oracle_best_iou_at_50": best_iou(50),
        }
    return output


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(
    name: str,
    metrics: dict[str, dict[str, float]],
    pair_ids: set[str],
    universe: str,
) -> dict[str, Any]:
    selected = [metrics[pair_id] for pair_id in sorted(pair_ids)]
    return {
        "universe": universe,
        "method": name,
        "pair_count": len(selected),
        "core_at_1": round(mean([row["core_at_1"] for row in selected]), 6),
        "core_at_3": round(mean([row["core_at_3"] for row in selected]), 6),
        "core_at_5": round(mean([row["core_at_5"] for row in selected]), 6),
        "tight_at_5": round(
            mean([row["tight_at_5"] for row in selected]), 6
        ),
        "best_iou_at_5": round(
            mean([row["best_iou_at_5"] for row in selected]), 6
        ),
        "core_at_50": "",
        "best_iou_at_50": "",
    }


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=METRICS_CSV")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare highlight selectors on full and common pair sets."
    )
    parser.add_argument(
        "--method",
        action="append",
        type=parse_method,
        required=True,
        help="Repeat NAME=METRICS_CSV.",
    )
    parser.add_argument(
        "--oracle",
        type=parse_method,
        required=True,
        help="NAME=METRICS_CSV for the wide candidate pool.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    methods = {
        name: pair_metrics(read_csv(path)) for name, path in args.method
    }
    common_pairs = set.intersection(
        *(set(metrics) for metrics in methods.values())
    )
    rows: list[dict[str, Any]] = []
    for name, metrics in methods.items():
        rows.append(summarize(name, metrics, set(metrics), "available"))
        rows.append(summarize(name, metrics, common_pairs, "common_pairs"))

    oracle_name, oracle_path = args.oracle
    oracle = pair_metrics(read_csv(oracle_path))
    oracle_pairs = set(oracle)
    oracle_row = {
        "universe": "candidate_pool_upper_bound",
        "method": oracle_name,
        "pair_count": len(oracle_pairs),
        "core_at_1": "",
        "core_at_3": "",
        "core_at_5": "",
        "tight_at_5": "",
        "best_iou_at_5": "",
        "core_at_50": round(
            mean([oracle[pair_id]["oracle_core_at_50"] for pair_id in oracle_pairs]),
            6,
        ),
        "best_iou_at_50": round(
            mean(
                [
                    oracle[pair_id]["oracle_best_iou_at_50"]
                    for pair_id in oracle_pairs
                ]
            ),
            6,
        ),
    }
    rows.append(oracle_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "comparison.csv", rows)
    summary = {
        "methods": list(methods),
        "common_pair_count": len(common_pairs),
        "common_pair_ids": sorted(common_pairs),
        "rows": rows,
        "interpretation": [
            "available rows use every pair available to each method",
            "common_pairs rows are the fair comparison with Vpick",
            "candidate_pool_upper_bound is an oracle diagnostic, not a selector",
        ],
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
