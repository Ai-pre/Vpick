from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from performance_judge_v1 import read_csv, read_jsonl, write_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    ROOT / "deliverables" / "2026-07-24" / "performance_judge_v1"
)
BLIND_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)


def build_rows(
    candidates: list[dict[str, Any]],
    targets: list[dict[str, str]],
) -> list[dict[str, Any]]:
    targets_by_id = {row["candidate_id"]: row for row in targets}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        target = targets_by_id.get(str(candidate["candidate_id"]))
        if not target:
            raise ValueError(
                f"Missing target mapping for {candidate['candidate_id']}"
            )
        row = {
            "candidate_id": target["source_candidate_id"],
            "duration_sec": round(
                (int(candidate["end_ms"]) - int(candidate["start_ms"])) / 1000.0,
                3,
            ),
            "description": str(candidate.get("description") or ""),
            "transcript": str(candidate.get("transcript") or ""),
            "before_context": str(candidate.get("before_context") or ""),
            "after_context": str(candidate.get("after_context") or ""),
        }
        if tuple(row) != BLIND_FIELDS:
            raise AssertionError("v7 blind columns changed unexpectedly")
        rows.append(row)
    if len(rows) != 60 or len({row["candidate_id"] for row in rows}) != 60:
        raise ValueError("Expected 60 unique v7 blind candidates")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Vpick-enriched structurally blind v7 Judge input."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "candidates_blind_v7_vpick.csv",
    )
    args = parser.parse_args()
    rows = build_rows(
        read_jsonl(args.dataset_dir / "candidates_blind.jsonl"),
        read_csv(args.dataset_dir / "candidate_targets_PRIVATE.csv"),
    )
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} blind candidates to {args.output}")


if __name__ == "__main__":
    main()
