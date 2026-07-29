from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def compact(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one blind listwise request per longform."
    )
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--description-chars", type=int, default=500)
    parser.add_argument("--transcript-chars", type=int, default=900)
    parser.add_argument("--context-chars", type=int, default=500)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.candidate_pool):
        if truthy(row.get("in_multislate_union")):
            grouped[row["longform_id"]].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate_count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for longform_id, rows in sorted(grouped.items()):
            candidates = []
            for row in sorted(
                rows,
                key=lambda item: (
                    int(float(item["timeline_bin"])),
                    float(item["start_sec"]),
                ),
            ):
                candidates.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "start_sec": float(row["start_sec"]),
                        "end_sec": float(row["end_sec"]),
                        "duration_sec": float(row["duration_sec"]),
                        "timeline_bin": int(float(row["timeline_bin"])),
                        "description": compact(
                            row["description"], args.description_chars
                        ),
                        "transcript": compact(
                            row["transcript"], args.transcript_chars
                        ),
                        "context_before_after": compact(
                            row["context_transcript"], args.context_chars
                        ),
                    }
                )
            payload = {
                "task": "rerank_shortform_candidates",
                "longform_id": longform_id,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            candidate_count += len(candidates)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "longform_count": len(grouped),
                "candidate_count": candidate_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
