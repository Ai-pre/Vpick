from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUCKET_PATTERN = re.compile(
    r"\[(\d{2}):(\d{2})-(\d{2}):(\d{2})\]\s*(.*?)(?=\n\[\d{2}:\d{2}-\d{2}:\d{2}\]|\Z)",
    re.DOTALL,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def seconds(minutes: str, seconds_text: str) -> int:
    return int(minutes) * 60 + int(seconds_text)


def overview_context(overview: str, midpoint: float) -> tuple[str, int]:
    buckets = [
        {
            "start": seconds(match.group(1), match.group(2)),
            "end": seconds(match.group(3), match.group(4)),
            "text": " ".join(match.group(5).split()),
        }
        for match in BUCKET_PATTERN.finditer(overview or "")
    ]
    if not buckets:
        return "", 0
    target_index = min(
        range(len(buckets)),
        key=lambda index: (
            0
            if buckets[index]["start"] <= midpoint <= buckets[index]["end"]
            else min(
                abs(midpoint - buckets[index]["start"]),
                abs(midpoint - buckets[index]["end"]),
            )
        ),
    )
    selected = buckets[max(0, target_index - 1) : min(len(buckets), target_index + 2)]
    rendered = "\n".join(
        f"[{item['start']//60:02d}:{item['start']%60:02d}-"
        f"{item['end']//60:02d}:{item['end']%60:02d}] {item['text'][:420]}"
        for item in selected
    )
    return rendered, len(buckets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT
        / "data/private/judge_validation_94/candidates_blind_94_with_overview_PRIVATE.jsonl",
    )
    parser.add_argument(
        "--package-input",
        type=Path,
        default=ROOT
        / "data/private/judge_validation_94/package_judge_v1/candidates_blind_94.jsonl",
    )
    parser.add_argument(
        "--v10-dimensions",
        type=Path,
        default=ROOT / "data/private/judge_validation_94/codex_direct_v10_dimensions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/private/judge_validation_94/source_salience_review_packet_94_PRIVATE.csv",
    )
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates)
    package = {row["candidate_id"]: row for row in read_jsonl(args.package_input)}
    dimensions = {
        row["candidate_id"]: row for row in read_csv(args.v10_dimensions)
    }
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: row["candidate_id"]):
        candidate_id = candidate["candidate_id"]
        midpoint = (
            float(candidate.get("start_ms", 0)) + float(candidate.get("end_ms", 0))
        ) / 2000.0
        context, bucket_count = overview_context(
            str(candidate.get("longform_overview", "")), midpoint
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "longform_id": candidate.get("longform_id", ""),
                "start_sec": round(float(candidate.get("start_ms", 0)) / 1000, 3),
                "end_sec": round(float(candidate.get("end_ms", 0)) / 1000, 3),
                "title": package[candidate_id].get("title", ""),
                "candidate_description": package[candidate_id].get(
                    "description", ""
                ),
                "candidate_transcript": package[candidate_id].get(
                    "transcript", ""
                )[:1000],
                "existing_reason": dimensions[candidate_id].get("reason", ""),
                "local_overview_context": context,
                "overview_bucket_count": bucket_count,
                "overview_available": bool(context),
                "source_salience_0_4": "",
                "source_salience_reason": "",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "overview_available": sum(row["overview_available"] for row in rows),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
