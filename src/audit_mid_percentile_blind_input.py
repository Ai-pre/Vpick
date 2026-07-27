"""Audit the mid-percentile blind evidence file before it reaches a judge.

Checks the three things that would silently invalidate a judge run: the column
schema must match what the judge scripts read, no label-bearing field may leak
into the blind file, and no candidate may carry empty evidence. Leakage is
checked against the private manifest's own values, not a keyword list, so a
channel name or view count that happens to appear verbatim is caught.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

EXPECTED_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)

# Manifest columns whose values must never appear in the blind evidence.
LEAK_SOURCE_COLUMNS = (
    "performance_label",
    "percentile_bucket",
    "channel_performance_percentile",
    "short_views",
    "short_video_id",
    "short_video_url",
    "longform_id",
    "long_video_url",
    "channel_name",
    "pair_id",
)

FORBIDDEN_KEY_PATTERN = re.compile(
    r"performance_label|channel_performance_percentile|short_views|short_likes"
    r"|percentile|youtube\.com|youtu\.be",
    flags=re.IGNORECASE,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit blind judge input for the mid-percentile set.")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    with evidence_path.open(encoding="utf-8-sig", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    evidence = read_csv(evidence_path)
    manifest = {row["candidate_id"]: row for row in read_csv(Path(args.manifest))}

    schema_ok = header == EXPECTED_FIELDS
    text_fields = ("description", "transcript", "before_context", "after_context")

    leaks: list[dict[str, Any]] = []
    for row in evidence:
        blob = " ".join(row.get(field, "") for field in text_fields)
        for match in FORBIDDEN_KEY_PATTERN.finditer(blob):
            leaks.append(
                {"candidate_id": row["candidate_id"], "kind": "forbidden_pattern",
                 "value": match.group(0)}
            )
        private = manifest.get(row["candidate_id"], {})
        for column in LEAK_SOURCE_COLUMNS:
            value = str(private.get(column, "")).strip()
            # Short numeric values collide with ordinary speech, so only treat
            # distinctive identifiers and long numbers as leakage evidence.
            if len(value) < 6:
                continue
            if value and value in blob:
                leaks.append(
                    {"candidate_id": row["candidate_id"], "kind": f"manifest_value:{column}",
                     "value": value[:60]}
                )

    empty = {
        field: [r["candidate_id"] for r in evidence if not r.get(field, "").strip()]
        for field in ("description", "transcript")
    }
    duplicate_ids = [
        cid for cid in {r["candidate_id"] for r in evidence}
        if sum(1 for r in evidence if r["candidate_id"] == cid) > 1
    ]
    unmatched = [r["candidate_id"] for r in evidence if r["candidate_id"] not in manifest]

    description_lengths = [
        len(r.get("description", "").replace(" ", "")) for r in evidence
    ]
    out_of_range = [
        r["candidate_id"]
        for r in evidence
        if not 45 <= len(r.get("description", "").replace(" ", "")) <= 150
    ]

    report: dict[str, Any] = {
        "evidence_rows": len(evidence),
        "manifest_rows": len(manifest),
        "schema_matches_judge_input": schema_ok,
        "observed_header": list(header),
        "forbidden_key_count": len(leaks),
        "leaks": leaks[:20],
        "empty_description_count": len(empty["description"]),
        "empty_transcript_count": len(empty["transcript"]),
        "duplicate_candidate_ids": duplicate_ids,
        "unmatched_candidate_ids": unmatched,
        "description_len_min": min(description_lengths) if description_lengths else 0,
        "description_len_max": max(description_lengths) if description_lengths else 0,
        "description_out_of_range": out_of_range,
    }
    report["audit_passed"] = bool(
        schema_ok
        and not leaks
        and not empty["description"]
        and not empty["transcript"]
        and not duplicate_ids
        and not unmatched
        and not out_of_range
    )

    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
