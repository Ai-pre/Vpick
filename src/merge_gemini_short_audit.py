from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No audit rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest_rows = read_csv(Path(args.manifest))
    manifest_ids = {row["candidate_id"] for row in manifest_rows}
    if len(manifest_ids) != len(manifest_rows):
        raise ValueError("Manifest contains duplicate candidate IDs")

    rows: list[dict[str, str]] = []
    for shard_dir in args.shard_dirs:
        path = Path(shard_dir) / "short_transcript_audit_60.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_csv(path))
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted(
            candidate_id
            for candidate_id, count in Counter(ids).items()
            if count > 1
        )
        raise ValueError(f"Duplicate audit candidate IDs: {duplicates}")
    if set(ids) != manifest_ids:
        raise ValueError(
            "Merged short audit must match the complete manifest. "
            f"missing={sorted(manifest_ids - set(ids))}, "
            f"extra={sorted(set(ids) - manifest_ids)}"
        )
    rows.sort(key=lambda row: row["candidate_id"])
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "short_transcript_audit_60.csv", rows)

    flagged = [
        row
        for row in rows
        if row["needs_longform_recheck"].strip().lower() == "true"
    ]
    summary = {
        "candidate_count": len(rows),
        "unique_candidate_count": len(set(ids)),
        "flagged_count": len(flagged),
        "flagged_candidate_ids": [
            row["candidate_id"] for row in flagged
        ],
        "recheck_reason_counts": dict(
            Counter(
                reason
                for row in flagged
                for reason in row["recheck_reasons"].split("|")
                if reason
            )
        ),
        "status_counts": dict(Counter(row["gemini_status"] for row in rows)),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
