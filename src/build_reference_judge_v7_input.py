from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BLIND_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the physically blind v7 Judge input.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--description-mode",
        choices=("source", "blank"),
        default="blank",
        help="Use blank for a fair transcript-only run when scene descriptions are incomplete.",
    )
    args = parser.parse_args()

    with Path(args.source).open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    rows: list[dict[str, str]] = []
    for source in source_rows:
        row = {field: source.get(field, "") for field in BLIND_FIELDS}
        if args.description_mode == "blank":
            row["description"] = ""
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLIND_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    description_count = sum(bool(row["description"].strip()) for row in rows)
    summary = {
        "candidate_count": len(rows),
        "columns": list(BLIND_FIELDS),
        "description_mode": args.description_mode,
        "description_nonempty_count": description_count,
        "description_empty_count": len(rows) - description_count,
        "description_empty_rate": (
            (len(rows) - description_count) / len(rows)
            if rows
            else 0.0
        ),
        "labels_in_input": False,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
