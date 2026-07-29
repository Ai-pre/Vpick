from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDS = (
    "candidate_id",
    "pair_id",
    "short_video_id",
    "channel_name",
    "channel_performance_percentile",
    "relative_performance_tier",
    "short_views",
    "short_likes",
    "label_confidence",
    "mapping_confidence",
    "alignment_status",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename private pos/neg labels as channel-relative performance tiers."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for source in read_csv(Path(args.source)):
        tier = {
            "pos": "relative_high",
            "neg": "relative_low",
            "relative_high": "relative_high",
            "relative_low": "relative_low",
        }.get(source.get("performance_label", ""), "unclassified")
        rows.append(
            {
                "candidate_id": source.get("candidate_id", ""),
                "pair_id": source.get("pair_id", ""),
                "short_video_id": source.get("short_video_id", ""),
                "channel_name": source.get("channel_name", ""),
                "channel_performance_percentile": source.get(
                    "channel_performance_percentile", ""
                ),
                "relative_performance_tier": tier,
                "short_views": source.get("short_views", ""),
                "short_likes": source.get("short_likes", ""),
                "label_confidence": source.get("label_confidence", ""),
                "mapping_confidence": source.get("mapping_confidence", ""),
                "alignment_status": source.get("alignment_status", ""),
            }
        )

    candidate_ids = [row["candidate_id"] for row in rows]
    if not all(candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id must be complete and unique")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
