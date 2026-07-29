from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = (
    "candidate_id",
    "long_video_id",
    "long_video_url",
    "start_sec",
    "end_sec",
    "start_time",
    "end_time",
    "duration_sec",
    "source_system",
    "source_run_id",
    "source_rank",
    "pair_id",
    "short_video_id",
    "short_video_url",
    "short_views",
    "short_likes",
    "channel_name",
    "channel_performance_percentile",
    "label_confidence",
    "mapping_confidence",
    "performance_evidence_status",
    "labeling_rule_version",
    "source_notes",
    "dataset_split",
    "evaluation_role",
    "performance_label",
    "alignment_status",
)


def display_timestamp(seconds: str | float) -> str:
    value = float(seconds)
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    remaining = value % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:06.3f}"
    return f"{minutes:02d}:{remaining:06.3f}"


def add_derived_fields(item: dict[str, Any]) -> None:
    start = item.get("start_sec", "")
    end = item.get("end_sec", "")
    if start != "" and end != "":
        item["start_time"] = display_timestamp(start)
        item["end_time"] = display_timestamp(end)
        item["duration_sec"] = round(float(end) - float(start), 3)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)


def build_rows(
    base_rows: list[dict[str, str]],
    accepted_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    excluded_pair_ids: set[str],
) -> list[dict[str, Any]]:
    audit_by_pair = {row.get("pair_id", ""): row for row in audit_rows}
    output: list[dict[str, Any]] = []
    for row in base_rows:
        if row.get("performance_label") not in {"pos", "neg"}:
            continue
        if row.get("pair_id") in excluded_pair_ids:
            continue
        item: dict[str, Any] = dict(row)
        long_id = row.get("long_video_id", "")
        short_id = row.get("short_video_id", "")
        item["long_video_url"] = f"https://www.youtube.com/watch?v={long_id}"
        item["short_video_url"] = f"https://www.youtube.com/shorts/{short_id}"
        item["alignment_status"] = audit_by_pair.get(row.get("pair_id", ""), {}).get("alignment_status", "")
        item["mapping_confidence"] = row.get("label_confidence", "")
        item["performance_evidence_status"] = "verified_channel_percentile_snapshot"
        item["labeling_rule_version"] = "channel_percentile_v1_alignment_v2"
        add_derived_fields(item)
        output.append(item)

    for row in accepted_rows:
        if str(row.get("auto_accept", "0")) != "1":
            continue
        status = row.get("alignment_status", "")
        item = {
                "candidate_id": row.get("candidate_id", ""),
                "long_video_id": row.get("long_video_id", ""),
                "long_video_url": row.get("long_video_url", ""),
                "start_sec": row.get("predicted_start", ""),
                "end_sec": row.get("predicted_end", ""),
                "source_system": "gold",
                "source_run_id": "subtitle_aligned_neg_2026-07-22",
                "source_rank": "1",
                "pair_id": row.get("pair_id", ""),
                "short_video_id": row.get("short_video_id", ""),
                "short_video_url": row.get("short_video_url", ""),
                "short_views": "",
                "short_likes": "",
                "channel_name": row.get("channel_name", ""),
                "channel_performance_percentile": "",
                "label_confidence": "medium",
                "mapping_confidence": "high" if status == "continuous" else "medium",
                "performance_evidence_status": "teammate_selected_pending_stats_snapshot",
                "labeling_rule_version": "channel_percentile_v1_alignment_v2",
                "source_notes": (
                    f"teammate_neg_candidate; subtitle_alignment={status}; "
                    f"coverage={row.get('coverage', '')}; match_score={row.get('mean_match_score', '')}"
                ),
                "dataset_split": "control",
                "evaluation_role": "gold",
                "performance_label": "neg",
                "alignment_status": status,
            }
        add_derived_fields(item)
        output.append(item)

    seen_pairs: set[str] = set()
    seen_shorts: set[str] = set()
    for row in output:
        pair_id = str(row.get("pair_id", ""))
        short_id = str(row.get("short_video_id", ""))
        if pair_id in seen_pairs:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        if short_id in seen_shorts:
            raise ValueError(f"Duplicate short_video_id: {short_id}")
        seen_pairs.add(pair_id)
        seen_shorts.add(short_id)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final balanced pos/neg reference dataset.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, action="append", default=[])
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--exclude-pair-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pos", type=int, default=30)
    parser.add_argument("--expected-neg", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted_rows = [row for path in args.accepted for row in read_csv(path)]
    rows = build_rows(
        read_csv(args.base),
        accepted_rows,
        read_csv(args.audit),
        set(args.exclude_pair_id),
    )
    counts = {
        label: sum(row.get("performance_label") == label for row in rows)
        for label in ("pos", "neg")
    }
    if counts != {"pos": args.expected_pos, "neg": args.expected_neg}:
        raise ValueError(f"Unexpected label counts: {counts}")
    write_csv(args.output, rows)
    summary = {
        "rows": len(rows),
        "label_counts": counts,
        "excluded_pair_ids": args.exclude_pair_id,
        "unique_long_videos": len({row.get("long_video_id", "") for row in rows}),
        "unique_short_videos": len({row.get("short_video_id", "") for row in rows}),
        "channel_label_counts": dict(
            sorted(
                Counter(
                    f"{row.get('channel_name', '')}|{row.get('performance_label', '')}"
                    for row in rows
                ).items()
            )
        ),
        "alignment_status_counts": dict(
            sorted(Counter(row.get("alignment_status", "") for row in rows).items())
        ),
        "performance_evidence_status_counts": dict(
            sorted(Counter(row.get("performance_evidence_status", "") for row in rows).items())
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
