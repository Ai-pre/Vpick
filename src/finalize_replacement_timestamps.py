from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prefixed(row: dict[str, str], prefix: str, fields: tuple[str, ...]) -> dict[str, str]:
    return {f"{prefix}{field}": row.get(field, "") for field in fields}


def finalize_replacement(
    metadata: dict[str, str],
    decision: dict[str, str],
    subtitle: dict[str, str],
) -> dict[str, Any]:
    output: dict[str, Any] = dict(metadata)
    output.update(decision)
    output.update(
        prefixed(
            subtitle,
            "subtitle_",
            (
                "alignment_status",
                "coverage",
                "mean_match_score",
                "predicted_start",
                "predicted_end",
                "source_span",
                "short_span",
                "segment_count",
                "backward_jumps",
                "excess_gap_seconds",
                "short_subtitle_source",
                "short_subtitle_language",
                "long_subtitle_source",
                "long_subtitle_language",
            ),
        )
    )
    return output


def update_full_row(
    row: dict[str, str],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = dict(row)
    usable = replacement["usable_for_gold"] == "yes"
    start = replacement.get("final_start_sec", "")
    end = replacement.get("final_end_sec", "")
    source_duration = float(end) - float(start) if usable and start and end else None
    short_duration_text = replacement.get("short_duration_sec_yt", "")
    short_duration = float(short_duration_text) if short_duration_text else None
    like_count = replacement.get("short_likes_yt", "")
    view_count = replacement.get("short_views_yt", "")

    output.update(
        {
            "long_video_id": replacement["final_long_video_id"],
            "long_video_url": replacement["final_long_video_url"],
            "short_views": view_count,
            "short_likes": like_count,
            "channel_name": replacement.get("short_channel_yt", output.get("channel_name", "")),
            "short_duration_sec": short_duration_text,
            "upload_date": replacement.get("short_upload_date_yt", ""),
            "stats_snapshot_date": replacement.get("youtube_metadata_snapshot_date", ""),
            "short_like_rate": (
                round(float(like_count) / float(view_count), 8)
                if like_count and view_count and float(view_count)
                else ""
            ),
            "start_sec": start if usable else "",
            "end_sec": end if usable else "",
            "start_time": replacement.get("final_start_time", "") if usable else "",
            "end_time": replacement.get("final_end_time", "") if usable else "",
            "duration_sec": round(source_duration, 3) if source_duration is not None else "",
            "gold_span_start_sec": start if usable else "",
            "gold_span_end_sec": end if usable else "",
            "gold_span_duration_sec": round(source_duration, 3)
            if source_duration is not None
            else "",
            "span_ratio": round(source_duration / short_duration, 4)
            if source_duration is not None and short_duration
            else "",
            "alignment_status": replacement["final_alignment_status"],
            "alignment_classification_v3": replacement["final_alignment_status"],
            "alignment_policy_version": "yt_dlp_timestamp_pipeline_v4",
            "mapping_confidence": replacement["timestamp_confidence"],
            "label_confidence": replacement["timestamp_confidence"] if usable else "rejected",
            "verification_status": "verified_usable" if usable else "rejected_heavy_edit",
            "verification_reason": replacement["decision_reason"],
            "next_action": "include_in_gold_dataset" if usable else "replace_pair",
            "ratio_status": "accepted_by_alignment_policy" if usable else "rejected_heavy_edit",
            "ratio_review_flag": "no" if usable else "yes",
            "source_notes": (
                f"{output.get('source_notes', '')}; "
                f"timestamp_method={replacement['timestamp_method']}; "
                f"{replacement['decision_reason']}"
            ).strip("; "),
            "timestamp_method": replacement["timestamp_method"],
            "timestamp_confidence": replacement["timestamp_confidence"],
            "usable_for_gold": replacement["usable_for_gold"],
            "observed_source_span_start_sec": replacement.get(
                "observed_source_span_start_sec", ""
            ),
            "observed_source_span_end_sec": replacement.get(
                "observed_source_span_end_sec", ""
            ),
            "short_title_yt": replacement.get("short_title_yt", ""),
            "long_title_yt": replacement.get("long_title_yt", ""),
            "youtube_metadata_source": replacement.get("youtube_metadata_source", ""),
            "youtube_metadata_snapshot_date": replacement.get(
                "youtube_metadata_snapshot_date", ""
            ),
        }
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge replacement metadata and timestamp decisions into the 60-pair draft."
    )
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--subtitle-audit", type=Path, required=True)
    parser.add_argument("--output-replacements", type=Path, required=True)
    parser.add_argument("--output-full", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_by_id = {row["pair_id"]: row for row in read_rows(args.metadata)}
    decisions_by_id = {row["pair_id"]: row for row in read_rows(args.decisions)}
    subtitle_by_id = {row["pair_id"]: row for row in read_rows(args.subtitle_audit)}

    replacements = [
        finalize_replacement(
            metadata_by_id[pair_id],
            decisions_by_id[pair_id],
            subtitle_by_id.get(pair_id, {}),
        )
        for pair_id in decisions_by_id
    ]
    replacement_by_id = {row["pair_id"]: row for row in replacements}
    full_rows = [
        update_full_row(row, replacement_by_id[row["pair_id"]])
        if row.get("pair_id") in replacement_by_id
        else row
        for row in read_rows(args.draft)
    ]
    write_rows(args.output_replacements, replacements)
    write_rows(args.output_full, full_rows)
    print(
        {
            "replacement_rows": len(replacements),
            "usable_replacements": sum(
                row["usable_for_gold"] == "yes" for row in replacements
            ),
            "rejected_replacements": sum(
                row["usable_for_gold"] != "yes" for row in replacements
            ),
            "full_rows": len(full_rows),
        }
    )


if __name__ == "__main__":
    main()
