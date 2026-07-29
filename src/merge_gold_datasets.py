from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from segments import seconds_to_clock


BASE_FIELDS = [
    "pair_id",
    "long_video_id",
    "short_video_id",
    "channel_name",
    "long_video_url",
    "short_video_url",
    "gold_start_sec",
    "gold_end_sec",
    "short_views",
    "short_likes",
    "label_confidence",
    "label_notes",
    "vpick_project_id",
    "vpick_asset_id",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must use SPLIT=PATH")
    split, raw_path = value.split("=", 1)
    if not split.strip():
        raise argparse.ArgumentTypeError("dataset split cannot be empty")
    return split.strip(), Path(raw_path)


def as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def clean_number(value: Any) -> str:
    parsed = as_float(value)
    if parsed is None:
        return ""
    if parsed.is_integer():
        return str(int(parsed))
    return str(round(parsed, 6))


def channel_percentile(notes: str) -> str:
    match = re.search(r"(?:채널내백분위|channel_percentile)\s*=\s*([0-9]+(?:\.[0-9]+)?)", str(notes or ""))
    return match.group(1) if match else ""


def quality_flags(row: dict[str, str]) -> str:
    flags = []
    for field in ("channel_name", "short_views", "short_likes", "vpick_project_id", "vpick_asset_id"):
        if not str(row.get(field, "")).strip():
            flags.append(f"missing_{field}")
    return "|".join(flags)


def performance_label(split: str, percentile: float | None = None) -> str:
    if percentile is not None:
        if percentile >= 75.0:
            return "pos"
        if percentile <= 25.0:
            return "neg"
        return "unlabeled"
    return {"main": "pos", "control": "neg"}.get(split, "unlabeled")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge and validate all Gold long-short pair CSV files.")
    parser.add_argument("--input", action="append", type=parse_input, required=True, help="Repeat SPLIT=PATH.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    merged: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split, path in args.input:
        rows = read_csv(path)
        split_counts[split] = len(rows)
        missing_columns = [field for field in BASE_FIELDS if field not in (rows[0].keys() if rows else [])]
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {missing_columns}")
        for row in rows:
            start = as_float(row.get("gold_start_sec"))
            end = as_float(row.get("gold_end_sec"))
            if start is None or end is None or end <= start:
                raise ValueError(f"Invalid Gold interval for pair_id={row.get('pair_id')}: {start}-{end}")
            views = as_float(row.get("short_views"))
            likes = as_float(row.get("short_likes"))
            percentile = as_float(channel_percentile(row.get("label_notes", "")))
            merged.append(
                {
                    "dataset_split": split,
                    "evaluation_role": "gold",
                    "performance_label": performance_label(split, percentile),
                    **{field: str(row.get(field, "")).strip() for field in BASE_FIELDS},
                    "gold_start_sec": clean_number(start),
                    "gold_end_sec": clean_number(end),
                    "gold_start_time": seconds_to_clock(start),
                    "gold_end_time": seconds_to_clock(end),
                    "gold_duration_sec": clean_number(end - start),
                    "short_views": clean_number(views),
                    "short_likes": clean_number(likes),
                    "short_like_rate": round(likes / views, 8) if views and likes is not None else "",
                    "channel_performance_percentile": channel_percentile(row.get("label_notes", "")),
                    "data_quality_flags": quality_flags(row),
                }
            )

    pair_ids = [str(row["pair_id"]) for row in merged]
    short_ids = [str(row["short_video_id"]) for row in merged]
    duplicate_pair_ids = sorted({value for value in pair_ids if pair_ids.count(value) > 1})
    duplicate_short_ids = sorted({value for value in short_ids if value and short_ids.count(value) > 1})
    if duplicate_pair_ids or duplicate_short_ids:
        raise ValueError(f"Duplicate IDs found: pair_ids={duplicate_pair_ids}, short_video_ids={duplicate_short_ids}")

    split_order = {split: index for index, (split, _path) in enumerate(args.input)}
    merged.sort(key=lambda row: (split_order[str(row["dataset_split"])], str(row["pair_id"])))
    fields = [
        "dataset_split", "evaluation_role", "performance_label", "pair_id", "long_video_id", "short_video_id", "channel_name",
        "long_video_url", "short_video_url", "gold_start_sec", "gold_end_sec", "gold_start_time", "gold_end_time",
        "gold_duration_sec", "short_views", "short_likes", "short_like_rate", "channel_performance_percentile",
        "label_confidence", "label_notes", "vpick_project_id", "vpick_asset_id", "data_quality_flags",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)

    summary = {
        "row_count": len(merged),
        "unique_long_video_count": len({row["long_video_id"] for row in merged}),
        "unique_short_video_count": len({row["short_video_id"] for row in merged}),
        "channel_count_excluding_blank": len({row["channel_name"] for row in merged if row["channel_name"]}),
        "split_counts": split_counts,
        "performance_label_counts": {
            label: sum(row["performance_label"] == label for row in merged)
            for label in ("pos", "neg", "unlabeled")
        },
        "missing_channel_count": sum(not bool(row["channel_name"]) for row in merged),
        "missing_views_count": sum(not bool(row["short_views"]) for row in merged),
        "missing_likes_count": sum(not bool(row["short_likes"]) for row in merged),
        "duplicate_pair_ids": duplicate_pair_ids,
        "duplicate_short_video_ids": duplicate_short_ids,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
