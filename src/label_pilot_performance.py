from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any

from youtube_metadata import channel_shorts_ids, collect_stats, oembed_metadata


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def midpoint_percentile(values: list[int], value: int) -> float:
    below = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    return round(100.0 * (below + 0.5 * equal) / len(values), 1)


def performance_label(percentile: float) -> str:
    if percentile >= 75.0:
        return "pos"
    if percentile <= 25.0:
        return "neg"
    return "unlabeled"


def replace_metadata_notes(notes: str, *, percentile: float, as_of: str, source: str, cohort_n: int) -> str:
    cleaned = str(notes or "")
    patterns = [
        r"(?:채널내백분위|channel_percentile)=[^;]+;?\s*",
        r"stats_as_of=[^;]+;?\s*",
        r"stats_source=[^;]+;?\s*",
        r"cohort_n=[^;]+;?\s*",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned).strip(" ;")
    metadata = (
        f"channel_percentile={percentile:.1f}; stats_as_of={as_of}; "
        f"stats_source={source}; cohort_n={cohort_n}"
    )
    return f"{cleaned}; {metadata}" if cleaned else metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh Pilot YouTube statistics and assign performance labels.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--as-of", default=dt.date.today().isoformat())
    parser.add_argument("--channel-shorts-url")
    parser.add_argument("--require-official-api", action="store_true")
    args = parser.parse_args()

    if args.require_official_api and not os.getenv("YOUTUBE_API_KEY"):
        raise SystemExit("YOUTUBE_API_KEY is required when --require-official-api is set.")

    rows, fieldnames = read_csv(Path(args.pairs))
    target_ids = [row["short_video_id"] for row in rows if row.get("short_video_id")]
    if not target_ids:
        raise SystemExit("No short_video_id values found.")

    channel_url = args.channel_shorts_url
    if not channel_url:
        metadata = oembed_metadata(f"https://www.youtube.com/shorts/{target_ids[0]}")
        channel_url = str(metadata.get("channel_url", ""))
    if not channel_url:
        raise SystemExit("Could not determine channel URL; pass --channel-shorts-url.")
    channel_shorts_url = channel_url.rstrip("/") + "/shorts"

    listed_ids = channel_shorts_ids(channel_shorts_url)
    cohort_ids = list(dict.fromkeys([*listed_ids, *target_ids]))
    stats_by_id = collect_stats(cohort_ids)
    if args.require_official_api and {
        str(value.get("source", "")) for value in stats_by_id.values()
    } != {"youtube_data_api_v3"}:
        raise RuntimeError("Official YouTube Data API v3 was required but a fallback source was used.")
    missing_ids = [video_id for video_id in cohort_ids if stats_by_id.get(video_id, {}).get("view_count") is None]
    if missing_ids:
        raise RuntimeError(f"Missing view counts for {missing_ids}")

    views = [int(stats_by_id[video_id]["view_count"]) for video_id in cohort_ids]
    target_by_id = {row["short_video_id"]: row for row in rows}
    snapshot_rows: list[dict[str, Any]] = []
    label_counts = {"pos": 0, "neg": 0, "unlabeled": 0}
    for video_id in cohort_ids:
        stats = stats_by_id[video_id]
        view_count = int(stats["view_count"])
        like_count = stats.get("like_count")
        percentile = midpoint_percentile(views, view_count)
        label = performance_label(percentile)
        pair = target_by_id.get(video_id)
        if pair:
            pair["channel_name"] = str(stats.get("channel_name") or pair.get("channel_name") or "")
            pair["short_views"] = str(view_count)
            pair["short_likes"] = "" if like_count is None else str(like_count)
            pair["label_notes"] = replace_metadata_notes(
                pair.get("label_notes", ""),
                percentile=percentile,
                as_of=args.as_of,
                source=str(stats.get("source", "unknown")),
                cohort_n=len(cohort_ids),
            )
            label_counts[label] += 1
        snapshot_rows.append(
            {
                "pair_id": pair.get("pair_id", "") if pair else "",
                "is_dataset_pair": bool(pair),
                "video_id": video_id,
                "title": stats.get("title", ""),
                "channel_name": stats.get("channel_name", ""),
                "published_at": stats.get("published_at", ""),
                "view_count": view_count,
                "like_count": "" if like_count is None else like_count,
                "like_rate": round(int(like_count) / view_count, 8) if like_count is not None and view_count else "",
                "channel_view_percentile": percentile,
                "performance_label": label if pair else "",
                "stats_source": stats.get("source", "unknown"),
                "stats_as_of": args.as_of,
                "cohort_n": len(cohort_ids),
            }
        )

    write_csv(Path(args.output), rows, fieldnames)
    write_csv(
        Path(args.snapshot),
        snapshot_rows,
        [
            "pair_id", "is_dataset_pair", "video_id", "title", "channel_name", "published_at",
            "view_count", "like_count", "like_rate", "channel_view_percentile", "performance_label",
            "stats_source", "stats_as_of", "cohort_n",
        ],
    )
    print(
        json.dumps(
            {
                "pair_count": len(rows),
                "listed_short_count": len(listed_ids),
                "cohort_count": len(cohort_ids),
                "performance_label_counts": label_counts,
                "stats_source": sorted({str(value.get("source", "unknown")) for value in stats_by_id.values()}),
                "output": args.output,
                "snapshot": args.snapshot,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
