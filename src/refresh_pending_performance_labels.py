from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from label_pilot_performance import midpoint_percentile, performance_label
from youtube_metadata import channel_shorts_ids, fetch_text, oembed_metadata, official_youtube_stats


PENDING_STATUS = "teammate_selected_pending_stats_snapshot"
VERIFIED_STATUS = "verified_channel_percentile_snapshot"


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def public_page_stats(video_id: str) -> dict[str, Any]:
    html = fetch_text(f"https://www.youtube.com/shorts/{video_id}")
    view_match = re.search(r'"viewCount"\s*:\s*"?(\d+)"?', html)
    like_match = re.search(r'"likeCount"\s*:\s*"?(\d+)"?', html)
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.DOTALL)
    if not view_match:
        raise RuntimeError(f"Missing public view count for {video_id}")
    return {
        "video_id": video_id,
        "view_count": int(view_match.group(1)),
        "like_count": int(like_match.group(1)) if like_match else None,
        "title": (
            re.sub(r"\s+-\s+YouTube\s*$", "", title_match.group(1)).strip()
            if title_match
            else ""
        ),
        "source": "youtube_public_page",
    }


def cached_public_stats(video_id: str, cache_dir: Path, as_of: str) -> dict[str, Any]:
    cache_path = cache_dir / f"{video_id}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("stats_as_of") == as_of and cached.get("view_count") is not None:
            return cached
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = {**public_page_stats(video_id), "stats_as_of": as_of}
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception as exc:  # Network retries are intentionally broad.
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not collect stats for {video_id}: {last_error}")


def collect_stats(video_ids: Iterable[str], cache_dir: Path, as_of: str, workers: int) -> dict[str, dict[str, Any]]:
    ids = list(dict.fromkeys(video_id for video_id in video_ids if video_id))
    if os.getenv("YOUTUBE_API_KEY"):
        official = official_youtube_stats(ids)
        return {
            video_id: {**official[video_id], "stats_as_of": as_of}
            for video_id in ids
            if video_id in official
        }
    output: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(cached_public_stats, video_id, cache_dir, as_of): video_id
            for video_id in ids
        }
        for future in as_completed(futures):
            video_id = futures[future]
            output[video_id] = future.result()
    return output


def append_stats_notes(notes: str, *, percentile: float, as_of: str, source: str, cohort_n: int) -> str:
    cleaned = str(notes or "")
    for pattern in (
        r"channel_percentile=[^;]+;?\s*",
        r"stats_as_of=[^;]+;?\s*",
        r"stats_source=[^;]+;?\s*",
        r"cohort_n=[^;]+;?\s*",
    ):
        cleaned = re.sub(pattern, "", cleaned).strip(" ;")
    suffix = (
        f"channel_percentile={percentile:.1f}; stats_as_of={as_of}; "
        f"stats_source={source}; cohort_n={cohort_n}"
    )
    return f"{cleaned}; {suffix}" if cleaned else suffix


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh pending Short performance evidence and relabel it by channel percentile.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-snapshot", type=Path, required=True)
    parser.add_argument("--cohort-snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--as-of", default=dt.date.today().isoformat())
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows, fields = read_csv(args.input)
    pending = [row for row in rows if row.get("performance_evidence_status") == PENDING_STATUS]
    if not pending:
        raise SystemExit("No pending performance rows found.")
    pending_by_channel: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pending:
        pending_by_channel[row.get("channel_name", "")].append(row)

    target_snapshot: list[dict[str, Any]] = []
    cohort_snapshot: list[dict[str, Any]] = []
    stats_by_target: dict[str, dict[str, Any]] = {}
    channel_details: dict[str, Any] = {}
    for channel_name, channel_rows in sorted(pending_by_channel.items()):
        target_ids = [row["short_video_id"] for row in channel_rows]
        metadata = oembed_metadata(f"https://www.youtube.com/shorts/{target_ids[0]}")
        channel_url = str(metadata.get("channel_url", ""))
        if not channel_url:
            raise RuntimeError(f"Could not determine channel URL for {channel_name}")
        shorts_url = channel_url.rstrip("/") + "/shorts"
        listed_ids = channel_shorts_ids(shorts_url)
        cohort_ids = list(dict.fromkeys([*listed_ids, *target_ids]))
        print(f"[{channel_name}] targets={len(target_ids)} cohort={len(cohort_ids)}", flush=True)
        stats_by_id = collect_stats(
            cohort_ids,
            args.cache_dir / re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", channel_name),
            args.as_of,
            args.workers,
        )
        missing = [video_id for video_id in cohort_ids if stats_by_id.get(video_id, {}).get("view_count") is None]
        if missing:
            raise RuntimeError(f"Missing view counts for {channel_name}: {missing}")
        views = [int(stats_by_id[video_id]["view_count"]) for video_id in cohort_ids]
        target_row_by_id = {row["short_video_id"]: row for row in channel_rows}
        for video_id in cohort_ids:
            stats = stats_by_id[video_id]
            view_count = int(stats["view_count"])
            percentile = midpoint_percentile(views, view_count)
            label = performance_label(percentile)
            pair = target_row_by_id.get(video_id)
            snapshot_row = {
                "pair_id": pair.get("pair_id", "") if pair else "",
                "is_target": bool(pair),
                "channel_name": channel_name,
                "channel_url": channel_url,
                "video_id": video_id,
                "short_url": f"https://www.youtube.com/shorts/{video_id}",
                "title": stats.get("title", ""),
                "view_count": view_count,
                "like_count": "" if stats.get("like_count") is None else stats["like_count"],
                "like_rate": (
                    round(int(stats["like_count"]) / view_count, 8)
                    if stats.get("like_count") is not None and view_count
                    else ""
                ),
                "channel_view_percentile": percentile,
                "performance_label": label if pair else "",
                "stats_source": stats.get("source", "unknown"),
                "stats_as_of": args.as_of,
                "cohort_n": len(cohort_ids),
            }
            cohort_snapshot.append(snapshot_row)
            if pair:
                target_snapshot.append(snapshot_row)
                stats_by_target[video_id] = snapshot_row
        channel_details[channel_name] = {
            "target_count": len(target_ids),
            "listed_short_count": len(listed_ids),
            "cohort_n": len(cohort_ids),
            "channel_url": channel_url,
        }

    relabel_changes: list[dict[str, str]] = []
    for row in rows:
        stats = stats_by_target.get(row.get("short_video_id", ""))
        if not stats:
            continue
        old_label = row.get("performance_label", "")
        new_label = str(stats["performance_label"])
        row["short_views"] = str(stats["view_count"])
        row["short_likes"] = str(stats["like_count"])
        row["channel_performance_percentile"] = str(stats["channel_view_percentile"])
        row["performance_label"] = new_label
        row["performance_evidence_status"] = VERIFIED_STATUS
        row["dataset_split"] = "main" if new_label == "pos" else "control" if new_label == "neg" else "pilot"
        row["source_notes"] = append_stats_notes(
            row.get("source_notes", ""),
            percentile=float(stats["channel_view_percentile"]),
            as_of=args.as_of,
            source=str(stats["stats_source"]),
            cohort_n=int(stats["cohort_n"]),
        )
        relabel_changes.append(
            {
                "pair_id": row.get("pair_id", ""),
                "short_video_id": row.get("short_video_id", ""),
                "old_label": old_label,
                "new_label": new_label,
            }
        )

    target_fields = [
        "pair_id", "is_target", "channel_name", "channel_url", "video_id", "short_url", "title",
        "view_count", "like_count", "like_rate", "channel_view_percentile", "performance_label",
        "stats_source", "stats_as_of", "cohort_n",
    ]
    write_csv(args.output, rows, fields)
    write_csv(args.target_snapshot, sorted(target_snapshot, key=lambda row: (row["channel_name"], row["pair_id"])), target_fields)
    write_csv(args.cohort_snapshot, sorted(cohort_snapshot, key=lambda row: (row["channel_name"], -int(row["view_count"]))), target_fields)
    summary = {
        "stats_as_of": args.as_of,
        "stats_source": "youtube_data_api_v3" if os.getenv("YOUTUBE_API_KEY") else "youtube_public_page",
        "pending_target_count": len(pending),
        "updated_target_count": len(stats_by_target),
        "pending_relabel_counts": dict(sorted(Counter(change["new_label"] for change in relabel_changes).items())),
        "final_label_counts": dict(sorted(Counter(row.get("performance_label", "") for row in rows).items())),
        "label_changes": relabel_changes,
        "channels": channel_details,
        "output": str(args.output),
        "target_snapshot": str(args.target_snapshot),
        "cohort_snapshot": str(args.cohort_snapshot),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
