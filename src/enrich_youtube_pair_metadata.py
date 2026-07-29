from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

from youtube_metadata import extract_youtube_id


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


def row_video_id(row: dict[str, str], kind: str) -> str:
    direct = row.get(f"{kind}_video_id", "").strip()
    if direct:
        return direct
    url = row.get(f"{kind}_video_url", "").strip()
    return extract_youtube_id(url) if url else ""


def compact_metadata(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": info.get("title", ""),
        "channel": info.get("channel", "") or info.get("uploader", ""),
        "channel_id": info.get("channel_id", ""),
        "duration": info.get("duration", ""),
        "view_count": info.get("view_count", ""),
        "like_count": info.get("like_count", ""),
        "upload_date": info.get("upload_date", ""),
        "availability": info.get("availability", ""),
    }


def fetch_metadata(video_ids: list[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with YoutubeDL(
        {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
        }
    ) as ydl:
        for index, video_id in enumerate(dict.fromkeys(video_ids), start=1):
            print(f"[{index}/{len(set(video_ids))}] metadata {video_id}", flush=True)
            try:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False,
                )
                output[video_id] = compact_metadata(info)
            except Exception as exc:
                output[video_id] = {
                    "metadata_error": type(exc).__name__,
                }
    return output


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    video_ids = [
        video_id
        for row in rows
        for video_id in (row_video_id(row, "short"), row_video_id(row, "long"))
        if video_id
    ]
    metadata = fetch_metadata(video_ids)
    snapshot = date.today().isoformat()
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched: dict[str, Any] = dict(row)
        short_id = row_video_id(row, "short")
        long_id = row_video_id(row, "long")
        short = metadata.get(short_id, {})
        long = metadata.get(long_id, {})
        enriched.update(
            {
                "short_video_id": short_id,
                "long_video_id": long_id,
                "short_title_yt": short.get("title", ""),
                "short_channel_yt": short.get("channel", ""),
                "short_channel_id_yt": short.get("channel_id", ""),
                "short_duration_sec_yt": short.get("duration", ""),
                "short_views_yt": short.get("view_count", ""),
                "short_likes_yt": short.get("like_count", ""),
                "short_upload_date_yt": short.get("upload_date", ""),
                "short_availability_yt": short.get("availability", ""),
                "short_metadata_error": short.get("metadata_error", ""),
                "long_title_yt": long.get("title", ""),
                "long_channel_yt": long.get("channel", ""),
                "long_channel_id_yt": long.get("channel_id", ""),
                "long_duration_sec_yt": long.get("duration", ""),
                "long_views_yt": long.get("view_count", ""),
                "long_likes_yt": long.get("like_count", ""),
                "long_upload_date_yt": long.get("upload_date", ""),
                "long_availability_yt": long.get("availability", ""),
                "long_metadata_error": long.get("metadata_error", ""),
                "youtube_metadata_source": "yt_dlp",
                "youtube_metadata_snapshot_date": snapshot,
            }
        )
        output.append(enriched)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich long-form/Short pairs with yt-dlp metadata.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_rows(args.output, enrich_rows(read_rows(args.input)))


if __name__ == "__main__":
    main()
