from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

from audit_short_long_alignment import (
    SubtitleCollector,
    align_transcripts,
    parse_json3,
)
from youtube_metadata import extract_youtube_id


def channel_video_candidates(channel_videos_url: str, limit: int) -> list[dict[str, Any]]:
    with YoutubeDL(
        {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "playlistend": limit,
        }
    ) as ydl:
        info = ydl.extract_info(channel_videos_url, download=False)
    return [
        {
            "video_id": str(entry.get("id", "")),
            "title": str(entry.get("title", "")),
            "duration": entry.get("duration", ""),
            "view_count": entry.get("view_count", ""),
        }
        for entry in info.get("entries", [])
        if entry and entry.get("id")
    ]


def rank_sources(
    short_video_id: str,
    candidates: list[dict[str, Any]],
    collector: SubtitleCollector,
) -> list[dict[str, Any]]:
    short_path, short_source, short_language = collector.fetch(short_video_id)
    if not short_path:
        raise RuntimeError(f"No subtitle track available for short {short_video_id}")
    short_cues = parse_json3(short_path)

    ranked: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        video_id = candidate["video_id"]
        print(f"[{index}/{len(candidates)}] {short_video_id} -> {video_id}", flush=True)
        long_path, long_source, long_language = collector.fetch(video_id)
        row = {
            "short_video_id": short_video_id,
            "short_subtitle_source": short_source,
            "short_subtitle_language": short_language,
            **candidate,
            "long_subtitle_source": long_source,
            "long_subtitle_language": long_language,
            "alignment_status": "missing_subtitle",
            "coverage": "",
            "mean_match_score": "",
            "predicted_start": "",
            "predicted_end": "",
            "source_span": "",
            "short_span": "",
            "segment_count": "",
            "backward_jumps": "",
            "excess_gap_seconds": "",
        }
        if long_path:
            result = align_transcripts(short_cues, parse_json3(long_path))
            row.update(
                {
                    "alignment_status": result.get("status", ""),
                    "coverage": result.get("coverage", ""),
                    "mean_match_score": result.get("mean_match_score", ""),
                    "predicted_start": result.get("predicted_start", ""),
                    "predicted_end": result.get("predicted_end", ""),
                    "source_span": result.get("source_span", ""),
                    "short_span": result.get("short_span", ""),
                    "segment_count": result.get("segment_count", ""),
                    "backward_jumps": result.get("backward_jumps", ""),
                    "excess_gap_seconds": result.get("excess_gap_seconds", ""),
                }
            )
        ranked.append(row)

    def ranking_key(row: dict[str, Any]) -> tuple[float, float]:
        coverage = float(row["coverage"]) if row["coverage"] != "" else -1.0
        mean_score = float(row["mean_match_score"]) if row["mean_match_score"] != "" else -1.0
        return coverage, mean_score

    return sorted(ranked, key=ranking_key, reverse=True)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the source long-form video for a Short by transcript alignment."
    )
    parser.add_argument("--short", required=True, help="Short video ID or URL")
    parser.add_argument("--channel-videos-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    short_id = extract_youtube_id(args.short) or args.short
    cache_dir = args.cache_dir or args.output.parent / "subtitles"
    candidates = channel_video_candidates(args.channel_videos_url, args.limit)
    collector = SubtitleCollector(cache_dir=cache_dir, sleep_seconds=args.sleep_seconds)
    ranked = rank_sources(short_id, candidates, collector)
    write_rows(args.output, ranked)
    for row in ranked[:10]:
        print(
            f"{row['video_id']}\t{row['coverage']}\t{row['mean_match_score']}\t"
            f"{row['predicted_start']}-{row['predicted_end']}\t{row['title']}"
        )


if __name__ == "__main__":
    main()
