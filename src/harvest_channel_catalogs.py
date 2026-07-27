"""Harvest per-channel long-form catalogs for short->long-form source discovery.

Stage 0 of the mid-percentile candidate mapping pipeline. Resolves each
channel_id from one probe short, then caches the full reverse-chronological
video listing (id, duration, title) via a single flat-playlist request per
channel. Subtitle downloads are handled by later stages; this stage only
needs two requests per channel, so it is safe to re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

FLAT_FIELDS = "%(id)s\t%(duration)s\t%(title)s"
META_FIELDS = "%(channel)s\t%(channel_id)s\t%(upload_date)s\t%(duration)s"


def run_ytdlp(args: list[str], timeout: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["yt-dlp", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def is_rate_limited(stderr: str) -> bool:
    return "429" in stderr or "Too Many Requests" in stderr


def polite_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def resolve_channel(video_id: str, sleep_sec: float, retries: int) -> dict[str, str]:
    """Resolve channel identity from a single video's metadata."""
    for attempt in range(retries + 1):
        code, out, err = run_ytdlp(
            [
                "--skip-download",
                "--no-warnings",
                "--print",
                META_FIELDS,
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            timeout=180,
        )
        line = next((l for l in out.splitlines() if "\t" in l), "")
        if code == 0 and line:
            channel, channel_id, upload_date, duration = (line.split("\t") + ["", "", "", ""])[:4]
            return {
                "channel": channel,
                "channel_id": channel_id,
                "probe_upload_date": upload_date,
                "probe_duration": duration,
            }
        backoff = sleep_sec * (2 ** attempt) if is_rate_limited(err) else sleep_sec
        print(
            json.dumps(
                {
                    "event": "resolve_retry",
                    "video_id": video_id,
                    "attempt": attempt,
                    "rate_limited": is_rate_limited(err),
                    "backoff_sec": round(backoff, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        polite_sleep(backoff)
    raise RuntimeError(f"could not resolve channel for {video_id}")


def list_channel_videos(
    channel_id: str, sleep_sec: float, retries: int, playlist_end: int | None
) -> list[dict[str, Any]]:
    """Fetch the channel's videos tab as a flat listing (newest first)."""
    args = [
        "--flat-playlist",
        "--skip-download",
        "--no-warnings",
        "--print",
        FLAT_FIELDS,
    ]
    if playlist_end:
        args += ["--playlist-end", str(playlist_end)]
    args.append(f"https://www.youtube.com/channel/{channel_id}/videos")

    for attempt in range(retries + 1):
        try:
            code, out, err = run_ytdlp(args, timeout=900)
        except subprocess.TimeoutExpired:
            code, out, err = 1, "", "timeout"
        rows: list[dict[str, Any]] = []
        for position, line in enumerate(out.splitlines()):
            if "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            video_id, duration, title = parts[0], parts[1], "\t".join(parts[2:])
            rows.append(
                {
                    "position": position,
                    "video_id": video_id,
                    "duration_sec": None if duration in {"NA", ""} else float(duration),
                    "title": title,
                }
            )
        if code == 0 and rows:
            return rows
        backoff = sleep_sec * (2 ** attempt) if is_rate_limited(err) else sleep_sec
        print(
            json.dumps(
                {
                    "event": "list_retry",
                    "channel_id": channel_id,
                    "attempt": attempt,
                    "rows": len(rows),
                    "rate_limited": is_rate_limited(err),
                    "backoff_sec": round(backoff, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        polite_sleep(backoff)
    raise RuntimeError(f"could not list videos for {channel_id}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harvest per-channel long-form catalogs for source discovery."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep-sec", type=float, default=6.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--playlist-end", type=int)
    parser.add_argument("--min-longform-sec", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    catalog_dir = out_dir / "catalogs"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    with open(args.candidates, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    channels: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["channel_name"]
        channels.setdefault(name, {"channel_name": name, "shorts": []})
        channels[name]["shorts"].append(row)

    summary: list[dict[str, Any]] = []
    for name, info in sorted(channels.items()):
        target = catalog_dir / f"{name}.json"
        if target.exists() and not args.force:
            cached = json.loads(target.read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {"event": "cached", "channel_name": name, "videos": len(cached["videos"])},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            summary.append(
                {
                    "channel_name": name,
                    "channel_id": cached["channel_id"],
                    "video_count": len(cached["videos"]),
                    "longform_count": cached["longform_count"],
                    "source": "cache",
                }
            )
            continue

        probe = info["shorts"][0]["short_video_id"]
        identity = resolve_channel(probe, args.sleep_sec, args.retries)
        polite_sleep(args.sleep_sec)
        videos = list_channel_videos(
            identity["channel_id"], args.sleep_sec, args.retries, args.playlist_end
        )
        polite_sleep(args.sleep_sec)

        longform = [
            v
            for v in videos
            if v["duration_sec"] is not None and v["duration_sec"] >= args.min_longform_sec
        ]
        payload = {
            "channel_name": name,
            "channel_id": identity["channel_id"],
            "channel_title": identity["channel"],
            "probe_short_id": probe,
            "short_count": len(info["shorts"]),
            "video_count": len(videos),
            "longform_count": len(longform),
            "min_longform_sec": args.min_longform_sec,
            "videos": videos,
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "event": "harvested",
                    "channel_name": name,
                    "channel_id": identity["channel_id"],
                    "videos": len(videos),
                    "longform": len(longform),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        summary.append(
            {
                "channel_name": name,
                "channel_id": identity["channel_id"],
                "video_count": len(videos),
                "longform_count": len(longform),
                "source": "fetched",
            }
        )

    (out_dir / "catalog_summary.json").write_text(
        json.dumps(
            {
                "channel_count": len(summary),
                "short_count": len(rows),
                "min_longform_sec": args.min_longform_sec,
                "channels": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", "channels": len(summary)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
