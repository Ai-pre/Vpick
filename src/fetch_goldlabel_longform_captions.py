"""Cache yt-dlp caption tracks for every long-form referenced by a gold-label master.

Needed so transcript rendering can be normalized across the whole gold set: the
vpick-sourced rows carry speaker labels but no per-line timestamps, so their
transcripts cannot be re-rendered into the same shape as the yt-dlp rows without
going back to the captions. Downloads are cached per (video, language) and the
script is safe to re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

LANGUAGES = ("ko-orig", "ko", "ko-ko")


def have_caption(cache: Path, video_id: str) -> Path | None:
    for language in LANGUAGES:
        path = cache / f"{video_id}.{language}.json3"
        if path.exists() and path.stat().st_size > 20:
            return path
    matches = sorted(cache.glob(f"{video_id}.*.json3"))
    return matches[0] if matches else None


def fetch(cache: Path, video_id: str, sleep_sec: float, retries: int) -> tuple[bool, str]:
    for attempt in range(retries + 1):
        proc = subprocess.run(
            [
                "yt-dlp",
                "--skip-download",
                "--write-auto-subs",
                "--sub-langs",
                ",".join(LANGUAGES),
                "--sub-format",
                "json3",
                "--no-warnings",
                "-o",
                "%(id)s",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True,
            text=True,
            cwd=str(cache),
            timeout=240,
        )
        if have_caption(cache, video_id):
            return True, "ok"
        blob = proc.stdout + proc.stderr
        rate_limited = "429" in blob or "Too Many Requests" in blob
        wait = sleep_sec * (3 ** attempt) if rate_limited else sleep_sec
        if attempt < retries:
            print(
                json.dumps(
                    {"event": "retry", "video_id": video_id, "attempt": attempt,
                     "rate_limited": rate_limited, "wait_sec": round(wait, 1)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(wait)
    return False, "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache captions for long-forms referenced by a gold-label master."
    )
    parser.add_argument("--master", action="append", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--extra-cache", action="append", default=[])
    parser.add_argument("--sleep-sec", type=float, default=7.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    wanted: list[str] = []
    for path in args.master:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                video_id = row.get("longform_id", "").strip()
                if video_id and video_id not in wanted:
                    wanted.append(video_id)

    # Reuse captions already downloaded by earlier stages.
    for extra in args.extra_cache:
        source = Path(extra)
        if not source.is_dir():
            continue
        for path in source.glob("*.json3"):
            target = cache / path.name
            if not target.exists():
                target.write_bytes(path.read_bytes())

    results = {}
    for index, video_id in enumerate(wanted, start=1):
        if have_caption(cache, video_id):
            results[video_id] = "cached"
            continue
        ok, status = fetch(cache, video_id, args.sleep_sec, args.retries)
        results[video_id] = "fetched" if ok else status
        print(
            json.dumps(
                {"event": "row", "index": index, "total": len(wanted),
                 "video_id": video_id, "status": results[video_id]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(args.sleep_sec)

    counts: dict[str, int] = {}
    for status in results.values():
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "longforms": len(wanted),
        "status_counts": counts,
        "unavailable": [v for v, s in results.items() if s == "unavailable"],
        "cache_dir": str(cache),
    }
    (cache.parent / "caption_fetch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
