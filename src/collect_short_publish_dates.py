from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from youtube_metadata import fetch_text


ROOT = Path(__file__).resolve().parents[1]
DATE_PATTERNS = (
    r'"publishDate"\s*:\s*"([^"]+)"',
    r'"uploadDate"\s*:\s*"([^"]+)"',
    r'<meta\s+itemprop="uploadDate"\s+content="([^"]+)"',
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fetch_date(video_id: str) -> dict[str, Any]:
    html = fetch_text(f"https://www.youtube.com/shorts/{video_id}")
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, html)
        if match:
            return {
                "short_video_id": video_id,
                "published_at": match.group(1),
                "source": "youtube_public_page",
                "error": "",
            }
    return {
        "short_video_id": video_id,
        "published_at": "",
        "source": "youtube_public_page",
        "error": "publish_date_not_found",
    }


def cached_fetch(video_id: str, cache_dir: Path) -> dict[str, Any]:
    cache_path = cache_dir / f"{video_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = fetch_date(video_id)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    return {
        "short_video_id": video_id,
        "published_at": "",
        "source": "youtube_public_page",
        "error": str(last_error),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "data/private/judge_validation_94/validation_targets_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/private/judge_validation_94/short_publish_dates_2026-07-29_PRIVATE.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/private/youtube_publish_date_cache",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    targets = read_csv(args.targets)
    ids = list(dict.fromkeys(row["short_video_id"] for row in targets))
    collected: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(cached_fetch, video_id, args.cache_dir): video_id
            for video_id in ids
        }
        for future in as_completed(futures):
            video_id = futures[future]
            collected[video_id] = future.result()
            print(
                json.dumps(
                    {
                        "completed": len(collected),
                        "total": len(ids),
                        "video_id": video_id,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    rows = [
        {
            "candidate_id": row["candidate_id"],
            **collected[row["short_video_id"]],
        }
        for row in targets
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "with_date": sum(bool(row["published_at"]) for row in rows),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
