"""Fill-in sheet for shorts that never got a source long-form, with suggestions.

These are the mid-percentile shorts whose origin was never found in the pinned
comments, so nothing about the source is known beyond the short itself. Filling
them by hand means scanning a channel catalog of up to a thousand videos, so the
sheet ships candidate long-forms scored by title overlap against the short's
title and hashtags. The suggestions are a starting point, not an answer: they are
title-only, and the pipeline still verifies any pasted URL by subtitle alignment.

Anything already carrying a long-form URL, and anything already in the gold set,
is omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

SUGGESTIONS = 3
# Tokens that appear in most titles of a channel and so carry no signal.
STOPWORDS = {
    "shorts", "the", "and", "for", "with", "that", "you", "your", "was", "are",
    "이거", "그거", "저거", "진짜", "너무", "정말", "우리", "근데", "그래서",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def short_id_of(row: dict[str, str]) -> str:
    stated = row.get("short_video_id", "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", stated):
        return stated
    for column in ("short_url", "short_video_url"):
        match = re.search(r"(?:shorts/|v=|youtu\.be/)([A-Za-z0-9_-]{11})", row.get(column, ""))
        if match:
            return match.group(1)
    return stated


def tokens(text: str) -> set[str]:
    """Content words and hashtags, long enough to be distinctive."""
    text = text.replace("#", " ")
    raw = re.split(r"[^0-9A-Za-z가-힣]+", text.lower())
    return {t for t in raw if len(t) >= 2 and t not in STOPWORDS}


def bigrams(text: str) -> set[str]:
    norm = re.sub(r"[^0-9A-Za-z가-힣]", "", text.lower())
    return {norm[i:i + 2] for i in range(len(norm) - 1)} if len(norm) >= 2 else set()


def score_title(short_title: str, long_title: str) -> float:
    """Token overlap plus character bigram overlap, so partial words still count."""
    st, lt = tokens(short_title), tokens(long_title)
    token_hit = len(st & lt) / len(st) if st else 0.0
    sb, lb = bigrams(short_title), bigrams(long_title)
    gram_hit = len(sb & lb) / len(sb) if sb else 0.0
    return round(0.6 * token_hit + 0.4 * gram_hit, 4)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill-in sheet for shorts with no known source long-form."
    )
    parser.add_argument("--pool-csv", action="append", required=True)
    parser.add_argument("--linked-csv", action="append", default=[],
                        help="CSVs whose rows already carry an origin URL.")
    parser.add_argument("--gold-master", action="append", default=[])
    parser.add_argument("--catalog-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-longform-sec", type=float, default=300.0)
    args = parser.parse_args()

    already: set[str] = set()
    for path in args.linked_csv:
        for row in read_csv(Path(path)):
            if (row.get("origin_long_video_url") or "").strip():
                already.add(short_id_of(row))
    for path in args.gold_master:
        for row in read_csv(Path(path)):
            already.add(row.get("short_video_id", "").strip())

    pool: dict[str, dict[str, str]] = {}
    for path in args.pool_csv:
        for row in read_csv(Path(path)):
            pool.setdefault(short_id_of(row), row)

    catalogs: dict[str, list[dict[str, Any]]] = {}
    for path in Path(args.catalog_dir).glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        catalogs[payload["channel_name"]] = [
            v
            for v in payload["videos"]
            if v.get("duration_sec") and v["duration_sec"] >= args.min_longform_sec
        ]

    rows: list[dict[str, Any]] = []
    for short_id, row in pool.items():
        if short_id in already:
            continue
        title = row.get("title", "")
        channel = row.get("channel_name", "")
        ranked = sorted(
            (
                (score_title(title, v["title"]), v)
                for v in catalogs.get(channel, [])
            ),
            key=lambda pair: -pair[0],
        )[:SUGGESTIONS]
        entry: dict[str, Any] = {
            "channel_name": channel,
            "short_video_id": short_id,
            "short_url": row.get("short_url") or f"https://www.youtube.com/shorts/{short_id}",
            "short_title": title,
            "short_upload_date": row.get("upload_date", ""),
            "short_duration_sec": row.get("metadata_duration_sec") or row.get("short_duration_sec", ""),
            "percentile_bucket": row.get("percentile_bucket", ""),
            "channel_performance_percentile": row.get("channel_performance_percentile", ""),
            "short_views": row.get("short_views", ""),
            "channel_longform_count": len(catalogs.get(channel, [])),
        }
        for index in range(SUGGESTIONS):
            if index < len(ranked):
                score, video = ranked[index]
                entry[f"suggest{index+1}_score"] = score
                entry[f"suggest{index+1}_url"] = f"https://www.youtube.com/watch?v={video['video_id']}"
                entry[f"suggest{index+1}_title"] = video["title"][:80]
                entry[f"suggest{index+1}_duration_sec"] = video["duration_sec"]
            else:
                entry[f"suggest{index+1}_score"] = ""
                entry[f"suggest{index+1}_url"] = ""
                entry[f"suggest{index+1}_title"] = ""
                entry[f"suggest{index+1}_duration_sec"] = ""
        entry["origin_long_video_url"] = ""
        entry["manual_start_sec"] = ""
        entry["manual_end_sec"] = ""
        rows.append(entry)

    rows.sort(key=lambda r: (r["channel_name"], r["percentile_bucket"], r["short_video_id"]))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    strong = [r for r in rows if isinstance(r["suggest1_score"], float) and r["suggest1_score"] >= 0.3]
    summary = {
        "rows_to_fill": len(rows),
        "channel_counts": {
            ch: sum(1 for r in rows if r["channel_name"] == ch)
            for ch in sorted({r["channel_name"] for r in rows})
        },
        "bucket_counts": {
            b: sum(1 for r in rows if r["percentile_bucket"] == b)
            for b in sorted({r["percentile_bucket"] for r in rows})
        },
        "with_suggestion": sum(1 for r in rows if r["suggest1_url"]),
        "strong_suggestion_ge_0_3": len(strong),
        "output": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
