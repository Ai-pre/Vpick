"""Locate a short's source span by constrained window search instead of free alignment.

The DP aligner in `audit_short_long_alignment` matches each short chunk to its best
long-form window independently, so when the captions are noisy it happily spreads
one 30-second short across ten minutes of source and still reports a high mean
score. That is the `needs_manual_review` failure mode: the long-form is correct
but the span is not.

This scores every contiguous window whose duration is plausible for the short and
returns the best one, so the answer cannot be wider than the short by construction.
Scoring is character-bigram overlap against the short's own transcript, which
tolerates ASR noise better than token matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from audit_short_long_alignment import parse_json3


def normalize(text: str) -> str:
    return re.sub(r"[^\w]", "", text, flags=re.UNICODE).replace("_", "").lower()


def bigrams(text: str) -> set[str]:
    norm = normalize(text)
    if len(norm) < 2:
        return {norm} if norm else set()
    return {norm[i:i + 2] for i in range(len(norm) - 1)}


def coverage(query: set[str], window: set[str]) -> float:
    """Share of the short's bigrams present in the window."""
    if not query:
        return 0.0
    return len(query & window) / len(query)


def window_score(query: set[str], window: set[str]) -> tuple[float, float, float]:
    """F1 of coverage and precision, so widening the window is not free.

    Coverage alone rises monotonically with window width, so a coverage-only
    search always returns the widest window the ratio cap allows rather than the
    span that actually matches. Precision (share of the window's bigrams that the
    short used) falls as unrelated speech is swept in, and the harmonic mean of
    the two has an interior optimum.
    """
    if not query or not window:
        return 0.0, 0.0, 0.0
    hits = len(query & window)
    recall = hits / len(query)
    precision = hits / len(window)
    if not (recall and precision):
        return 0.0, recall, precision
    return 2 * recall * precision / (recall + precision), recall, precision


def load_short_transcript(paths: list[Path], short_id: str) -> str:
    for path in paths:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("short_video_id") == short_id:
                return " ".join(
                    str(seg.get("text", "")).strip() for seg in payload.get("segments", [])
                )
    return ""


def long_caption(dirs: list[Path], long_id: str) -> Path | None:
    for directory in dirs:
        for language in ("ko-orig", "ko", "ko-ko", "ko-KR", "en-orig", "en"):
            candidate = directory / f"{long_id}.{language}.json3"
            if candidate.exists() and candidate.stat().st_size > 20:
                return candidate
        matches = sorted(directory.glob(f"{long_id}.*.json3"))
        if matches:
            return matches[0]
    return None


def search(
    cues: list[Any],
    query: str,
    short_duration: float,
    min_ratio: float,
    max_ratio: float,
    step: int,
) -> dict[str, Any] | None:
    query_grams = bigrams(query)
    if not query_grams or not cues:
        return None
    lo = short_duration * min_ratio
    hi = short_duration * max_ratio
    best: dict[str, Any] | None = None
    for start_index in range(0, len(cues), step):
        start = cues[start_index].start
        window_grams: set[str] = set()
        for end_index in range(start_index, len(cues)):
            span = cues[end_index].end - start
            if span > hi:
                break
            window_grams |= bigrams(cues[end_index].text)
            if span < lo:
                continue
            score, recall, precision = window_score(query_grams, window_grams)
            if best is None or score > best["f1"]:
                best = {
                    "f1": round(score, 4),
                    "coverage": round(recall, 4),
                    "precision": round(precision, 4),
                    "start_sec": round(start, 3),
                    "end_sec": round(cues[end_index].end, 3),
                    "span_sec": round(span, 3),
                    "cue_count": end_index - start_index + 1,
                }
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Locate source spans by constrained window search."
    )
    parser.add_argument("--input", required=True, help="remap sheet with confirmed origin URLs")
    parser.add_argument("--subtitle-dir", action="append", required=True)
    parser.add_argument("--gemini-transcripts", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-ratio", type=float, default=0.8)
    parser.add_argument("--max-ratio", type=float, default=1.6)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--min-coverage", type=float, default=0.35)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [Path(p) for p in args.subtitle_dir]
    hint_paths = [Path(p) for p in args.gemini_transcripts]

    with open(args.input, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    results: list[dict[str, Any]] = []
    for row in rows:
        url = (row.get("origin_long_video_url") or "").strip()
        match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
        if not match:
            continue
        long_id = match.group(1)
        short_id = row["short_video_id"]
        try:
            short_duration = float(row.get("short_duration_sec") or 0)
        except ValueError:
            short_duration = 0.0
        record: dict[str, Any] = {
            "channel_name": row.get("channel_name", ""),
            "short_video_id": short_id,
            "long_video_id": long_id,
            "short_duration_sec": short_duration,
        }
        query = load_short_transcript(hint_paths, short_id) or row.get("short_speech_hint", "")
        caption = long_caption(dirs, long_id)
        if not query or not caption or short_duration <= 0:
            record.update(
                {"status": "missing_input", "has_query": bool(query),
                 "has_caption": bool(caption)}
            )
            results.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            continue

        cues = parse_json3(caption)
        best = search(cues, query, short_duration, args.min_ratio, args.max_ratio, args.step)
        if not best:
            record.update({"status": "no_window", "long_cue_count": len(cues)})
        else:
            record.update(
                {
                    "status": "located" if best["coverage"] >= args.min_coverage else "low_coverage",
                    "long_cue_count": len(cues),
                    "query_len": len(query),
                    **best,
                    "span_ratio": round(best["span_sec"] / short_duration, 3),
                }
            )
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if results:
        fields = list(dict.fromkeys(k for r in results for k in r))
        with (out_dir / "windowed_spans.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "rows": len(results),
        "status_counts": {
            status: sum(1 for r in results if r.get("status") == status)
            for status in sorted({str(r.get("status")) for r in results})
        },
        "min_coverage": args.min_coverage,
        "ratio_window": [args.min_ratio, args.max_ratio],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
