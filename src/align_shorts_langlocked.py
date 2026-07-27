"""Align shorts to their source long-form using a language-locked subtitle pair.

`audit_short_long_alignment.py` picks a preferred subtitle track for each video
independently, so a short can end up matched in Japanese against a long-form in
Korean. Cross-language fuzzy matching either fails outright or, worse, produces a
short high-scoring span that looks like a clean hit. This module picks one
language both videos share before downloading anything, caches per language, and
scores the resulting span against the short's own duration so implausible
alignments are flagged rather than auto-accepted.

Pure parsing/alignment helpers are reused from audit_short_long_alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

from audit_short_long_alignment import (
    align_transcripts,
    display_timestamp,
    parse_json3,
)

# Preference order for the shared language. Korean originals first because every
# channel in this dataset speaks Korean; auto-translated tracks are last resort.
LANGUAGE_PREFERENCE = (
    "ko-orig",
    "ko",
    "ko-KR",
    "en-orig",
    "en",
)


def track_languages(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map language -> json3 track, preferring automatic captions over manual."""
    available: dict[str, dict[str, Any]] = {}
    for source_name in ("automatic_captions", "subtitles"):
        for language, formats in (info.get(source_name) or {}).items():
            track = next(
                (item for item in formats if item.get("ext") == "json3" and item.get("url")),
                None,
            )
            if track and language not in available:
                available[language] = {"source": source_name, "track": track}
    return available


def shared_languages(
    short_tracks: dict[str, Any], long_tracks: dict[str, Any]
) -> list[str]:
    """Languages present in both videos, most promising first.

    Korean first because every channel here speaks Korean, but English stays in
    the list: when the clip's audio is actually English, the Korean track is a
    machine translation on both sides and matches far worse than the original.
    """
    shared = set(short_tracks) & set(long_tracks)
    ordered = [language for language in LANGUAGE_PREFERENCE if language in shared]
    ordered += sorted(l for l in shared if l.startswith("ko") and l not in ordered)
    return ordered


def choose_shared_language(
    short_tracks: dict[str, Any], long_tracks: dict[str, Any]
) -> tuple[str | None, str]:
    """Pick one language present in both videos; report why it was chosen."""
    ordered = shared_languages(short_tracks, long_tracks)
    if not ordered:
        return None, "no_shared_language"
    return ordered[0], "preference_order"


class LanguageLockedCollector:
    def __init__(self, cache_dir: Path, sleep_seconds: float = 6.0) -> None:
        self.cache_dir = cache_dir
        self.sleep_seconds = sleep_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ydl = YoutubeDL(
            {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
            }
        )
        self._info_cache: dict[str, dict[str, Any]] = {}

    def info(self, video_id: str) -> dict[str, Any] | None:
        if video_id in self._info_cache:
            return self._info_cache[video_id]
        path = self.cache_dir / f"{video_id}.info.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._info_cache[video_id] = payload
            return payload
        try:
            info = self.ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
        except Exception as exc:  # Network/bot-check failures are expected.
            print(
                json.dumps(
                    {"event": "info_error", "video_id": video_id, "error": str(exc)[:160]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return None
        payload = {
            "video_id": video_id,
            "title": info.get("title", ""),
            "duration": info.get("duration"),
            "upload_date": info.get("upload_date", ""),
            "languages": {
                language: {"source": meta["source"]}
                for language, meta in track_languages(info).items()
            },
            "_tracks": {
                language: meta["track"]["url"]
                for language, meta in track_languages(info).items()
            },
        }
        path.write_text(
            json.dumps({k: v for k, v in payload.items() if k != "_tracks"},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._info_cache[video_id] = payload
        time.sleep(self.sleep_seconds)
        return payload

    def subtitle(self, video_id: str, language: str) -> Path | None:
        path = self.cache_dir / f"{video_id}.{language}.json3"
        if path.exists() and path.stat().st_size > 20:
            return path
        info = self.info(video_id)
        if not info:
            return None
        url = (info.get("_tracks") or {}).get(language)
        if not url:
            return None
        for attempt in range(3):
            try:
                with self.ydl.urlopen(url) as response:
                    payload = response.read()
                json.loads(payload.decode("utf-8"))
                path.write_bytes(payload)
                time.sleep(self.sleep_seconds)
                return path
            except Exception:
                time.sleep((attempt + 1) * self.sleep_seconds)
        return None


def span_verdict(span: float, short_duration: float) -> tuple[str, float | str]:
    """Judge the aligned span against how long the short actually is."""
    if not short_duration or not span:
        return "unknown", ""
    ratio = span / short_duration
    if 0.6 <= ratio <= 1.6:
        return "plausible", round(ratio, 3)
    if ratio < 0.6:
        return "span_too_short", round(ratio, 3)
    if ratio <= 2.5:
        return "span_long_review", round(ratio, 3)
    return "span_implausible", round(ratio, 3)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align shorts to long-form sources with a language-locked subtitle pair."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=6.0)
    parser.add_argument("--only-short-ids", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collector = LanguageLockedCollector(out_dir / "subtitles", args.sleep_seconds)

    with open(args.input, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    only = {v.strip() for v in args.only_short_ids.split(",") if v.strip()}
    if only:
        rows = [r for r in rows if r["short_video_id"] in only]

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        short_id = row["short_video_id"]
        long_id = row["long_video_id"]
        short_info = collector.info(short_id)
        long_info = collector.info(long_id)
        record: dict[str, Any] = {
            "pair_id": row.get("pair_id", ""),
            "channel_name": row.get("channel_name", ""),
            "short_video_id": short_id,
            "long_video_id": long_id,
            "short_duration_sec": row.get("metadata_duration_sec", ""),
            "long_duration_sec": (long_info or {}).get("duration", ""),
            "long_title": (long_info or {}).get("title", ""),
            "short_languages": ",".join(sorted((short_info or {}).get("languages", {}))),
            "long_languages": ",".join(sorted((long_info or {}).get("languages", {}))),
        }
        if not short_info or not long_info:
            record.update(
                {"alignment_status": "metadata_unavailable", "chosen_language": "",
                 "language_choice_reason": "", "needs_gemini": 1}
            )
            results.append(record)
            print(json.dumps({"event": "row", "index": index, **record}, ensure_ascii=False), flush=True)
            continue

        language, reason = choose_shared_language(
            short_info.get("languages", {}), long_info.get("languages", {})
        )
        record["chosen_language"] = language or ""
        record["language_choice_reason"] = reason
        if not language:
            record.update({"alignment_status": "no_shared_language", "needs_gemini": 1})
            results.append(record)
            print(json.dumps({"event": "row", "index": index, **record}, ensure_ascii=False), flush=True)
            continue

        short_path = collector.subtitle(short_id, language)
        long_path = collector.subtitle(long_id, language)
        if not short_path or not long_path:
            record.update({"alignment_status": "missing_subtitle", "needs_gemini": 1})
            results.append(record)
            print(json.dumps({"event": "row", "index": index, **record}, ensure_ascii=False), flush=True)
            continue

        short_cues = parse_json3(short_path)
        long_cues = parse_json3(long_path)
        record["short_cue_count"] = len(short_cues)
        record["long_cue_count"] = len(long_cues)
        result = align_transcripts(short_cues, long_cues)
        predicted_start = result.get("predicted_start", "")
        predicted_end = result.get("predicted_end", "")
        span = float(result.get("source_span") or 0.0)
        try:
            short_duration = float(row.get("metadata_duration_sec") or 0.0)
        except ValueError:
            short_duration = 0.0
        verdict, ratio = span_verdict(span, short_duration)
        record.update(
            {
                "alignment_status": result.get("status", ""),
                "coverage": result.get("coverage", ""),
                "mean_match_score": result.get("mean_match_score", ""),
                "predicted_start": predicted_start,
                "predicted_end": predicted_end,
                "predicted_start_time": display_timestamp(predicted_start),
                "predicted_end_time": display_timestamp(predicted_end),
                "source_span": result.get("source_span", ""),
                "segment_count": result.get("segment_count", ""),
                "backward_jumps": result.get("backward_jumps", ""),
                "span_verdict": verdict,
                "span_ratio": ratio,
                "accept": int(
                    result.get("status") in {"continuous", "light_edit"}
                    and verdict == "plausible"
                ),
                "needs_gemini": int(
                    result.get("status") in {"insufficient_alignment", "insufficient_transcript"}
                    or verdict in {"span_too_short", "span_implausible"}
                ),
            }
        )
        results.append(record)
        print(json.dumps({"event": "row", "index": index, **record}, ensure_ascii=False), flush=True)

    fieldnames = list(dict.fromkeys(key for record in results for key in record))
    with (out_dir / "alignment_langlocked.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    counts: dict[str, int] = {}
    languages: dict[str, int] = {}
    for record in results:
        counts[str(record.get("alignment_status"))] = counts.get(str(record.get("alignment_status")), 0) + 1
        languages[str(record.get("chosen_language"))] = languages.get(str(record.get("chosen_language")), 0) + 1
    summary = {
        "rows": len(results),
        "accepted": sum(int(r.get("accept") or 0) for r in results),
        "needs_gemini": sum(int(r.get("needs_gemini") or 0) for r in results),
        "alignment_status_counts": counts,
        "chosen_language_counts": languages,
        "span_verdict_counts": {
            v: sum(1 for r in results if r.get("span_verdict") == v)
            for v in {str(r.get("span_verdict")) for r in results}
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
