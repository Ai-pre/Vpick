"""Emit the final gold set as two files: a label/metadata table and a subtitle table.

The label table always carries the long-form URL, the short URL, and the span
timestamps, so a reader can locate any candidate in the source video without
touching the subtitle table. The subtitle table carries only what a judge reads.

Transcript rendering is normalized to a single shape across the whole set:
`S{speaker}: utterance` lines, no per-line timestamps. The vpick-sourced rows
never had per-line times (only a span header) while the yt-dlp rows always did,
and that difference alone identified which pipeline produced a row — which
correlated with the label. Span timing lives in the label table instead, so
nothing is lost by dropping it here.

Evidence priority follows vpick_scene_api > yt_dlp_transcript_fallback >
gemini, and the source actually used is recorded per row.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

LABEL_FIELDS = (
    "candidate_id",
    "pair_id",
    "channel_name",
    "longform_id",
    "long_video_url",
    "short_video_id",
    "short_video_url",
    "start_sec",
    "end_sec",
    "duration_sec",
    "start_time",
    "end_time",
    "performance_label_PRIVATE",
    "channel_performance_percentile_PRIVATE",
    "percentile_bucket",
    "label_confidence",
    "mapping_confidence",
    "timestamp_method",
    "timestamp_confidence",
    "transcript_source",
    "dataset_role_v2",
    "split_lock_version",
)

SUBTITLE_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)

SPEAKER_LINE = re.compile(r"^(?:\[[^\]]*\]\s*)?(S[0-9?]+)\s*:\s*(.+)$")
BARE_BRACKET_LINE = re.compile(r"^\[[^\]]*\]\s*(.*)$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clock(seconds: str | float) -> str:
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return ""
    return f"{total // 60:02d}:{total % 60:02d}"


def normalize_block(text: str) -> str:
    """Render any of the historical transcript shapes as `S{n}: utterance` lines."""
    out: list[str] = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        match = SPEAKER_LINE.match(line)
        if match:
            speaker, utterance = match.group(1), match.group(2).strip()
            if utterance:
                out.append(f"{speaker}: {utterance}")
            continue
        bare = BARE_BRACKET_LINE.match(line)
        if bare:
            # A span header such as "[원본 구간 13:27-15:03]" carries no utterance.
            remainder = bare.group(1).strip()
            if remainder:
                out.append(f"S?: {remainder}")
            continue
        out.append(f"S?: {line}")
    return "\n".join(out)


def strip_speakers(text: str) -> str:
    """Utterances as running text: drops speaker tokens and line granularity."""
    parts: list[str] = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^S[0-9?]+:\s*(.+)$", line)
        parts.append(match.group(1).strip() if match else line)
    return " ".join(parts)


def percentile_bucket(value: str) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    for hi, name in ((20, "p0_20"), (40, "p20_40"), (60, "p40_60"), (80, "p60_80")):
        if x < hi:
            return name
    return "p80_100"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split the final gold set into a label table and a subtitle table."
    )
    parser.add_argument("--master", action="append", required=True)
    parser.add_argument("--descriptions", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for path in args.master:
        rows += read_csv(Path(path))

    overrides: dict[str, str] = {}
    if args.descriptions and Path(args.descriptions).exists():
        overrides = json.loads(Path(args.descriptions).read_text(encoding="utf-8"))

    labels: list[dict[str, Any]] = []
    subtitles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        provider = row.get("evidence_provider", "")
        source = (
            "vpick_scene_api"
            if provider == "vpick_scene_api"
            else "gemini"
            if "gemini" in (row.get("final_transcript_source", "") or "")
            and provider != "yt_dlp_transcript_fallback"
            else "yt_dlp_transcript_fallback"
        )
        labels.append(
            {
                "candidate_id": candidate_id,
                "pair_id": row.get("pair_id", ""),
                "channel_name": row.get("channel_name", ""),
                "longform_id": row.get("longform_id", ""),
                "long_video_url": row.get("long_video_url")
                or f"https://www.youtube.com/watch?v={row.get('longform_id','')}",
                "short_video_id": row.get("short_video_id", ""),
                "short_video_url": row.get("short_video_url")
                or f"https://www.youtube.com/shorts/{row.get('short_video_id','')}",
                "start_sec": row.get("start_sec", ""),
                "end_sec": row.get("end_sec", ""),
                "duration_sec": row.get("duration_sec", ""),
                "start_time": clock(row.get("start_sec", "")),
                "end_time": clock(row.get("end_sec", "")),
                "performance_label_PRIVATE": row.get("performance_label_PRIVATE", ""),
                "channel_performance_percentile_PRIVATE": row.get(
                    "channel_performance_percentile_PRIVATE", ""
                ),
                "percentile_bucket": percentile_bucket(
                    row.get("channel_performance_percentile_PRIVATE", "")
                ),
                "label_confidence": row.get("label_confidence", ""),
                "mapping_confidence": row.get("mapping_confidence", ""),
                "timestamp_method": row.get("timestamp_method", ""),
                "timestamp_confidence": row.get("timestamp_confidence", ""),
                "transcript_source": source,
                "dataset_role_v2": row.get("dataset_role_v2", ""),
                "split_lock_version": row.get("split_lock_version", ""),
            }
        )
        subtitles.append(
            {
                "candidate_id": candidate_id,
                "duration_sec": row.get("duration_sec", ""),
                "description": overrides.get(candidate_id, row.get("description", "")),
                "transcript": normalize_block(row.get("transcript", "")),
                "before_context": normalize_block(row.get("before_context", "")),
                "after_context": normalize_block(row.get("after_context", "")),
            }
        )

    label_path = out_dir / "vpick_goldlabel_final_PRIVATE.csv"
    with label_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LABEL_FIELDS))
        writer.writeheader()
        writer.writerows(labels)

    subtitle_path = out_dir / "vpick_short_subtitles_final.csv"
    with subtitle_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUBTITLE_FIELDS))
        writer.writeheader()
        writer.writerows(subtitles)

    # Judge-facing variant. Speaker labels survive only on vpick rows, and their
    # presence identifies the producing pipeline (which correlates with the
    # label), so the judge copy carries utterances as running text instead.
    plain_rows = [
        {
            **row,
            **{
                field: strip_speakers(row[field])
                for field in ("transcript", "before_context", "after_context")
            },
        }
        for row in subtitles
    ]
    plain_path = out_dir / "vpick_short_subtitles_final_plain.csv"
    with plain_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUBTITLE_FIELDS))
        writer.writeheader()
        writer.writerows(plain_rows)

    def counts(key: str, source: list[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in source:
            out[str(item.get(key))] = out.get(str(item.get(key)), 0) + 1
        return dict(sorted(out.items()))

    missing_span = [
        r["candidate_id"] for r in labels if not (r["start_sec"] and r["end_sec"])
    ]
    missing_url = [
        r["candidate_id"]
        for r in labels
        if not (r["long_video_url"] and r["short_video_url"])
    ]
    summary = {
        "rows": len(labels),
        "label_fields": len(LABEL_FIELDS),
        "subtitle_fields": len(SUBTITLE_FIELDS),
        "missing_span": missing_span,
        "missing_url": missing_url,
        "label_counts": counts("performance_label_PRIVATE", labels),
        "bucket_counts": counts("percentile_bucket", labels),
        "transcript_source_counts": counts("transcript_source", labels),
        "channel_counts": counts("channel_name", labels),
        "empty_description": [s["candidate_id"] for s in subtitles if not s["description"].strip()],
        "empty_transcript": [s["candidate_id"] for s in subtitles if not s["transcript"].strip()],
        "label_path": str(label_path),
        "subtitle_path": str(subtitle_path),
        "plain_subtitle_path": str(plain_path),
    }
    (out_dir / "final_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
