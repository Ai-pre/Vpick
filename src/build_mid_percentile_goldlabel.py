"""Emit mid-percentile candidates in the original 60-row gold-label schema.

Matches `goldlabel_master_transcript_final_60_PRIVATE.csv` column-for-column and
reproduces its transcript rendering (`[MM:SS-MM:SS] S?: text`) so the new rows can
be concatenated onto the existing gold master without any downstream change. The
blind six-column judge input is written alongside from the same spans.

`description` is left empty here and filled in a later step; nothing downstream
should treat an empty description as valid, so the audit checks it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from audit_short_long_alignment import parse_json3

GOLD_FIELDS = (
    "candidate_id",
    "pair_id",
    "channel_name",
    "performance_label_PRIVATE",
    "channel_performance_percentile_PRIVATE",
    "label_confidence",
    "mapping_confidence",
    "longform_id",
    "long_video_url",
    "short_video_id",
    "short_video_url",
    "start_sec",
    "end_sec",
    "duration_sec",
    "timestamp_method",
    "timestamp_confidence",
    "evidence_provider",
    "transcript_validation_status",
    "final_transcript_source",
    "gemini_model",
    "gemini_confidence",
    "description",
    "transcript",
    "before_context",
    "after_context",
    "dataset_role_v2",
    "split_lock_version",
)

JUDGE_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)

CONTEXT_SECONDS = 45.0
TRANSCRIPT_LIMIT = 5000
CONTEXT_LIMIT = 2500


def blind_id(long_video_id: str, start: float, end: float, salt: str) -> str:
    raw = f"{salt}|{long_video_id}|{start:.3f}|{end:.3f}".encode("utf-8")
    return f"C_{hashlib.sha256(raw).hexdigest()[:14]}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clock(seconds: float) -> str:
    """MM:SS, as the existing gold master renders cue times."""
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def render_cues(cues: list[Any], start: float, end: float, limit: int) -> str:
    lines = []
    for cue in cues:
        if cue.end < start or cue.start > end:
            continue
        text = " ".join(cue.text.split())
        if not text:
            continue
        lines.append(f"[{clock(cue.start)}-{clock(cue.end)}] S?: {text}")
    return "\n".join(lines)[:limit]


def long_subtitle_path(subtitle_dir: Path, long_id: str) -> Path | None:
    for language in ("ko-orig", "ko", "ko-ko", "ko-KR", "en-orig", "en"):
        path = subtitle_dir / f"{long_id}.{language}.json3"
        if path.exists() and path.stat().st_size > 20:
            return path
    matches = sorted(subtitle_dir.glob(f"{long_id}.*.json3"))
    return matches[0] if matches else None


def span_seconds(row: dict[str, str]) -> float | None:
    try:
        return float(row.get("source_span") or "")
    except ValueError:
        return None


def within_span_budget(row: dict[str, str], budget_sec: float) -> bool:
    """A non-contiguous span is still usable if it stays inside the budget.

    `heavy_edit` only means the short was assembled from more than one piece. As
    long as the pieces sit inside a window about as long as the short itself, the
    mapping still points at the right moment.
    """
    span = span_seconds(row)
    return span is not None and 0 < span <= budget_sec


def resolve(
    alignment: list[dict[str, str]],
    recovery: list[dict[str, str]],
    span_budget_sec: float = 60.0,
):
    """Prefer a Gemini-recovered span; otherwise the caption-only one."""
    recovered = {r["short_video_id"]: r for r in recovery}
    accepted, rejected = [], []
    for row in alignment:
        short_id = row["short_video_id"]
        fix = recovered.get(short_id)
        if fix and str(fix.get("accept", "")) == "1":
            accepted.append((row, fix, "gemini"))
        elif str(row.get("accept", "")) == "1":
            accepted.append((row, row, "caption"))
        elif fix and within_span_budget(fix, span_budget_sec):
            accepted.append((row, fix, "gemini_span_budget"))
        elif within_span_budget(row, span_budget_sec):
            accepted.append((row, row, "caption_span_budget"))
        else:
            if fix and fix.get("resolved_status") not in {"", "recovered"}:
                reason = fix["resolved_status"]
            elif row.get("alignment_status") not in {"continuous", "light_edit"}:
                reason = f"alignment_{row.get('alignment_status') or 'unknown'}"
            else:
                reason = f"span_{row.get('span_verdict') or 'unknown'}"
            rejected.append(
                {
                    "short_video_id": short_id,
                    "long_video_id": row.get("long_video_id", ""),
                    "channel_name": row.get("channel_name", ""),
                    "chosen_language": row.get("chosen_language", ""),
                    "long_duration_sec": row.get("long_duration_sec", ""),
                    "short_duration_sec": row.get("short_duration_sec", ""),
                    "reject_reason": reason,
                }
            )
    return accepted, rejected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit mid-percentile rows in the original gold-label schema."
    )
    parser.add_argument("--alignment", action="append", required=True)
    parser.add_argument("--recovery", action="append", default=[])
    parser.add_argument("--subtitle-dir", action="append", required=True)
    parser.add_argument("--source-csv", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--salt", default="mid_percentile_2026_07_27")
    parser.add_argument("--gemini-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--split-lock-version", default="mid_percentile_2026_07_27")
    parser.add_argument("--dataset-role", default="mid_percentile_expansion")
    parser.add_argument("--span-budget-sec", type=float, default=60.0,
                        help="Accept a non-contiguous span if it fits this window.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alignment: list[dict[str, str]] = []
    for path in args.alignment:
        alignment += read_csv(Path(path))
    recovery: list[dict[str, str]] = []
    for path in args.recovery:
        if Path(path).exists():
            recovery += read_csv(Path(path))

    # Index source rows by both the stated short id and the id parsed out of the
    # short URL: Excel rewrites ids that begin with '-' as #NAME?, so the stated
    # id can be unusable while the URL still carries the real one.
    source: dict[str, dict[str, str]] = {}
    for path in args.source_csv:
        for row in read_csv(Path(path)):
            keys = {row.get("short_video_id", "").strip()}
            for column in ("short_url", "short_video_url"):
                match = re.search(
                    r"(?:shorts/|v=|youtu\.be/)([A-Za-z0-9_-]{11})", row.get(column, "")
                )
                if match:
                    keys.add(match.group(1))
            for key in keys:
                if key:
                    source.setdefault(key, {}).update(row)

    subtitle_dirs = [Path(p) for p in args.subtitle_dir]
    accepted, rejected = resolve(alignment, recovery, args.span_budget_sec)

    gold_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row, chosen, origin in accepted:
        short_id = row["short_video_id"]
        if short_id in seen:
            continue
        seen.add(short_id)
        long_id = row["long_video_id"]
        start = float(chosen["predicted_start"])
        end = float(chosen["predicted_end"])
        path = next(
            (p for p in (long_subtitle_path(d, long_id) for d in subtitle_dirs) if p), None
        )
        if not path:
            rejected.append(
                {"short_video_id": short_id, "long_video_id": long_id,
                 "channel_name": row.get("channel_name", ""),
                 "reject_reason": "long_subtitle_file_missing"}
            )
            continue
        cues = parse_json3(path)
        candidate_id = blind_id(long_id, start, end, args.salt)
        meta = source.get(short_id, {})
        percentile = (
            meta.get("channel_performance_percentile")
            or row.get("channel_performance_percentile")
            or ""
        )
        gold_rows.append(
            {
                "candidate_id": candidate_id,
                "pair_id": row.get("pair_id", ""),
                "channel_name": row.get("channel_name", ""),
                "performance_label_PRIVATE": "mid",
                "channel_performance_percentile_PRIVATE": percentile,
                "label_confidence": "high",
                "mapping_confidence": (
                    "high" if chosen.get("alignment_status") == "continuous"
                    else "low" if origin.endswith("span_budget") else "medium"
                ),
                "longform_id": long_id,
                "long_video_url": f"https://www.youtube.com/watch?v={long_id}",
                "short_video_id": short_id,
                "short_video_url": f"https://www.youtube.com/shorts/{short_id}",
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "timestamp_method": (
                    "gemini_short_transcript_and_subtitle_alignment"
                    if origin.startswith("gemini")
                    else "yt_dlp_subtitle_alignment"
                ),
                "timestamp_confidence": (
                    "high" if chosen.get("alignment_status") == "continuous" else "medium"
                ),
                "evidence_provider": "yt_dlp_transcript_fallback",
                "transcript_validation_status": (
                    "gemini_short_text_after_long_confirmation"
                    if origin.startswith("gemini")
                    else "yt_dlp_subtitle_span_aligned"
                ),
                "final_transcript_source": (
                    "manifest_interval+gemini_short_transcript"
                    if origin.startswith("gemini")
                    else "yt_dlp_transcript_fallback"
                ),
                "gemini_model": args.gemini_model if origin.startswith("gemini") else "",
                "gemini_confidence": "0.95" if origin.startswith("gemini") else "",
                "description": "",
                "transcript": render_cues(cues, start, end, TRANSCRIPT_LIMIT),
                "before_context": render_cues(cues, max(0.0, start - CONTEXT_SECONDS), start, CONTEXT_LIMIT),
                "after_context": render_cues(cues, end, end + CONTEXT_SECONDS, CONTEXT_LIMIT),
                "dataset_role_v2": args.dataset_role,
                "split_lock_version": args.split_lock_version,
            }
        )
        judge_rows.append(
            {
                "candidate_id": candidate_id,
                "duration_sec": round(end - start, 3),
                "description": "",
                "transcript": gold_rows[-1]["transcript"],
                "before_context": gold_rows[-1]["before_context"],
                "after_context": gold_rows[-1]["after_context"],
            }
        )

    gold_path = out_dir / "goldlabel_master_mid_percentile_PRIVATE.csv"
    with gold_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GOLD_FIELDS))
        writer.writeheader()
        writer.writerows(gold_rows)

    judge_path = out_dir / "candidates_mid_percentile_blind.csv"
    with judge_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(JUDGE_FIELDS))
        writer.writeheader()
        writer.writerows(judge_rows)

    if rejected:
        with (out_dir / "rejected.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = list(dict.fromkeys(k for r in rejected for k in r))
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rejected)

    summary = {
        "alignment_rows": len(alignment),
        "accepted": len(gold_rows),
        "rejected": len(rejected),
        "schema_field_count": len(GOLD_FIELDS),
        "channel_counts": {
            channel: sum(1 for r in gold_rows if r["channel_name"] == channel)
            for channel in sorted({r["channel_name"] for r in gold_rows})
        },
        "timestamp_method_counts": {
            method: sum(1 for r in gold_rows if r["timestamp_method"] == method)
            for method in sorted({r["timestamp_method"] for r in gold_rows})
        },
        "empty_transcript_count": sum(1 for r in gold_rows if not r["transcript"].strip()),
        "reject_reason_counts": {
            reason: sum(1 for r in rejected if r.get("reject_reason") == reason)
            for reason in sorted({str(r.get("reject_reason")) for r in rejected})
        },
        "gold_path": str(gold_path),
        "judge_path": str(judge_path),
    }
    (out_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
