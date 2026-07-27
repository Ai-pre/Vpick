"""Build the manual re-mapping sheet: candidates still lacking a usable long-form span.

A row lands here for one of three distinct reasons, and the sheet says which so the
right fix can be applied: the linked origin was disproved by transcript alignment
(needs a different URL), the aligned span is far larger than the short (needs a
narrower URL or a manual span), or the evidence itself is unusable (destroyed ASR
or no speech at all, so the span cannot be confirmed from captions).

Rows already present in the accepted gold master are never emitted, so the sheet
shrinks as mappings land.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

FILL_COLUMNS = ("origin_long_video_url", "manual_start_sec", "manual_end_sec")

REASON_LABELS = {
    "origin_mismatch_suspect": "링크된 롱폼과 자막이 전혀 일치하지 않음 (링크 오류) — 올바른 롱폼 URL 필요",
    "needs_manual_review": "정렬 구간이 숏폼 길이의 2배 이상으로 벌어짐 — 올바른 롱폼 URL 또는 수동 구간 필요",
    "gemini_empty_transcript": "숏폼에 발화가 없음 (음악·노래) — 자막으로 구간 확인 불가, 수동 구간 필요",
    "asr_unusable": "롱폼 자막이 음성인식 오류로 복원 불가 — 올바른 롱폼 URL 필요",
    "origin_not_longform": "origin 영상이 롱폼이 아님 (길이 부족) — 올바른 롱폼 URL 필요",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_hints(paths: list[Path]) -> dict[str, str]:
    """Gemini transcript of the short, used as a search hint for the right source."""
    hints: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            payload = json.loads(line)
            text = " ".join(
                str(seg.get("text", "")).strip() for seg in payload.get("segments", [])
            )
            text = " ".join(text.split())
            if text:
                hints[payload["short_video_id"]] = text
    return hints


def short_id_from(row: dict[str, str]) -> str:
    stated = row.get("short_video_id", "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", stated):
        return stated
    for column in ("short_url", "short_video_url"):
        match = re.search(r"(?:shorts/|v=|youtu\.be/)([A-Za-z0-9_-]{11})", row.get(column, ""))
        if match:
            return match.group(1)
    return stated


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the long-form re-mapping sheet.")
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--accepted-master", action="append", required=True)
    parser.add_argument("--source-csv", action="append", required=True)
    parser.add_argument("--alignment", action="append", default=[])
    parser.add_argument("--gemini-transcripts", action="append", default=[])
    parser.add_argument("--extra-reason", action="append", default=[],
                        help="short_video_id=reason_key, for gates outside the reject file.")
    parser.add_argument("--drop", action="append", default=[],
                        help="short_video_id to omit (e.g. excluded by language).")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rejected = read_csv(Path(args.rejected))
    accepted: set[str] = set()
    for path in args.accepted_master:
        for row in read_csv(Path(path)):
            accepted.add(row.get("short_video_id", "").strip())

    source: dict[str, dict[str, str]] = {}
    for path in args.source_csv:
        for row in read_csv(Path(path)):
            key = short_id_from(row)
            if key:
                source.setdefault(key, {}).update(row)

    alignment: dict[str, dict[str, str]] = {}
    for path in args.alignment:
        if Path(path).exists():
            for row in read_csv(Path(path)):
                alignment.setdefault(row["short_video_id"], row)

    hints = load_hints([Path(p) for p in args.gemini_transcripts])
    extra = dict(item.split("=", 1) for item in args.extra_reason if "=" in item)
    dropped = {v.strip() for v in args.drop if v.strip()}

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    queue = [(r["short_video_id"], r.get("reject_reason", "")) for r in rejected]
    queue += [(short_id, reason) for short_id, reason in extra.items()]

    for short_id, reason_key in queue:
        if short_id in accepted and short_id not in extra:
            continue
        if short_id in dropped or short_id in seen:
            continue
        seen.add(short_id)
        meta = source.get(short_id, {})
        align = alignment.get(short_id, {})
        rows.append(
            {
                "channel_name": meta.get("channel_name", "") or align.get("channel_name", ""),
                "short_video_id": short_id,
                "short_url": meta.get("short_url") or f"https://www.youtube.com/shorts/{short_id}",
                "short_title": meta.get("title", ""),
                "short_upload_date": meta.get("upload_date") or meta.get("metadata_upload_date", ""),
                "short_duration_sec": meta.get("metadata_duration_sec") or meta.get("short_duration_sec", ""),
                "percentile_bucket": meta.get("percentile_bucket", ""),
                "channel_performance_percentile": meta.get("channel_performance_percentile", ""),
                "need_reason": REASON_LABELS.get(reason_key, reason_key),
                "reason_key": reason_key,
                "previous_origin_url": meta.get("origin_long_video_url", ""),
                "previous_origin_duration_sec": align.get("long_duration_sec", ""),
                "previous_origin_title": align.get("long_title", ""),
                "aligned_span_sec": align.get("source_span", ""),
                "short_speech_hint": hints.get(short_id, "")[:220],
                "origin_long_video_url": "",
                "manual_start_sec": "",
                "manual_end_sec": "",
            }
        )

    rows.sort(key=lambda r: (r["reason_key"], r["channel_name"], r["short_video_id"]))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows_to_remap": len(rows),
        "reason_counts": {
            key: sum(1 for r in rows if r["reason_key"] == key)
            for key in sorted({r["reason_key"] for r in rows})
        },
        "channel_counts": {
            ch: sum(1 for r in rows if r["channel_name"] == ch)
            for ch in sorted({r["channel_name"] for r in rows})
        },
        "with_speech_hint": sum(1 for r in rows if r["short_speech_hint"]),
        "fill_columns": list(FILL_COLUMNS),
        "output": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
