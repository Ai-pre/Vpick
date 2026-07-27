"""Build the long-form URL fill-in sheet for the 18 origin-linked shorts only.

Every one of the 18 gets a row so the sheet doubles as a status view: 9 already
have a verified span and need nothing, the rest need a URL pasted into
`origin_long_video_url`. `needs_url` says which is which, and rows that were
refuted carry the Gemini transcript that refuted them so the correct source can
be searched for without re-listening to the short.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

FILL_COLUMN = "origin_long_video_url"

REASON_LABELS = {
    "origin_mismatch_suspect": "링크된 롱폼과 자막 불일치(공통 부분문자열 0자) — 링크 오류",
    "gemini_empty_transcript": "숏폼에 발화 없음(음악/노래) — 자막으로 검증 불가",
    "alignment_heavy_edit": "정렬은 되지만 비연속 조립 구간",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_hints(paths: list[Path]) -> dict[str, str]:
    hints: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            payload = json.loads(line)
            text = " ".join(
                str(segment.get("text", "")).strip() for segment in payload.get("segments", [])
            )
            text = " ".join(text.split())
            if text:
                hints[payload["short_video_id"]] = text
    return hints


def origin_duration(value: str) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill-in sheet for the 18 origin-linked mid-percentile shorts."
    )
    parser.add_argument("--linked-csv", required=True)
    parser.add_argument("--alignment-csv", required=True)
    parser.add_argument("--rejected-csv", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--gemini-transcripts", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    linked = read_csv(Path(args.linked_csv))
    alignment = {r["short_video_id"]: r for r in read_csv(Path(args.alignment_csv))}
    rejected = {r["short_video_id"]: r for r in read_csv(Path(args.rejected_csv))}
    manifest = {r["short_video_id"]: r for r in read_csv(Path(args.manifest_csv))}
    hints = load_hints([Path(p) for p in args.gemini_transcripts])

    rows: list[dict[str, Any]] = []
    for source in linked:
        short_id = source["short_video_id"]
        resolved = manifest.get(short_id)
        rejection = rejected.get(short_id)
        align = alignment.get(short_id, {})
        long_duration = origin_duration(align.get("long_duration_sec", ""))

        if resolved:
            needs_url = 0
            status = "resolved"
            reason = ""
        else:
            reason_key = (rejection or {}).get("reject_reason", "unresolved")
            reason = REASON_LABELS.get(reason_key, reason_key)
            # A heavy-edit row whose origin is a real long-form already has the
            # right URL; only the span policy is open, so no URL is needed.
            if reason_key == "alignment_heavy_edit" and long_duration >= 300.0:
                needs_url = 0
                status = "origin_ok_span_policy_pending"
            else:
                needs_url = 1
                status = "needs_url"

        rows.append(
            {
                "needs_url": needs_url,
                "status": status,
                "channel_name": source["channel_name"],
                "short_video_id": short_id,
                "short_url": source.get("short_url", ""),
                "short_title": source.get("title", ""),
                "short_upload_date": source.get("metadata_upload_date", ""),
                "short_duration_sec": source.get("metadata_duration_sec", ""),
                "percentile_bucket": source.get("percentile_bucket", ""),
                "channel_performance_percentile": source.get("channel_performance_percentile", ""),
                "short_views": source.get("short_views", ""),
                "need_reason": reason,
                "previous_origin_video_id": source.get("origin_long_video_id", ""),
                "previous_origin_url": source.get("origin_long_video_url", ""),
                "previous_origin_duration_sec": align.get("long_duration_sec", ""),
                "previous_origin_title": align.get("long_title", ""),
                "resolved_start_time": (resolved or {}).get("start_time", ""),
                "resolved_end_time": (resolved or {}).get("end_time", ""),
                "short_speech_hint": hints.get(short_id, "")[:220],
                FILL_COLUMN: "",
            }
        )

    rows.sort(key=lambda r: (-r["needs_url"], r["channel_name"], r["short_video_id"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "needs_url": sum(r["needs_url"] for r in rows),
        "status_counts": {
            status: sum(1 for r in rows if r["status"] == status)
            for status in sorted({r["status"] for r in rows})
        },
        "reason_counts": {
            reason: sum(1 for r in rows if r["need_reason"] == reason)
            for reason in sorted({r["need_reason"] for r in rows if r["need_reason"]})
        },
        "with_speech_hint": sum(1 for r in rows if r["needs_url"] and r["short_speech_hint"]),
        "output": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
