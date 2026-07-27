"""Whole-dataset sanity audit for a gold label + subtitle pair.

Checks the failure modes that have actually occurred in this project rather than a
generic schema check:

- span sanity: a span must be plausible against the short it claims to represent,
  and must fit inside the long-form.
- rolling-caption duplication: VTT auto-captions repeat each line across
  consecutive cues, so a parser that does not dedupe yields transcripts where the
  same phrase appears three times. That silently destroys alignment scores.
- caption truncation: a long-form transcript built from a handful of cues cannot
  cover a real span; cue counts far below the span length are a parser failure.
- join integrity, duplicate ids, empty evidence, label/percentile agreement.

Every check reports the offending rows, not just a count, so a failure is
actionable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def utterances(transcript: str) -> list[str]:
    out = []
    for line in (transcript or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\[[^\]]*\]\s*", "", line)
        line = re.sub(r"^S[0-9?]+:\s*", "", line)
        if line:
            out.append(line)
    return out


def duplication_ratio(transcript: str) -> float:
    """Share of utterances that repeat an earlier one.

    Rolling VTT captions produce near-total duplication; ordinary speech repeats
    some short interjections, so a high value is the signal, not any value.
    """
    lines = [l for l in utterances(transcript) if len(l) >= 6]
    if len(lines) < 3:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return round(repeated / len(lines), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a gold label + subtitle dataset.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--subtitles", action="append", required=True)
    parser.add_argument("--short-durations", default="",
                        help="Optional CSV of short_video_id,duration_sec for ratio checks.")
    parser.add_argument("--min-span-sec", type=float, default=10.0)
    parser.add_argument("--max-span-sec", type=float, default=120.0)
    parser.add_argument("--min-ratio", type=float, default=0.5)
    parser.add_argument("--max-ratio", type=float, default=1.8)
    parser.add_argument("--max-duplication", type=float, default=0.35)
    parser.add_argument("--min-chars-per-sec", type=float, default=1.5,
                        help="Speech density floor. Characters, not lines: the "
                             "judge-facing variant joins utterances into one line, "
                             "so a line-count floor would flag every row.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = read_csv(Path(args.labels))
    subtitle_sets = {Path(p).name: {r["candidate_id"]: r for r in read_csv(Path(p))}
                     for p in args.subtitles}
    durations: dict[str, float] = {}
    if args.short_durations and Path(args.short_durations).exists():
        for row in read_csv(Path(args.short_durations)):
            value = number(row.get("duration_sec"))
            if value:
                durations[row["short_video_id"]] = value

    findings: dict[str, list[dict[str, Any]]] = {
        "duplicate_candidate_id": [],
        "duplicate_short_video_id": [],
        "missing_url_or_span": [],
        "span_out_of_range": [],
        "span_ratio_out_of_range": [],
        "span_exceeds_longform": [],
        "label_percentile_disagreement": [],
        "subtitle_row_missing": [],
        "empty_evidence": [],
        "rolling_caption_duplication": [],
        "sparse_transcript": [],
    }

    seen_candidates: Counter = Counter(r["candidate_id"] for r in labels)
    seen_shorts: Counter = Counter(r["short_video_id"] for r in labels)
    for cid, count in seen_candidates.items():
        if count > 1:
            findings["duplicate_candidate_id"].append({"candidate_id": cid, "count": count})
    for sid, count in seen_shorts.items():
        if count > 1:
            findings["duplicate_short_video_id"].append({"short_video_id": sid, "count": count})

    for row in labels:
        cid = row["candidate_id"]
        start, end = number(row.get("start_sec")), number(row.get("end_sec"))
        span = number(row.get("duration_sec"))
        if span is None and start is not None and end is not None:
            span = end - start
        if not (row.get("long_video_url") and row.get("short_video_url")) or start is None or end is None:
            findings["missing_url_or_span"].append({"candidate_id": cid})
        if span is not None and not (args.min_span_sec <= span <= args.max_span_sec):
            findings["span_out_of_range"].append(
                {"candidate_id": cid, "short_video_id": row["short_video_id"], "span_sec": span}
            )

        actual = durations.get(row["short_video_id"])
        if actual and span:
            ratio = span / actual
            if not (args.min_ratio <= ratio <= args.max_ratio):
                findings["span_ratio_out_of_range"].append(
                    {"candidate_id": cid, "short_video_id": row["short_video_id"],
                     "span_sec": round(span, 2), "short_duration_sec": actual,
                     "ratio": round(ratio, 3)}
                )

        long_duration = number(row.get("long_duration_sec"))
        if long_duration and end and end > long_duration + 1:
            findings["span_exceeds_longform"].append(
                {"candidate_id": cid, "end_sec": end, "long_duration_sec": long_duration}
            )

        label = row.get("performance_label_PRIVATE", "")
        percentile = number(row.get("channel_performance_percentile_PRIVATE"))
        if percentile is not None:
            expected = "pos" if percentile >= 80 else "neg" if percentile <= 20 else "mid"
            if label and label != expected:
                findings["label_percentile_disagreement"].append(
                    {"candidate_id": cid, "label": label,
                     "percentile": percentile, "expected": expected}
                )

        for name, table in subtitle_sets.items():
            entry = table.get(cid)
            if entry is None:
                findings["subtitle_row_missing"].append({"candidate_id": cid, "file": name})
                continue
            if not entry.get("transcript", "").strip() or not entry.get("description", "").strip():
                findings["empty_evidence"].append({"candidate_id": cid, "file": name})
            dup = duplication_ratio(entry.get("transcript", ""))
            if dup > args.max_duplication:
                findings["rolling_caption_duplication"].append(
                    {"candidate_id": cid, "file": name, "duplication_ratio": dup}
                )
            body = " ".join(utterances(entry.get("transcript", "")))
            chars = len(re.sub(r"\s+", "", body))
            if span and span > 0:
                density = chars / span
                if density < args.min_chars_per_sec:
                    findings["sparse_transcript"].append(
                        {"candidate_id": cid, "file": name, "chars": chars,
                         "span_sec": round(span, 2), "chars_per_sec": round(density, 2)}
                    )

    summary = {
        "label_rows": len(labels),
        "subtitle_files": {name: len(table) for name, table in subtitle_sets.items()},
        "short_durations_supplied": len(durations),
        "thresholds": {
            "span_sec": [args.min_span_sec, args.max_span_sec],
            "span_ratio": [args.min_ratio, args.max_ratio],
            "max_duplication": args.max_duplication,
            "min_chars_per_sec": args.min_chars_per_sec,
        },
        "finding_counts": {k: len(v) for k, v in findings.items()},
        "findings": findings,
    }
    summary["audit_passed"] = all(not v for v in findings.values())
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"라벨 {len(labels)}행 | 자막파일 {list(summary['subtitle_files'].items())}")
    print(f"숏폼 실측 길이 제공: {len(durations)}건")
    print()
    for name, items in findings.items():
        mark = "OK  " if not items else "FAIL"
        print(f"  [{mark}] {name:34} {len(items)}")
        for item in items[:5]:
            print(f"           {json.dumps(item, ensure_ascii=False)}")
        if len(items) > 5:
            print(f"           ... 외 {len(items)-5}건")
    print()
    print("전체 통과" if summary["audit_passed"] else "실패 항목 있음")
    print("->", args.out)
    return 0 if summary["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
