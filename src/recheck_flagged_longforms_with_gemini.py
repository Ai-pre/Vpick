from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai

from audit_all_shorts_with_gemini import ngram_overlap
from repair_transcripts_with_gemini import (
    build_prompt,
    meaningful_text,
    normalized_similarity,
    request_transcript,
    transcript_from_payload,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def agreement(left: str, right: str) -> float:
    return max(
        normalized_similarity(left, right),
        ngram_overlap(left, right),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-audit", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument(
        "--candidate-ids",
        nargs="*",
        help="Also recheck these candidate IDs even if the automatic audit did not flag them.",
    )
    parser.add_argument(
        "--retry-nonreplace",
        action="store_true",
        help="Retry existing rows whose decision is not replace_with_gemini_long.",
    )
    args = parser.parse_args()

    audit_rows = read_csv(Path(args.short_audit))
    requested_ids = set(args.candidate_ids or [])
    flagged = {
        row["candidate_id"]: row
        for row in audit_rows
        if row["needs_longform_recheck"].strip().lower() == "true"
        or row["candidate_id"] in requested_ids
    }
    unknown_requested = requested_ids - {row["candidate_id"] for row in audit_rows}
    if unknown_requested:
        raise ValueError(
            f"Requested IDs missing from short audit: {sorted(unknown_requested)}"
        )
    candidates = {
        row["candidate_id"]: row
        for row in read_csv(Path(args.candidates))
    }
    manifest = {
        row["candidate_id"]: row
        for row in read_csv(Path(args.manifest))
    }
    missing = set(flagged) - set(candidates) | set(flagged) - set(manifest)
    if missing:
        raise ValueError(f"Flagged IDs missing from inputs: {sorted(missing)}")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    client = genai.Client(api_key=api_key)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "longform_recheck_results.csv"
    existing_rows = read_csv(result_path) if result_path.exists() else []
    existing = {
        row["candidate_id"]: row
        for row in existing_rows
        if not (
            args.retry_nonreplace
            and row.get("decision") != "replace_with_gemini_long"
        )
    }
    results = list(existing.values())

    for candidate_id, short_row in flagged.items():
        if candidate_id in existing:
            continue
        candidate = candidates[candidate_id]
        metadata = manifest[candidate_id]
        try:
            payload, raw_text = request_transcript(
                client,
                model=args.model,
                video_url=metadata["long_video_url"],
                prompt=build_prompt(
                    media_kind="long_interval",
                    start_sec=float(metadata["start_sec"]),
                    end_sec=float(metadata["end_sec"]),
                ),
                attempts=args.attempts,
            )
            error_text = ""
        except Exception as error:
            payload = {
                "status": "unavailable",
                "confidence": 0.0,
                "segments": [],
                "notes": f"{type(error).__name__}: {error}",
            }
            raw_text = ""
            error_text = payload["notes"]
        long_text = transcript_from_payload(payload, focus_only=True)
        short_text = short_row["gemini_short_transcript"]
        canonical_text = candidate["transcript"]
        canonical_short = agreement(canonical_text, short_text)
        gemini_long_short = agreement(long_text, short_text)
        canonical_long = agreement(canonical_text, long_text)
        decision = "manual_review"
        if payload["status"] == "ok" and float(payload["confidence"]) >= 0.7:
            if (
                gemini_long_short >= 0.3
                and gemini_long_short >= canonical_short + 0.1
            ):
                decision = "replace_with_gemini_long"
            elif canonical_long >= 0.5:
                decision = "retain_canonical_short_edit_difference"
            elif gemini_long_short < 0.3 and canonical_long < 0.3:
                decision = "mapping_or_heavy_edit_review"
        elif (
            payload["status"] == "no_speech"
            and len(meaningful_text(short_text)) < 20
        ):
            decision = "verified_no_speech"

        raw_path = raw_dir / f"{candidate_id}_long_interval.json"
        raw_path.write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "video_url": metadata["long_video_url"],
                    "payload": payload,
                    "raw_text": raw_text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        results.append(
            {
                "candidate_id": candidate_id,
                "pair_id": metadata["pair_id"],
                "short_recheck_reasons": short_row["recheck_reasons"],
                "long_status": payload["status"],
                "long_confidence": payload["confidence"],
                "canonical_short_agreement": round(canonical_short, 4),
                "gemini_long_short_agreement": round(gemini_long_short, 4),
                "canonical_long_agreement": round(canonical_long, 4),
                "decision": decision,
                "canonical_transcript": canonical_text,
                "gemini_long_transcript": long_text,
                "gemini_short_transcript": short_text,
                "notes": payload.get("notes", ""),
                "error": error_text,
                "model": args.model,
            }
        )
        results.sort(key=lambda row: row["candidate_id"])
        write_csv(result_path, results)
        print(
            json.dumps(
                {
                    "event": "longform_recheck_complete",
                    "completed": len(results),
                    "target": len(flagged),
                    "candidate_id": candidate_id,
                    "decision": decision,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(max(0.0, args.sleep_sec))

    decision_counts: dict[str, int] = {}
    for row in results:
        decision_counts[row["decision"]] = (
            decision_counts.get(row["decision"], 0) + 1
        )
    summary = {
        "flagged_count": len(flagged),
        "completed_count": len(results),
        "decision_counts": decision_counts,
        "replace_candidate_ids": [
            row["candidate_id"]
            for row in results
            if row["decision"] == "replace_with_gemini_long"
        ],
        "unresolved_candidate_ids": [
            row["candidate_id"]
            for row in results
            if row["decision"]
            in {"manual_review", "mapping_or_heavy_edit_review"}
        ],
        "model": args.model,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
