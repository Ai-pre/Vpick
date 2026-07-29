from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai

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


def character_ngrams(text: str, size: int = 2) -> set[str]:
    normalized = meaningful_text(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {
        normalized[index : index + size]
        for index in range(len(normalized) - size + 1)
    }


def ngram_overlap(left: str, right: str) -> float:
    left_set = character_ngrams(left)
    right_set = character_ngrams(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def load_cached_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument(
        "--retry-unavailable",
        action="store_true",
        help="Retry rows whose previous Gemini status was not ok.",
    )
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    candidates = read_csv(Path(args.candidates))
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be within the shard range")
    candidates = [
        row
        for index, row in enumerate(candidates)
        if index % args.shard_count == args.shard_index
    ]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    manifest_rows = read_csv(Path(args.manifest))
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Manifest contains duplicate candidate IDs")
    missing = [
        row["candidate_id"]
        for row in candidates
        if row["candidate_id"] not in manifest
    ]
    if missing:
        raise ValueError(f"Candidates missing from manifest: {missing[:10]}")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    client = genai.Client(api_key=api_key)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "short_transcript_audit_60.csv"
    existing_rows = read_csv(score_path) if score_path.exists() else []
    existing = {
        row["candidate_id"]: row
        for row in existing_rows
        if not (
            args.retry_unavailable
            and row.get("gemini_status") != "ok"
        )
    }

    rows = list(existing.values())
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate["candidate_id"]
        if candidate_id in existing:
            continue
        metadata = manifest[candidate_id]
        cache_path = raw_dir / f"{candidate_id}_short_full.json"
        payload = load_cached_payload(cache_path)
        raw_text = ""
        error_text = ""
        if payload is None:
            try:
                payload, raw_text = request_transcript(
                    client,
                    model=args.model,
                    video_url=metadata["short_video_url"],
                    prompt=build_prompt(
                        media_kind="short_full",
                        start_sec=float(metadata["start_sec"]),
                        end_sec=float(metadata["end_sec"]),
                    ),
                    attempts=args.attempts,
                )
                cache_path.write_text(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "video_url": metadata["short_video_url"],
                            "model": args.model,
                            "payload": payload,
                            "raw_text": raw_text,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
                payload = {
                    "status": "unavailable",
                    "confidence": 0.0,
                    "segments": [],
                    "notes": error_text,
                }
        short_text = transcript_from_payload(payload, focus_only=False)
        sequence_similarity = normalized_similarity(
            candidate["transcript"],
            short_text,
        )
        containment_overlap = ngram_overlap(
            candidate["transcript"],
            short_text,
        )
        original_chars = len(meaningful_text(candidate["transcript"]))
        short_chars = len(meaningful_text(short_text))
        reasons = []
        if payload["status"] in {"unavailable", "uncertain"}:
            reasons.append(f"gemini_{payload['status']}")
        if payload["status"] == "ok" and short_chars >= 20:
            if original_chars < 20:
                reasons.append("canonical_missing_speech")
            elif max(sequence_similarity, containment_overlap) < 0.3:
                reasons.append("low_short_long_text_agreement")
        if payload["status"] == "no_speech" and original_chars >= 40:
            reasons.append("canonical_speech_but_short_no_speech")
        rows.append(
            {
                "candidate_id": candidate_id,
                "pair_id": metadata["pair_id"],
                "channel_name": metadata["channel_name"],
                "performance_label_PRIVATE": metadata["performance_label"],
                "short_video_id": metadata["short_video_id"],
                "gemini_status": payload["status"],
                "gemini_confidence": payload["confidence"],
                "canonical_speech_chars": original_chars,
                "gemini_short_speech_chars": short_chars,
                "sequence_similarity": round(sequence_similarity, 4),
                "containment_overlap": round(containment_overlap, 4),
                "needs_longform_recheck": bool(reasons),
                "recheck_reasons": "|".join(reasons),
                "canonical_transcript": candidate["transcript"],
                "gemini_short_transcript": short_text,
                "notes": payload.get("notes", ""),
                "error": error_text,
                "model": args.model,
            }
        )
        rows.sort(key=lambda row: row["candidate_id"])
        write_csv(score_path, rows)
        print(
            json.dumps(
                {
                    "event": "short_audit_complete",
                    "completed": len(rows),
                    "target": len(candidates),
                    "candidate_id": candidate_id,
                    "status": payload["status"],
                    "needs_longform_recheck": bool(reasons),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(max(0.0, args.sleep_sec))

    flagged = [
        row for row in rows if str(row["needs_longform_recheck"]).lower() == "true"
    ]
    summary = {
        "candidate_count": len(candidates),
        "completed_count": len(rows),
        "flagged_count": len(flagged),
        "flagged_candidate_ids": [
            row["candidate_id"] for row in flagged
        ],
        "status_counts": {
            status: sum(row["gemini_status"] == status for row in rows)
            for status in sorted({row["gemini_status"] for row in rows})
        },
        "model": args.model,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
