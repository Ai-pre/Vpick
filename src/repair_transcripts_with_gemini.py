from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from google import genai


TIMESTAMP_RE = re.compile(
    r"\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]"
)
MARKER_RE = re.compile(
    r"\[(?:음악|박수|웃음|효과음|music|applause|laughter)[^\]]*\]",
    re.IGNORECASE,
)
ALLOWED_STATUS = {"ok", "no_speech", "uncertain", "unavailable"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def meaningful_text(text: str) -> str:
    without_timestamps = TIMESTAMP_RE.sub(" ", text)
    without_markers = MARKER_RE.sub(" ", without_timestamps)
    return "".join(re.findall(r"[가-힣A-Za-z0-9]+", without_markers))


def transcript_span_sec(text: str) -> float:
    intervals = []
    for match in TIMESTAMP_RE.finditer(text):
        start = int(match.group(1)) * 60 + float(match.group(2))
        end = int(match.group(3)) * 60 + float(match.group(4))
        intervals.append((start, end))
    if not intervals:
        return 0.0
    return max(end for _, end in intervals) - min(start for start, _ in intervals)


def audit_row(
    candidate: dict[str, str],
    metadata: dict[str, str],
) -> dict[str, Any]:
    transcript = candidate["transcript"]
    duration = max(float(candidate["duration_sec"]), 1.0)
    speech_chars = len(meaningful_text(transcript))
    span = transcript_span_sec(transcript)
    coverage = min(1.0, span / duration)
    chars_per_sec = speech_chars / duration
    reasons = []
    if speech_chars < 40:
        reasons.append("very_low_speech_text")
    if coverage < 0.35:
        reasons.append("low_timestamp_coverage")
    if chars_per_sec < 0.75:
        reasons.append("low_text_density")
    return {
        "candidate_id": candidate["candidate_id"],
        "pair_id": metadata["pair_id"],
        "channel_name": metadata["channel_name"],
        "duration_sec": round(duration, 3),
        "speech_chars": speech_chars,
        "chars_per_sec": round(chars_per_sec, 4),
        "timestamp_span_sec": round(span, 3),
        "timestamp_coverage": round(coverage, 4),
        "repair_flag": bool(reasons),
        "repair_reasons": "|".join(reasons),
    }


def seconds_label(value: float) -> str:
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


def build_prompt(
    *,
    media_kind: str,
    start_sec: float,
    end_sec: float,
) -> str:
    if media_kind == "long_interval":
        scope = f"""Focus interval in the source video:
- start: {seconds_label(start_sec)} ({start_sec:.3f} seconds)
- end: {seconds_label(end_sec)} ({end_sec:.3f} seconds)

Listen from 5 seconds before the start through 5 seconds after the end, but
mark only speech overlapping the focus interval as in_focus=true. Timestamps
must be absolute positions in the source video."""
    else:
        scope = """Transcribe the entire short video. Timestamps must be
relative to the beginning of the short video and every speech segment must
have in_focus=true."""
    return f"""You are producing evidence for a Korean video dataset.
{scope}

Return a faithful audio transcription, not a summary.

Rules:
1. Transcribe what is actually spoken. Do not infer dialogue from the title,
   description, comments, channel identity, or visible captions.
2. Do not translate. Korean speech stays Korean; English speech stays English.
3. Keep fillers and repeated words when audible.
4. Assign neutral speaker IDs S1, S2, and so on. Do not guess names.
5. Record laughter, applause, and music as non-speech events, not dialogue.
6. If speech cannot be heard, return no_speech or uncertain instead of
   inventing text.
7. Use numeric seconds for all timestamps.

Return exactly one JSON object:
{{
  "status": "ok|no_speech|uncertain|unavailable",
  "detected_languages": ["ko"],
  "segments": [
    {{
      "start_sec": 0.0,
      "end_sec": 1.0,
      "speaker": "S1",
      "text": "verbatim speech",
      "in_focus": true
    }}
  ],
  "non_speech_events": [
    {{"start_sec": 0.0, "end_sec": 1.0, "label": "웃음"}}
  ],
  "confidence": 0.0,
  "notes": "short factual note"
}}"""


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    decoder = json.JSONDecoder()
    for candidate in fenced + [text]:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("No JSON object found in Gemini response")


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "")).strip()
    if status not in ALLOWED_STATUS:
        raise ValueError(f"Unexpected Gemini status: {status!r}")
    confidence = float(payload.get("confidence", 0))
    if not 0 <= confidence <= 1:
        raise ValueError("Gemini confidence must be between 0 and 1")
    segments = payload.get("segments") or []
    if not isinstance(segments, list):
        raise ValueError("Gemini segments must be a list")
    cleaned_segments = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment["start_sec"])
        end = float(segment["end_sec"])
        if end < start:
            raise ValueError("Gemini segment end precedes start")
        cleaned_segments.append(
            {
                "start_sec": start,
                "end_sec": end,
                "speaker": str(segment.get("speaker") or "S?"),
                "text": text,
                "in_focus": bool(segment.get("in_focus", True)),
            }
        )
    if status == "ok" and not cleaned_segments:
        raise ValueError("Gemini returned status=ok without transcript segments")
    return {
        "status": status,
        "detected_languages": payload.get("detected_languages") or [],
        "segments": cleaned_segments,
        "non_speech_events": payload.get("non_speech_events") or [],
        "confidence": confidence,
        "notes": str(payload.get("notes") or "").strip(),
    }


def request_transcript(
    client: genai.Client,
    *,
    model: str,
    video_url: str,
    prompt: str,
    attempts: int,
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            interaction = client.interactions.create(
                model=model,
                input=[
                    {"type": "video", "uri": video_url},
                    {"type": "text", "text": prompt},
                ],
            )
            raw_text = interaction.output_text
            return validate_payload(extract_json(raw_text)), raw_text
        except Exception as error:  # API and schema failures share retry policy.
            last_error = error
            if attempt < attempts:
                retry_match = re.search(
                    r"retry in\s+([0-9.]+)s",
                    str(error),
                    flags=re.IGNORECASE,
                )
                if retry_match:
                    wait_seconds = float(retry_match.group(1)) + 2.0
                else:
                    wait_seconds = min(30, 2**attempt)
                time.sleep(wait_seconds)
    error_detail = (
        f"{type(last_error).__name__}: {last_error}"
        if last_error is not None
        else "unknown error"
    )
    raise RuntimeError(
        f"Gemini transcription failed after {attempts} attempts; "
        f"last_error={error_detail}"
    ) from last_error


def normalized_similarity(left: str, right: str) -> float:
    left_norm = meaningful_text(left)
    right_norm = meaningful_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def format_timestamp(seconds: float) -> str:
    rounded_seconds = round(max(0.0, seconds), 1)
    minutes = int(rounded_seconds // 60)
    remainder = rounded_seconds - minutes * 60
    if abs(remainder - round(remainder)) < 0.05:
        return f"{minutes}:{int(round(remainder)):02d}"
    return f"{minutes}:{remainder:04.1f}"


def transcript_from_payload(
    payload: dict[str, Any],
    *,
    focus_only: bool,
) -> str:
    lines = []
    for segment in payload["segments"]:
        if focus_only and not segment["in_focus"]:
            continue
        lines.append(
            f"[{format_timestamp(segment['start_sec'])}-"
            f"{format_timestamp(segment['end_sec'])}] "
            f"{segment['speaker']}: {segment['text']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--candidate-ids", nargs="*")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--apply-output")
    args = parser.parse_args()

    candidates = read_csv(Path(args.candidates))
    manifest_rows = read_csv(Path(args.manifest))
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Manifest contains duplicate candidate IDs")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("Candidate file contains duplicate candidate IDs")
    if set(row["candidate_id"] for row in candidates) != set(manifest):
        raise ValueError("Candidate and manifest ID sets must match exactly")

    audit_rows = [
        audit_row(row, manifest[row["candidate_id"]])
        for row in candidates
    ]
    selected_ids = set(args.candidate_ids or [])
    if not selected_ids:
        selected_ids = {
            row["candidate_id"]
            for row in audit_rows
            if row["repair_flag"]
        }
    unknown = selected_ids - set(manifest)
    if unknown:
        raise ValueError(f"Unknown candidate IDs: {sorted(unknown)}")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    client = genai.Client(api_key=api_key)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "transcript_quality_audit_60.csv", audit_rows)

    evidence_rows: list[dict[str, Any]] = []
    replacements: dict[str, str] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        if candidate_id not in selected_ids:
            continue
        metadata = manifest[candidate_id]
        start_sec = float(metadata["start_sec"])
        end_sec = float(metadata["end_sec"])
        payloads: dict[str, dict[str, Any]] = {}
        raw_texts: dict[str, str] = {}
        for media_kind, video_url in (
            ("long_interval", metadata["long_video_url"]),
            ("short_full", metadata["short_video_url"]),
        ):
            payload, raw_text = request_transcript(
                client,
                model=args.model,
                video_url=video_url,
                prompt=build_prompt(
                    media_kind=media_kind,
                    start_sec=start_sec,
                    end_sec=end_sec,
                ),
                attempts=args.attempts,
            )
            payloads[media_kind] = payload
            raw_texts[media_kind] = raw_text
            (raw_dir / f"{candidate_id}_{media_kind}.json").write_text(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "media_kind": media_kind,
                        "video_url": video_url,
                        "model": args.model,
                        "payload": payload,
                        "raw_text": raw_text,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        long_text = transcript_from_payload(
            payloads["long_interval"],
            focus_only=True,
        )
        short_text = transcript_from_payload(
            payloads["short_full"],
            focus_only=False,
        )
        agreement = normalized_similarity(long_text, short_text)
        long_ok = (
            payloads["long_interval"]["status"] == "ok"
            and payloads["long_interval"]["confidence"] >= 0.7
            and len(meaningful_text(long_text)) >= 20
        )
        short_ok = (
            payloads["short_full"]["status"] == "ok"
            and payloads["short_full"]["confidence"] >= 0.7
            and len(meaningful_text(short_text)) >= 20
        )
        apply_status = "review"
        if long_ok and short_ok and agreement >= 0.35:
            replacements[candidate_id] = long_text
            apply_status = "auto_replace"
        elif (
            payloads["long_interval"]["status"] == "no_speech"
            and payloads["short_full"]["status"] == "no_speech"
        ):
            apply_status = "verified_no_speech"
        evidence_rows.append(
            {
                "candidate_id": candidate_id,
                "pair_id": metadata["pair_id"],
                "long_status": payloads["long_interval"]["status"],
                "long_confidence": payloads["long_interval"]["confidence"],
                "short_status": payloads["short_full"]["status"],
                "short_confidence": payloads["short_full"]["confidence"],
                "long_short_text_similarity": round(agreement, 4),
                "original_speech_chars": len(
                    meaningful_text(candidate["transcript"])
                ),
                "gemini_long_speech_chars": len(meaningful_text(long_text)),
                "gemini_short_speech_chars": len(meaningful_text(short_text)),
                "apply_status": apply_status,
                "original_transcript": candidate["transcript"],
                "gemini_long_transcript": long_text,
                "gemini_short_transcript": short_text,
                "long_notes": payloads["long_interval"]["notes"],
                "short_notes": payloads["short_full"]["notes"],
                "model": args.model,
            }
        )
        print(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "long_status": payloads["long_interval"]["status"],
                    "short_status": payloads["short_full"]["status"],
                    "similarity": round(agreement, 4),
                    "apply_status": apply_status,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if evidence_rows:
        write_csv(out_dir / "gemini_transcript_repair_evidence.csv", evidence_rows)

    if args.apply_output:
        repaired_rows = []
        for row in candidates:
            updated = dict(row)
            if row["candidate_id"] in replacements:
                updated["transcript"] = replacements[row["candidate_id"]]
            repaired_rows.append(updated)
        write_csv(Path(args.apply_output), repaired_rows)

    summary = {
        "candidate_count": len(candidates),
        "audited_count": len(audit_rows),
        "repair_flag_count": sum(bool(row["repair_flag"]) for row in audit_rows),
        "requested_candidate_ids": sorted(selected_ids),
        "gemini_completed_count": len(evidence_rows),
        "auto_replace_count": len(replacements),
        "auto_replace_candidate_ids": sorted(replacements),
        "model": args.model,
        "apply_output": args.apply_output or "",
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
