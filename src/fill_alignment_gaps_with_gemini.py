"""Recover shorts whose YouTube captions were missing, wrong-language, or unalignable.

The language-locked aligner marks a row `needs_gemini` when the short has no
usable caption track in the long-form's language, or when the aligned span is
implausible against the short's own duration. This stage transcribes the short
directly with Gemini (which reads the YouTube URL), then re-aligns that clean
transcript against the long-form's Korean caption track. A row that still fails
after a real transcript is evidence the linked origin video is wrong, not that
the captions were bad, so it is reported as `origin_mismatch_suspect` instead of
being silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from audit_short_long_alignment import Cue, align_transcripts, display_timestamp, parse_json3, normalize_text

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

TRANSCRIBE_PROMPT = """이 영상의 한국어 음성을 그대로 전사하십시오.

규칙:
- 들리는 발화만 적으십시오. 추측하거나 요약하지 마십시오.
- 화면에만 보이는 자막 텍스트는 발화가 아니면 제외하십시오.
- 발화가 없는 구간은 건너뛰십시오.
- 배경음악 가사는 발화로 취급하지 마십시오.

출력은 JSON 객체 하나만 출력하십시오. 코드블록이나 설명을 붙이지 마십시오.
{"segments": [{"start_sec": 0.0, "end_sec": 3.2, "text": "발화 내용"}]}
"""


def gemini_generate(
    api_key: str, model: str, video_url: str, prompt: str, attempts: int, timeout: int
) -> tuple[dict[str, Any] | None, str]:
    payload = {
        "contents": [
            {
                "parts": [
                    {"file_data": {"file_uri": video_url}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192},
    }
    body = json.dumps(payload).encode("utf-8")
    last_error = ""
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            f"{API_ROOT}/{model}:generateContent",
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            candidate = (data.get("candidates") or [{}])[0]
            text = "".join(
                part.get("text", "") for part in (candidate.get("content") or {}).get("parts", [])
            )
            return data, text
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {error.code} {detail[:200]}"
            retry_after = re.search(r"retry in\s+([0-9.]+)s", detail, flags=re.IGNORECASE)
            wait = float(retry_after.group(1)) + 2.0 if retry_after else min(60, 5 * 2**attempt)
        except Exception as error:  # Network/timeout share the retry policy.
            last_error = f"{type(error).__name__}: {error}"
            wait = min(60, 5 * 2**attempt)
        if attempt < attempts:
            print(
                json.dumps(
                    {"event": "gemini_retry", "attempt": attempt, "wait_sec": round(wait, 1),
                     "error": last_error[:160]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(wait)
    return None, last_error


def extract_segments(text: str) -> list[dict[str, Any]]:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL) or [text]
    decoder = json.JSONDecoder()
    for block in blocks:
        block = block.strip()
        for match in re.finditer(r"\{", block):
            try:
                parsed, _ = decoder.raw_decode(block[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("segments"), list):
                return parsed["segments"]
    return []


def segments_to_cues(segments: list[dict[str, Any]]) -> list[Cue]:
    cues: list[Cue] = []
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if not text or not normalize_text(text):
            continue
        try:
            start = float(segment.get("start_sec", 0.0))
            end = float(segment.get("end_sec", start + 2.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + 2.0
        cues.append(Cue(start=start, end=end, text=text))
    return sorted(cues, key=lambda cue: cue.start)


def span_verdict(span: float, short_duration: float) -> tuple[str, float | str]:
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


def long_subtitle_path(subtitle_dir: Path, long_id: str) -> Path | None:
    for language in ("ko-orig", "ko", "ko-KR", "en-orig", "en"):
        path = subtitle_dir / f"{long_id}.{language}.json3"
        if path.exists() and path.stat().st_size > 20:
            return path
    matches = sorted(subtitle_dir.glob(f"{long_id}.*.json3"))
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe unalignable shorts with Gemini and re-align them."
    )
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--subtitle-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--sleep-sec", type=float, default=4.0)
    parser.add_argument("--short-ids", default="", help="Override which shorts to process.")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set.")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    subtitle_dir = Path(args.subtitle_dir)

    with open(args.alignment, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    override = {v.strip() for v in args.short_ids.split(",") if v.strip()}
    targets = (
        [r for r in rows if r["short_video_id"] in override]
        if override
        else [r for r in rows if str(r.get("needs_gemini", "")) == "1"]
    )
    print(
        json.dumps({"event": "start", "total_rows": len(rows), "targets": len(targets)}, ensure_ascii=False),
        flush=True,
    )

    transcript_path = out_dir / "gemini_short_transcripts.jsonl"
    results: list[dict[str, Any]] = []
    for index, row in enumerate(targets, start=1):
        short_id = row["short_video_id"]
        long_id = row["long_video_id"]
        record: dict[str, Any] = {
            "pair_id": row.get("pair_id", ""),
            "channel_name": row.get("channel_name", ""),
            "short_video_id": short_id,
            "long_video_id": long_id,
            "short_duration_sec": row.get("short_duration_sec", ""),
            "prior_alignment_status": row.get("alignment_status", ""),
            "prior_span_verdict": row.get("span_verdict", ""),
            "gemini_model": args.model,
        }
        data, text = gemini_generate(
            api_key,
            args.model,
            f"https://www.youtube.com/watch?v={short_id}",
            TRANSCRIBE_PROMPT,
            args.attempts,
            args.timeout,
        )
        if data is None:
            record.update({"gemini_status": "failed", "gemini_error": text[:300],
                           "resolved_status": "gemini_failed"})
            results.append(record)
            print(json.dumps({"event": "row", "index": index, **record}, ensure_ascii=False), flush=True)
            continue

        (raw_dir / f"{short_id}.json").write_text(
            json.dumps({"short_video_id": short_id, "response_text": text,
                        "usage": data.get("usageMetadata", {})}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        segments = extract_segments(text)
        short_cues = segments_to_cues(segments)
        record["gemini_status"] = "ok"
        record["gemini_segment_count"] = len(segments)
        record["gemini_cue_count"] = len(short_cues)
        record["gemini_total_tokens"] = data.get("usageMetadata", {}).get("totalTokenCount", "")
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"short_video_id": short_id, "segments": segments}, ensure_ascii=False) + "\n"
            )

        long_path = long_subtitle_path(subtitle_dir, long_id)
        if not short_cues or not long_path:
            record.update(
                {
                    "resolved_status": "long_subtitle_missing" if short_cues else "gemini_empty_transcript",
                    "long_subtitle_file": long_path.name if long_path else "",
                }
            )
            results.append(record)
            print(json.dumps({"event": "row", "index": index, **record}, ensure_ascii=False), flush=True)
            time.sleep(args.sleep_sec)
            continue

        record["long_subtitle_file"] = long_path.name
        long_cues = parse_json3(long_path)
        record["long_cue_count"] = len(long_cues)
        result = align_transcripts(short_cues, long_cues)
        span = float(result.get("source_span") or 0.0)
        try:
            short_duration = float(row.get("short_duration_sec") or 0.0)
        except ValueError:
            short_duration = 0.0
        verdict, ratio = span_verdict(span, short_duration)
        status = result.get("status", "")
        accepted = status in {"continuous", "light_edit"} and verdict == "plausible"
        record.update(
            {
                "alignment_status": status,
                "coverage": result.get("coverage", ""),
                "mean_match_score": result.get("mean_match_score", ""),
                "predicted_start": result.get("predicted_start", ""),
                "predicted_end": result.get("predicted_end", ""),
                "predicted_start_time": display_timestamp(result.get("predicted_start", "")),
                "predicted_end_time": display_timestamp(result.get("predicted_end", "")),
                "source_span": result.get("source_span", ""),
                "segment_count": result.get("segment_count", ""),
                "span_verdict": verdict,
                "span_ratio": ratio,
                "accept": int(accepted),
                "resolved_status": (
                    "recovered"
                    if accepted
                    else "origin_mismatch_suspect"
                    if status in {"insufficient_alignment", "insufficient_transcript"}
                    else "needs_manual_review"
                ),
            }
        )
        results.append(record)
        print(json.dumps({"event": "row", "index": index, **record}, ensure_ascii=False), flush=True)
        time.sleep(args.sleep_sec)

    if results:
        fieldnames = list(dict.fromkeys(key for record in results for key in record))
        with (out_dir / "gemini_fill_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "targets": len(targets),
        "recovered": sum(1 for r in results if r.get("resolved_status") == "recovered"),
        "resolved_status_counts": {
            status: sum(1 for r in results if r.get("resolved_status") == status)
            for status in {str(r.get("resolved_status")) for r in results}
        },
        "model": args.model,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
