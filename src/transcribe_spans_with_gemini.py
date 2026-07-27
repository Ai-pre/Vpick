"""Re-transcribe gold spans with Gemini where the YouTube caption is untrustworthy.

Auto-generated captions fail in two ways that matter here: they degrade into
word salad (`"아무리 상기 기판의 누가 이거 돈 우리나라 2미터 시리아 사태"`), or they
return nothing at all for music-heavy segments. Both produce a transcript that a
judge cannot reason over, and neither is visible from the caption metadata.

Gemini reads the YouTube URL directly and accepts a clip window, so each span can
be transcribed in isolation. The existing caption is kept as the comparison
baseline: when the two agree the caption stands, and when they diverge the Gemini
text replaces it. Divergence is reported per row rather than silently applied.
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

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

PROMPT = """이 영상 구간의 한국어 음성을 그대로 전사하십시오.

규칙:
- 들리는 발화만 적고, 추측하거나 요약하지 마십시오.
- 화면 자막만 있고 발화가 없으면 제외하십시오.
- 배경음악 가사는 발화로 취급하지 마십시오.
- 발화가 전혀 없으면 segments 를 빈 배열로 두십시오.

start_sec 과 end_sec 은 이 구간의 시작을 0 으로 둔 상대 시간입니다.
출력은 JSON 객체 하나만, 코드블록 없이 출력하십시오.
{"segments": [{"start_sec": 0.0, "end_sec": 3.2, "text": "발화"}], "speech_present": true}
"""


def normalize(text: str) -> str:
    return re.sub(r"[^\w]", "", text, flags=re.UNICODE).replace("_", "").lower()


def similarity(left: str, right: str) -> float:
    """Character-bigram overlap; robust to reordering and ASR noise."""
    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    grams_a = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    grams_b = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    return len(grams_a & grams_b) / max(1, len(grams_a | grams_b))


def extract_payload(text: str) -> dict[str, Any] | None:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL) or [text]
    decoder = json.JSONDecoder()
    for block in blocks:
        block = block.strip()
        for match in re.finditer(r"\{", block):
            try:
                parsed, _ = decoder.raw_decode(block[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "segments" in parsed:
                return parsed
    return None


def gemini_transcribe(
    api_key: str,
    model: str,
    video_id: str,
    start_sec: float,
    end_sec: float,
    attempts: int,
    timeout: int,
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {
                            "file_data": {
                                "file_uri": f"https://www.youtube.com/watch?v={video_id}"
                            },
                            # Clip to the span so long sources stay affordable.
                            "video_metadata": {
                                "start_offset": {"seconds": int(max(0, start_sec))},
                                "end_offset": {"seconds": int(end_sec) + 1},
                            },
                        },
                        {"text": PROMPT},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 8192},
        }
    ).encode("utf-8")

    last = ""
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
                part.get("text", "")
                for part in (candidate.get("content") or {}).get("parts", [])
            )
            return extract_payload(text), text, data.get("usageMetadata", {})
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            last = f"HTTP {error.code} {detail[:160]}"
            retry = re.search(r"retry in\s+([0-9.]+)s", detail, flags=re.IGNORECASE)
            wait = float(retry.group(1)) + 2.0 if retry else min(90, 6 * 2**attempt)
        except Exception as error:
            last = f"{type(error).__name__}: {error}"
            wait = min(90, 6 * 2**attempt)
        if attempt < attempts:
            print(
                json.dumps({"event": "retry", "video_id": video_id, "attempt": attempt,
                            "wait_sec": round(wait, 1), "error": last[:140]}, ensure_ascii=False),
                flush=True,
            )
            time.sleep(wait)
    return None, last, {}


def render(segments: list[dict[str, Any]], offset: float) -> str:
    lines = []
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if not text:
            continue
        try:
            start = float(segment.get("start_sec", 0.0)) + offset
        except (TypeError, ValueError):
            start = offset
        total = int(round(start))
        lines.append(f"[{total // 60:02d}:{total % 60:02d}] {text}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-transcribe uncertain gold spans with Gemini."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--subtitles", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--candidate-ids", default="", help="Restrict to these candidate_ids.")
    parser.add_argument("--max-similarity", type=float, default=0.35,
                        help="Below this, the caption is treated as untrustworthy.")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--sleep-sec", type=float, default=12.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set.")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labels, encoding="utf-8-sig", newline="") as handle:
        labels = {r["candidate_id"]: r for r in csv.DictReader(handle)}
    with open(args.subtitles, encoding="utf-8-sig", newline="") as handle:
        subtitles = {r["candidate_id"]: r for r in csv.DictReader(handle)}

    wanted = [v.strip() for v in args.candidate_ids.split(",") if v.strip()]
    targets = wanted or list(labels)
    if args.limit:
        targets = targets[: args.limit]

    results: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(targets, start=1):
        meta = labels.get(candidate_id)
        if not meta:
            continue
        try:
            start = float(meta["start_sec"])
            end = float(meta["end_sec"])
        except (KeyError, ValueError):
            continue
        baseline = (subtitles.get(candidate_id) or {}).get("transcript", "")
        payload, raw, usage = gemini_transcribe(
            api_key, args.model, meta["longform_id"], start, end,
            args.attempts, args.timeout,
        )
        record: dict[str, Any] = {
            "candidate_id": candidate_id,
            "channel_name": meta.get("channel_name", ""),
            "longform_id": meta.get("longform_id", ""),
            "start_sec": meta.get("start_sec", ""),
            "end_sec": meta.get("end_sec", ""),
            "model": args.model,
            "baseline_len": len(baseline),
        }
        if payload is None:
            record.update({"status": "gemini_failed", "error": raw[:200]})
            results.append(record)
            print(json.dumps({"event": "row", "index": index, **record}, ensure_ascii=False), flush=True)
            continue

        (raw_dir / f"{candidate_id}.json").write_text(
            json.dumps({"candidate_id": candidate_id, "payload": payload,
                        "raw": raw, "usage": usage}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        segments = payload.get("segments") or []
        gemini_text = render(segments, start)
        score = similarity(baseline, gemini_text)
        speech = bool(payload.get("speech_present", bool(segments)))
        if not segments:
            status = "no_speech_confirmed"
        elif score < args.max_similarity:
            status = "caption_untrustworthy_replace"
        else:
            status = "caption_confirmed"
        record.update(
            {
                "status": status,
                "speech_present": int(speech),
                "gemini_segments": len(segments),
                "gemini_len": len(gemini_text),
                "similarity_to_caption": round(score, 4),
                "gemini_transcript": gemini_text,
                "total_tokens": usage.get("totalTokenCount", ""),
            }
        )
        results.append(record)
        print(
            json.dumps(
                {"event": "row", "index": index, "candidate_id": candidate_id,
                 "status": status, "similarity": round(score, 4),
                 "segments": len(segments)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(args.sleep_sec)

    if results:
        fields = list(dict.fromkeys(k for r in results for k in r))
        with (out_dir / "gemini_span_transcripts.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "targets": len(targets),
        "processed": len(results),
        "status_counts": {
            status: sum(1 for r in results if r.get("status") == status)
            for status in sorted({str(r.get("status")) for r in results})
        },
        "replace_candidates": [
            r["candidate_id"] for r in results if r.get("status") == "caption_untrustworthy_replace"
        ],
        "model": args.model,
        "similarity_threshold": args.max_similarity,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
