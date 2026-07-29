from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from run_anthropic_listwise_v2 import extract_json
from run_openai_listwise_v2 import (
    append_jsonl,
    read_jsonl,
    validate_result,
)


def call_generate_content(
    *,
    api_key: str,
    model: str,
    instructions: str,
    batch: dict[str, Any],
    timeout_sec: int,
) -> dict[str, Any]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='')}:generateContent"
        f"?key={urllib.parse.quote(api_key, safe='')}"
    )
    payload = {
        "systemInstruction": {
            "parts": [{"text": instructions}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "아래 입력을 평가하십시오. 반드시 지침의 JSON "
                            "형식만 출력하고 모든 candidate_id를 정확히 한 번 "
                            "포함하십시오.\n\n"
                            + json.dumps(batch, ensure_ascii=False)
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = json.load(response)
    candidates = body.get("candidates", [])
    if not candidates:
        raise ValueError(f"Gemini returned no candidate: {body}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(str(part["text"]) for part in parts if "text" in part)
    if not text:
        raise ValueError("Gemini returned no text")
    return extract_json(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the v2 intrinsic listwise rubric through Gemini."
    )
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--longform-ids", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")
    instructions = args.prompt.read_text(encoding="utf-8")
    batches = read_jsonl(args.batches)
    if args.longform_ids:
        allowed = {
            line.strip()
            for line in args.longform_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        batches = [
            row for row in batches if str(row["longform_id"]) in allowed
        ]
    completed = {
        str(row["longform_id"]) for row in read_jsonl(args.output)
    }
    pending = [
        row for row in batches if str(row["longform_id"]) not in completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    failures: list[dict[str, str]] = []
    for index, batch in enumerate(pending, start=1):
        longform_id = str(batch["longform_id"])
        for attempt in range(1, args.max_retries + 1):
            try:
                result = call_generate_content(
                    api_key=api_key,
                    model=args.model,
                    instructions=instructions,
                    batch=batch,
                    timeout_sec=args.timeout_sec,
                )
                append_jsonl(
                    args.output, validate_result(batch, result)
                )
                print(
                    json.dumps(
                        {
                            "status": "completed",
                            "index": index,
                            "pending_total": len(pending),
                            "longform_id": longform_id,
                            "candidate_count": len(batch["candidates"]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                if isinstance(error, urllib.error.HTTPError):
                    detail = error.read().decode("utf-8", errors="replace")
                    message = f"HTTP {error.code}: {detail[:1000]}"
                else:
                    message = str(error)
                print(
                    json.dumps(
                        {
                            "status": "retry",
                            "longform_id": longform_id,
                            "attempt": attempt,
                            "error": message,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if attempt == args.max_retries:
                    failures.append(
                        {"longform_id": longform_id, "error": message}
                    )
                else:
                    time.sleep(args.sleep_sec * attempt)
        time.sleep(args.sleep_sec)

    summary = {
        "requested": len(pending),
        "completed_now": len(pending) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
