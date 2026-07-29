from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from run_openai_listwise_v2 import (
    append_jsonl,
    read_jsonl,
    validate_result,
)


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            stripped = stripped[first_newline + 1 : last_fence].strip()
    return json.loads(stripped)


def call_messages_api(
    *,
    api_key: str,
    model: str,
    instructions: str,
    batch: dict[str, Any],
    timeout_sec: int,
    max_tokens: int,
) -> dict[str, Any]:
    user_input = (
        "아래 입력을 평가하십시오. 반드시 지침의 JSON 형식만 출력하고 "
        "모든 candidate_id를 정확히 한 번 포함하십시오.\n\n"
        + json.dumps(batch, ensure_ascii=False)
    )
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": instructions,
        "messages": [{"role": "user", "content": user_input}],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = json.load(response)
    text_blocks = [
        str(block["text"])
        for block in body.get("content", [])
        if block.get("type") == "text"
    ]
    if not text_blocks:
        raise ValueError("Anthropic API returned no text block")
    return extract_json("\n".join(text_blocks))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the v2 intrinsic listwise rubric through Anthropic."
    )
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--longform-ids", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
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
                result = call_messages_api(
                    api_key=api_key,
                    model=args.model,
                    instructions=instructions,
                    batch=batch,
                    timeout_sec=args.timeout_sec,
                    max_tokens=args.max_tokens,
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
