from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCORE_FIELDS = [
    "opening_clarity_pull_0_4",
    "event_reaction_change_0_4",
    "progression_payoff_0_4",
    "self_contained_0_4",
    "boundary_integrity_0_4",
    "titleability_0_4",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def response_schema() -> dict[str, Any]:
    score_properties: dict[str, Any] = {
        "candidate_id": {"type": "string"},
    }
    score_properties.update(
        {
            field: {
                "type": "integer",
                "minimum": 0,
                "maximum": 4,
            }
            for field in SCORE_FIELDS
        }
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["longform_id", "content_mode", "candidate_scores"],
        "properties": {
            "longform_id": {"type": "string"},
            "content_mode": {
                "type": "string",
                "enum": [
                    "entertainment_vlog",
                    "interview_conversation",
                    "lecture_information",
                    "mixed_other",
                ],
            },
            "candidate_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["candidate_id", *SCORE_FIELDS],
                    "properties": score_properties,
                },
            },
        },
    }


def extract_output_text(response: dict[str, Any]) -> str:
    for output in response.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                return str(content["text"])
    raise ValueError("Responses API returned no output_text")


def validate_result(
    batch: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    expected_longform = str(batch["longform_id"])
    if str(result.get("longform_id")) != expected_longform:
        raise ValueError(f"longform_id mismatch: {expected_longform}")
    expected_ids = {
        str(candidate["candidate_id"]) for candidate in batch["candidates"]
    }
    scored = result.get("candidate_scores")
    if not isinstance(scored, list):
        raise ValueError(f"candidate_scores missing: {expected_longform}")
    actual_ids = [str(item["candidate_id"]) for item in scored]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError(f"duplicate candidate_id: {expected_longform}")
    if set(actual_ids) != expected_ids:
        missing = sorted(expected_ids - set(actual_ids))
        extra = sorted(set(actual_ids) - expected_ids)
        raise ValueError(
            f"candidate coverage mismatch: {expected_longform}; "
            f"missing={missing}; extra={extra}"
        )
    for item in scored:
        for field in SCORE_FIELDS:
            value = item[field]
            if not isinstance(value, int) or not 0 <= value <= 4:
                raise ValueError(
                    f"invalid {field}: {expected_longform}/"
                    f"{item['candidate_id']}={value}"
                )
    return result


def call_responses_api(
    *,
    api_key: str,
    model: str,
    instructions: str,
    batch: dict[str, Any],
    timeout_sec: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": json.dumps(batch, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "shortform_listwise_scores",
                "strict": True,
                "schema": response_schema(),
            }
        },
    }
    if reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = json.load(response)
    return json.loads(extract_output_text(body))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the v2 intrinsic listwise rubric through Responses API."
    )
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--longform-ids", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
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
                result = call_responses_api(
                    api_key=api_key,
                    model=args.model,
                    instructions=instructions,
                    batch=batch,
                    timeout_sec=args.timeout_sec,
                    reasoning_effort=args.reasoning_effort,
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
