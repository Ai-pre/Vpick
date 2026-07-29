from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCORE_FIELDS = (
    "first_glance_clarity_1_10",
    "curiosity_click_pull_1_10",
    "title_thumbnail_complementarity_1_10",
    "content_alignment_1_10",
    "joint_package_score_1_10",
)


class JudgeError(RuntimeError):
    pass


def parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise JudgeError("No JSON object found in response")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise JudgeError("Response root must be an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_block(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        },
    }


def post_anthropic(
    model: str,
    system_prompt: str,
    content: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise JudgeError("ANTHROPIC_API_KEY or CLAUDE_API_KEY is not set")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": content}],
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
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise JudgeError(f"Anthropic HTTP {exc.code}: {detail[:1200]}") from exc
    text = "\n".join(
        item.get("text", "")
        for item in data.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ).strip()
    if not text:
        raise JudgeError("Anthropic response contained no text")
    return {
        "json": parse_json_text(text),
        "usage": data.get("usage", {}),
        "stop_reason": data.get("stop_reason", ""),
    }


def candidate_text(row: dict[str, Any], alias: str) -> str:
    payload = {
        "blind_id": alias,
        "title": str(row.get("title", ""))[:300],
        "description": str(row.get("description", ""))[:900],
        "transcript": str(row.get("transcript", ""))[:2200],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def validate_response(
    response: dict[str, Any],
    expected_aliases: set[str],
) -> list[dict[str, Any]]:
    judgments = response.get("judgments")
    if not isinstance(judgments, list):
        raise JudgeError("Response must contain a judgments list")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in judgments:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("blind_id", "")).strip()
        if alias not in expected_aliases or alias in seen:
            continue
        normalized = {
            "blind_id": alias,
            "evidence_first": str(item.get("evidence_first", ""))[:1200],
            "confidence_1_5": int(item.get("confidence_1_5", 0)),
        }
        if not 1 <= normalized["confidence_1_5"] <= 5:
            raise JudgeError(f"Invalid confidence for {alias}")
        for field in SCORE_FIELDS:
            score = int(item.get(field, 0))
            if not 1 <= score <= 10:
                raise JudgeError(f"Invalid {field} for {alias}: {score}")
            normalized[field] = score
        output.append(normalized)
        seen.add(alias)
    if seen != expected_aliases:
        raise JudgeError(f"Missing aliases: {sorted(expected_aliases - seen)}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt", default="prompts/package_success_judge_v1_ko.md")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    if args.max_candidates is not None:
        rows = rows[: max(0, args.max_candidates)]
    prompt = (ROOT / args.prompt).read_text(encoding="utf-8")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[int, int, list[dict[str, Any]]]] = []
    for repeat in range(1, args.repeat_count + 1):
        shuffled = list(rows)
        random.Random(f"package-v1-repeat-{repeat}").shuffle(shuffled)
        for start in range(0, len(shuffled), args.batch_size):
            tasks.append((repeat, start // args.batch_size + 1, shuffled[start : start + args.batch_size]))

    def process(task: tuple[int, int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        repeat, batch_index, batch = task
        aliases = {
            row["candidate_id"]: "PKG_"
            + hashlib.sha256(
                f"{repeat}:{row['candidate_id']}:package-v1".encode("utf-8")
            ).hexdigest()[:12]
            for row in batch
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"반복 {repeat}, 배치 {batch_index}. "
                    "각 이미지 바로 앞의 후보 정보에 대해서만 독립 채점하십시오."
                ),
            }
        ]
        content[0]["text"] = (
            f"반복 {repeat}, 배치 {batch_index}. "
            "각 이미지 바로 앞의 후보 정보와 연결해서만 독립 채점하십시오."
        )
        for row in batch:
            content.append({"type": "text", "text": candidate_text(row, aliases[row["candidate_id"]])})
            thumbnail_path = Path(row["thumbnail_path"])
            if not thumbnail_path.is_absolute():
                thumbnail_path = ROOT / thumbnail_path
            content.append(image_block(thumbnail_path))
        content.append(
            {
                "type": "text",
                "text": (
                    "위 후보를 모두 채점하고, 입력된 blind_id를 정확히 한 번씩 "
                    "포함한 JSON 객체만 반환하십시오."
                ),
            }
        )
        content[-1]["text"] = (
            "위 후보를 모두 채점하고, 입력의 blind_id를 정확하게 한 번씩 "
            "포함한 JSON 객체만 반환하십시오."
        )
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "model": args.model,
                    "repeat": repeat,
                    "batch": batch_index,
                    "candidate_ids": [row["candidate_id"] for row in batch],
                    "prompt": prompt,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        cache_path = output_dir / "raw" / f"{cache_key}.json"
        if cache_path.exists():
            result = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            last_error: Exception | None = None
            for attempt in range(args.retries + 1):
                try:
                    result = post_anthropic(args.model, prompt, content, args.max_tokens)
                    break
                except (JudgeError, OSError) as exc:
                    last_error = exc
                    if attempt >= args.retries:
                        raise
                    time.sleep(5 * (2**attempt))
            else:
                raise JudgeError(f"Request failed: {last_error}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        normalized = validate_response(
            result["json"], set(aliases.values())
        )
        candidate_by_alias = {alias: candidate_id for candidate_id, alias in aliases.items()}
        for row in normalized:
            row["candidate_id"] = candidate_by_alias[row["blind_id"]]
            row["repeat_index"] = repeat
            row["batch_index"] = batch_index
            row["model"] = args.model
        return normalized

    all_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(process, task): task for task in tasks}
        for future in as_completed(future_map):
            repeat, batch_index, _ = future_map[future]
            batch_rows = future.result()
            all_rows.extend(batch_rows)
            print(
                json.dumps(
                    {
                        "repeat": repeat,
                        "batch": batch_index,
                        "completed_rows": len(all_rows),
                        "expected_rows": len(rows) * args.repeat_count,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    all_rows.sort(key=lambda row: (int(row["repeat_index"]), str(row["candidate_id"])))
    result_path = output_dir / "package_judgments.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "model": args.model,
        "candidate_count": len(rows),
        "repeat_count": args.repeat_count,
        "judgment_count": len(all_rows),
        "result_path": str(result_path),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
