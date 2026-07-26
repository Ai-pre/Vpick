from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "rubricreward/mR3-Qwen3-8B-tgt-prompt-en-thinking"
INPUT_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INPUT_FIELDS:
            raise ValueError(
                f"candidate columns must be exactly {INPUT_FIELDS}, "
                f"got {reader.fieldnames}"
            )
        rows = list(reader)
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate input contains duplicate candidate_id values")
    return rows


def extract_json(text: str) -> dict[str, Any]:
    candidates = [text]
    if "</think>" in text:
        candidates.insert(0, text.rsplit("</think>", 1)[-1])
    candidates = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL
    ) + candidates
    decoder = json.JSONDecoder()
    for candidate in candidates:
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
    raise ValueError("no JSON object found")


def required_keys(version: str) -> tuple[str, ...]:
    if version == "v1":
        return ("candidate_id", "verdict", "saliency_market_1_5", "checks")
    if version in {"v2", "v3"}:
        return ("candidate_id", "verdict", "saliency_market_0_100", "checks")
    if version == "v4":
        return ("candidate_id", "verdict", "channel_percentile_0_100")
    if version == "v5":
        return ("candidate_id", "verdict", "p_stop", "p_watch", "p_share")
    raise ValueError(f"unsupported version: {version}")


def validate(judgment: dict[str, Any], version: str, candidate_id: str) -> None:
    missing = [key for key in required_keys(version) if key not in judgment]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    if str(judgment["candidate_id"]) != candidate_id:
        raise ValueError("candidate_id mismatch")
    if judgment["verdict"] not in {"score", "abstain"}:
        score_keys = set(required_keys(version)) - {"candidate_id", "verdict"}
        if all(judgment.get(key) is not None for key in score_keys):
            judgment["verdict"] = "score"
    if judgment["verdict"] not in {"score", "abstain"}:
        raise ValueError("invalid verdict")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["candidate_id"]))
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one prompt-ablation version with mR3."
    )
    parser.add_argument("--version", required=True, choices=("v1", "v2", "v3", "v4", "v5"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--format-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rows = read_candidates(Path(args.candidates))
    if args.max_candidates is not None:
        rows = rows[: max(0, args.max_candidates)]
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / "judgments.jsonl"
    done = completed_ids(judgments_path) if args.resume else set()
    if not args.resume and judgments_path.exists():
        judgments_path.unlink()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "version": args.version,
                "device": args.device,
                "load_sec": round(time.perf_counter() - load_started, 3),
            }
        ),
        flush=True,
    )

    generated_count = 0
    format_retry_count = 0
    total_elapsed = 0.0
    for index, row in enumerate(rows, start=1):
        candidate_id = row["candidate_id"]
        if candidate_id in done:
            continue
        payload = {field: row[field] for field in INPUT_FIELDS}
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(
            json.dumps(
                {
                    "model": args.model,
                    "version": args.version,
                    "prompt": prompt,
                    "payload": payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]
        attempts: list[dict[str, Any]] = []
        judgment: dict[str, Any] | None = None
        for attempt_index in range(args.format_retries + 1):
            attempt_messages = list(messages)
            if attempt_index:
                format_retry_count += 1
                attempt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "출력 형식 오류를 수정하십시오. 분석 과정이나 마크다운 없이 "
                            "요구한 스키마의 유효한 JSON 객체 하나만 출력하십시오."
                        ),
                    }
                )
            chat_text = tokenizer.apply_chat_template(
                attempt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer([chat_text], return_tensors="pt").to(args.device)
            input_tokens = int(inputs.input_ids.shape[-1])
            started = time.perf_counter()
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            total_elapsed += elapsed
            output_ids = generated_ids[0, input_tokens:]
            raw_text = tokenizer.decode(output_ids, skip_special_tokens=False)
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "input_tokens": input_tokens,
                    "generated_tokens": int(output_ids.shape[-1]),
                    "elapsed_sec": round(elapsed, 3),
                    "raw_text": raw_text,
                }
            )
            try:
                parsed = extract_json(raw_text)
                validate(parsed, args.version, candidate_id)
                judgment = parsed
                break
            except (ValueError, TypeError, KeyError):
                judgment = None

        raw_record = {
            "candidate_id": candidate_id,
            "version": args.version,
            "model": args.model,
            "digest": digest,
            "attempts": attempts,
            "judgment": judgment,
        }
        (raw_dir / f"{digest}.json").write_text(
            json.dumps(raw_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if judgment is None:
            raise ValueError(
                f"no valid judgment after {len(attempts)} attempts for {candidate_id}"
            )
        append_jsonl(judgments_path, judgment)
        generated_count += 1
        print(
            json.dumps(
                {
                    "event": "candidate_scored",
                    "version": args.version,
                    "index": index,
                    "total": len(rows),
                    "candidate_id": candidate_id,
                    "verdict": judgment["verdict"],
                    "elapsed_sec": attempts[-1]["elapsed_sec"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "version": args.version,
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "candidate_count": len(rows),
        "generated_count": generated_count,
        "resume_skipped_count": len(done),
        "format_retry_count": format_retry_count,
        "generation_elapsed_sec": round(total_elapsed, 3),
        "seed": args.seed,
        "do_sample": False,
        "thinking": False,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
