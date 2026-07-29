from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from reference_judge_v7 import (
    CHECK_DIMENSIONS,
    EVIDENCE_DIMENSIONS,
    normalize_judgment,
)


ROOT = Path(__file__).resolve().parents[1]
BLIND_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)


def read_blind_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BLIND_FIELDS:
            raise ValueError(
                f"v7 input columns must be exactly {BLIND_FIELDS}, got {reader.fieldnames}"
            )
        rows = list(reader)
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("v7 input contains duplicate candidate_id values")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_json(text: str) -> dict[str, Any]:
    candidates = [text]
    if "</think>" in text:
        candidates.insert(0, text.rsplit("</think>", 1)[-1])
    candidates = (
        re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        + candidates
    )
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
            if isinstance(parsed, dict) and "candidate_id" in parsed:
                return parsed
    raise ValueError("No valid judgment JSON object found in model output")


def post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server HTTP {exc.code}: {body}") from exc


def score_fields() -> list[str]:
    fields = [
        "judge_run_id",
        "provider",
        "model",
        "prompt_id",
        "input_modality",
        "request_id",
        "repeat_index",
        "dry_run",
        "candidate_id",
        "verdict",
    ]
    fields.extend(f"evidence_{name}" for name in EVIDENCE_DIMENSIONS)
    fields.append("saliency_market_1_5")
    fields.extend(f"check_{name}" for name in CHECK_DIMENSIONS)
    fields.extend(
        [
            "checklist_score_100",
            "overall_shortform_suitable",
            "confidence_1_5",
            "failure_flags",
            "reason",
        ]
    )
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the blind v7 Judge through a CPU llama.cpp mR3 server."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument(
        "--prompt",
        default=str(ROOT / "prompts" / "shortform_reference_judge_v7_ko.md"),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8093/v1/chat/completions",
    )
    parser.add_argument("--model", default="mR3-Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    candidates = read_blind_csv(Path(args.candidates))
    if args.max_candidates is not None:
        candidates = candidates[: max(0, args.max_candidates)]
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_id = "mr3_qwen3_8b_q4km_cpu_reference_judge_v7"
    score_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for index, row in enumerate(candidates, start=1):
        request_id = f"R01_C{index:03d}"
        candidate_payload = {field: row[field] for field in BLIND_FIELDS}
        user_prompt = json.dumps(
            candidate_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(
            json.dumps(
                {"model": args.model, "prompt": prompt, "payload": candidate_payload},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        raw_path = raw_dir / f"{digest}.json"

        if args.resume and raw_path.exists():
            raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
            judgment = raw_record["judgment"]
            elapsed_sec = float(raw_record.get("elapsed_sec", 0))
            prompt_tokens = int(raw_record.get("prompt_tokens", 0))
            completion_tokens = int(raw_record.get("completion_tokens", 0))
        else:
            request_payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            last_error: Exception | None = None
            for attempt in range(args.retries + 1):
                started = time.perf_counter()
                try:
                    response = post_json(
                        args.server_url,
                        request_payload,
                        args.timeout_sec,
                    )
                    elapsed_sec = time.perf_counter() - started
                    message = response["choices"][0]["message"]
                    content = str(message.get("content") or "")
                    reasoning = str(message.get("reasoning_content") or "")
                    judgment = extract_json(content or reasoning)
                    normalize_judgment(judgment, row["candidate_id"])
                    break
                except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
                    last_error = exc
                    if attempt >= args.retries:
                        raise
                    time.sleep(2)
            else:
                raise RuntimeError(f"mR3 request failed: {last_error}")

            usage = response.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            raw_record = {
                "candidate_id": row["candidate_id"],
                "model": args.model,
                "elapsed_sec": elapsed_sec,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_sec": (
                    completion_tokens / elapsed_sec if elapsed_sec > 0 else None
                ),
                "response": response,
                "judgment": judgment,
            }
            raw_path.write_text(
                json.dumps(raw_record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        normalized = normalize_judgment(judgment, row["candidate_id"])
        score_rows.append(
            {
                "judge_run_id": run_id,
                "provider": "llama_cpp_cpu",
                "model": args.model,
                "prompt_id": "shortform_reference_judge_v7_ko",
                "input_modality": "yt_dlp_transcript_only_structurally_blind",
                "request_id": request_id,
                "repeat_index": 1,
                "dry_run": False,
                **normalized,
            }
        )
        timing_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_sec": round(elapsed_sec, 3),
                "tokens_per_sec": round(
                    completion_tokens / elapsed_sec if elapsed_sec > 0 else 0,
                    4,
                ),
            }
        )
        write_csv(
            out_dir / "reference_judge_v7_scores.csv",
            score_rows,
            score_fields(),
        )
        write_csv(
            out_dir / "reference_judge_v7_timing.csv",
            timing_rows,
            [
                "candidate_id",
                "prompt_tokens",
                "completion_tokens",
                "elapsed_sec",
                "tokens_per_sec",
            ],
        )
        print(
            json.dumps(
                {
                    "event": "candidate_scored",
                    "index": index,
                    "total": len(candidates),
                    "candidate_id": row["candidate_id"],
                    "verdict": normalized["verdict"],
                    "saliency_market_1_5": normalized["saliency_market_1_5"],
                    "completion_tokens": completion_tokens,
                    "elapsed_sec": round(elapsed_sec, 3),
                    "tokens_per_sec": round(
                        completion_tokens / elapsed_sec if elapsed_sec > 0 else 0,
                        4,
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "run_id": run_id,
        "model": args.model,
        "base_checkpoint": (
            "rubricreward/mR3-Qwen3-8B-tgt-prompt-en-thinking"
        ),
        "runtime": "llama.cpp CPU",
        "quantization": "Q4_K_M directly from BF16",
        "candidate_count": len(candidates),
        "score_row_count": len(score_rows),
        "abstain_count": sum(row["verdict"] == "abstain" for row in score_rows),
        "total_generation_sec": round(
            sum(float(row["elapsed_sec"]) for row in timing_rows),
            3,
        ),
        "mean_tokens_per_sec": round(
            sum(float(row["tokens_per_sec"]) for row in timing_rows)
            / max(1, len(timing_rows)),
            4,
        ),
        "prompt_id": "shortform_reference_judge_v7_ko",
        "prompt_language": "ko",
        "input_language": "ko",
        "input_modality": "yt_dlp_transcript_only_structurally_blind",
        "generation": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
