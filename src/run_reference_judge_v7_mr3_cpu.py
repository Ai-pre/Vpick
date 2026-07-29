from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
DEFAULT_MODEL = "rubricreward/mR3-Qwen3-8B-tgt-prompt-en-thinking"


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
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced + candidates

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
        description="Run the structurally blind v7 reference Judge with mR3 on CPU."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument(
        "--prompt",
        default=str(ROOT / "prompts" / "shortform_reference_judge_v7_ko.md"),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    candidates = read_blind_csv(Path(args.candidates))
    if args.max_candidates is not None:
        candidates = candidates[: max(0, args.max_candidates)]
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        torch.set_num_threads(max(1, args.threads))
        torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        attn_implementation="eager" if args.device == "cpu" else "sdpa",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    load_sec = time.perf_counter() - load_started
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model": args.model,
                "dtype": str(model.dtype),
                "device": args.device,
                "threads": args.threads,
                "load_sec": round(load_sec, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    device_id = args.device.replace(":", "_")
    run_id = f"mr3_qwen3_8b_{device_id}_{args.dtype}_reference_judge_v7"
    score_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for index, row in enumerate(candidates, start=1):
        request_id = f"R01_C{index:03d}"
        payload = {field: row[field] for field in BLIND_FIELDS}
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(
            json.dumps(
                {"model": args.model, "prompt": prompt, "payload": payload},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        raw_path = raw_dir / f"{digest}.json"

        if args.resume and raw_path.exists():
            raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
            judgment = raw_record["judgment"]
            elapsed_sec = float(raw_record.get("elapsed_sec", 0))
            generated_tokens = int(raw_record.get("generated_tokens", 0))
            input_tokens = int(raw_record.get("input_tokens", 0))
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ]
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            model_inputs = tokenizer([chat_text], return_tensors="pt").to(args.device)
            input_tokens = int(model_inputs.input_ids.shape[-1])

            started = time.perf_counter()
            with torch.inference_mode():
                generation_args: dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.do_sample,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if args.do_sample:
                    generation_args.update(
                        {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                        }
                    )
                generated_ids = model.generate(
                    **model_inputs,
                    **generation_args,
                )
            elapsed_sec = time.perf_counter() - started
            output_ids = generated_ids[0, input_tokens:]
            generated_tokens = int(output_ids.shape[-1])
            raw_text = tokenizer.decode(output_ids, skip_special_tokens=False)
            raw_record = {
                "candidate_id": row["candidate_id"],
                "model": args.model,
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "elapsed_sec": elapsed_sec,
                "tokens_per_sec": (
                    generated_tokens / elapsed_sec if elapsed_sec > 0 else None
                ),
                "raw_text": raw_text,
            }
            raw_path.write_text(
                json.dumps(raw_record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            judgment = extract_json(raw_text)
            normalize_judgment(judgment, row["candidate_id"])
            raw_record["judgment"] = judgment
            raw_path.write_text(
                json.dumps(raw_record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        normalized = normalize_judgment(judgment, row["candidate_id"])
        score_rows.append(
            {
                "judge_run_id": run_id,
                "provider": f"huggingface_transformers_{device_id}",
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
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "elapsed_sec": round(elapsed_sec, 3),
                "tokens_per_sec": round(
                    generated_tokens / elapsed_sec if elapsed_sec > 0 else 0, 4
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
                "input_tokens",
                "generated_tokens",
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
                    "generated_tokens": generated_tokens,
                    "elapsed_sec": round(elapsed_sec, 3),
                    "tokens_per_sec": round(
                        generated_tokens / elapsed_sec if elapsed_sec > 0 else 0, 4
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "run_id": run_id,
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "candidate_count": len(candidates),
        "score_row_count": len(score_rows),
        "abstain_count": sum(row["verdict"] == "abstain" for row in score_rows),
        "model_load_sec": round(load_sec, 3),
        "total_generation_sec": round(
            sum(float(row["elapsed_sec"]) for row in timing_rows), 3
        ),
        "mean_tokens_per_sec": round(
            sum(float(row["tokens_per_sec"]) for row in timing_rows)
            / max(1, len(timing_rows)),
            4,
        ),
        "prompt_id": "shortform_reference_judge_v7_ko",
        "input_modality": "yt_dlp_transcript_only_structurally_blind",
        "generation": {
            "enable_thinking": True,
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "top_k": args.top_k if args.do_sample else None,
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
