from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from highlight_quality_judge_v1 import (
    DIMENSIONS,
    flip_pairwise_winner,
    load_config,
    normalize_pairwise,
    normalize_pointwise,
)
from llm_client import LLMError, call_llm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "highlight_quality_judge_v1.json"
DEFAULT_POINTWISE_INPUT = (
    ROOT / "deliverables" / "2026-07-24" / "highlight_quality_v1" / "candidates_blind.jsonl"
)
DEFAULT_PAIRWISE_INPUT = (
    ROOT / "deliverables" / "2026-07-24" / "highlight_quality_v1" / "pairs_blind.jsonl"
)
DEFAULT_POINTWISE_PROMPT = ROOT / "prompts" / "highlight_quality_pointwise_v1_ko.md"
DEFAULT_PAIRWISE_PROMPT = ROOT / "prompts" / "highlight_quality_pairwise_v1_ko.md"
DEFAULT_OUTPUT = ROOT / "results" / "highlight_quality_judge_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL row must be an object")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def request_hash(
    provider: str,
    model: str,
    prompt: str,
    payload: dict[str, Any],
) -> str:
    raw = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def swapped_pair(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        **pair,
        "candidate_a": pair["candidate_b"],
        "candidate_b": pair["candidate_a"],
    }


def restore_swapped_result(row: dict[str, Any]) -> dict[str, Any]:
    restored = dict(row)
    restored["winner"] = flip_pairwise_winner(str(row["winner"]))
    for name in DIMENSIONS:
        restored[f"{name}_winner"] = flip_pairwise_winner(
            str(row[f"{name}_winner"])
        )
    return restored


def pairwise_consensus(
    presentation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(presentation_results) != 2:
        return {
            "winner_ab": presentation_results[0]["winner"],
            "winner_ba_restored": "",
            "order_inconsistent": False,
            "consensus_status": "single_pass_unverified",
            "consensus_winner": "abstain",
            "consensus_confidence_1_5": "",
        }
    winner_ab = presentation_results[0]["winner"]
    winner_ba = presentation_results[1]["winner"]
    consistent = winner_ab == winner_ba
    return {
        "winner_ab": winner_ab,
        "winner_ba_restored": winner_ba,
        "order_inconsistent": not consistent,
        "consensus_status": "accepted" if consistent else "abstain_order_inconsistent",
        "consensus_winner": winner_ab if consistent else "abstain",
        "consensus_confidence_1_5": (
            min(
                int(presentation_results[0]["confidence_1_5"]),
                int(presentation_results[1]["confidence_1_5"]),
            )
            if consistent
            else ""
        ),
    }


def mock_pointwise(candidate_id: str) -> dict[str, Any]:
    dimensions = {}
    for index, name in enumerate(DIMENSIONS):
        score = int(hashlib.sha256(f"{candidate_id}:{name}".encode()).hexdigest()[:2], 16) % 5
        dimensions[name] = {
            "score": score,
            "reason": "mock pipeline validation",
            "scene_ids": [],
            "insufficient_information": False,
        }
    return {
        "candidate_id": candidate_id,
        "verdict": "score",
        "dimensions": dimensions,
        "fatal_flags": [],
        "confidence_1_5": 3,
        "overall_reason": "mock pipeline validation",
    }


def mock_pairwise(pair_id: str) -> dict[str, Any]:
    comparisons = {
        name: {
            "winner": "tie",
            "reason": "mock pipeline validation",
            "scene_ids": [],
        }
        for name in DIMENSIONS
    }
    return {
        "pair_id": pair_id,
        "dimension_comparisons": comparisons,
        "winner": "tie",
        "fatal_flags_a": [],
        "fatal_flags_b": [],
        "confidence_1_5": 3,
        "reason": "mock pipeline validation",
    }


def call_with_retry(
    provider: str,
    model: str,
    prompt: str,
    payload: dict[str, Any],
    *,
    retries: int,
    max_tokens: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    user_prompt = json.dumps(payload, ensure_ascii=False)
    compact_json_retry = False
    for attempt in range(retries + 1):
        try:
            attempt_prompt = system_prompt = prompt
            if compact_json_retry:
                attempt_prompt = (
                    system_prompt
                    + "\n\n이전 응답의 JSON이 중간에 잘렸습니다. 모든 reason을 한국어 40자 "
                    "이내 한 문장으로 줄이고 scene_ids는 최대 2개만 사용하여 완결된 JSON "
                    "객체 하나만 반환하십시오."
                )
            return call_llm(
                provider,
                model,
                attempt_prompt,
                user_prompt,
                max_tokens=max_tokens,
            )
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if "did not return valid JSON" in str(exc):
                compact_json_retry = True
            if attempt < retries:
                message = str(exc)
                retry_delay = (
                    45.0
                    if any(
                        marker in message
                        for marker in ("429", "RESOURCE_EXHAUSTED", "retryDelay")
                    )
                    else min(8.0, 2.0 ** attempt)
                )
                time.sleep(retry_delay)
    raise RuntimeError(f"Judge request failed after retries: {last_error}")


def cached_call(
    cache_dir: Path,
    provider: str,
    model: str,
    prompt: str,
    payload: dict[str, Any],
    *,
    retries: int,
    max_tokens: int,
) -> tuple[dict[str, Any], bool]:
    key = request_hash(provider, model, prompt, payload)
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    result = call_with_retry(
        provider,
        model,
        prompt,
        payload,
        retries=retries,
        max_tokens=max_tokens,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result, False


def run_pointwise(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = []
    usage = []
    for index, row in enumerate(rows, start=1):
        candidate_id = str(row["candidate_id"])
        started = time.perf_counter()
        if args.mock:
            raw = mock_pointwise(candidate_id)
            result = {"usage": {}, "provider": "mock", "model": "mock"}
            cache_hit = False
        else:
            result, cache_hit = cached_call(
                args.cache_dir,
                args.provider,
                args.model,
                prompt,
                row,
                retries=args.retries,
                max_tokens=args.max_tokens,
            )
            raw = result["json"]
        normalized = normalize_pointwise(
            raw,
            candidate_id,
            config["weights"],
        )
        output.append(normalized)
        usage.append(
            {
                "item_id": candidate_id,
                "index": index,
                "cache_hit": cache_hit,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "provider": result.get("provider", args.provider),
                "model": result.get("model", args.model),
                "usage": result.get("usage", {}),
            }
        )
        print(f"[{index}/{len(rows)}] {candidate_id}", flush=True)
        if args.request_delay and not cache_hit:
            time.sleep(args.request_delay)
    return output, usage


def run_pairwise(
    rows: list[dict[str, Any]],
    prompt: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = []
    usage = []
    for index, row in enumerate(rows, start=1):
        pair_id = str(row["pair_id"])
        presentations = [("A-B", row)]
        if args.repeat_swapped:
            presentations.append(("B-A", swapped_pair(row)))
        presentation_results = []
        for display_order, payload in presentations:
            started = time.perf_counter()
            if args.mock:
                raw = mock_pairwise(pair_id)
                result = {"usage": {}, "provider": "mock", "model": "mock"}
                cache_hit = False
            else:
                result, cache_hit = cached_call(
                    args.cache_dir,
                    args.provider,
                    args.model,
                    prompt,
                    payload,
                    retries=args.retries,
                    max_tokens=args.max_tokens,
                )
                raw = result["json"]
            normalized = normalize_pairwise(raw, pair_id)
            if display_order == "B-A":
                normalized = restore_swapped_result(normalized)
            normalized["display_order"] = display_order
            presentation_results.append(normalized)
            usage.append(
                {
                    "item_id": pair_id,
                    "display_order": display_order,
                    "cache_hit": cache_hit,
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "provider": result.get("provider", args.provider),
                    "model": result.get("model", args.model),
                    "usage": result.get("usage", {}),
                }
            )
            if args.request_delay and not cache_hit:
                time.sleep(args.request_delay)
        base = dict(presentation_results[0])
        consensus = pairwise_consensus(presentation_results)
        base.update(consensus)
        base["swapped_winner_restored"] = consensus["winner_ba_restored"]
        output.append(base)
        print(f"[{index}/{len(rows)}] {pair_id}", flush=True)
    return output, usage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Highlight Quality Judge v1.")
    parser.add_argument("--mode", choices=("pointwise", "pairwise"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_OUTPUT / "cache")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to wait after each non-cached model request.",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repeat-swapped", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = args.input or (
        DEFAULT_POINTWISE_INPUT
        if args.mode == "pointwise"
        else DEFAULT_PAIRWISE_INPUT
    )
    prompt_path = args.prompt or (
        DEFAULT_POINTWISE_PROMPT
        if args.mode == "pointwise"
        else DEFAULT_PAIRWISE_PROMPT
    )
    rows = read_jsonl(input_path)
    if args.limit:
        rows = rows[: args.limit]
    prompt = prompt_path.read_text(encoding="utf-8")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "input": str(input_path),
                    "prompt": str(prompt_path),
                    "item_count": len(rows),
                    "provider": args.provider,
                    "model": args.model,
                    "repeat_swapped": args.repeat_swapped,
                    "status": "validated_without_model_calls",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.mode == "pointwise":
        judgments, usage = run_pointwise(rows, config, prompt, args)
    else:
        judgments, usage = run_pairwise(rows, prompt, args)
    suffix = "mock" if args.mock else args.model.replace("/", "_")
    write_jsonl(
        args.output_dir / f"{args.mode}_{suffix}_judgments.jsonl",
        judgments,
    )
    write_jsonl(
        args.output_dir / f"{args.mode}_{suffix}_usage.jsonl",
        usage,
    )
    summary = {
        "mode": args.mode,
        "item_count": len(judgments),
        "provider": "mock" if args.mock else args.provider,
        "model": "mock" if args.mock else args.model,
        "repeat_swapped": args.repeat_swapped,
        "cache_hit_count": sum(bool(row["cache_hit"]) for row in usage),
        "order_inconsistent_count": sum(
            bool(row.get("order_inconsistent")) for row in judgments
        ),
    }
    (args.output_dir / f"{args.mode}_{suffix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
