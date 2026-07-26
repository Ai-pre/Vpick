from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_client import LLMError, call_llm
from shortform_judge_v9 import (
    EDITORIAL_DIMENSIONS,
    ENGAGEMENT_DIMENSIONS,
    EVIDENCE_DIMENSIONS,
    load_config,
    normalize_judgment,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "shortform_judge_v9_opus.json"
DEFAULT_INPUT = (
    ROOT
    / "deliverables"
    / "2026-07-24"
    / "performance_judge_v1"
    / "candidates_blind.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results" / "shortform_judge_v9"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL row must be an object")
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_overview(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for scene in value:
        if not isinstance(scene, dict):
            continue
        compact.append(
            {
                "scene_id": scene.get("scene_id", ""),
                "start_ms": scene.get("start_ms", ""),
                "end_ms": scene.get("end_ms", ""),
                "scene_name": str(scene.get("scene_name", ""))[:160],
                "description": str(scene.get("description", ""))[:500],
            }
        )
    return compact


def candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    start_ms = int(row["start_ms"])
    end_ms = int(row["end_ms"])
    return {
        "candidate_id": row["candidate_id"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_sec": round((end_ms - start_ms) / 1000.0, 3),
        "longform_overview": compact_overview(row.get("longform_overview")),
        "scene_ids": row.get("scene_ids") or [],
        "description": str(row.get("description", ""))[:3000],
        "transcript": str(row.get("transcript", ""))[:7000],
        "before_context": str(row.get("before_context", ""))[:2500],
        "after_context": str(row.get("after_context", ""))[:2500],
        "visual_evidence_available": bool(
            row.get("visual_evidence_available", False)
        ),
    }


def request_hash(
    run_id: str,
    repeat_index: int,
    prompt: str,
    payload: dict[str, Any],
) -> str:
    material = {
        "run_id": run_id,
        "repeat_index": repeat_index,
        "prompt": prompt,
        "payload": payload,
    }
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def mock_response(candidate_id: str) -> dict[str, Any]:
    def axis(dimensions: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        return {
            dimension: {
                "score": 2,
                "reason": "mock pipeline validation",
            }
            for dimension in dimensions
        }

    return {
        "candidate_id": candidate_id,
        "verdict": "score",
        "evidence": {dimension: 3 for dimension in EVIDENCE_DIMENSIONS},
        "editorial": axis(EDITORIAL_DIMENSIONS),
        "engagement": axis(ENGAGEMENT_DIMENSIONS),
        "confidence_1_5": 3,
        "failure_flags": [],
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
    for attempt in range(retries + 1):
        try:
            return call_llm(
                provider,
                model,
                prompt,
                user_prompt,
                max_tokens=max_tokens,
            )
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(16.0, 2.0**attempt))
    raise RuntimeError(f"Judge request failed after retries: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Shortform Judge v9.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run = config["run"]
    prompt = (
        ROOT / "prompts" / f"{config['prompt_id']}.md"
    ).read_text(encoding="utf-8")
    candidates = read_jsonl(args.input)
    if args.limit:
        candidates = candidates[: args.limit]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "candidate_count": len(candidates),
                    "repeat_count": args.repeat_count,
                    "provider": run["provider"],
                    "model": run["model"],
                    "status": "validated_without_model_calls",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    tasks: list[tuple[int, dict[str, Any]]] = []
    for repeat_index in range(1, max(1, args.repeat_count) + 1):
        shuffled = list(candidates)
        random.Random(f"{run['run_id']}:{repeat_index}").shuffle(shuffled)
        tasks.extend((repeat_index, row) for row in shuffled)

    def process(
        task: tuple[int, dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        repeat_index, row = task
        payload = candidate_payload(row)
        key = request_hash(
            run["run_id"],
            repeat_index,
            prompt,
            payload,
        )
        cache_path = (
            args.output_dir / "cache" / run["run_id"] / f"{key}.json"
        )
        started = time.perf_counter()
        cache_hit = not args.no_cache and cache_path.exists()
        if args.mock:
            result = {
                "json": mock_response(str(row["candidate_id"])),
                "usage": {},
                "provider": "mock",
                "model": "mock",
            }
            cache_hit = False
        elif cache_hit:
            result = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            result = call_with_retry(
                str(run["provider"]),
                str(run["model"]),
                prompt,
                payload,
                retries=args.retries,
                max_tokens=args.max_tokens,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
        if args.request_delay and not cache_hit and not args.mock:
            time.sleep(args.request_delay)

        normalized = normalize_judgment(
            result["json"],
            str(row["candidate_id"]),
            config,
        )
        score_row = {
            "judge_run_id": run["run_id"],
            "provider": result.get("provider", run["provider"]),
            "model": result.get("model", run["model"]),
            "prompt_id": config["prompt_id"],
            "repeat_index": repeat_index,
            "longform_id": row.get("longform_id", ""),
            **normalized,
        }
        usage_row = {
            "item_id": f"{row['candidate_id']}#R{repeat_index}",
            "candidate_id": row["candidate_id"],
            "repeat_index": repeat_index,
            "cache_hit": cache_hit,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "usage": result.get("usage", {}),
        }
        return score_row, usage_row

    score_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            score_row, usage_row = future.result()
            score_rows.append(score_row)
            usage_rows.append(usage_row)
            print(
                f"[{completed}/{len(tasks)}] "
                f"R{score_row['repeat_index']} {score_row['candidate_id']}",
                flush=True,
            )

    score_rows.sort(
        key=lambda item: (int(item["repeat_index"]), item["candidate_id"])
    )
    write_csv(args.output_dir / "shortform_judge_v9_scores.csv", score_rows)
    write_jsonl(args.output_dir / "shortform_judge_v9_usage.jsonl", usage_rows)
    summary = {
        "run_id": run["run_id"],
        "candidate_count": len(candidates),
        "repeat_count": max(1, args.repeat_count),
        "request_count": len(score_rows),
        "scored_count": sum(row["verdict"] == "score" for row in score_rows),
        "abstain_count": sum(
            row["verdict"] == "abstain" for row in score_rows
        ),
        "mock": args.mock,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
