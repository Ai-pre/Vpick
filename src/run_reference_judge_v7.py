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


def dry_response(row: dict[str, str]) -> dict[str, Any]:
    return {
        "json": {
            "candidate_id": row["candidate_id"],
            "verdict": "score",
            "evidence": {name: 3 for name in EVIDENCE_DIMENSIONS},
            "saliency_market_1_5": 3,
            "checks": {name: 1 for name in CHECK_DIMENSIONS},
            "overall_shortform_suitable": True,
            "confidence_1_5": 3,
            "failure_flags": [],
            "reason": "dry_run",
        },
        "usage": {},
        "dry_run": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the structurally blind v7 reference Judge.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-interval-sec", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    candidates = read_blind_csv(Path(args.candidates))
    if args.max_candidates is not None:
        candidates = candidates[: max(0, args.max_candidates)]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run = dict(config["runs"][0])
    prompt_id = str(config["prompt_id"])
    prompt = (ROOT / "prompts" / f"{prompt_id}.md").read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)

    tasks: list[tuple[int, int, dict[str, str]]] = []
    for repeat_index in range(1, max(1, args.repeat_count) + 1):
        shuffled = list(candidates)
        random.Random(f"{run['run_id']}:{repeat_index}:v7").shuffle(shuffled)
        tasks.extend(
            (repeat_index, index, row)
            for index, row in enumerate(shuffled, start=1)
        )

    def process(task: tuple[int, int, dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
        repeat_index, request_index, row = task
        request_id = f"R{repeat_index:02d}_C{request_index:03d}"
        payload = {field: row[field] for field in BLIND_FIELDS}
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(
            json.dumps(
                {"run": run, "prompt": prompt, "payload": payload},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        cache_file = out_dir / "raw_responses" / str(run["run_id"]) / f"{digest}.json"

        if not args.no_cache and cache_file.exists():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
        elif args.dry_run:
            result = dry_response(row)
        else:
            last_error: Exception | None = None
            for attempt in range(max(0, args.retries) + 1):
                try:
                    result = call_llm(
                        str(run["provider"]),
                        str(run["model"]),
                        prompt,
                        user_prompt,
                        max_tokens=args.max_tokens,
                    )
                    normalize_judgment(result.get("json", {}), row["candidate_id"])
                    break
                except (LLMError, OSError, KeyError, TypeError) as exc:
                    last_error = exc
                    if attempt >= max(0, args.retries):
                        raise
                    time.sleep(5 * (2 ** attempt))
            else:
                raise LLMError(f"v7 Judge failed: {last_error}")
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if args.request_interval_sec > 0:
                time.sleep(args.request_interval_sec)

        normalized = normalize_judgment(result.get("json", {}), row["candidate_id"])
        score_row = {
            "judge_run_id": run["run_id"],
            "provider": run["provider"],
            "model": run["model"],
            "prompt_id": prompt_id,
            "input_modality": config.get("input_modality", "transcript_only"),
            "request_id": request_id,
            "repeat_index": repeat_index,
            "dry_run": bool(result.get("dry_run", False)),
            **normalized,
        }
        usage = {
            "judge_run_id": run["run_id"],
            "request_id": request_id,
            "repeat_index": repeat_index,
            "candidate_id": row["candidate_id"],
            "usage_json": json.dumps(result.get("usage", {}), ensure_ascii=False),
            "dry_run": bool(result.get("dry_run", False)),
        }
        return score_row, usage

    rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process, task) for task in tasks]
        for future in as_completed(futures):
            score_row, usage = future.result()
            rows.append(score_row)
            usage_rows.append(usage)

    rows.sort(key=lambda row: (int(row["repeat_index"]), row["candidate_id"]))
    usage_rows.sort(key=lambda row: (int(row["repeat_index"]), row["candidate_id"]))
    fields = [
        "judge_run_id", "provider", "model", "prompt_id", "input_modality",
        "request_id", "repeat_index", "dry_run", "candidate_id", "verdict",
    ]
    fields.extend(f"evidence_{name}" for name in EVIDENCE_DIMENSIONS)
    fields.append("saliency_market_1_5")
    fields.extend(f"check_{name}" for name in CHECK_DIMENSIONS)
    fields.extend(
        [
            "checklist_score_100", "overall_shortform_suitable",
            "confidence_1_5", "failure_flags", "reason",
        ]
    )
    write_csv(out_dir / "reference_judge_v7_scores.csv", rows, fields)
    write_csv(
        out_dir / "reference_judge_v7_usage.csv",
        usage_rows,
        [
            "judge_run_id", "request_id", "repeat_index",
            "candidate_id", "usage_json", "dry_run",
        ],
    )
    summary = {
        "run_id": run["run_id"],
        "candidate_count": len(candidates),
        "repeat_count": max(1, args.repeat_count),
        "api_request_count": len(tasks),
        "score_row_count": len(rows),
        "description_nonempty_count": sum(bool(row["description"].strip()) for row in candidates),
        "description_empty_count": sum(not row["description"].strip() for row in candidates),
        "dry_run": args.dry_run,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
