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

from llm_client import LLMError, call_gemini_video_pointwise_batch, call_llm
from run_pairwise_judge import EDITORIAL_DIMENSIONS, PERFORMANCE_DIMENSIONS, score, score_map, weighted_score


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIMENSIONS = ("description_support", "transcript_intelligibility", "boundary_observability")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def candidate_payload(row: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "duration_sec": float(row["duration_sec"]),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "language": row.get("language", "ko"),
        "genre": row.get("genre", "general"),
        "description": row.get("description", "")[:1800],
        "transcript": row.get("transcript", "")[:5000],
        "before_context": row.get("before_context", "")[:2500],
        "after_context": row.get("after_context", "")[:2500],
    }


def normalize_judgments(
    response: dict[str, Any],
    expected_ids: set[str],
    editorial_weights: dict[str, Any],
    performance_weights: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_items = response.get("judgments")
    if not isinstance(raw_items, list):
        raise LLMError("Pointwise response must contain a judgments list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id not in expected_ids or candidate_id in seen:
            continue
        verdict = str(item.get("verdict", "score")).strip().lower()
        if verdict not in {"score", "abstain"}:
            raise LLMError(f"Invalid verdict for {candidate_id}: {verdict}")
        evidence = score_map(item.get("evidence"), EVIDENCE_DIMENSIONS)
        flags = [str(value) for value in (item.get("failure_flags") or [])[:13]]
        if verdict == "abstain":
            if "insufficient_evidence" not in flags:
                flags.append("insufficient_evidence")
            editorial: dict[str, int] = {}
            performance: dict[str, int] = {}
        else:
            editorial = score_map(item.get("editorial"), EDITORIAL_DIMENSIONS)
            performance = score_map(item.get("performance"), PERFORMANCE_DIMENSIONS)
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "verdict": verdict,
            **{f"evidence_{name}": value for name, value in evidence.items()},
            "confidence": score(item.get("confidence", 1)),
            "editorial_score": weighted_score(editorial, editorial_weights) if editorial else "",
            "performance_score": weighted_score(performance, performance_weights) if performance else "",
            "failure_flags": "|".join(flags),
            "reason": str(item.get("reason", ""))[:1200],
        }
        for name in EDITORIAL_DIMENSIONS:
            row[f"editorial_{name}"] = editorial.get(name, "")
        for name in PERFORMANCE_DIMENSIONS:
            row[f"performance_{name}"] = performance.get(name, "")
        rows.append(row)
        seen.add(candidate_id)
    if seen != expected_ids:
        raise LLMError(f"Pointwise Judge omitted candidate IDs: {sorted(expected_ids - seen)}")
    return rows


def dry_result(batch: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "json": {
            "judgments": [
                {
                    "candidate_id": row["candidate_id"],
                    "verdict": "score",
                    "evidence": {name: 3 for name in EVIDENCE_DIMENSIONS},
                    "editorial": {name: 3 for name in EDITORIAL_DIMENSIONS},
                    "performance": {name: 3 for name in PERFORMANCE_DIMENSIONS},
                    "confidence": 3,
                    "failure_flags": [],
                    "reason": "dry_run",
                }
                for row in batch
            ]
        },
        "usage": {},
        "dry_run": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run independent pointwise Gold candidate evaluation.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-interval-sec", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    candidates = read_csv(Path(args.candidates))
    if args.max_candidates is not None:
        candidates = candidates[: max(0, args.max_candidates)]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run = dict(config["runs"][0])
    prompt_id = str(config["prompt_id"])
    prompt = (ROOT / "prompts" / f"{prompt_id}.md").read_text(encoding="utf-8")
    output_dir = Path(args.out_dir)
    input_modality = str(config.get("input_modality", "scene_description_transcript"))
    video_mode = "video" in input_modality or "youtube" in input_modality
    default_batch_size = int(config.get("batch_size", 1))
    batch_size = max(1, min(5 if video_mode else 10, args.batch_size or default_batch_size))
    fps = float(config.get("video_fps", 2.0)) if video_mode else None

    tasks: list[tuple[int, int, list[dict[str, str]]]] = []
    for repeat_index in range(1, max(1, args.repeat_count) + 1):
        shuffled = list(candidates)
        random.Random(f"{run['run_id']}:{repeat_index}:pointwise").shuffle(shuffled)
        batches = [shuffled[index:index + batch_size] for index in range(0, len(shuffled), batch_size)]
        for batch_index, batch in enumerate(batches, start=1):
            tasks.append((repeat_index, batch_index, batch))
    if args.max_tasks is not None:
        tasks = tasks[: max(0, args.max_tasks)]

    def process(task: tuple[int, int, list[dict[str, str]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        repeat_index, batch_index, batch = task
        batch_id = f"R{repeat_index:02d}_B{batch_index:03d}"
        payload = {
            "task": "score_each_fixed_shortform_candidate_independently",
            "rubric_version": prompt_id,
            "batch_id": batch_id,
            "independence_rule": "Apply the absolute rubric independently; never rank or compare candidates.",
            "candidates": [candidate_payload(row) for row in batch],
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        media = [
            {
                "candidate_id": row["candidate_id"],
                "url": f"https://www.youtube.com/watch?v={row['long_video_id']}",
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
            }
            for row in batch
        ]
        cache_material = {
            "run_id": run["run_id"],
            "batch_id": batch_id,
            "prompt": prompt,
            "payload": payload,
            "media": media if video_mode else [],
            "fps": fps,
        }
        digest = hashlib.sha256(
            json.dumps(cache_material, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        cache_file = output_dir / "raw_responses" / str(run["run_id"]) / f"{digest}.json"
        if not args.no_cache and cache_file.exists():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
        elif args.dry_run:
            result = dry_result(batch)
        else:
            last_error: Exception | None = None
            for attempt in range(max(0, args.retries) + 1):
                try:
                    if video_mode:
                        result = call_gemini_video_pointwise_batch(
                            str(run["model"]), prompt, user_prompt, media, args.max_tokens, fps=fps or 2.0
                        )
                    else:
                        result = call_llm(
                            str(run["provider"]), str(run["model"]), prompt, user_prompt, max_tokens=args.max_tokens
                        )
                    break
                except (LLMError, OSError) as exc:
                    last_error = exc
                    if attempt >= max(0, args.retries):
                        raise
                    time.sleep(5 * (2 ** attempt))
            else:
                raise LLMError(f"Pointwise Judge failed: {last_error}")
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.request_interval_sec > 0:
                time.sleep(args.request_interval_sec)

        expected_ids = {row["candidate_id"] for row in batch}
        normalized = normalize_judgments(
            result.get("json", {}),
            expected_ids,
            dict(config["editorial_weights"]),
            dict(config["performance_weights"]),
        )
        source_by_id = {row["candidate_id"]: row for row in batch}
        rows = [
            {
                "judge_run_id": run["run_id"],
                "provider": run["provider"],
                "model": run["model"],
                "judge_role": run.get("judge_role", "pointwise"),
                "prompt_id": prompt_id,
                "input_modality": input_modality,
                "video_fps": fps if video_mode else "",
                "batch_id": batch_id,
                "batch_size": len(batch),
                "repeat_index": repeat_index,
                "dry_run": bool(result.get("dry_run", False)),
                "long_video_id": source_by_id[row["candidate_id"]]["long_video_id"],
                **row,
            }
            for row in normalized
        ]
        usage = {
            "judge_run_id": run["run_id"],
            "batch_id": batch_id,
            "repeat_index": repeat_index,
            "candidate_ids": "|".join(sorted(expected_ids)),
            "usage_json": json.dumps(result.get("usage", {}), ensure_ascii=False),
            "dry_run": bool(result.get("dry_run", False)),
        }
        return rows, usage

    rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process, task) for task in tasks]
        for future in as_completed(futures):
            batch_rows, usage = future.result()
            rows.extend(batch_rows)
            usage_rows.append(usage)
    rows.sort(key=lambda row: (row["judge_run_id"], int(row["repeat_index"]), row["candidate_id"]))
    usage_rows.sort(key=lambda row: (int(row["repeat_index"]), row["batch_id"]))

    fields = [
        "judge_run_id", "provider", "model", "judge_role", "prompt_id", "input_modality", "video_fps",
        "batch_id", "batch_size", "repeat_index", "dry_run", "candidate_id", "long_video_id", "verdict",
    ]
    fields.extend(f"evidence_{name}" for name in EVIDENCE_DIMENSIONS)
    fields.extend(f"editorial_{name}" for name in EDITORIAL_DIMENSIONS)
    fields.append("editorial_score")
    fields.extend(f"performance_{name}" for name in PERFORMANCE_DIMENSIONS)
    fields.extend(["performance_score", "confidence", "failure_flags", "reason"])
    write_csv(output_dir / "pointwise_judge_scores.csv", rows, fields)
    write_csv(
        output_dir / "pointwise_judge_usage.csv",
        usage_rows,
        ["judge_run_id", "batch_id", "repeat_index", "candidate_ids", "usage_json", "dry_run"],
    )
    summary = {
        "run_id": run["run_id"],
        "candidate_count": len(candidates),
        "repeat_count": max(1, args.repeat_count),
        "batch_size": batch_size,
        "api_request_count": len(tasks),
        "score_row_count": len(rows),
        "input_modality": input_modality,
        "video_fps": fps,
        "dry_run": args.dry_run,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
