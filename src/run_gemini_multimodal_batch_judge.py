from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_client import LLMError, call_gemini_video_batch
from run_pairwise_judge import (
    EDITORIAL_DIMENSIONS,
    EVIDENCE_DIMENSIONS,
    PERFORMANCE_DIMENSIONS,
    normalize_response,
    read_jsonl,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def youtube_url(candidate: dict[str, str]) -> str:
    video_id = str(candidate.get("long_video_id") or "").strip()
    if not video_id:
        raise ValueError(f"Missing long_video_id for {candidate.get('candidate_id')}")
    return f"https://www.youtube.com/watch?v={video_id}"


def chunks(rows: list[Any], size: int) -> list[list[Any]]:
    return [rows[index:index + size] for index in range(0, len(rows), size)]


def dry_run_result(batch: list[dict[str, Any]]) -> dict[str, Any]:
    judgments = []
    for item in batch:
        side = {
            "evidence": {name: 3 for name in EVIDENCE_DIMENSIONS},
            "editorial": {name: 3 for name in EDITORIAL_DIMENSIONS},
            "performance": {name: 3 for name in PERFORMANCE_DIMENSIONS},
        }
        judgments.append(
            {
                "comparison_id": item["comparison_id"],
                "verdict": "score",
                "left": side,
                "right": side,
                "editorial_preference": "tie",
                "performance_preference": "tie",
                "confidence": 3,
                "failure_flags": [],
                "reason": "dry_run",
            }
        )
    return {"json": {"judgments": judgments}, "usage": {}, "dry_run": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batched Gemini Judge on up to ten YouTube clips per request.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-comparisons", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-interval-sec", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    pairs = read_jsonl(Path(args.pairs))
    if args.max_comparisons is not None:
        pairs = pairs[: max(0, args.max_comparisons)]
    candidates = {row["candidate_id"]: row for row in read_csv(Path(args.candidates))}
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run = dict(config["runs"][0])
    prompt_ids = [
        str(config["pairwise_prompt_id"]),
        str(config["multimodal_addendum_id"]),
        str(config["batch_addendum_id"]),
    ]
    system_prompt = "\n\n".join(
        (ROOT / "prompts" / f"{prompt_id}.md").read_text(encoding="utf-8")
        for prompt_id in prompt_ids
    )
    output_dir = Path(args.out_dir)
    repeat_count = max(1, args.repeat_count)
    fps = float(config.get("video_fps", 2.0))
    batch_size = max(1, min(5, args.batch_size or int(config.get("batch_size", 5))))

    tasks: list[tuple[int, int, list[dict[str, Any]]]] = []
    for repeat_index in range(1, repeat_count + 1):
        for batch_index, batch_pairs in enumerate(chunks(pairs, batch_size), start=1):
            tasks.append((repeat_index, batch_index, batch_pairs))
    if args.max_batches is not None:
        tasks = tasks[: max(0, args.max_batches)]

    def process(task: tuple[int, int, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        repeat_index, batch_index, batch_pairs = task
        swapped = repeat_index % 2 == 0
        media_batch: list[dict[str, Any]] = []
        auxiliary: list[dict[str, Any]] = []
        for pair in batch_pairs:
            left = pair["right"] if swapped else pair["left"]
            right = pair["left"] if swapped else pair["right"]
            left_source = candidates[left["candidate_id"]]
            right_source = candidates[right["candidate_id"]]
            media_batch.append(
                {
                    "comparison_id": pair["comparison_id"],
                    "left": {
                        "url": youtube_url(left_source),
                        "start_sec": float(left_source["start_sec"]),
                        "end_sec": float(left_source["end_sec"]),
                    },
                    "right": {
                        "url": youtube_url(right_source),
                        "start_sec": float(right_source["start_sec"]),
                        "end_sec": float(right_source["end_sec"]),
                    },
                }
            )
            auxiliary.append(
                {
                    "comparison_id": pair["comparison_id"],
                    "left_auxiliary": left,
                    "right_auxiliary": right,
                }
            )

        batch_id = f"R{repeat_index:02d}_B{batch_index:02d}"
        user_prompt = json.dumps(
            {
                "task": "compare_each_fixed_shortform_pair_from_video",
                "batch_id": batch_id,
                "rubric_version": "+".join(prompt_ids),
                "comparisons": auxiliary,
            },
            ensure_ascii=False,
            indent=2,
        )
        cache_material = {
            "run_id": run["run_id"],
            "batch_id": batch_id,
            "fps": fps,
            "media": media_batch,
            "prompt": auxiliary,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_material, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        cache_file = output_dir / "raw_responses" / str(run["run_id"]) / f"{cache_key}.json"
        if not args.no_cache and cache_file.exists():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
        elif args.dry_run:
            result = dry_run_result(media_batch)
        else:
            last_error: LLMError | None = None
            for attempt in range(max(0, args.retries) + 1):
                try:
                    result = call_gemini_video_batch(
                        str(run["model"]),
                        system_prompt,
                        user_prompt,
                        media_batch,
                        max_tokens=args.max_tokens,
                        fps=fps,
                    )
                    break
                except LLMError as exc:
                    last_error = exc
                    if attempt >= max(0, args.retries):
                        raise
                    time.sleep(10 * (2 ** attempt))
            else:
                raise last_error or LLMError("Gemini batch Judge failed without an error")
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.request_interval_sec > 0:
                time.sleep(args.request_interval_sec)

        raw_judgments = result.get("json", {}).get("judgments")
        if not isinstance(raw_judgments, list):
            raise LLMError(f"Batch {batch_id} is missing judgments")
        by_id = {str(item.get("comparison_id")): item for item in raw_judgments if isinstance(item, dict)}
        expected_ids = {str(pair["comparison_id"]) for pair in batch_pairs}
        if set(by_id) != expected_ids:
            raise LLMError(f"Batch {batch_id} returned IDs {sorted(by_id)}; expected {sorted(expected_ids)}")

        rows: list[dict[str, Any]] = []
        for pair in batch_pairs:
            comparison_id = str(pair["comparison_id"])
            normalized = normalize_response(
                by_id[comparison_id],
                comparison_id,
                dict(config["editorial_weights"]),
                dict(config["performance_weights"]),
                swapped,
            )
            rows.append(
                {
                    "judge_run_id": run["run_id"],
                    "provider": run["provider"],
                    "model": run["model"],
                    "judge_role": run.get("judge_role", "primary_multimodal"),
                    "prompt_id": "+".join(prompt_ids),
                    "input_modality": config.get("input_modality"),
                    "video_fps": fps,
                    "batch_id": batch_id,
                    "repeat_index": repeat_index,
                    "presentation_swapped": swapped,
                    "dry_run": bool(result.get("dry_run", False)),
                    **normalized,
                }
            )
        usage = {
            "judge_run_id": run["run_id"],
            "batch_id": batch_id,
            "repeat_index": repeat_index,
            "comparison_ids": "|".join(sorted(expected_ids)),
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
    rows.sort(key=lambda row: (int(row["repeat_index"]), row["comparison_id"]))
    usage_rows.sort(key=lambda row: (int(row["repeat_index"]), row["batch_id"]))

    fields = [
        "judge_run_id", "provider", "model", "judge_role", "prompt_id", "input_modality", "video_fps",
        "batch_id", "repeat_index", "presentation_swapped", "dry_run", "comparison_id", "verdict",
        "editorial_preference", "performance_preference", "confidence",
    ]
    for side in ("left", "right"):
        fields.extend(f"{side}_evidence_{name}" for name in EVIDENCE_DIMENSIONS)
        fields.extend(f"{side}_editorial_{name}" for name in EDITORIAL_DIMENSIONS)
        fields.append(f"{side}_editorial_score")
        fields.extend(f"{side}_performance_{name}" for name in PERFORMANCE_DIMENSIONS)
        fields.append(f"{side}_performance_score")
    fields.extend(["failure_flags", "reason"])
    write_csv(output_dir / "pairwise_judge_scores.csv", rows, fields)
    write_csv(
        output_dir / "pairwise_judge_usage.csv",
        usage_rows,
        ["judge_run_id", "batch_id", "repeat_index", "comparison_ids", "usage_json", "dry_run"],
    )
    summary = {
        "run_id": run["run_id"],
        "comparison_count": len(pairs),
        "repeat_count": repeat_count,
        "batch_size": batch_size,
        "api_request_count": len(tasks),
        "score_row_count": len(rows),
        "input_modality": config.get("input_modality"),
        "video_fps": fps,
        "dry_run": args.dry_run,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
