from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_client import LLMError, call_gemini_video_pair
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


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric timestamp, got {value!r}") from exc


def youtube_url(candidate: dict[str, str]) -> str:
    video_id = str(candidate.get("long_video_id") or "").strip()
    if not video_id:
        raise ValueError(f"Missing long_video_id for {candidate.get('candidate_id')}")
    return f"https://www.youtube.com/watch?v={video_id}"


def dry_run_response(comparison_id: str) -> dict[str, Any]:
    side = {
        "evidence": {name: 3 for name in EVIDENCE_DIMENSIONS},
        "editorial": {name: 3 for name in EDITORIAL_DIMENSIONS},
        "performance": {name: 3 for name in PERFORMANCE_DIMENSIONS},
    }
    return {
        "provider": "gemini",
        "model": "dry-run",
        "json": {
            "comparison_id": comparison_id,
            "verdict": "score",
            "left": side,
            "right": side,
            "editorial_preference": "tie",
            "performance_preference": "tie",
            "confidence": 3,
            "failure_flags": [],
            "reason": "dry_run",
        },
        "usage": {},
        "dry_run": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gemini pairwise Judge on public YouTube video clips.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-comparisons", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    pairs = read_jsonl(Path(args.pairs))
    if args.max_comparisons is not None:
        pairs = pairs[: max(0, args.max_comparisons)]
    candidates = {row["candidate_id"]: row for row in read_csv(Path(args.candidates))}
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run = dict(config["runs"][0])
    prompt_id = str(config["pairwise_prompt_id"])
    addendum_id = str(config["multimodal_addendum_id"])
    system_prompt = "\n\n".join(
        [
            (ROOT / "prompts" / f"{prompt_id}.md").read_text(encoding="utf-8"),
            (ROOT / "prompts" / f"{addendum_id}.md").read_text(encoding="utf-8"),
        ]
    )
    output_dir = Path(args.out_dir)
    repeat_count = max(1, args.repeat_count)
    fps = float(config.get("video_fps", 2.0))

    tasks = [(repeat_index, pair) for repeat_index in range(1, repeat_count + 1) for pair in pairs]

    def process(task: tuple[int, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        repeat_index, pair = task
        swapped = repeat_index % 2 == 0
        left = pair["right"] if swapped else pair["left"]
        right = pair["left"] if swapped else pair["right"]
        left_media = candidates[left["candidate_id"]]
        right_media = candidates[right["candidate_id"]]
        media = {
            "left": {
                "url": youtube_url(left_media),
                "start_sec": as_float(left_media["start_sec"]),
                "end_sec": as_float(left_media["end_sec"]),
            },
            "right": {
                "url": youtube_url(right_media),
                "start_sec": as_float(right_media["start_sec"]),
                "end_sec": as_float(right_media["end_sec"]),
            },
        }
        presented = {
            "task": "compare_fixed_shortform_candidates_from_video",
            "rubric_version": f"{prompt_id}+{addendum_id}",
            "comparison_id": pair["comparison_id"],
            "left_auxiliary": left,
            "right_auxiliary": right,
        }
        cache_material = {
            "run_id": run["run_id"],
            "repeat_index": repeat_index,
            "comparison_id": pair["comparison_id"],
            "fps": fps,
            "media": media,
            "prompt": presented,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_material, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        cache_file = output_dir / "raw_responses" / str(run["run_id"]) / f"{cache_key}.json"
        if not args.no_cache and cache_file.exists():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
        elif args.dry_run:
            result = dry_run_response(str(pair["comparison_id"]))
        else:
            last_error: LLMError | None = None
            for attempt in range(max(0, args.retries) + 1):
                try:
                    result = call_gemini_video_pair(
                        str(run["model"]),
                        system_prompt,
                        json.dumps(presented, ensure_ascii=False, indent=2),
                        media["left"]["url"],
                        media["left"]["start_sec"],
                        media["left"]["end_sec"],
                        media["right"]["url"],
                        media["right"]["start_sec"],
                        media["right"]["end_sec"],
                        max_tokens=args.max_tokens,
                        fps=fps,
                    )
                    break
                except LLMError as exc:
                    last_error = exc
                    if attempt >= max(0, args.retries):
                        raise
                    time.sleep(5 * (2 ** attempt))
            else:
                raise last_error or LLMError("Gemini video Judge failed without an error")
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        normalized = normalize_response(
            result["json"],
            str(pair["comparison_id"]),
            dict(config["editorial_weights"]),
            dict(config["performance_weights"]),
            swapped,
        )
        row = {
            "judge_run_id": run["run_id"],
            "provider": run["provider"],
            "model": run["model"],
            "judge_role": run.get("judge_role", "primary_multimodal"),
            "prompt_id": f"{prompt_id}+{addendum_id}",
            "input_modality": config.get("input_modality", "public_youtube_video_clips"),
            "video_fps": fps,
            "repeat_index": repeat_index,
            "presentation_swapped": swapped,
            "dry_run": bool(result.get("dry_run", False)),
            **normalized,
        }
        usage = {
            "judge_run_id": run["run_id"],
            "repeat_index": repeat_index,
            "comparison_id": pair["comparison_id"],
            "usage_json": json.dumps(result.get("usage", {}), ensure_ascii=False),
            "dry_run": bool(result.get("dry_run", False)),
        }
        return row, usage

    rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process, task) for task in tasks]
        for future in as_completed(futures):
            row, usage = future.result()
            rows.append(row)
            usage_rows.append(usage)
    rows.sort(key=lambda row: (int(row["repeat_index"]), row["comparison_id"]))

    fields = [
        "judge_run_id", "provider", "model", "judge_role", "prompt_id", "input_modality", "video_fps",
        "repeat_index", "presentation_swapped", "dry_run", "comparison_id", "verdict", "editorial_preference",
        "performance_preference", "confidence",
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
        ["judge_run_id", "repeat_index", "comparison_id", "usage_json", "dry_run"],
    )
    summary = {
        "run_id": run["run_id"],
        "comparison_count": len(pairs),
        "repeat_count": repeat_count,
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
