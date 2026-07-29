from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from run_shortform_judge_v9 import (
    DEFAULT_CONFIG,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    candidate_payload,
    read_jsonl,
    request_hash,
    write_csv,
    write_jsonl,
)
from shortform_judge_v9 import load_config, normalize_judgment


ROOT = Path(__file__).resolve().parents[1]


def recover(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    prompt: str,
    cache_root: Path,
    repeat_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run = config["run"]
    scores: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for repeat_index in range(1, max(1, repeat_count) + 1):
        shuffled = list(candidates)
        random.Random(f"{run['run_id']}:{repeat_index}").shuffle(shuffled)
        for row in shuffled:
            payload = candidate_payload(row)
            key = request_hash(
                run["run_id"],
                repeat_index,
                prompt,
                payload,
            )
            cache_path = cache_root / run["run_id"] / f"{key}.json"
            if not cache_path.exists():
                missing.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "repeat_index": repeat_index,
                    }
                )
                continue
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            normalized = normalize_judgment(
                result["json"],
                str(row["candidate_id"]),
                config,
            )
            scores.append(
                {
                    "judge_run_id": run["run_id"],
                    "provider": result.get("provider", run["provider"]),
                    "model": result.get("model", run["model"]),
                    "prompt_id": config["prompt_id"],
                    "repeat_index": repeat_index,
                    "longform_id": row.get("longform_id", ""),
                    **normalized,
                }
            )
            usage.append(
                {
                    "item_id": f"{row['candidate_id']}#R{repeat_index}",
                    "candidate_id": row["candidate_id"],
                    "repeat_index": repeat_index,
                    "cache_hit": True,
                    "usage": result.get("usage", {}),
                }
            )
    scores.sort(
        key=lambda row: (int(row["repeat_index"]), row["candidate_id"])
    )
    usage.sort(
        key=lambda row: (int(row["repeat_index"]), row["candidate_id"])
    )
    missing.sort(
        key=lambda row: (int(row["repeat_index"]), row["candidate_id"])
    )
    return scores, usage, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover completed Shortform Judge v9 responses from cache."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat-count", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    prompt = (
        ROOT / "prompts" / f"{config['prompt_id']}.md"
    ).read_text(encoding="utf-8")
    candidates = read_jsonl(args.input)
    scores, usage, missing = recover(
        candidates,
        config,
        prompt,
        args.output_dir / "cache",
        args.repeat_count,
    )
    write_csv(
        args.output_dir / "shortform_judge_v9_scores_partial.csv",
        scores,
    )
    write_jsonl(
        args.output_dir / "shortform_judge_v9_usage_partial.jsonl",
        usage,
    )
    write_jsonl(
        args.output_dir / "shortform_judge_v9_missing_requests.jsonl",
        missing,
    )
    summary = {
        "candidate_count": len(candidates),
        "expected_request_count": len(candidates) * args.repeat_count,
        "recovered_request_count": len(scores),
        "missing_request_count": len(missing),
        "repeat_counts": {
            str(repeat_index): sum(
                int(row["repeat_index"]) == repeat_index for row in scores
            )
            for repeat_index in range(1, args.repeat_count + 1)
        },
        "status": "complete" if not missing else "partial",
    }
    (args.output_dir / "recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
