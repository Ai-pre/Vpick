from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_client import LLMError, call_llm


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIMENSIONS = (
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
    "visual_dependency",
)
EDITORIAL_DIMENSIONS = (
    "context_clarity",
    "event_progression",
    "completeness",
    "boundary_naturalness",
    "content_density",
    "standalone",
)
PERFORMANCE_DIMENSIONS = (
    "emotional_intensity",
    "change_or_surprise",
    "specificity_novelty",
    "relatability_shareability",
    "payoff_strength",
    "hook_title_potential",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score(value: Any) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise LLMError(f"Invalid 1-5 score: {value!r}") from exc
    if not 1 <= parsed <= 5:
        raise LLMError(f"Score outside 1-5: {parsed}")
    return parsed


def score_map(raw: Any, dimensions: tuple[str, ...]) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise LLMError("Expected a score object")
    return {name: score(raw.get(name)) for name in dimensions}


def weighted_score(values: dict[str, int], weights: dict[str, Any]) -> float:
    total_weight = sum(float(weights.get(name, 0.0)) for name in values)
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive number")
    value = sum(float(weights.get(name, 0.0)) * ((item - 1.0) / 4.0) for name, item in values.items())
    return round(100.0 * value / total_weight, 3)


def preference(value: Any) -> str:
    normalized = str(value or "tie").strip().lower()
    aliases = {"l": "left", "r": "right", "왼쪽": "left", "오른쪽": "right", "동점": "tie"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"left", "right", "tie"}:
        raise LLMError(f"Invalid preference: {value!r}")
    return normalized


def flip_preference(value: str) -> str:
    return {"left": "right", "right": "left", "tie": "tie"}[value]


def normalize_response(
    raw: dict[str, Any],
    comparison_id: str,
    editorial_weights: dict[str, Any],
    performance_weights: dict[str, Any],
    swapped: bool,
) -> dict[str, Any]:
    response_id = str(raw.get("comparison_id", comparison_id))
    if response_id != comparison_id:
        raise LLMError(f"Unexpected comparison_id: {response_id}")
    verdict = str(raw.get("verdict", "score")).strip().lower()
    if verdict not in {"score", "abstain"}:
        raise LLMError(f"Invalid verdict: {verdict}")

    side_values: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        side_raw = raw.get(side)
        if not isinstance(side_raw, dict):
            raise LLMError(f"Missing {side} object")
        if verdict == "abstain" and not isinstance(side_raw.get("evidence"), dict):
            evidence: dict[str, int | str] = {name: "" for name in EVIDENCE_DIMENSIONS}
        else:
            evidence = score_map(side_raw.get("evidence"), EVIDENCE_DIMENSIONS)
        if verdict == "abstain":
            editorial: dict[str, int] = {}
            performance: dict[str, int] = {}
        else:
            editorial = score_map(side_raw.get("editorial"), EDITORIAL_DIMENSIONS)
            performance = score_map(side_raw.get("performance"), PERFORMANCE_DIMENSIONS)
        side_values[side] = {
            "evidence": evidence,
            "editorial": editorial,
            "performance": performance,
            "editorial_score": weighted_score(editorial, editorial_weights) if editorial else "",
            "performance_score": weighted_score(performance, performance_weights) if performance else "",
        }

    editorial_preference = preference(raw.get("editorial_preference"))
    performance_preference = preference(raw.get("performance_preference"))
    if swapped:
        side_values = {"left": side_values["right"], "right": side_values["left"]}
        editorial_preference = flip_preference(editorial_preference)
        performance_preference = flip_preference(performance_preference)
    if verdict == "abstain":
        editorial_preference = "tie"
        performance_preference = "tie"

    output: dict[str, Any] = {
        "comparison_id": comparison_id,
        "verdict": verdict,
        "editorial_preference": editorial_preference,
        "performance_preference": performance_preference,
        "confidence": score(raw.get("confidence", 1)),
        "failure_flags": "|".join(str(value) for value in (raw.get("failure_flags") or [])[:8]),
        "reason": str(raw.get("reason", ""))[:1200],
    }
    for side in ("left", "right"):
        for name, value in side_values[side]["evidence"].items():
            output[f"{side}_evidence_{name}"] = value
        for name in EDITORIAL_DIMENSIONS:
            output[f"{side}_editorial_{name}"] = side_values[side]["editorial"].get(name, "")
        for name in PERFORMANCE_DIMENSIONS:
            output[f"{side}_performance_{name}"] = side_values[side]["performance"].get(name, "")
        output[f"{side}_editorial_score"] = side_values[side]["editorial_score"]
        output[f"{side}_performance_score"] = side_values[side]["performance_score"]
    return output


def cached_call(
    *, cache_file: Path, provider: str, model: str, system_prompt: str, user_prompt: str,
    max_tokens: int, dry_run: bool, use_cache: bool, comparison_id: str,
) -> dict[str, Any]:
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    if dry_run:
        side = {
            "evidence": {name: 3 for name in EVIDENCE_DIMENSIONS},
            "editorial": {name: 3 for name in EDITORIAL_DIMENSIONS},
            "performance": {name: 3 for name in PERFORMANCE_DIMENSIONS},
        }
        result = {
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
    else:
        result = call_llm(provider, model, system_prompt, user_prompt, max_tokens=max_tokens)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched Gold pairwise LLM judges.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    pairs = read_jsonl(Path(args.pairs))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    selected_ids = set(args.run_id or [])
    runs = [run for run in config["runs"] if not selected_ids or run["run_id"] in selected_ids]
    if not runs:
        raise SystemExit("No judge runs selected")
    prompt_id = str(config["pairwise_prompt_id"])
    system_prompt = (ROOT / "prompts" / f"{prompt_id}.md").read_text(encoding="utf-8")
    output_dir = Path(args.out_dir)

    tasks = [
        (run, repeat_index, pair)
        for run in runs
        for repeat_index in range(1, max(1, args.repeat_count) + 1)
        for pair in pairs
    ]

    def process(task: tuple[dict[str, Any], int, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        run, repeat_index, pair = task
        swapped = repeat_index % 2 == 0
        presented = {
            "task": "compare_fixed_shortform_candidates",
            "rubric_version": prompt_id,
            "comparison_id": pair["comparison_id"],
            "same_channel": pair.get("same_channel", True),
            "left": pair["right"] if swapped else pair["left"],
            "right": pair["left"] if swapped else pair["right"],
        }
        cache_key = hashlib.sha256(
            f"{run['run_id']}|{repeat_index}|{pair['comparison_id']}".encode("utf-8")
        ).hexdigest()[:24]
        result = cached_call(
            cache_file=output_dir / "raw_responses" / str(run["run_id"]) / f"{cache_key}.json",
            provider=str(run["provider"]),
            model=str(run["model"]),
            system_prompt=system_prompt,
            user_prompt=json.dumps(presented, ensure_ascii=False, indent=2),
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
            use_cache=not args.no_cache,
            comparison_id=str(pair["comparison_id"]),
        )
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
            "prompt_id": prompt_id,
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
    rows.sort(key=lambda row: (row["judge_run_id"], int(row["repeat_index"]), row["comparison_id"]))

    fields = [
        "judge_run_id", "provider", "model", "prompt_id", "repeat_index", "presentation_swapped", "dry_run",
        "comparison_id", "verdict", "editorial_preference", "performance_preference", "confidence",
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
        "run_ids": [run["run_id"] for run in runs],
        "comparison_count": len(pairs),
        "repeat_count": max(1, args.repeat_count),
        "score_row_count": len(rows),
        "dry_run": args.dry_run,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
