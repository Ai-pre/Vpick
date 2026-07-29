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


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIMENSIONS = (
    "opening_strength",
    "standalone",
    "completeness",
    "engagement_value",
    "boundary_naturalness",
    "titleability",
)
EVIDENCE_DIMENSIONS = (
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
)
SET_DIMENSIONS = (
    "redundancy_control",
    "event_diversity",
    "timeline_coverage",
    "portfolio_quality",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def load_prompt(prompt_id: str) -> str:
    return (ROOT / "prompts" / f"{prompt_id}.md").read_text(encoding="utf-8")


def clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise LLMError(f"Invalid judge score: {value!r}") from exc
    if not 1 <= score <= 5:
        raise LLMError(f"Judge score must be between 1 and 5: {score}")
    return score


def weighted_score(scores: dict[str, int], weights: dict[str, Any], dimensions: tuple[str, ...]) -> float:
    def weight_for(name: str) -> float:
        if name == "opening_strength" and name not in weights:
            return float(weights.get("hook_clarity", 0.0))
        return float(weights.get(name, 0.0))

    total_weight = sum(weight_for(name) for name in dimensions)
    if total_weight <= 0:
        raise ValueError("Judge weights must sum to a positive number.")
    weighted = sum(weight_for(name) * ((scores[name] - 1.0) / 4.0) for name in dimensions)
    return round(100.0 * weighted / total_weight, 3)


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


def normalize_candidate_judgments(
    response: dict[str, Any], valid_ids: set[str], weights: dict[str, Any], require_all: bool = True
) -> list[dict[str, Any]]:
    raw = response.get("judgments")
    if not isinstance(raw, list):
        raise LLMError("Candidate judge response must contain a judgments list.")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id not in valid_ids or candidate_id in seen:
            continue
        verdict = str(item.get("verdict", "score")).strip().lower()
        if verdict not in {"score", "abstain"}:
            raise LLMError(f"Invalid verdict for candidate {candidate_id}: {verdict!r}")
        evidence_obj = item.get("evidence")
        evidence = {
            name: clamp_score(evidence_obj.get(name)) if isinstance(evidence_obj, dict) and evidence_obj.get(name) is not None else ""
            for name in EVIDENCE_DIMENSIONS
        }
        confidence = clamp_score(item.get("confidence", 3))
        flags = item.get("failure_flags", [])
        if not isinstance(flags, list):
            flags = []
        if verdict == "abstain":
            normalized_flags = [str(flag) for flag in flags[:8]]
            if "insufficient_evidence" not in normalized_flags:
                normalized_flags.append("insufficient_evidence")
            output.append(
                {
                    "candidate_id": candidate_id,
                    "verdict": verdict,
                    **evidence,
                    **{name: "" for name in CANDIDATE_DIMENSIONS},
                    "confidence": confidence,
                    "overall_score": "",
                    "failure_flags": "|".join(normalized_flags),
                    "reason": str(item.get("reason", ""))[:1000],
                }
            )
            seen.add(candidate_id)
            continue
        score_obj = item.get("scores")
        if not isinstance(score_obj, dict):
            raise LLMError(f"Missing scores for candidate {candidate_id}")
        scores = {
            name: clamp_score(score_obj.get(name, score_obj.get("hook_clarity") if name == "opening_strength" else None))
            for name in CANDIDATE_DIMENSIONS
        }
        output.append(
            {
                "candidate_id": candidate_id,
                "verdict": verdict,
                **evidence,
                **scores,
                "confidence": confidence,
                "overall_score": weighted_score(scores, weights, CANDIDATE_DIMENSIONS),
                "failure_flags": "|".join(str(flag) for flag in flags[:8]),
                "reason": str(item.get("reason", ""))[:1000],
            }
        )
        seen.add(candidate_id)
    if require_all and seen != valid_ids:
        missing = sorted(valid_ids - seen)
        raise LLMError(f"Candidate judge omitted IDs: {missing}")
    return output


def normalize_set_judgment(
    response: dict[str, Any], expected_set_id: str, weights: dict[str, Any]
) -> dict[str, Any]:
    raw = response.get("set_judgment")
    if not isinstance(raw, dict):
        raise LLMError("Set judge response must contain set_judgment.")
    returned_id = str(raw.get("set_id", "")).strip()
    if returned_id != expected_set_id:
        raise LLMError(f"Set judge returned unexpected set_id={returned_id!r}")
    score_obj = raw.get("scores")
    if not isinstance(score_obj, dict):
        raise LLMError("Set judgment is missing scores.")
    scores = {name: clamp_score(score_obj.get(name)) for name in SET_DIMENSIONS}
    return {
        "set_id": expected_set_id,
        **scores,
        "overall_set_score": weighted_score(scores, weights, SET_DIMENSIONS),
        "reason": str(raw.get("reason", ""))[:1000],
    }


def call_with_retry(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    attempts: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call_llm(provider, model, system_prompt, user_prompt, max_tokens=max_tokens)
        except (LLMError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise LLMError(f"Judge call failed after {attempts} attempts: {last_error}")


def cached_call(
    *,
    cache_dir: Path,
    cache_key: str,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    dry_run: bool,
    dry_response: dict[str, Any],
    use_cache: bool,
) -> dict[str, Any]:
    digest = hashlib.sha256(f"{cache_key}\n{system_prompt}\n{user_prompt}".encode("utf-8")).hexdigest()[:20]
    path = cache_dir / f"{digest}.json"
    if use_cache and path.exists():
        return load_json(path)
    if dry_run:
        result = {
            "provider": provider,
            "model": model,
            "json": dry_response,
            "usage": {},
            "cached": False,
            "dry_run": True,
        }
    else:
        result = call_with_retry(provider, model, system_prompt, user_prompt, max_tokens)
        result["cached"] = False
        result["dry_run"] = False
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def batches(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def build_sets(
    candidates: dict[str, dict[str, str]], sources: list[dict[str, str]], set_sources: set[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sources:
        source_system = row.get("source_system", "")
        if source_system == "gold" or (set_sources and source_system not in set_sources):
            continue
        key = (source_system, row.get("long_video_id", ""))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (source_system, long_video_id), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(float(row.get("source_rank") or 999)))
        candidate_ids: list[str] = []
        for row in ordered:
            candidate_id = row.get("candidate_id", "")
            if candidate_id in candidates and candidate_id not in candidate_ids:
                candidate_ids.append(candidate_id)
        if len(candidate_ids) < 2:
            continue
        raw = f"judge-set-v1|{source_system}|{long_video_id}|{'|'.join(candidate_ids)}"
        set_id = f"S_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:14]}"
        output.append(
            {
                "set_id": set_id,
                "source_system": source_system,
                "long_video_id": long_video_id,
                "candidate_ids": candidate_ids,
            }
        )
    return output


def run_candidate_mode(
    *,
    candidates: list[dict[str, str]],
    runs: list[dict[str, Any]],
    prompt_id: str,
    weights: dict[str, Any],
    output_dir: Path,
    repeat_count: int,
    batch_size: int,
    max_tokens: int,
    dry_run: bool,
    use_cache: bool,
    workers: int,
) -> list[dict[str, Any]]:
    prompt = load_prompt(prompt_id)
    output: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    tasks: list[tuple[dict[str, Any], int, int, list[dict[str, str]]]] = []
    for run in runs:
        for repeat_index in range(1, repeat_count + 1):
            shuffled = list(candidates)
            random.Random(f"{run['run_id']}:{repeat_index}:candidate").shuffle(shuffled)
            for batch_index, batch in enumerate(batches(shuffled, batch_size), start=1):
                tasks.append((run, repeat_index, batch_index, batch))

    def process_task(task: tuple[dict[str, Any], int, int, list[dict[str, str]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        run, repeat_index, batch_index, batch = task
        payload = {
            "task": "score_fixed_shortform_candidates",
            "rubric_version": prompt_id,
            "candidates": [candidate_payload(row) for row in batch],
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        dry_response = {
            "judgments": [
                {
                    "candidate_id": row["candidate_id"],
                    "verdict": "score",
                    "evidence": {name: 3 for name in EVIDENCE_DIMENSIONS},
                    "scores": {name: 3 for name in CANDIDATE_DIMENSIONS},
                    "confidence": 3,
                    "failure_flags": [],
                    "reason": "dry_run",
                }
                for row in batch
            ]
        }
        result = cached_call(
            cache_dir=output_dir / "raw_responses" / "candidate",
            cache_key=f"candidate|{run['run_id']}|{repeat_index}|{batch_index}",
            provider=str(run["provider"]),
            model=str(run["model"]),
            system_prompt=prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            dry_run=dry_run,
            dry_response=dry_response,
            use_cache=use_cache,
        )
        valid_ids = {row["candidate_id"] for row in batch}
        try:
            judgments = normalize_candidate_judgments(result["json"], valid_ids, weights, require_all=False)
        except LLMError:
            judgments = []
        returned_ids = {row["candidate_id"] for row in judgments}
        for missing_row in [row for row in batch if row["candidate_id"] not in returned_ids]:
            repair_payload = {
                "task": "score_fixed_shortform_candidates",
                "rubric_version": prompt_id,
                "candidates": [candidate_payload(missing_row)],
            }
            repair_dry_response = {
                "judgments": [
                    {
                        "candidate_id": missing_row["candidate_id"],
                        "verdict": "score",
                        "evidence": {name: 3 for name in EVIDENCE_DIMENSIONS},
                        "scores": {name: 3 for name in CANDIDATE_DIMENSIONS},
                        "confidence": 3,
                        "failure_flags": [],
                        "reason": "dry_run_repair",
                    }
                ]
            }
            repair_result = cached_call(
                cache_dir=output_dir / "raw_responses" / "candidate_repair",
                cache_key=f"candidate-repair|{run['run_id']}|{repeat_index}|{batch_index}|{missing_row['candidate_id']}",
                provider=str(run["provider"]),
                model=str(run["model"]),
                system_prompt=prompt,
                user_prompt=json.dumps(repair_payload, ensure_ascii=False, indent=2),
                max_tokens=max_tokens,
                dry_run=dry_run,
                dry_response=repair_dry_response,
                use_cache=use_cache,
            )
            judgments.extend(
                normalize_candidate_judgments(
                    repair_result["json"], {missing_row["candidate_id"]}, weights, require_all=True
                )
            )
        by_id = {row["candidate_id"]: row for row in batch}
        rows = []
        for judgment in judgments:
            candidate = by_id[judgment["candidate_id"]]
            rows.append(
                {
                    "judge_run_id": run["run_id"],
                    "provider": run["provider"],
                    "model": run["model"],
                    "prompt_id": prompt_id,
                    "repeat_index": repeat_index,
                    "candidate_id": judgment["candidate_id"],
                    "long_video_id": candidate["long_video_id"],
                    "verdict": judgment["verdict"],
                    **{name: judgment[name] for name in EVIDENCE_DIMENSIONS},
                    **{name: judgment[name] for name in CANDIDATE_DIMENSIONS},
                    "confidence": judgment["confidence"],
                    "overall_score": judgment["overall_score"],
                    "failure_flags": judgment["failure_flags"],
                    "reason": judgment["reason"],
                }
            )
        usage = {
            "mode": "candidate",
            "judge_run_id": run["run_id"],
            "repeat_index": repeat_index,
            "batch_index": batch_index,
            "item_count": len(batch),
            "usage_json": json.dumps(result.get("usage", {}), ensure_ascii=False),
            "dry_run": bool(result.get("dry_run")),
        }
        return rows, usage

    random.Random("candidate-task-order-v1").shuffle(tasks)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for future in as_completed(futures):
            rows, usage = future.result()
            output.extend(rows)
            usage_rows.append(usage)
    output.sort(key=lambda row: (row["judge_run_id"], row["repeat_index"], row["candidate_id"]))
    usage_rows.sort(key=lambda row: (row["judge_run_id"], row["repeat_index"], row["batch_index"]))
    write_csv(
        output_dir / "candidate_judge_usage.csv",
        usage_rows,
        ["mode", "judge_run_id", "repeat_index", "batch_index", "item_count", "usage_json", "dry_run"],
    )
    return output


def run_set_mode(
    *,
    candidates_by_id: dict[str, dict[str, str]],
    sets: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    prompt_id: str,
    weights: dict[str, Any],
    output_dir: Path,
    repeat_count: int,
    max_tokens: int,
    dry_run: bool,
    use_cache: bool,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = load_prompt(prompt_id)
    output: list[dict[str, Any]] = []
    private_manifest: list[dict[str, Any]] = []
    for set_item in sets:
        for rank, candidate_id in enumerate(set_item["candidate_ids"], start=1):
            private_manifest.append(
                {
                    "set_id": set_item["set_id"],
                    "source_system": set_item["source_system"],
                    "long_video_id": set_item["long_video_id"],
                    "rank": rank,
                    "candidate_id": candidate_id,
                }
            )
    tasks: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    for run in runs:
        for repeat_index in range(1, repeat_count + 1):
            ordered_sets = list(sets)
            random.Random(f"{run['run_id']}:{repeat_index}:set").shuffle(ordered_sets)
            for set_item in ordered_sets:
                tasks.append((run, repeat_index, set_item))

    def process_set_task(task: tuple[dict[str, Any], int, dict[str, Any]]) -> dict[str, Any]:
        run, repeat_index, set_item = task
        rows = [candidates_by_id[candidate_id] for candidate_id in set_item["candidate_ids"]]
        long_duration = max(float(row.get("long_duration_sec") or 0.0) for row in rows)
        payload = {
            "task": "score_fixed_topk_set",
            "rubric_version": prompt_id,
            "set_id": set_item["set_id"],
            "long_duration_sec": long_duration,
            "candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "duration_sec": float(row["duration_sec"]),
                    "genre": row.get("genre", "general"),
                    "description": row.get("description", "")[:1200],
                    "transcript": row.get("transcript", "")[:2800],
                }
                for row in rows
            ],
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        dry_response = {
            "set_judgment": {
                "set_id": set_item["set_id"],
                "scores": {name: 3 for name in SET_DIMENSIONS},
                "reason": "dry_run",
            }
        }
        result = cached_call(
            cache_dir=output_dir / "raw_responses" / "set",
            cache_key=f"set|{run['run_id']}|{repeat_index}|{set_item['set_id']}",
            provider=str(run["provider"]),
            model=str(run["model"]),
            system_prompt=prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            dry_run=dry_run,
            dry_response=dry_response,
            use_cache=use_cache,
        )
        try:
            judgment = normalize_set_judgment(result["json"], set_item["set_id"], weights)
        except LLMError:
            repair_result = cached_call(
                cache_dir=output_dir / "raw_responses" / "set_repair",
                cache_key=f"set-repair|{run['run_id']}|{repeat_index}|{set_item['set_id']}",
                provider=str(run["provider"]),
                model=str(run["model"]),
                system_prompt=prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                dry_run=dry_run,
                dry_response=dry_response,
                use_cache=use_cache,
            )
            judgment = normalize_set_judgment(repair_result["json"], set_item["set_id"], weights)
        return {
            "judge_run_id": run["run_id"],
            "provider": run["provider"],
            "model": run["model"],
            "prompt_id": prompt_id,
            "repeat_index": repeat_index,
            "set_id": set_item["set_id"],
            "source_system": set_item["source_system"],
            "long_video_id": set_item["long_video_id"],
            "candidate_count": len(rows),
            **{name: judgment[name] for name in SET_DIMENSIONS},
            "overall_set_score": judgment["overall_set_score"],
            "reason": judgment["reason"],
        }

    random.Random("set-task-order-v1").shuffle(tasks)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process_set_task, task) for task in tasks]
        for future in as_completed(futures):
            output.append(future.result())
    output.sort(key=lambda row: (row["judge_run_id"], row["repeat_index"], row["set_id"]))
    return output, private_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run blind candidate and Top-K LLM-as-a-Judge evaluation.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sources", required=True, help="Private source manifest from build_judge_candidates.py.")
    parser.add_argument("--config", default=str(ROOT / "config" / "llm_judge_v2.json"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=("candidate", "set", "both"), default="both")
    parser.add_argument("--provider", action="append")
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--set-source", action="append", default=[])
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    providers = {value.lower() for value in args.provider or []}
    run_ids = set(args.run_id or [])
    runs = [
        run
        for run in config.get("runs", [])
        if (not providers or str(run.get("provider", "")).lower() in providers)
        and (not run_ids or str(run.get("run_id", "")) in run_ids)
    ]
    if not runs:
        raise SystemExit("No judge runs selected.")

    candidates = read_csv(Path(args.candidates))
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    sources = [row for row in read_csv(Path(args.sources)) if row.get("candidate_id") in candidates_by_id]
    out_dir = Path(args.out_dir)

    candidate_scores: list[dict[str, Any]] = []
    if args.mode in {"candidate", "both"}:
        candidate_scores = run_candidate_mode(
            candidates=candidates,
            runs=runs,
            prompt_id=str(config["candidate_prompt_id"]),
            weights=dict(config["candidate_weights"]),
            output_dir=out_dir,
            repeat_count=max(1, args.repeat_count),
            batch_size=max(1, args.batch_size),
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
            use_cache=not args.no_cache,
            workers=max(1, args.workers),
        )
        write_csv(
            out_dir / "candidate_judge_scores.csv",
            candidate_scores,
            [
                "judge_run_id", "provider", "model", "prompt_id", "repeat_index", "candidate_id", "long_video_id",
                "verdict", *EVIDENCE_DIMENSIONS, *CANDIDATE_DIMENSIONS, "confidence", "overall_score", "failure_flags", "reason",
            ],
        )

    set_scores: list[dict[str, Any]] = []
    set_manifest: list[dict[str, Any]] = []
    if args.mode in {"set", "both"}:
        sets = build_sets(candidates_by_id, sources, set(args.set_source))
        set_scores, set_manifest = run_set_mode(
            candidates_by_id=candidates_by_id,
            sets=sets,
            runs=runs,
            prompt_id=str(config["set_prompt_id"]),
            weights=dict(config["set_weights"]),
            output_dir=out_dir,
            repeat_count=max(1, args.repeat_count),
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
            use_cache=not args.no_cache,
            workers=max(1, args.workers),
        )
        write_csv(
            out_dir / "set_judge_scores.csv",
            set_scores,
            [
                "judge_run_id", "provider", "model", "prompt_id", "repeat_index", "set_id", "source_system",
                "long_video_id", "candidate_count", *SET_DIMENSIONS, "overall_set_score", "reason",
            ],
        )
        write_csv(
            out_dir / "set_sources_private.csv",
            set_manifest,
            ["set_id", "source_system", "long_video_id", "rank", "candidate_id"],
        )

    summary = {
        "judge_runs": [str(run["run_id"]) for run in runs],
        "repeat_count": max(1, args.repeat_count),
        "candidate_score_rows": len(candidate_scores),
        "set_score_rows": len(set_scores),
        "dry_run": args.dry_run,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
