from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from language_utils import choose_prompt_id_by_language_and_genre, detect_content_genre, detect_prompt_language
from llm_client import LLMError, call_llm
from vpick_client import save_json


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "llm_rerank_top5"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompt(prompt_id: str) -> str:
    return (ROOT / "prompts" / f"{prompt_id}.md").read_text(encoding="utf-8")


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get(key, ""), []).append(row)
    return grouped


def approximate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def compact_candidate(row: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "rank": int(float(row.get("rank", "999"))),
        "start_sec": float(row["start_sec"]),
        "end_sec": float(row["end_sec"]),
        "duration_sec": float(row["duration_sec"]),
        "context_start_sec": float(row.get("context_start_sec") or row["start_sec"]),
        "context_end_sec": float(row.get("context_end_sec") or row["end_sec"]),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "context_start_time": row.get("context_start_time", ""),
        "context_end_time": row.get("context_end_time", ""),
        "source_rank": row.get("source_rank", ""),
        "rerank_score": row.get("rerank_score", ""),
        "description": row.get("description", "")[:1200],
        "transcript": row.get("transcript", "")[:3500],
        "context_transcript": row.get("context_transcript", "")[:5000],
    }


def make_user_prompt(long_video_id: str, candidates: list[dict[str, str]], language: str, genre: str, top_k: int) -> str:
    payload = {
        "task": "rerank_shortform_candidates",
        "long_video_id": long_video_id,
        "detected_language": language,
        "detected_genre": genre,
        "input_candidate_count": len(candidates),
        "output_top_k": top_k,
        "candidates": [compact_candidate(row) for row in candidates],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_choices(response_json: dict[str, Any], valid_ids: set[str], top_k: int) -> list[dict[str, Any]]:
    raw_choices = response_json.get("choices") or response_json.get("ranked_candidates") or response_json.get("candidates") or []
    choices: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(raw_choices, list):
        return []
    for idx, item in enumerate(raw_choices, start=1):
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id not in valid_ids or candidate_id in seen:
            continue
        choices.append(
            {
                "rank": int(item.get("rank") or idx),
                "candidate_id": candidate_id,
                "score": item.get("score", item.get("confidence", "")),
                "reason": str(item.get("reason", ""))[:800],
                "suggested_start_sec": item.get("suggested_start_sec", ""),
                "suggested_end_sec": item.get("suggested_end_sec", ""),
            }
        )
        seen.add(candidate_id)
        if len(choices) >= top_k:
            break
    choices.sort(key=lambda item: int(item["rank"]))
    for rank, item in enumerate(choices, start=1):
        item["rank"] = rank
    return choices


def dry_run_choices(candidates: list[dict[str, str]], top_k: int) -> list[dict[str, Any]]:
    return [
        {
            "rank": idx,
            "candidate_id": row["candidate_id"],
            "score": row.get("rerank_score", ""),
            "reason": "dry_run_kept_deterministic_order",
            "suggested_start_sec": row["start_sec"],
            "suggested_end_sec": row["end_sec"],
        }
        for idx, row in enumerate(candidates[:top_k], start=1)
    ]


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def adjusted_interval(choice: dict[str, Any], candidate: dict[str, str]) -> tuple[float, float, bool]:
    original_start = float(candidate["start_sec"])
    original_end = float(candidate["end_sec"])
    suggested_start = to_float(choice.get("suggested_start_sec"))
    suggested_end = to_float(choice.get("suggested_end_sec"))
    if suggested_start is None or suggested_end is None:
        return original_start, original_end, False

    context_start = float(candidate.get("context_start_sec") or original_start)
    context_end = float(candidate.get("context_end_sec") or original_end)
    start = max(context_start, min(context_end, suggested_start))
    end = max(context_start, min(context_end, suggested_end))
    duration = end - start
    if duration < 15.0 or duration > 90.0:
        return original_start, original_end, False
    adjusted = abs(start - original_start) > 0.001 or abs(end - original_end) > 0.001
    return round(start, 3), round(end, 3), adjusted


def choices_to_prediction_rows(
    choices: list[dict[str, Any]],
    candidates_by_id: dict[str, dict[str, str]],
    pair_rows: list[dict[str, str]],
    run: dict[str, Any],
    prompt_id: str,
    language: str,
    genre: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for choice in choices:
        candidate = candidates_by_id[str(choice["candidate_id"])]
        pred_start, pred_end, adjusted = adjusted_interval(choice, candidate)
        for pair in pair_rows:
            output.append(
                {
                    "pair_id": pair["pair_id"],
                    "long_video_id": pair.get("long_video_id", ""),
                    "short_video_id": pair.get("short_video_id", ""),
                    "run_id": f"{candidate.get('source_run_id', 'longform_slate')}__llm_top5_{run['run_id']}",
                    "selector_type": "llm_top5_longform_reranker",
                    "prompt_id": prompt_id,
                    "model_name": run["model"],
                    "rank": int(choice["rank"]),
                    "pred_start_sec": pred_start,
                    "pred_end_sec": pred_end,
                    "selected_scene_ids": candidate.get("selected_scene_ids", ""),
                    "confidence": choice.get("score", ""),
                    "notes": (
                        f"candidate_id={candidate['candidate_id']}; detected_language={language}; "
                        f"detected_genre={genre}; original_start={candidate['start_sec']}; "
                        f"original_end={candidate['end_sec']}; adjusted={adjusted}; {choice.get('reason', '')}"
                    ),
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerank one-longform-to-many-shortform top5 slates with a genre-routed LLM prompt.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--slate", required=True)
    parser.add_argument("--runs", default=str(ROOT / "config" / "llm_rerank_top5_genre_lang.json"))
    parser.add_argument("--output", default=str(OUT_DIR / "predictions.csv"))
    parser.add_argument("--artifact-dir", default=str(OUT_DIR))
    parser.add_argument("--input-top-k", type=int, default=5)
    parser.add_argument("--output-top-k", type=int, default=5)
    parser.add_argument("--provider", action="append")
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_by_long = group_rows(read_csv(Path(args.dataset)), "long_video_id")
    slate_by_long = group_rows(read_csv(Path(args.slate)), "long_video_id")
    config = load_json(Path(args.runs))
    providers = {item.lower() for item in args.provider or []}
    run_ids = set(args.run_id or [])
    runs = [
        run
        for run in config.get("runs", [])
        if (not providers or str(run.get("provider", "")).lower() in providers)
        and (not run_ids or str(run.get("run_id", "")) in run_ids)
    ]
    if not runs:
        raise SystemExit("No LLM reranker runs selected.")

    artifact_dir = Path(args.artifact_dir)
    prompt_dir = artifact_dir / "prompts"
    raw_dir = artifact_dir / "raw_responses"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []

    for long_video_id, pair_rows in sorted(dataset_by_long.items()):
        candidates = sorted(
            slate_by_long.get(long_video_id, []),
            key=lambda row: int(float(row.get("rank", "999"))),
        )[: args.input_top_k]
        if not candidates:
            continue
        candidate_texts = [f"{row.get('description', '')}\n{row.get('transcript', '')}" for row in candidates]
        language = detect_prompt_language(candidate_texts, default="ko")
        genre = detect_content_genre(candidate_texts, default="variety_vlog")
        prompt_id = choose_prompt_id_by_language_and_genre(
            config.get("prompt_id", "highlight_reranker_v1_ko_variety_vlog"),
            config.get("prompt_id_by_language_and_genre"),
            language,
            genre,
        )
        system_prompt = load_prompt(prompt_id)
        user_prompt = make_user_prompt(long_video_id, candidates, language, genre, args.output_top_k)
        safe_id = hashlib.sha1(f"{long_video_id}:{prompt_id}".encode("utf-8")).hexdigest()[:16]
        (prompt_dir / f"{safe_id}_{long_video_id}_{prompt_id}.json").write_text(user_prompt, encoding="utf-8")
        candidates_by_id = {row["candidate_id"]: row for row in candidates}
        valid_ids = set(candidates_by_id)

        for run in runs:
            provider = str(run["provider"]).lower()
            model = str(run["model"])
            if args.dry_run:
                response_json = {"choices": dry_run_choices(candidates, args.output_top_k)}
                raw = {"dry_run": True, "json": response_json, "usage": {}}
            else:
                try:
                    raw = call_llm(provider, model, system_prompt, user_prompt, max_tokens=args.max_tokens)
                except LLMError as exc:
                    raise SystemExit(f"{run['run_id']} rerank failed for long_video_id={long_video_id}: {exc}") from exc
                response_json = raw["json"]
            choices = normalize_choices(response_json, valid_ids, args.output_top_k)
            save_json(raw_dir / f"{safe_id}_{long_video_id}_{run['run_id']}_{prompt_id}.json", raw)
            prediction_rows.extend(
                choices_to_prediction_rows(choices, candidates_by_id, pair_rows, run, prompt_id, language, genre)
            )
            usage_rows.append(
                {
                    "long_video_id": long_video_id,
                    "run_id": run["run_id"],
                    "provider": provider,
                    "model": model,
                    "prompt_id": prompt_id,
                    "detected_language": language,
                    "detected_genre": genre,
                    "input_candidate_count": len(candidates),
                    "input_tokens_est": approximate_tokens(system_prompt + "\n" + user_prompt),
                    "output_tokens_est": args.max_tokens,
                    "usage_json": json.dumps(raw.get("usage", {}), ensure_ascii=False),
                }
            )

    write_csv(Path(args.output), prediction_rows)
    write_csv(artifact_dir / "usage.csv", usage_rows)
    print(json.dumps({"predictions": args.output, "rows": len(prediction_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
