from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Each candidate JSONL row must be an object")
                rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["pair_id", "rank"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any, field: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert anonymous Judge scores into ranked segment predictions."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-field", default="score")
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Rank smaller score values first, for source ranks or losses.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selector-type", default="judge_guided_rerank")
    parser.add_argument("--prompt-id", default="shortform_judge_v10_plus_v14")
    parser.add_argument("--model-name", default="pointwise_v10_plus_v14")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    candidates = read_jsonl(args.candidates)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    by_longform: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        candidate_id = str(row.get("candidate_id", "")).strip()
        longform_id = str(row.get("longform_id", "")).strip()
        if not candidate_id or not longform_id or candidate_id in candidate_by_id:
            raise ValueError(
                "Candidate IDs must be unique and longform_id must be present"
            )
        candidate_by_id[candidate_id] = row
        by_longform[longform_id].append(candidate_id)

    scores: dict[str, float] = {}
    for row in read_csv(args.scores):
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id in scores:
            raise ValueError("Score candidate IDs must be complete and unique")
        if str(row.get("verdict", "score")).lower() == "abstain":
            continue
        scores[candidate_id] = number(
            row.get(args.score_field),
            args.score_field,
        )
    if set(scores) != set(candidate_by_id):
        raise ValueError(
            "Candidate and score sets differ: "
            f"missing={sorted(set(candidate_by_id) - set(scores))[:5]}, "
            f"unexpected={sorted(set(scores) - set(candidate_by_id))[:5]}"
        )

    ranked_by_longform: dict[str, list[str]] = {}
    for longform_id, candidate_ids in by_longform.items():
        direction = 1.0 if args.lower_is_better else -1.0
        ranked_by_longform[longform_id] = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                direction * scores[candidate_id],
                candidate_id,
            ),
        )[: args.top_k]

    output: list[dict[str, Any]] = []
    for gold in read_csv(args.dataset):
        longform_id = str(gold.get("long_video_id", "")).strip()
        for rank, candidate_id in enumerate(
            ranked_by_longform.get(longform_id, []),
            start=1,
        ):
            candidate = candidate_by_id[candidate_id]
            scene_ids = candidate.get("scene_ids") or []
            output.append(
                {
                    "pair_id": gold["pair_id"],
                    "long_video_id": longform_id,
                    "short_video_id": gold.get("short_video_id", ""),
                    "run_id": args.run_id,
                    "selector_type": args.selector_type,
                    "prompt_id": args.prompt_id,
                    "model_name": args.model_name,
                    "rank": rank,
                    "pred_start_sec": round(
                        number(candidate["start_ms"], "start_ms") / 1000.0,
                        3,
                    ),
                    "pred_end_sec": round(
                        number(candidate["end_ms"], "end_ms") / 1000.0,
                        3,
                    ),
                    "selected_scene_ids": "|".join(
                        str(scene_id) for scene_id in scene_ids
                    ),
                    "confidence": round(scores[candidate_id], 6),
                    "notes": f"candidate_id={candidate_id}",
                }
            )
    write_csv(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prediction_count": len(output),
                "longform_count": len(ranked_by_longform),
                "top_k": args.top_k,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
