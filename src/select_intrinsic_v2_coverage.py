from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from apply_hierarchical_listwise_results_v1 import calculate_v2_score


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["pair_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def center(row: dict[str, str]) -> float:
    return (float(row["start_sec"]) + float(row["end_sec"])) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select one intrinsic-judge candidate per timeline bin."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, nargs="+")
    parser.add_argument("--longform-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-weight", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()
    if not 0.0 <= args.judge_weight <= 1.0:
        raise ValueError("--judge-weight must be between 0 and 1")
    if args.bins < 1:
        raise ValueError("--bins must be positive")

    allowed_longforms = {
        line.strip()
        for line in args.longform_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    judge_by_candidate: dict[str, float] = {}
    for result_path in args.results:
        for result in read_jsonl(result_path):
            if str(result["longform_id"]) not in allowed_longforms:
                continue
            for score in result["candidate_scores"]:
                current_id = str(score["candidate_id"])
                if current_id in judge_by_candidate:
                    raise ValueError(f"Duplicate candidate score: {current_id}")
                judge_by_candidate[current_id] = calculate_v2_score(score)[0]

    candidates_by_longform: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.candidate_pool):
        if (
            row["longform_id"] in allowed_longforms
            and row["candidate_id"] in judge_by_candidate
        ):
            candidates_by_longform[row["longform_id"]].append(row)

    selected_by_longform: dict[str, list[dict[str, Any]]] = {}
    for longform_id, candidates in candidates_by_longform.items():
        video_end = max(float(row["end_sec"]) for row in candidates)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            bin_index = min(
                args.bins - 1,
                max(0, int((center(row) / max(video_end, 1.0)) * args.bins)),
            )
            judge = judge_by_candidate[row["candidate_id"]]
            prior = float(row["hierarchical_score"])
            combined = (
                args.judge_weight * judge
                + (1.0 - args.judge_weight) * prior
            )
            grouped[bin_index].append(
                {
                    "candidate": row,
                    "judge": judge,
                    "prior": prior,
                    "combined": combined,
                    "bin_index": bin_index,
                }
            )
        selected = [
            max(
                bucket,
                key=lambda item: (
                    item["combined"],
                    item["judge"],
                    item["prior"],
                    item["candidate"]["candidate_id"],
                ),
            )
            for _, bucket in sorted(grouped.items())
            if bucket
        ]
        if len(selected) < args.bins:
            chosen = {
                item["candidate"]["candidate_id"] for item in selected
            }
            remaining = sorted(
                [
                    item
                    for bucket in grouped.values()
                    for item in bucket
                    if item["candidate"]["candidate_id"] not in chosen
                ],
                key=lambda item: (
                    item["combined"],
                    item["judge"],
                    item["prior"],
                ),
                reverse=True,
            )
            selected.extend(remaining[: args.bins - len(selected)])
        selected_by_longform[longform_id] = sorted(
            selected[: args.bins],
            key=lambda item: (
                item["combined"],
                item["judge"],
                item["prior"],
            ),
            reverse=True,
        )

    gold_by_longform: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.dataset):
        gold_by_longform[row["long_video_id"]].append(row)
    output: list[dict[str, Any]] = []
    for longform_id, selected in selected_by_longform.items():
        for gold in gold_by_longform.get(longform_id, []):
            for rank, item in enumerate(selected, start=1):
                candidate = item["candidate"]
                output.append(
                    {
                        "pair_id": gold["pair_id"],
                        "long_video_id": longform_id,
                        "short_video_id": gold.get("short_video_id", ""),
                        "run_id": "intrinsic_v2_adaptive_coverage",
                        "selector_type": "five_bin_intrinsic_v2_rerank",
                        "prompt_id": "hierarchical_multislate_listwise_v2_ko",
                        "model_name": "codex_direct_development",
                        "rank": rank,
                        "pred_start_sec": candidate["start_sec"],
                        "pred_end_sec": candidate["end_sec"],
                        "selected_scene_ids": "",
                        "confidence": round(item["combined"], 6),
                        "notes": (
                            f"candidate_id={candidate['candidate_id']};"
                            f"bin={item['bin_index'] + 1}/{args.bins};"
                            f"intrinsic={item['judge']:.6f};"
                            f"candidate_prior={item['prior']:.6f};"
                            f"judge_weight={args.judge_weight:.2f}"
                        ),
                    }
                )
    write_csv(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "longform_count": len(selected_by_longform),
                "prediction_rows": len(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
