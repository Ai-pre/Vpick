from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from apply_hierarchical_listwise_results_v1 import calculate_v2_score


CANDIDATE_ID_PATTERN = re.compile(r"(?:^|;)candidate_id=([^;]+)")


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


def candidate_id(row: dict[str, str]) -> str:
    match = CANDIDATE_ID_PATTERN.search(row.get("notes", ""))
    if not match:
        raise ValueError(f"Missing candidate_id in notes: {row.get('pair_id')}")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preserve B2 Top5 and use the intrinsic judge only for ordering."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, nargs="+")
    parser.add_argument("--longform-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-weight", type=float, default=0.5)
    parser.add_argument(
        "--prior-mode",
        choices=("confidence", "rank"),
        default="confidence",
        help="Use B2 confidence or the original rank as the ordering prior.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.judge_weight <= 1.0:
        raise ValueError("--judge-weight must be between 0 and 1")

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

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.predictions):
        if row["long_video_id"] in allowed_longforms and int(row["rank"]) <= 5:
            groups[row["pair_id"]].append(row)

    output: list[dict[str, Any]] = []
    for pair_id, rows in sorted(groups.items()):
        scored = []
        for row in rows:
            current_id = candidate_id(row)
            if current_id not in judge_by_candidate:
                raise ValueError(f"Missing judge score: {current_id}")
            if args.prior_mode == "rank":
                prior = (6.0 - float(row["rank"])) / 5.0
            else:
                prior = float(row["confidence"])
            judge = judge_by_candidate[current_id]
            combined = (
                (1.0 - args.judge_weight) * prior
                + args.judge_weight * judge
            )
            scored.append((combined, judge, prior, current_id, row))
        scored.sort(
            key=lambda item: (item[0], item[1], item[2], item[3]),
            reverse=True,
        )
        for rank, (combined, judge, prior, current_id, row) in enumerate(
            scored, start=1
        ):
            output.append(
                {
                    **row,
                    "run_id": "b2_top5_intrinsic_v2_rerank",
                    "selector_type": "b2_top5_preserved_intrinsic_rerank",
                    "prompt_id": "hierarchical_multislate_listwise_v2_ko",
                    "model_name": "codex_direct_development",
                    "rank": rank,
                    "confidence": round(combined, 6),
                    "notes": (
                        f"candidate_id={current_id};"
                        f"b2_prior={prior:.6f};"
                        f"prior_mode={args.prior_mode};"
                        f"intrinsic={judge:.6f};"
                        f"judge_weight={args.judge_weight:.2f}"
                    ),
                }
            )
    write_csv(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pair_count": len(groups),
                "prediction_rows": len(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
