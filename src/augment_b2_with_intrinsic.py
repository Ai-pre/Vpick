from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["pair_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overlap_ratio(left: dict[str, str], right: dict[str, str]) -> float:
    left_start = float(left["pred_start_sec"])
    left_end = float(left["pred_end_sec"])
    right_start = float(right["pred_start_sec"])
    right_end = float(right["pred_end_sec"])
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    shortest = min(left_end - left_start, right_end - right_start)
    return overlap / shortest if shortest > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anchor early B2 ranks and fill remaining slots with judge picks."
    )
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--longform-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-overlap", type=float, default=0.58)
    args = parser.parse_args()

    allowed_longforms = {
        line.strip()
        for line in args.longform_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    anchor_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    supplement_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.anchor):
        if row["long_video_id"] in allowed_longforms:
            anchor_by_pair[row["pair_id"]].append(row)
    for row in read_csv(args.supplement):
        if row["long_video_id"] in allowed_longforms:
            supplement_by_pair[row["pair_id"]].append(row)

    output: list[dict[str, Any]] = []
    for pair_id, anchor_rows in sorted(anchor_by_pair.items()):
        ordered_anchor = sorted(anchor_rows, key=lambda row: int(row["rank"]))
        ordered_supplement = sorted(
            supplement_by_pair.get(pair_id, []),
            key=lambda row: int(row["rank"]),
        )
        selected = list(ordered_anchor[: args.anchor_count])
        for row in ordered_supplement:
            if len(selected) >= args.top_k:
                break
            if all(
                overlap_ratio(row, chosen) <= args.max_overlap
                for chosen in selected
            ):
                selected.append(row)
        for row in ordered_anchor[args.anchor_count :]:
            if len(selected) >= args.top_k:
                break
            if all(
                overlap_ratio(row, chosen) <= args.max_overlap
                for chosen in selected
            ):
                selected.append(row)

        for rank, row in enumerate(selected[: args.top_k], start=1):
            output.append(
                {
                    **row,
                    "run_id": "b2_anchor_intrinsic_v2_augment",
                    "selector_type": "b2_anchor_then_intrinsic_coverage",
                    "prompt_id": "hierarchical_multislate_listwise_v2_ko",
                    "model_name": "codex_direct_development",
                    "rank": rank,
                    "notes": (
                        f"source={'b2_anchor' if row in ordered_anchor[:args.anchor_count] else 'intrinsic_or_fill'};"
                        f"anchor_count={args.anchor_count};"
                        f"{row.get('notes', '')}"
                    ),
                }
            )
    write_csv(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pair_count": len(anchor_by_pair),
                "prediction_rows": len(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
