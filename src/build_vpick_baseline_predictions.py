from __future__ import annotations

import argparse
import csv
import json
import math
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


def number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def source_interval(shortform: dict[str, Any]) -> tuple[float, float] | None:
    metadata = shortform.get("generation_metadata") or {}
    scenes = metadata.get("scenes") or []
    intervals = []
    for scene in scenes:
        start = number(scene.get("source_start_ms"), math.nan)
        end = number(scene.get("source_end_ms"), math.nan)
        if math.isfinite(start) and math.isfinite(end) and end > start:
            intervals.append((start / 1000.0, end / 1000.0))
    if not intervals:
        return None
    return min(start for start, _ in intervals), max(end for _, end in intervals)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a fair unique-interval Vpick auto-shortform baseline."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    gold_by_longform: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.dataset):
        gold_by_longform[row["long_video_id"]].append(row)

    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for longform_id, gold_rows in sorted(gold_by_longform.items()):
        details_path = args.raw_dir / f"{longform_id}_shortforms_details.json"
        if not details_path.exists():
            audit.append(
                {
                    "longform_id": longform_id,
                    "status": "missing_details",
                    "raw_shortform_count": 0,
                    "valid_interval_count": 0,
                    "unique_interval_count": 0,
                    "selected_count": 0,
                }
            )
            continue
        payload = json.loads(details_path.read_text(encoding="utf-8"))
        shortforms = payload if isinstance(payload, list) else []
        candidates: list[dict[str, Any]] = []
        for source_index, shortform in enumerate(shortforms):
            interval = source_interval(shortform)
            if interval is None:
                continue
            candidates.append(
                {
                    "start": interval[0],
                    "end": interval[1],
                    "overall_score": number(shortform.get("overall_score")),
                    "source_index": source_index,
                    "shortform_id": str(shortform.get("id", "")),
                    "scene_count": len(
                        (shortform.get("generation_metadata") or {}).get(
                            "scenes"
                        )
                        or []
                    ),
                }
            )
        candidates.sort(
            key=lambda row: (-row["overall_score"], row["source_index"])
        )
        unique: list[dict[str, Any]] = []
        seen: set[tuple[float, float]] = set()
        for candidate in candidates:
            key = (
                round(candidate["start"], 3),
                round(candidate["end"], 3),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        selected = unique[: args.top_k]
        for pair in gold_rows:
            for rank, candidate in enumerate(selected, start=1):
                output.append(
                    {
                        "pair_id": pair["pair_id"],
                        "long_video_id": longform_id,
                        "short_video_id": pair.get("short_video_id", ""),
                        "run_id": "b0_vpick_auto_unique_top5",
                        "selector_type": "vpick_auto_shortform",
                        "prompt_id": "vpick_internal",
                        "model_name": "vpick",
                        "rank": rank,
                        "pred_start_sec": round(candidate["start"], 3),
                        "pred_end_sec": round(candidate["end"], 3),
                        "selected_scene_ids": "",
                        "confidence": candidate["overall_score"],
                        "notes": (
                            f"vpick_shortform_id={candidate['shortform_id']};"
                            f"source_scene_count={candidate['scene_count']}"
                        ),
                    }
                )
        audit.append(
            {
                "longform_id": longform_id,
                "status": "ready" if selected else "empty_details",
                "raw_shortform_count": len(shortforms),
                "valid_interval_count": len(candidates),
                "unique_interval_count": len(unique),
                "selected_count": len(selected),
            }
        )

    write_csv(args.output, output)
    write_csv(args.audit, audit)
    summary = {
        "dataset_longform_count": len(gold_by_longform),
        "ready_longform_count": sum(row["status"] == "ready" for row in audit),
        "missing_details_count": sum(
            row["status"] == "missing_details" for row in audit
        ),
        "empty_details_count": sum(
            row["status"] == "empty_details" for row in audit
        ),
        "prediction_rows": len(output),
        "top_k": args.top_k,
        "duplicate_intervals_removed": sum(
            int(row["valid_interval_count"]) - int(row["unique_interval_count"])
            for row in audit
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
