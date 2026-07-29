from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import random
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_rank(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("source_rank") or 999))
    except ValueError:
        return 999


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a blind human pairwise preference label sheet.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--system", action="append", default=[])
    parser.add_argument("--pairs-per-long", type=int, default=3)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    candidates = {row["candidate_id"]: row for row in read_csv(Path(args.candidates))}
    source_rows = [row for row in read_csv(Path(args.sources)) if row.get("candidate_id") in candidates]
    allowed = set(args.system)
    if allowed:
        source_rows = [row for row in source_rows if row.get("source_system") in allowed or row.get("source_system") == "gold"]

    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in source_rows:
        grouped.setdefault(row.get("long_video_id", ""), {}).setdefault(row.get("source_system", ""), []).append(row)
    for by_system in grouped.values():
        for rows in by_system.values():
            rows.sort(key=source_rank)

    rng = random.Random(args.seed)
    label_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for long_video_id, by_system in sorted(grouped.items()):
        systems = sorted(by_system, key=lambda value: (value != "gold", value))
        proposed: list[tuple[dict[str, str], dict[str, str]]] = []
        for left_system, right_system in itertools.combinations(systems, 2):
            left_rows = by_system[left_system]
            right_rows = by_system[right_system]
            max_depth = max(len(left_rows), len(right_rows))
            for depth in range(max_depth):
                left = left_rows[min(depth, len(left_rows) - 1)]
                right = right_rows[min(depth, len(right_rows) - 1)]
                if left["candidate_id"] != right["candidate_id"]:
                    proposed.append((left, right))
        rng.shuffle(proposed)
        selected = proposed[: max(0, args.pairs_per_long)]

        for local_index, (left_source, right_source) in enumerate(selected, start=1):
            if rng.random() < 0.5:
                left_source, right_source = right_source, left_source
            raw_id = f"{args.seed}|{long_video_id}|{local_index}|{left_source['candidate_id']}|{right_source['candidate_id']}"
            comparison_id = f"H_{hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:14]}"
            left = candidates[left_source["candidate_id"]]
            right = candidates[right_source["candidate_id"]]
            manifest_rows.append(
                {
                    "comparison_id": comparison_id,
                    "long_video_id": long_video_id,
                    "left_candidate_id": left["candidate_id"],
                    "left_source_system": left_source["source_system"],
                    "left_source_rank": left_source.get("source_rank", ""),
                    "right_candidate_id": right["candidate_id"],
                    "right_source_system": right_source["source_system"],
                    "right_source_rank": right_source.get("source_rank", ""),
                }
            )
            for annotator_index in range(1, max(1, args.annotators) + 1):
                label_rows.append(
                    {
                        "comparison_id": comparison_id,
                        "annotator_id": f"A{annotator_index:02d}",
                        "long_video_id": long_video_id,
                        "left_candidate_id": left["candidate_id"],
                        "left_start": left.get("start_time", ""),
                        "left_end": left.get("end_time", ""),
                        "left_url": left.get("candidate_url", ""),
                        "left_description": left.get("description", "")[:700],
                        "left_transcript": left.get("transcript", "")[:1800],
                        "right_candidate_id": right["candidate_id"],
                        "right_start": right.get("start_time", ""),
                        "right_end": right.get("end_time", ""),
                        "right_url": right.get("candidate_url", ""),
                        "right_description": right.get("description", "")[:700],
                        "right_transcript": right.get("transcript", "")[:1800],
                        "preference": "",
                        "confidence_1_to_5": "",
                        "notes": "",
                    }
                )

    write_csv(
        Path(args.out_dir) / "human_pairwise_labels.csv",
        label_rows,
        [
            "comparison_id", "annotator_id", "long_video_id", "left_candidate_id", "left_start", "left_end",
            "left_url", "left_description", "left_transcript", "right_candidate_id", "right_start", "right_end",
            "right_url", "right_description", "right_transcript", "preference", "confidence_1_to_5", "notes",
        ],
    )
    write_csv(
        Path(args.out_dir) / "human_pairwise_sources_private.csv",
        manifest_rows,
        [
            "comparison_id", "long_video_id", "left_candidate_id", "left_source_system", "left_source_rank",
            "right_candidate_id", "right_source_system", "right_source_rank",
        ],
    )
    print(f"comparisons={len(manifest_rows)} label_rows={len(label_rows)} annotators={max(1, args.annotators)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
