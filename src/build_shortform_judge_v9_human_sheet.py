from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from shortform_judge_v9 import EDITORIAL_DIMENSIONS, ENGAGEMENT_DIMENSIONS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (
    ROOT
    / "deliverables"
    / "2026-07-24"
    / "performance_judge_v1"
    / "candidate_targets_PRIVATE.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "deliverables"
    / "2026-07-24"
    / "shortform_judge_v9"
    / "human_anchor_scores.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_time(seconds: float) -> str:
    rounded = int(round(seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours}:{minutes:02d}:{secs:02d}"
        if hours
        else f"{minutes}:{secs:02d}"
    )


def select_anchors(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in targets:
        label = row.get("performance_label", "")
        channel = row.get("channel_name", "")
        if channel and label in {"pos", "neg"}:
            grouped[channel][label].append(row)

    selected: list[dict[str, str]] = []
    for channel in sorted(grouped):
        labels = grouped[channel]
        if not labels["pos"] or not labels["neg"]:
            continue
        selected.append(
            max(
                labels["pos"],
                key=lambda row: float(
                    row["channel_performance_percentile"]
                ),
            )
        )
        selected.append(
            min(
                labels["neg"],
                key=lambda row: float(
                    row["channel_performance_percentile"]
                ),
            )
        )
    return selected


def build_rows(
    targets: list[dict[str, str]],
    annotators: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    anchors = select_anchors(targets)
    rows: list[dict[str, Any]] = []
    for annotator in annotators:
        ordered = list(anchors)
        random.Random(f"{seed}:{annotator}").shuffle(ordered)
        for display_order, source in enumerate(ordered, start=1):
            start_sec = float(source["start_sec"])
            end_sec = float(source["end_sec"])
            row: dict[str, Any] = {
                "candidate_id": source["candidate_id"],
                "annotator_id": annotator,
                "display_order": display_order,
                "candidate_url": (
                    "https://www.youtube.com/watch?v="
                    f"{source['longform_id']}&t={int(start_sec)}s"
                ),
                "start_time": format_time(start_sec),
                "end_time": format_time(end_sec),
            }
            for axis, dimensions in (
                ("editorial", EDITORIAL_DIMENSIONS),
                ("engagement", ENGAGEMENT_DIMENSIONS),
            ):
                for dimension in dimensions:
                    row[f"{axis}_{dimension}_score_0_4"] = ""
            row["insufficient_evidence"] = ""
            row["notes"] = ""
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a blind 12-candidate human anchor sheet for Judge v9."
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotators", default="H1,H2")
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    annotators = [
        value.strip()
        for value in args.annotators.split(",")
        if value.strip()
    ]
    rows = build_rows(read_csv(args.targets), annotators, args.seed)
    write_csv(args.output, rows)
    summary = {
        "anchor_candidate_count": len(
            {row["candidate_id"] for row in rows}
        ),
        "annotator_count": len(annotators),
        "rating_row_count": len(rows),
        "selection": (
            "one highest-pos and one lowest-neg per eligible channel; "
            "labels and channels omitted from the human sheet"
        ),
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
