from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

from reference_judge import CHECKLIST_DIMENSIONS


FIELDS = [
    "candidate_id", "annotator_id", "display_order", "candidate_url", "start_time", "end_time",
    "highlight_saliency_1_5",
]
FIELDS.extend(f"check_{name}" for name in CHECKLIST_DIMENSIONS)
FIELDS.extend(["overall_shortform_suitable", "insufficient_evidence", "notes"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stratified_sample(
    candidates: list[dict[str, str]], sources: list[dict[str, str]], sample_size: int
) -> list[dict[str, str]]:
    if sample_size <= 0 or sample_size >= len(candidates):
        return candidates
    source_by_id = {row.get("candidate_id", ""): row for row in sources}
    quotas = {"pos": round(sample_size * 0.4), "neg": round(sample_size * 0.4)}
    quotas["unlabeled"] = sample_size - quotas["pos"] - quotas["neg"]
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    for label in ("pos", "neg", "unlabeled"):
        group = [
            row for row in candidates
            if source_by_id.get(row["candidate_id"], {}).get("performance_label", "").lower() == label
        ]
        by_channel: dict[str, list[dict[str, str]]] = {}
        for row in group:
            channel = source_by_id.get(row["candidate_id"], {}).get("channel_name", "unknown")
            by_channel.setdefault(channel, []).append(row)
        for channel, rows in by_channel.items():
            random.Random(f"reference-human:{label}:{channel}").shuffle(rows)
        channels = sorted(by_channel)
        random.Random(f"reference-human-channels:{label}").shuffle(channels)
        while channels and sum(
            source_by_id.get(row["candidate_id"], {}).get("performance_label", "").lower() == label
            for row in selected
        ) < quotas[label]:
            remaining_channels: list[str] = []
            for channel in channels:
                rows = by_channel[channel]
                if rows:
                    row = rows.pop()
                    if row["candidate_id"] not in selected_ids:
                        selected.append(row)
                        selected_ids.add(row["candidate_id"])
                if rows:
                    remaining_channels.append(channel)
                if sum(
                    source_by_id.get(row["candidate_id"], {}).get("performance_label", "").lower() == label
                    for row in selected
                ) >= quotas[label]:
                    break
            channels = remaining_channels

    if len(selected) < sample_size:
        remainder = [row for row in candidates if row["candidate_id"] not in selected_ids]
        random.Random("reference-human-remainder").shuffle(remainder)
        selected.extend(remainder[: sample_size - len(selected)])
    return selected[:sample_size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a blind three-rater reference Judge validation sheet.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=15)
    parser.add_argument("--annotator", action="append", default=[])
    args = parser.parse_args()

    candidates = stratified_sample(read_csv(Path(args.candidates)), read_csv(Path(args.sources)), args.sample_size)
    annotators = args.annotator or ["H1", "H2", "H3"]
    output_path = Path(args.output)
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if output_path.exists():
        existing = {
            (row.get("candidate_id", ""), row.get("annotator_id", "")): row
            for row in read_csv(output_path)
        }

    rows: list[dict[str, Any]] = []
    for annotator in annotators:
        ordered = list(candidates)
        random.Random(f"reference-human-order:{annotator}").shuffle(ordered)
        for display_order, candidate in enumerate(ordered, start=1):
            preserved = existing.get((candidate["candidate_id"], annotator), {})
            start_sec = int(float(candidate["start_sec"]))
            row: dict[str, Any] = {
                "candidate_id": candidate["candidate_id"],
                "annotator_id": annotator,
                "display_order": display_order,
                "candidate_url": f"https://www.youtube.com/watch?v={candidate['long_video_id']}&t={start_sec}s",
                "start_time": candidate.get("start_time", ""),
                "end_time": candidate.get("end_time", ""),
                "highlight_saliency_1_5": preserved.get("highlight_saliency_1_5", ""),
                "overall_shortform_suitable": preserved.get("overall_shortform_suitable", ""),
                "insufficient_evidence": preserved.get("insufficient_evidence", ""),
                "notes": preserved.get("notes", ""),
            }
            for name in CHECKLIST_DIMENSIONS:
                row[f"check_{name}"] = preserved.get(f"check_{name}", "")
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows for {len(candidates)} candidates to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
