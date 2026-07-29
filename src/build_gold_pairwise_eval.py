from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


HUMAN_RESPONSE_FIELDS = (
    "editorial_preference",
    "performance_preference",
    "confidence_1_to_5",
    "insufficient_evidence",
    "notes",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def candidate_payload(row: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "duration_sec": number(row.get("duration_sec", "")),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "language": row.get("language", "ko"),
        "genre": row.get("genre", "general"),
        "description": row.get("description", "")[:1800],
        "transcript": row.get("transcript", "")[:5000],
        "before_context": row.get("before_context", "")[:2500],
        "after_context": row.get("after_context", "")[:2500],
    }


def preserve_human_responses(path: Path, rows: list[dict[str, Any]]) -> int:
    if not path.exists():
        return 0
    existing = {
        (row.get("comparison_id", ""), row.get("annotator_id", "")): row
        for row in read_csv(path)
    }
    preserved = 0
    for row in rows:
        previous = existing.get((str(row["comparison_id"]), str(row["annotator_id"])))
        if not previous:
            continue
        if (
            previous.get("left_candidate_id") != row.get("left_candidate_id")
            or previous.get("right_candidate_id") != row.get("right_candidate_id")
        ):
            continue
        if not any(str(previous.get(field, "")).strip() for field in HUMAN_RESPONSE_FIELDS):
            continue
        for field in HUMAN_RESPONSE_FIELDS:
            row[field] = previous.get(field, "")
        preserved += 1
    return preserved


def main() -> int:
    parser = argparse.ArgumentParser(description="Build blind, channel-matched Pos/Neg Gold comparisons.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--matches-per-neg", type=int, default=2)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--seed", default="gold-pairwise-v4")
    args = parser.parse_args()

    dataset = {row["pair_id"]: row for row in read_csv(Path(args.dataset))}
    candidates = {row["candidate_id"]: row for row in read_csv(Path(args.candidates))}
    sources = [
        row
        for row in read_csv(Path(args.sources))
        if row.get("source_system") == "gold"
        and row.get("performance_label") in {"pos", "neg"}
        and row.get("candidate_id") in candidates
        and row.get("pair_id") in dataset
    ]

    records: list[dict[str, Any]] = []
    for source in sources:
        pair = dataset[source["pair_id"]]
        candidate = candidates[source["candidate_id"]]
        records.append(
            {
                "source": source,
                "pair": pair,
                "candidate": candidate,
                "pair_id": source["pair_id"],
                "candidate_id": source["candidate_id"],
                "label": source["performance_label"],
                "channel_name": pair.get("channel_name", ""),
                "long_video_id": pair.get("long_video_id", ""),
                "duration": number(candidate.get("duration_sec", "")),
                "genre": candidate.get("genre", "general"),
            }
        )

    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_channel[record["channel_name"]].append(record)

    blind_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    positive_usage: dict[str, int] = defaultdict(int)

    for channel_name, group in sorted(by_channel.items()):
        positives = [row for row in group if row["label"] == "pos"]
        negatives = [row for row in group if row["label"] == "neg"]
        if not positives or not negatives:
            continue
        for negative in sorted(negatives, key=lambda row: row["pair_id"]):
            selected_ids: set[str] = set()
            for match_index in range(1, max(1, args.matches_per_neg) + 1):
                available = [row for row in positives if row["candidate_id"] not in selected_ids]
                if not available:
                    available = positives
                positive = min(
                    available,
                    key=lambda row: (
                        row["long_video_id"] != negative["long_video_id"],
                        row["genre"] != negative["genre"],
                        positive_usage[row["candidate_id"]],
                        abs(row["duration"] - negative["duration"]),
                        row["pair_id"],
                    ),
                )
                selected_ids.add(positive["candidate_id"])
                positive_usage[positive["candidate_id"]] += 1

                orientation = hashlib.sha256(
                    f"{args.seed}|{negative['candidate_id']}|{positive['candidate_id']}|{match_index}".encode("utf-8")
                ).digest()[0] % 2
                left, right = (positive, negative) if orientation == 0 else (negative, positive)
                raw_id = f"{args.seed}|{negative['candidate_id']}|{positive['candidate_id']}|{match_index}"
                comparison_id = f"PW_{hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:14]}"
                matching_rule = (
                    "same_channel_same_longform"
                    if positive["long_video_id"] == negative["long_video_id"]
                    else "same_channel_genre_duration"
                )

                blind_rows.append(
                    {
                        "comparison_id": comparison_id,
                        "same_channel": True,
                        "matching_rule": matching_rule,
                        "left": candidate_payload(left["candidate"]),
                        "right": candidate_payload(right["candidate"]),
                    }
                )
                manifest_rows.append(
                    {
                        "comparison_id": comparison_id,
                        "channel_name": channel_name,
                        "matching_rule": matching_rule,
                        "left_candidate_id": left["candidate_id"],
                        "left_pair_id": left["pair_id"],
                        "left_performance_label": left["label"],
                        "left_views": left["pair"].get("short_views", ""),
                        "left_likes": left["pair"].get("short_likes", ""),
                        "right_candidate_id": right["candidate_id"],
                        "right_pair_id": right["pair_id"],
                        "right_performance_label": right["label"],
                        "right_views": right["pair"].get("short_views", ""),
                        "right_likes": right["pair"].get("short_likes", ""),
                        "positive_side": "left" if left["label"] == "pos" else "right",
                        "negative_pair_id": negative["pair_id"],
                        "positive_pair_id": positive["pair_id"],
                        "duration_difference_sec": round(abs(positive["duration"] - negative["duration"]), 3),
                    }
                )
                for annotator_index in range(1, max(1, args.annotators) + 1):
                    human_rows.append(
                        {
                            "comparison_id": comparison_id,
                            "annotator_id": f"A{annotator_index:02d}",
                            "left_candidate_id": left["candidate_id"],
                            "left_start": left["candidate"].get("start_time", ""),
                            "left_end": left["candidate"].get("end_time", ""),
                            "left_url": left["candidate"].get("candidate_url", ""),
                            "left_description": left["candidate"].get("description", "")[:900],
                            "left_transcript": left["candidate"].get("transcript", "")[:2200],
                            "right_candidate_id": right["candidate_id"],
                            "right_start": right["candidate"].get("start_time", ""),
                            "right_end": right["candidate"].get("end_time", ""),
                            "right_url": right["candidate"].get("candidate_url", ""),
                            "right_description": right["candidate"].get("description", "")[:900],
                            "right_transcript": right["candidate"].get("transcript", "")[:2200],
                            "editorial_preference": "",
                            "performance_preference": "",
                            "confidence_1_to_5": "",
                            "insufficient_evidence": "",
                            "notes": "",
                        }
                    )

    out_dir = Path(args.out_dir)
    human_path = out_dir / "human_pairwise_labels.csv"
    preserved_human_rows = preserve_human_responses(human_path, human_rows)
    write_jsonl(out_dir / "pairwise_candidates_blind.jsonl", blind_rows)
    write_csv(
        out_dir / "pairwise_sources_private.csv",
        manifest_rows,
        [
            "comparison_id", "channel_name", "matching_rule", "left_candidate_id", "left_pair_id",
            "left_performance_label", "left_views", "left_likes", "right_candidate_id", "right_pair_id",
            "right_performance_label", "right_views", "right_likes", "positive_side", "negative_pair_id",
            "positive_pair_id", "duration_difference_sec",
        ],
    )
    write_csv(
        human_path,
        human_rows,
        [
            "comparison_id", "annotator_id", "left_candidate_id", "left_start", "left_end", "left_url",
            "left_description", "left_transcript", "right_candidate_id", "right_start", "right_end", "right_url",
            "right_description", "right_transcript", "editorial_preference", "performance_preference",
            "confidence_1_to_5", "insufficient_evidence", "notes",
        ],
    )
    summary = {
        "comparison_count": len(blind_rows),
        "negative_case_count": len({row["negative_pair_id"] for row in manifest_rows}),
        "positive_case_count": len({row["positive_pair_id"] for row in manifest_rows}),
        "channel_count": len({row["channel_name"] for row in manifest_rows}),
        "human_label_row_count": len(human_rows),
        "preserved_human_label_row_count": preserved_human_rows,
        "matches_per_negative": max(1, args.matches_per_neg),
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
