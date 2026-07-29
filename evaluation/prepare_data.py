from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    ROOT,
    assert_blind_payload,
    deterministic_group_split,
    load_config,
    read_csv,
    read_jsonl,
    resolve_path,
    upload_age_days,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_CONFIG = ROOT / "configs" / "evaluation.yaml"


def _index_unique(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value and value not in result:
            result[value] = row
    return result


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    output_dir = resolve_path(config["output_dir"]) / "prepared"

    candidates = read_jsonl(resolve_path(dataset["candidates_blind"]))
    source_targets = read_csv(resolve_path(dataset["candidate_targets"]))
    gold_rows = read_csv(resolve_path(dataset["gold_metadata"]))

    if len(candidates) != len(source_targets):
        raise ValueError(
            f"Candidate/target count mismatch: {len(candidates)} != {len(source_targets)}"
        )

    candidate_ids = [str(row.get("candidate_id", "")) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate_id in blind candidate input")
    for candidate in candidates:
        assert_blind_payload(candidate)

    targets_by_candidate = _index_unique(source_targets, "candidate_id")
    gold_by_short = _index_unique(gold_rows, "short_video_id")
    gold_by_candidate = _index_unique(gold_rows, "candidate_id")

    split_by_longform = deterministic_group_split(
        source_targets,
        group_key=config["split"]["group_key"],
        train_fraction=float(config["split"]["train_fraction"]),
        validation_fraction=float(config["split"]["validation_fraction"]),
        seed=str(config["split"]["seed"]),
    )

    private_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        source = targets_by_candidate.get(candidate_id)
        if not source:
            raise ValueError(f"Missing private target row for {candidate_id}")
        gold = (
            gold_by_short.get(str(source.get("short_video_id", "")))
            or gold_by_candidate.get(str(source.get("source_candidate_id", "")))
            or {}
        )
        longform_id = str(source.get("longform_id", ""))
        upload_date = gold.get("upload_date", "")
        snapshot_date = gold.get("stats_snapshot_date", "")
        age_days = upload_age_days(upload_date, snapshot_date)
        private_rows.append(
            {
                "candidate_id": candidate_id,
                "source_candidate_id": source.get("source_candidate_id", ""),
                "pair_id": source.get("pair_id", ""),
                "longform_id": longform_id,
                "short_video_id": source.get("short_video_id", ""),
                "channel_name": gold.get("channel_name") or source.get("channel_name", ""),
                "channel_name_raw": gold.get("channel_name_raw", ""),
                "short_views": gold.get("short_views", ""),
                "short_likes": gold.get("short_likes", ""),
                "short_like_rate": gold.get("short_like_rate", ""),
                "upload_date": upload_date,
                "stats_snapshot_date": snapshot_date,
                "upload_age_days": "" if age_days is None else age_days,
                "legacy_performance_label": source.get("performance_label", ""),
                "legacy_channel_percentile": (
                    gold.get("channel_performance_percentile")
                    or source.get("channel_performance_percentile", "")
                ),
                "start_sec": source.get("start_sec", ""),
                "end_sec": source.get("end_sec", ""),
                "evidence_provider": source.get("evidence_provider", ""),
                "evidence_path": source.get("evidence_path", ""),
                "dataset_split": split_by_longform[longform_id],
                "performance_observation_basis": (
                    "cumulative_views_with_upload_age"
                    if age_days is not None
                    else "cumulative_views_without_upload_age"
                ),
                "fixed_window_view_status": "unavailable",
            }
        )

    blind_output = output_dir / "candidates_blind.jsonl"
    private_output = output_dir / "targets_private.csv"
    split_output = output_dir / "longform_group_split.csv"
    write_jsonl(blind_output, candidates)
    write_csv(private_output, private_rows)
    write_csv(
        split_output,
        [
            {"longform_id": group, "dataset_split": split}
            for group, split in sorted(split_by_longform.items())
        ],
    )

    channel_counts = Counter(row["channel_name"] for row in private_rows)
    split_counts = Counter(row["dataset_split"] for row in private_rows)
    summary = {
        "evaluation_id": config["evaluation_id"],
        "candidate_count": len(candidates),
        "longform_count": len(split_by_longform),
        "channel_counts": dict(sorted(channel_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "upload_age_available_count": sum(row["upload_age_days"] != "" for row in private_rows),
        "fixed_window_view_count": 0,
        "fixed_window_view_status": "N/A: source data has no 7-day or 30-day snapshots",
        "blind_payload_leak_check": "passed",
        "outputs": {
            "candidates_blind": str(blind_output),
            "targets_private": str(private_output),
            "longform_group_split": str(split_output),
        },
    }
    write_json(output_dir / "prepare_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the common 60-candidate evaluation dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = prepare(load_config(args.config))
    print(f"Prepared {summary['candidate_count']} candidates across {summary['longform_count']} longforms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
