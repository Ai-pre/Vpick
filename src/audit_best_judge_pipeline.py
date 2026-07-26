#!/usr/bin/env python3
"""Audit the frozen Best Judge package without calling an LLM."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config" / "best_judge_pipeline.json"
EXPECTED_CHANNELS = {"BDNS", "OOTB", "숏박스", "안원잘부", "워크맨", "피식대학"}
FORBIDDEN_MODEL_KEYS = {
    "channel_name",
    "short_video_id",
    "short_video_url",
    "short_views",
    "short_likes",
    "short_title_yt",
    "performance_label",
    "channel_performance_percentile",
    "candidate_rank_in_channel",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve(path: str) -> Path:
    return ROOT / path


def require_unique(rows: list[dict[str, Any]], field: str) -> None:
    values = [str(row.get(field, "")).strip() for row in rows]
    if any(not value for value in values):
        raise ValueError(f"{field} contains an empty value")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} contains duplicate values")


def audit_dataset(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    if len(rows) != 60:
        raise ValueError(f"Expected 60 dataset rows, found {len(rows)}")
    for field in ("candidate_id", "pair_id", "short_video_id", "short_video_url"):
        require_unique(rows, field)
    channels = {row["channel_name"].strip() for row in rows}
    if channels != EXPECTED_CHANNELS:
        raise ValueError(f"Unexpected normalized channels: {sorted(channels)}")
    if any(
        not str(row.get(field, "")).strip()
        for row in rows
        for field in ("long_video_id", "start_sec", "end_sec")
    ):
        raise ValueError("Every frozen pair must have a longform ID and timestamps")
    return {
        "row_count": len(rows),
        "channel_count": len(channels),
        "channels": sorted(channels),
        "judge_evidence_ready_count": 60,
        "legacy_alignment_verified_count": sum(
            row.get("usable_for_gold") == "yes" for row in rows
        ),
        "legacy_alignment_note": (
            "usable_for_gold is the earlier strict source-alignment field; "
            "Judge readiness is audited against the blind evidence input."
        ),
        "performance_labels": {
            label: sum(row.get("performance_label") == label for row in rows)
            for label in ("pos", "neg")
        },
    }


def audit_blind_input(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 60:
        raise ValueError(f"Expected 60 blind candidates, found {len(rows)}")
    require_unique(rows, "candidate_id")
    leaked = sorted(
        {
            key
            for row in rows
            for key in FORBIDDEN_MODEL_KEYS
            if key in row
        }
    )
    if leaked:
        raise ValueError(f"Blind input contains forbidden keys: {leaked}")
    required = {
        "candidate_id",
        "start_ms",
        "end_ms",
        "description",
        "transcript",
        "before_context",
        "after_context",
        "visual_evidence_available",
    }
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Blind row {index} is missing {sorted(missing)}")
        if not str(row["description"]).strip():
            raise ValueError(f"Blind row {index} has an empty description")
        if not str(row["transcript"]).strip():
            raise ValueError(f"Blind row {index} has an empty transcript")
    providers: dict[str, int] = {}
    for row in rows:
        provider = str(row.get("evidence_provider", "unknown"))
        providers[provider] = providers.get(provider, 0) + 1
    return {
        "row_count": len(rows),
        "candidate_ids": sorted(str(row["candidate_id"]) for row in rows),
        "forbidden_key_count": 0,
        "empty_description_count": 0,
        "empty_transcript_count": 0,
        "evidence_provider_counts": providers,
    }


def audit_validation_targets(
    path: Path,
    blind_candidate_ids: set[str],
) -> dict[str, Any]:
    rows = read_csv(path)
    if len(rows) != 60:
        raise ValueError(f"Expected 60 validation targets, found {len(rows)}")
    for field in ("candidate_id", "source_candidate_id", "pair_id", "short_video_id"):
        require_unique(rows, field)
    target_ids = {row["candidate_id"] for row in rows}
    if target_ids != blind_candidate_ids:
        raise ValueError("Validation target IDs do not match blind candidate IDs")
    labels = {
        label: sum(row.get("performance_label") == label for row in rows)
        for label in ("pos", "neg")
    }
    if labels != {"pos": 30, "neg": 30}:
        raise ValueError(f"Expected balanced 30/30 validation labels, found {labels}")
    return {
        "row_count": len(rows),
        "label_counts": labels,
        "role": "post_hoc_validation_only",
        "passed_to_model": False,
    }


def audit_scores(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    if len(rows) != 60:
        raise ValueError(f"Expected 60 score rows, found {len(rows)}")
    require_unique(rows, "candidate_id")
    abstain_count = 0
    scores: list[float] = []
    for row in rows:
        if row.get("verdict") == "abstain":
            abstain_count += 1
            continue
        editorial = float(row["editorial_score_100"])
        engagement = float(row["engagement_score_100"])
        total = float(row["judge_score_100"])
        expected = 0.5 * editorial + 0.5 * engagement
        if not math.isclose(total, expected, abs_tol=1e-6):
            raise ValueError(
                f"Fixed formula mismatch for {row['candidate_id']}: "
                f"{total} != {expected}"
            )
        scores.append(total)
    frequencies = {score: scores.count(score) for score in set(scores)}
    return {
        "row_count": len(rows),
        "scored_count": len(scores),
        "abstain_count": abstain_count,
        "unique_score_count": len(set(scores)),
        "largest_tie_group": max(frequencies.values()) if frequencies else 0,
        "score_mean": round(sum(scores) / len(scores), 4) if scores else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--write-summary",
        type=Path,
        default=ROOT / "results" / "best_judge_pipeline" / "audit_summary.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    blind_input = audit_blind_input(resolve(manifest["blind_input"]))
    blind_ids = set(blind_input.pop("candidate_ids"))
    summary = {
        "pipeline_id": manifest["pipeline_id"],
        "status": manifest["status"],
        "dataset": audit_dataset(resolve(manifest["dataset"])),
        "blind_input": blind_input,
        "validation_targets": audit_validation_targets(
            resolve(manifest["validation_targets"]),
            blind_ids,
        ),
        "reference_result": audit_scores(resolve(manifest["reference_result"])),
        "fixed_formula": (
            "judge_score_100 = 0.5 * editorial_score_100 "
            "+ 0.5 * engagement_score_100"
        ),
        "performance_prediction_validated": False,
        "audit_passed": True,
    }
    args.write_summary.parent.mkdir(parents=True, exist_ok=True)
    args.write_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
