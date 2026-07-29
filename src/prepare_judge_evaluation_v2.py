from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


BLIND_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)
FALLBACK_FIELDS = ("duration_sec", "transcript", "before_context", "after_context")
DEFAULT_CANARY_IDS = ("C_519202486aef76",)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def unique_index(
    rows: list[dict[str, str]],
    key: str,
    source_name: str,
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "").strip()
        if not value:
            raise ValueError(f"{source_name} contains an empty {key}")
        if value in index:
            raise ValueError(f"{source_name} contains duplicate {key}: {value}")
        index[value] = row
    return index


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_equal(
    label: str,
    left: str,
    right: str,
    candidate_id: str,
) -> None:
    if left.strip() != right.strip():
        raise ValueError(
            f"{label} mismatch for {candidate_id}: {left!r} != {right!r}"
        )


def choose_dev_ids(
    manifest: list[dict[str, Any]],
    transcript_by_id: dict[str, dict[str, str]],
    forced_ids: set[str],
) -> tuple[set[str], dict[str, str]]:
    longform_counts: dict[str, int] = defaultdict(int)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        longform_counts[str(row["longform_id"])] += 1
        strata[(str(row["channel_name"]), str(row["performance_label"]))].append(
            row
        )

    expected_strata = {
        (str(row["channel_name"]), label)
        for row in manifest
        for label in ("pos", "neg")
    }
    if set(strata) != expected_strata:
        missing = sorted(expected_strata - set(strata))
        raise ValueError(f"Cannot build channel-label dev strata; missing={missing}")

    selected: set[str] = set()
    reasons: dict[str, str] = {}
    for candidate_id in forced_ids:
        row = next(
            (item for item in manifest if item["candidate_id"] == candidate_id),
            None,
        )
        if row is None:
            raise ValueError(f"Forced dev candidate does not exist: {candidate_id}")
        if longform_counts[str(row["longform_id"])] != 1:
            raise ValueError(
                f"Forced dev candidate is not a singleton longform: {candidate_id}"
            )
        selected.add(candidate_id)
        reasons[candidate_id] = "forced_low_evidence_canary"

    for stratum in sorted(strata):
        existing = [
            row
            for row in strata[stratum]
            if str(row["candidate_id"]) in selected
        ]
        if existing:
            continue
        eligible = [
            row
            for row in strata[stratum]
            if longform_counts[str(row["longform_id"])] == 1
        ]
        if not eligible:
            raise ValueError(f"No singleton longform candidate for stratum={stratum}")
        lengths = sorted(
            len(transcript_by_id[str(row["candidate_id"])]["transcript"])
            for row in eligible
        )
        target_length = statistics.median(lengths)
        chosen = min(
            eligible,
            key=lambda row: (
                abs(
                    len(
                        transcript_by_id[str(row["candidate_id"])]["transcript"]
                    )
                    - target_length
                ),
                str(row["candidate_id"]),
            ),
        )
        candidate_id = str(chosen["candidate_id"])
        selected.add(candidate_id)
        reasons[candidate_id] = "median_transcript_length_by_channel_label"

    if len(selected) != len(expected_strata):
        raise ValueError(
            f"Expected {len(expected_strata)} dev candidates, got {len(selected)}"
        )
    return selected, reasons


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical 60-candidate manifest, repair Vpick evidence, "
            "and lock a longform-isolated dev/test split."
        )
    )
    parser.add_argument("--targets", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--transcript-candidates", required=True)
    parser.add_argument("--vpick-candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--canary-id",
        action="append",
        default=list(DEFAULT_CANARY_IDS),
    )
    args = parser.parse_args()

    targets_path = Path(args.targets)
    gold_path = Path(args.gold)
    transcript_path = Path(args.transcript_candidates)
    vpick_path = Path(args.vpick_candidates)
    out_dir = Path(args.out_dir)

    targets = read_csv(targets_path)
    gold = read_csv(gold_path)
    transcript_rows = read_csv(transcript_path)
    vpick_rows = read_csv(vpick_path)

    targets_by_id = unique_index(
        targets, "source_candidate_id", "candidate targets"
    )
    gold_by_pair = unique_index(gold, "pair_id", "gold dataset")
    transcript_by_id = unique_index(
        transcript_rows, "candidate_id", "transcript candidates"
    )
    vpick_by_id = unique_index(vpick_rows, "candidate_id", "Vpick candidates")

    candidate_ids = set(transcript_by_id)
    if set(targets_by_id) != candidate_ids:
        raise ValueError("candidate targets and transcript candidates differ")
    if set(vpick_by_id) != candidate_ids:
        raise ValueError("Vpick and transcript candidate ID sets differ")
    if len(candidate_ids) != 60:
        raise ValueError(f"Expected 60 candidates, got {len(candidate_ids)}")

    manifest: list[dict[str, Any]] = []
    for candidate_id in transcript_by_id:
        target = targets_by_id[candidate_id]
        pair_id = target["pair_id"]
        gold_row = gold_by_pair.get(pair_id)
        if gold_row is None:
            raise ValueError(f"Gold pair missing for {candidate_id}: {pair_id}")
        assert_equal(
            "short_video_id",
            target["short_video_id"],
            gold_row["short_video_id"],
            candidate_id,
        )
        assert_equal(
            "longform_id",
            target["longform_id"],
            gold_row["long_video_id"],
            candidate_id,
        )
        assert_equal(
            "performance_label",
            target["performance_label"],
            gold_row["performance_label"],
            candidate_id,
        )
        assert_equal(
            "channel_name",
            target["channel_name"],
            gold_row["channel_name"],
            candidate_id,
        )
        manifest.append(
            {
                "candidate_id": candidate_id,
                "pair_id": pair_id,
                "longform_id": target["longform_id"],
                "long_video_url": gold_row["long_video_url"],
                "short_video_id": target["short_video_id"],
                "short_video_url": gold_row["short_video_url"],
                "channel_name": target["channel_name"],
                "performance_label": target["performance_label"],
                "channel_performance_percentile": target[
                    "channel_performance_percentile"
                ],
                "start_sec": target["start_sec"],
                "end_sec": target["end_sec"],
                "duration_sec": transcript_by_id[candidate_id]["duration_sec"],
                "label_confidence": gold_row["label_confidence"],
                "mapping_confidence": gold_row["mapping_confidence"],
                "timestamp_method": gold_row["timestamp_method"],
                "timestamp_confidence": gold_row["timestamp_confidence"],
                "verification_status_legacy": gold_row["verification_status"],
                "alignment_classification_legacy": gold_row[
                    "alignment_classification_v3"
                ],
                "evidence_provider": target["evidence_provider"],
                "evidence_path": target["evidence_path"],
            }
        )

    dev_ids, dev_reasons = choose_dev_ids(
        manifest,
        transcript_by_id,
        set(args.canary_id),
    )
    dev_longforms = {
        str(row["longform_id"])
        for row in manifest
        if row["candidate_id"] in dev_ids
    }
    for row in manifest:
        candidate_id = str(row["candidate_id"])
        is_dev = candidate_id in dev_ids
        if not is_dev and str(row["longform_id"]) in dev_longforms:
            raise ValueError(
                f"Longform leakage into locked test: {row['longform_id']}"
            )
        row["dataset_role_v2"] = "dev" if is_dev else "locked_test"
        row["split_lock_version"] = "judge_dev12_grouped_v1"
        row["dev_selection_reason"] = dev_reasons.get(candidate_id, "")

    repaired_vpick: list[dict[str, str]] = []
    repair_counts = {field: 0 for field in FALLBACK_FIELDS}
    repaired_candidate_ids: dict[str, list[str]] = defaultdict(list)
    for transcript_row in transcript_rows:
        candidate_id = transcript_row["candidate_id"]
        row = dict(vpick_by_id[candidate_id])
        for field in FALLBACK_FIELDS:
            if not row[field].strip() and transcript_row[field].strip():
                row[field] = transcript_row[field]
                repair_counts[field] += 1
                repaired_candidate_ids[field].append(candidate_id)
        repaired_vpick.append(row)

    for row in repaired_vpick:
        for field in BLIND_FIELDS:
            if not row[field].strip():
                raise ValueError(
                    f"Repaired Vpick input still has empty {field}: "
                    f"{row['candidate_id']}"
                )

    manifest_fields = list(manifest[0])
    write_csv(
        out_dir / "canonical_candidate_manifest_PRIVATE.csv",
        manifest,
        manifest_fields,
    )
    write_csv(
        out_dir / "candidates_vpick_enriched_repaired.csv",
        repaired_vpick,
        list(BLIND_FIELDS),
    )

    for role in ("dev", "locked_test"):
        role_ids = {
            str(row["candidate_id"])
            for row in manifest
            if row["dataset_role_v2"] == role
        }
        write_csv(
            out_dir / f"{role}_manifest_PRIVATE.csv",
            [row for row in manifest if str(row["candidate_id"]) in role_ids],
            manifest_fields,
        )
        write_csv(
            out_dir / f"{role}_candidates_transcript.csv",
            [
                row
                for row in transcript_rows
                if row["candidate_id"] in role_ids
            ],
            list(BLIND_FIELDS),
        )
        write_csv(
            out_dir / f"{role}_candidates_vpick.csv",
            [
                row
                for row in repaired_vpick
                if row["candidate_id"] in role_ids
            ],
            list(BLIND_FIELDS),
        )

    summary = {
        "candidate_count": len(manifest),
        "dev_count": sum(row["dataset_role_v2"] == "dev" for row in manifest),
        "locked_test_count": sum(
            row["dataset_role_v2"] == "locked_test" for row in manifest
        ),
        "dev_longform_count": len(dev_longforms),
        "longform_overlap_count": 0,
        "dev_channel_label_strata": sorted(
            {
                f"{row['channel_name']}|{row['performance_label']}"
                for row in manifest
                if row["dataset_role_v2"] == "dev"
            }
        ),
        "repair_counts": repair_counts,
        "repaired_candidate_ids": {
            key: sorted(value) for key, value in repaired_candidate_ids.items()
        },
        "source_sha256": {
            "targets": sha256(targets_path),
            "gold": sha256(gold_path),
            "transcript_candidates": sha256(transcript_path),
            "vpick_candidates": sha256(vpick_path),
        },
        "split_lock_version": "judge_dev12_grouped_v1",
        "performance_labels_visible_to_judge": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
