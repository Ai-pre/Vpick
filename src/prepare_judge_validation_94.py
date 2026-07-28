from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PRIVATE_INPUT_KEYS = {
    "channel_name",
    "performance_label_PRIVATE",
    "channel_performance_percentile_PRIVATE",
    "percentile_bucket",
    "transcript_source",
    "dataset_role_v2",
    "split_lock_version",
}
VALID_LABELS = {"neg", "mid", "pos"}
VALID_CHANNELS = {"BDNS", "OOTB", "숏박스", "안원잘부", "워크맨", "피식대학"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_index(
    rows: list[dict[str, str]],
    key: str,
    source_name: str,
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            raise ValueError(f"{source_name} contains an empty {key}")
        if value in output:
            raise ValueError(f"{source_name} contains duplicate {key}: {value}")
        output[value] = row
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def normalize_context(value: str, side: str, max_chars: int) -> str:
    text = "\n".join(line.rstrip() for line in str(value).strip().splitlines())
    if len(text) <= max_chars:
        return text
    if side == "before":
        return text[-max_chars:]
    return text[:max_chars]


def assign_group_split(
    rows: list[dict[str, str]],
    *,
    dev_fraction: float,
    seed: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    by_longform: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_longform[row["longform_id"]].append(row)

    group_roles: dict[str, str] = {}
    locked_reason: dict[str, str] = {}
    for longform_id, members in by_longform.items():
        roles = {
            row.get("dataset_role_v2", "")
            for row in members
            if row.get("dataset_role_v2", "") in {"dev", "locked_test"}
        }
        if roles == {"dev", "locked_test"}:
            raise ValueError(
                f"Existing dev/test roles already leak across {longform_id}"
            )
        if "locked_test" in roles:
            group_roles[longform_id] = "locked_test"
            locked_reason[longform_id] = "preserve_existing_locked_test_group"
        elif "dev" in roles:
            group_roles[longform_id] = "dev"
            locked_reason[longform_id] = "preserve_existing_dev_group"

    target_total = round(len(rows) * dev_fraction)
    channel_totals = Counter(row["channel_name"] for row in rows)
    label_totals = Counter(row["performance_label_PRIVATE"] for row in rows)
    target_channels = {
        key: value * dev_fraction for key, value in channel_totals.items()
    }
    target_labels = {
        key: value * dev_fraction for key, value in label_totals.items()
    }

    def selected_members(roles: dict[str, str]) -> list[dict[str, str]]:
        return [
            row
            for longform_id, members in by_longform.items()
            if roles.get(longform_id) == "dev"
            for row in members
        ]

    unassigned = [
        longform_id
        for longform_id in by_longform
        if longform_id not in group_roles
    ]
    while len(selected_members(group_roles)) < target_total and unassigned:
        current = selected_members(group_roles)
        current_channels = Counter(row["channel_name"] for row in current)
        current_labels = Counter(
            row["performance_label_PRIVATE"] for row in current
        )

        def objective(longform_id: str) -> tuple[float, int, str]:
            members = by_longform[longform_id]
            next_total = len(current) + len(members)
            next_channels = current_channels + Counter(
                row["channel_name"] for row in members
            )
            next_labels = current_labels + Counter(
                row["performance_label_PRIVATE"] for row in members
            )
            score = 5.0 * abs(next_total - target_total)
            score += sum(
                abs(next_channels[key] - target_channels[key])
                for key in target_channels
            )
            score += sum(
                abs(next_labels[key] - target_labels[key])
                for key in target_labels
            )
            return score, len(members), stable_hash(longform_id, seed)

        chosen = min(unassigned, key=objective)
        group_roles[chosen] = "dev"
        locked_reason[chosen] = "deterministic_group_stratified_expansion"
        unassigned.remove(chosen)

    for longform_id in unassigned:
        group_roles[longform_id] = "locked_test"
        locked_reason[longform_id] = "deterministic_group_stratified_holdout"

    candidate_roles = {
        row["candidate_id"]: group_roles[row["longform_id"]]
        for row in rows
    }
    dev = [row for row in rows if candidate_roles[row["candidate_id"]] == "dev"]
    locked = [
        row
        for row in rows
        if candidate_roles[row["candidate_id"]] == "locked_test"
    ]
    overlap = {
        row["longform_id"] for row in dev
    } & {
        row["longform_id"] for row in locked
    }
    if overlap:
        raise ValueError(f"Longform group leakage remains: {sorted(overlap)}")

    summary = {
        "split_lock_version": "judge94_grouped_v1",
        "seed": seed,
        "dev_fraction_target": dev_fraction,
        "dev_candidate_count": len(dev),
        "locked_test_candidate_count": len(locked),
        "dev_longform_count": len({row["longform_id"] for row in dev}),
        "locked_test_longform_count": len(
            {row["longform_id"] for row in locked}
        ),
        "longform_overlap_count": 0,
        "dev_label_counts": dict(
            sorted(Counter(row["performance_label_PRIVATE"] for row in dev).items())
        ),
        "locked_test_label_counts": dict(
            sorted(
                Counter(
                    row["performance_label_PRIVATE"] for row in locked
                ).items()
            )
        ),
        "dev_channel_counts": dict(
            sorted(Counter(row["channel_name"] for row in dev).items())
        ),
        "group_assignment_reasons": dict(
            sorted(Counter(locked_reason.values()).items())
        ),
    }
    return candidate_roles, summary


def length_stats(
    labels: list[dict[str, str]],
    text_by_id: dict[str, dict[str, str]],
    field: str,
    *,
    normalized: bool,
    max_chars: int,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for label in sorted(VALID_LABELS):
        values: list[int] = []
        for row in labels:
            if row["performance_label_PRIVATE"] != label:
                continue
            text = text_by_id[row["candidate_id"]].get(field, "")
            if normalized:
                side = "before" if field == "before_context" else "after"
                text = normalize_context(text, side, max_chars)
            values.append(len(text))
        output[label] = round(sum(values) / len(values), 2) if values else 0.0
    return output


def build(
    labels: list[dict[str, str]],
    texts: list[dict[str, str]],
    *,
    max_context_chars: int,
    dev_fraction: float,
    seed: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    label_by_id = unique_index(labels, "candidate_id", "labels")
    text_by_id = unique_index(texts, "candidate_id", "texts")
    if set(label_by_id) != set(text_by_id):
        raise ValueError(
            "Label/text candidate IDs differ: "
            f"missing_text={sorted(set(label_by_id) - set(text_by_id))[:10]}, "
            f"extra_text={sorted(set(text_by_id) - set(label_by_id))[:10]}"
        )
    if len(labels) != 94:
        raise ValueError(f"Expected 94 labels, found {len(labels)}")
    if {row["performance_label_PRIVATE"] for row in labels} != VALID_LABELS:
        raise ValueError("Expected neg/mid/pos performance labels")
    if {row["channel_name"] for row in labels} != VALID_CHANNELS:
        raise ValueError("Channel normalization is incomplete")

    candidate_roles, split_summary = assign_group_split(
        labels,
        dev_fraction=dev_fraction,
        seed=seed,
    )
    blind_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for label in sorted(labels, key=lambda row: row["candidate_id"]):
        candidate_id = label["candidate_id"]
        text = text_by_id[candidate_id]
        required_text = ("description", "transcript")
        if any(not str(text.get(field, "")).strip() for field in required_text):
            raise ValueError(f"{candidate_id} has empty description/transcript")
        start_sec = float(label["start_sec"])
        end_sec = float(label["end_sec"])
        if end_sec <= start_sec:
            raise ValueError(f"{candidate_id} has an invalid interval")

        blind = {
            "candidate_id": candidate_id,
            "longform_id": label["longform_id"],
            "start_ms": round(start_sec * 1000),
            "end_ms": round(end_sec * 1000),
            "duration_sec": round(end_sec - start_sec, 3),
            "longform_overview": [],
            "scene_ids": [],
            "description": text["description"].strip(),
            "transcript": text["transcript"].strip(),
            "before_context": normalize_context(
                text.get("before_context", ""),
                "before",
                max_context_chars,
            ),
            "after_context": normalize_context(
                text.get("after_context", ""),
                "after",
                max_context_chars,
            ),
            "visual_evidence_available": False,
            "input_policy": f"uniform_context_{max_context_chars}chars_v1",
        }
        leaked = PRIVATE_INPUT_KEYS & set(blind)
        if leaked:
            raise ValueError(f"Blind input leak for {candidate_id}: {sorted(leaked)}")
        blind_rows.append(blind)

        role = candidate_roles[candidate_id]
        target_rows.append(
            {
                **label,
                "dataset_role_v3": role,
                "split_lock_version_v3": "judge94_grouped_v1",
            }
        )
        split_rows.append(
            {
                "candidate_id": candidate_id,
                "longform_id": label["longform_id"],
                "channel_name": label["channel_name"],
                "performance_label_PRIVATE": label[
                    "performance_label_PRIVATE"
                ],
                "percentile_bucket": label["percentile_bucket"],
                "dataset_role_v3": role,
                "split_lock_version_v3": "judge94_grouped_v1",
            }
        )

    source_by_label: dict[str, dict[str, int]] = {}
    for label in sorted(VALID_LABELS):
        source_by_label[label] = dict(
            sorted(
                Counter(
                    row["transcript_source"]
                    for row in labels
                    if row["performance_label_PRIVATE"] == label
                ).items()
            )
        )
    summary = {
        "dataset_version": "judge_validation_94_v1",
        "candidate_count": len(blind_rows),
        "longform_count": len({row["longform_id"] for row in labels}),
        "label_counts_PRIVATE": dict(
            sorted(Counter(row["performance_label_PRIVATE"] for row in labels).items())
        ),
        "channel_counts": dict(
            sorted(Counter(row["channel_name"] for row in labels).items())
        ),
        "transcript_source_by_label_PRIVATE": source_by_label,
        "context_policy": {
            "before": f"rightmost {max_context_chars} characters",
            "after": f"leftmost {max_context_chars} characters",
        },
        "raw_context_mean_chars_PRIVATE": {
            "before_context": length_stats(
                labels,
                text_by_id,
                "before_context",
                normalized=False,
                max_chars=max_context_chars,
            ),
            "after_context": length_stats(
                labels,
                text_by_id,
                "after_context",
                normalized=False,
                max_chars=max_context_chars,
            ),
        },
        "normalized_context_mean_chars_PRIVATE": {
            "before_context": length_stats(
                labels,
                text_by_id,
                "before_context",
                normalized=True,
                max_chars=max_context_chars,
            ),
            "after_context": length_stats(
                labels,
                text_by_id,
                "after_context",
                normalized=True,
                max_chars=max_context_chars,
            ),
        },
        "split": split_summary,
        "blind_input_private_key_count": 0,
        "production_equivalent": False,
        "blocking_issue": (
            "The 94-row handover input has no full longform_overview. "
            "The normalized file is valid for the auxiliary performance-consistency "
            "experiment, but it must not be reported as an exact production-equivalent "
            "v10 validation until overviews are rebuilt uniformly."
        ),
    }
    return blind_rows, target_rows, split_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join the 94-candidate handover dataset, normalize context length, "
            "and create a longform-grouped dev/locked-test split."
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-context-chars", type=int, default=200)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--seed", default="20260728")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_csv(args.labels)
    texts = read_csv(args.texts)
    blind, targets, split, summary = build(
        labels,
        texts,
        max_context_chars=args.max_context_chars,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "candidates_blind_94.jsonl", blind)
    write_csv(args.output_dir / "validation_targets_94_PRIVATE.csv", targets)
    write_csv(args.output_dir / "group_split_94_PRIVATE.csv", split)
    summary.update(
        {
            "source_sha256": {
                "labels": sha256(args.labels),
                "texts": sha256(args.texts),
            }
        }
    )
    (args.output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
