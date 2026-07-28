from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "reproduction"
DIAGNOSTICS = ROOT / "data" / "diagnostics"

REQUIRED_FILES = [
    DATA / "candidates_blind_94.jsonl",
    DATA / "codex_direct_v10_dimensions.csv",
    DATA / "codex_direct_v10_judgments_94.jsonl",
    DATA / "codex_direct_v10_scores_94.csv",
    DATA / "validation_targets_94_PRIVATE.csv",
    DATA / "group_split_94_PRIVATE.csv",
    DATA / "performance_controls_94.json",
    DATA / "preparation_summary.json",
    DIAGNOSTICS / "oof_predictions_PRIVATE.csv",
    DIAGNOSTICS / "tuning_log_PRIVATE.json",
]

FORBIDDEN_BLIND_KEYS = {
    "channel_name",
    "views",
    "likes",
    "performance_label_PRIVATE",
    "channel_performance_percentile_PRIVATE",
    "percentile_bucket",
    "dataset_role_v2",
    "dataset_role_v3",
    "short_video_id",
    "short_video_url",
    "long_video_url",
    "transcript_source",
}

SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{12,}|api[_-]?key|password\s*[:=])",
    re.IGNORECASE,
)


def fail(message: str) -> None:
    raise AssertionError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                fail(f"{path.name}:{line_number}: invalid JSON: {exc}")
    return records


def ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["candidate_id"]) for record in records]


def assert_unique(name: str, values: list[str]) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        fail(f"{name}: duplicate candidate_id values: {duplicates[:5]}")


def scan_for_secrets(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".md", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        match = SECRET_PATTERN.search(content)
        if match:
            fail(f"{path}: possible secret token: {match.group(0)[:20]}")


def verify_hash_manifest() -> None:
    manifest = ROOT / "DATA_SHA256SUMS.txt"
    if not manifest.is_file():
        fail("Missing DATA_SHA256SUMS.txt")
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            expected, relative_path = line.split(maxsplit=1)
        except ValueError as exc:
            raise AssertionError(
                f"Invalid hash manifest line {line_number}: {line}"
            ) from exc
        path = ROOT / relative_path
        if not path.is_file():
            fail(f"Hash manifest path missing: {relative_path}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            fail(
                f"SHA256 mismatch for {relative_path}: "
                f"expected {expected}, observed {observed}"
            )


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        fail(f"Missing required files: {missing}")

    candidates = read_jsonl(DATA / "candidates_blind_94.jsonl")
    dimensions = read_csv(DATA / "codex_direct_v10_dimensions.csv")
    judgments = read_jsonl(DATA / "codex_direct_v10_judgments_94.jsonl")
    scores = read_csv(DATA / "codex_direct_v10_scores_94.csv")
    targets = read_csv(DATA / "validation_targets_94_PRIVATE.csv")
    groups = read_csv(DATA / "group_split_94_PRIVATE.csv")
    oof = read_csv(DIAGNOSTICS / "oof_predictions_PRIVATE.csv")

    datasets = {
        "candidates": candidates,
        "dimensions": dimensions,
        "judgments": judgments,
        "scores": scores,
        "targets": targets,
        "groups": groups,
        "oof": oof,
    }
    for name, records in datasets.items():
        if len(records) != 94:
            fail(f"{name}: expected 94 rows, got {len(records)}")
        assert_unique(name, ids(records))

    canonical_ids = set(ids(candidates))
    for name, records in datasets.items():
        if set(ids(records)) != canonical_ids:
            fail(f"{name}: candidate_id set differs from candidates")

    blind_keys = set().union(*(record.keys() for record in candidates))
    leaked_keys = sorted(blind_keys & FORBIDDEN_BLIND_KEYS)
    if leaked_keys:
        fail(f"Blind input contains forbidden keys: {leaked_keys}")

    for field in ("description", "transcript", "before_context", "after_context"):
        empty = [
            record["candidate_id"]
            for record in candidates
            if not str(record.get(field, "")).strip()
        ]
        if empty:
            fail(f"Blind input has empty {field}: {empty[:5]}")

    longforms = {str(record["longform_id"]) for record in targets}
    channels = Counter(str(record["channel_name"]) for record in targets)
    labels = Counter(str(record["performance_label_PRIVATE"]) for record in targets)
    sources = Counter(str(record["transcript_source"]) for record in targets)
    roles = Counter(str(record["dataset_role_v3"]) for record in targets)

    if len(longforms) != 85:
        fail(f"Expected 85 longforms, got {len(longforms)}")
    if len(channels) != 6:
        fail(f"Expected 6 channels, got {len(channels)}")
    if labels != Counter({"mid": 34, "pos": 30, "neg": 30}):
        fail(f"Unexpected label distribution: {dict(labels)}")
    if sources != Counter(
        {"vpick_scene_api": 47, "yt_dlp_transcript_fallback": 47}
    ):
        fail(f"Unexpected transcript source distribution: {dict(sources)}")

    target_by_id = {
        row["candidate_id"]: float(row["channel_performance_percentile_PRIVATE"])
        for row in targets
    }
    for row in oof:
        expected = target_by_id[row["candidate_id"]]
        observed = float(row["channel_performance_percentile_PRIVATE"])
        if abs(expected - observed) > 1e-9:
            fail(f"OOF target mismatch for {row['candidate_id']}")

    package_text_files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and "workspace" not in path.parts
    ]
    scan_for_secrets(package_text_files)
    verify_hash_manifest()

    summary = {
        "status": "PASS",
        "candidates": len(candidates),
        "longforms": len(longforms),
        "channels": dict(sorted(channels.items())),
        "performance_labels_diagnostic_only": dict(sorted(labels.items())),
        "transcript_sources_diagnostic_only": dict(sorted(sources.items())),
        "dataset_roles_historical_only": dict(sorted(roles.items())),
        "blind_input_forbidden_key_count": len(leaked_keys),
        "secret_scan": "PASS",
        "data_sha256_manifest": "PASS",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"PACKAGE VERIFICATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
