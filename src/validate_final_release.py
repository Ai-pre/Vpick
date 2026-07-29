#!/usr/bin/env python3
"""Validate the public Vpick release manifest and headline metrics."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(relative_path: str) -> list[dict[str, str]]:
    path = ROOT / relative_path
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_release() -> list[str]:
    errors: list[str] = []
    manifest = load_json("config/final_system.json")

    referenced_paths = [
        manifest["evaluation_track"]["config"],
        manifest["evaluation_track"]["pointwise_prompt"],
        manifest["evaluation_track"]["report"],
        manifest["evaluation_track"]["public_metrics"],
        manifest["selection_track"]["config"],
        manifest["selection_track"]["listwise_prompt"],
        manifest["selection_track"]["report"],
        manifest["selection_track"]["public_metrics"],
        manifest["release_validation"]["script"],
        manifest["release_validation"]["test"],
    ]
    for relative_path in referenced_paths:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"Unsafe public path: {relative_path}")
        if not (ROOT / path).is_file():
            errors.append(f"Missing referenced file: {relative_path}")

    dataset = manifest["dataset"]
    tier_total = sum(dataset["performance_tiers"].values())
    if tier_total != dataset["shorts"]:
        errors.append(
            f"Performance tier total {tier_total} does not match "
            f"short count {dataset['shorts']}."
        )
    expected_counts = {"channels": 6, "longforms": 85, "shorts": 94}
    for key, expected in expected_counts.items():
        if dataset.get(key) != expected:
            errors.append(
                f"Dataset {key} is {dataset.get(key)}, expected {expected}."
            )
    if dataset.get("private_gold_in_repository") is not False:
        errors.append("Public manifest must declare private_gold_in_repository=false.")

    judge = load_json(manifest["evaluation_track"]["config"])
    public_judge = load_json(manifest["evaluation_track"]["public_metrics"])
    pointwise_weights = [
        dimension["weight"] for dimension in judge["pointwise_dimensions"].values()
    ]
    if abs(sum(pointwise_weights) - 1.0) > 1e-9:
        errors.append(
            f"Pointwise dimension weights sum to {sum(pointwise_weights):.6f}, not 1."
        )
    selected_weights = public_judge["selected_pointwise_formula"]
    expected_weights = {
        "change_or_surprise": 0.4,
        "title_packaging": 0.15,
        "thumbnail_packaging": 0.45,
        "source_salience": 0.0,
    }
    if selected_weights != expected_weights:
        errors.append("Public Judge weights differ from the frozen 40:15:45 formula.")

    improvement = load_json(manifest["selection_track"]["config"])
    listwise_weight_sum = sum(
        improvement["selector"]["judge_dimensions"].values()
    )
    if abs(listwise_weight_sum - 1.0) > 1e-9:
        errors.append(
            f"Listwise Judge weights sum to {listwise_weight_sum:.6f}, not 1."
        )
    if improvement["selector"]["overlap_ratio"]["reject_above"] != 0.58:
        errors.append("Final supplement overlap threshold must be 0.58.")

    public_improvement = load_csv(manifest["selection_track"]["public_metrics"])
    selected_rows = [
        row
        for row in public_improvement
        if row["method"] == "ac_top4_plus_blended_supplement"
    ]
    if len(selected_rows) != 1:
        errors.append("Expected exactly one selected improvement result row.")
    else:
        selected = selected_rows[0]
        frozen = improvement["development_evaluation"]["selected_pipeline"]
        metric_map = {
            "core_at_1": "core_at_1",
            "core_at_3": "core_at_3",
            "core_at_5": "core_at_5",
            "tight_at_5": "tight_at_5",
            "best_iou_at_5": "best_iou_at_5",
        }
        for csv_key, config_key in metric_map.items():
            observed = float(selected[csv_key])
            expected = float(frozen[config_key])
            if abs(observed - expected) > 1e-6:
                errors.append(
                    f"Improvement metric {csv_key} is {observed}, expected {expected}."
                )

    return errors


def main() -> int:
    errors = validate_release()
    summary = {
        "release": "vpick_final_2026_07_30",
        "status": "ok" if not errors else "failed",
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
