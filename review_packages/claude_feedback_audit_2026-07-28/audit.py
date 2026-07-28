from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_run_summary(path: Path) -> dict[str, float | int]:
    runs = read_json(path)["runs"]
    pair_count = sum(int(run["pair_count"]) for run in runs)
    prediction_count = sum(int(run["prediction_count"]) for run in runs)
    metric_names = [
        "top1_core_hit_rate",
        "top1_tight_hit_rate",
        "hit_at_3_core_rate",
        "hit_at_3_tight_rate",
        "best_iou_at_3_mean",
        "hit_at_5_core_rate",
        "hit_at_5_tight_rate",
        "best_iou_at_5_mean",
    ]
    output: dict[str, float | int] = {
        "pair_count": pair_count,
        "prediction_count": prediction_count,
    }
    for name in metric_names:
        output[name] = sum(
            float(run[name]) * int(run["pair_count"]) for run in runs
        ) / pair_count
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    historical = weighted_run_summary(
        ROOT / "results" / "vpick_historical" / "summary.json"
    )
    corrected = weighted_run_summary(
        ROOT / "results" / "vpick_deduplicated" / "summary.json"
    )
    ours = weighted_run_summary(ROOT / "results" / "ours" / "summary.json")
    duplicate_audit = read_json(
        ROOT / "results" / "vpick_deduplicated" / "summary.json"
    )["prediction_audit"]
    validity = read_json(
        REPO
        / "review_packages"
        / "fable5_v13_cross_validation_2026-07-28"
        / "reference_results"
        / "independent_oof_audit_corrected.json"
    )["metrics"]["v13_repeated_grouped_oof"]

    files = sorted((ROOT / "data").glob("*.csv"))
    output = {
        "experiment3_duplicate_audit": {
            "historical_vpick": historical,
            "deduplicated_vpick": corrected,
            "ours": ours,
            "duplicate_rows_removed": duplicate_audit["duplicate_rows_removed"],
            "groups_with_duplicates": duplicate_audit["groups_with_duplicates"],
            "groups_with_fewer_than_5_unique_candidates": duplicate_audit[
                "groups_with_fewer_than_5_unique_candidates"
            ],
            "core_at_5_changed": (
                corrected["hit_at_5_core_rate"]
                != historical["hit_at_5_core_rate"]
            ),
            "best_iou_at_3_delta": (
                corrected["best_iou_at_3_mean"]
                - historical["best_iou_at_3_mean"]
            ),
        },
        "v13_validity_audit": {
            "channel_centered_spearman_all": validity[
                "channel_centered_spearman"
            ],
            "mid_only_pooled_spearman": validity[
                "mid_only_pooled_spearman"
            ],
            "mid_only_channel_centered_spearman": validity[
                "mid_only_channel_centered_spearman"
            ],
            "extremes_pos_neg_channel_centered_spearman": validity[
                "extremes_pos_neg_channel_centered_spearman"
            ],
            "extremes_pos_neg_auc": validity["extremes_pos_neg_auc"],
            "same_channel_local_pairwise_accuracy": validity[
                "same_channel_local_pairwise_accuracy"
            ],
            "within_label_bucket_pairwise_accuracy": validity[
                "within_label_bucket_pairwise_accuracy"
            ],
        },
        "data_sha256": {path.name: sha256(path) for path in files},
    }
    destination = ROOT / "audit_summary.json"
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
