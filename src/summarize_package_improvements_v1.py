from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_package_and_context_v1 import spearman


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["candidate_id"]: row for row in rows}


def group_bootstrap_delta(
    rows: list[dict[str, Any]],
    prediction_a: str,
    prediction_b: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["longform_id"]].append(row)
    groups = sorted(grouped)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = [row for group in sampled_groups for row in grouped[group]]
        target = np.array([float(row["raw_views"]) for row in sample])
        left = np.array([float(row[prediction_a]) for row in sample])
        right = np.array([float(row[prediction_b]) for row in sample])
        deltas.append(spearman(left, target) - spearman(right, target))
    values = np.array(deltas)
    return {
        "mean_delta": round(float(values.mean()), 6),
        "ci95_low": round(float(np.quantile(values, 0.025)), 6),
        "ci95_high": round(float(np.quantile(values, 0.975)), 6),
        "probability_delta_above_zero": round(float(np.mean(values > 0)), 6),
        "iterations": iterations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeat-scores",
        type=Path,
        default=ROOT
        / "results/package_success_judge_v1_codex_direct_2026-07-29/codex_package_judge_3pass_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--aggregate-scores",
        type=Path,
        default=ROOT
        / "results/package_success_judge_v1_codex_direct_2026-07-29/codex_package_judge_aggregate_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "data/private/judge_validation_94/validation_targets_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=ROOT
        / "data/private/judge_validation_94/raw_short_performance_2026-07-29_PRIVATE.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT
        / "results/package_context_performance_v1_2026-07-29/candidate_predictions_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/package_context_performance_v1_2026-07-29/robustness_summary_PRIVATE.json",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    targets = by_id(read_csv(args.targets))
    performance = by_id(read_csv(args.performance))
    repeat_rows = read_csv(args.repeat_scores)
    aggregate = read_csv(args.aggregate_scores)
    predictions = read_csv(args.predictions)

    by_pass: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in repeat_rows:
        by_pass[row["pass_name"]].append(row)
    pass_metrics: dict[str, dict[str, float]] = {}
    pass_vectors: dict[str, dict[str, float]] = {}
    for pass_name, rows in by_pass.items():
        vector = {
            row["candidate_id"]: float(row["joint_package_score_1_10"])
            for row in rows
        }
        ids = sorted(vector)
        score = np.array([vector[candidate_id] for candidate_id in ids])
        views = np.array(
            [float(performance[candidate_id]["view_count"]) for candidate_id in ids]
        )
        percentiles = np.array(
            [
                float(
                    targets[candidate_id][
                        "channel_performance_percentile_PRIVATE"
                    ]
                )
                for candidate_id in ids
            ]
        )
        pass_metrics[pass_name] = {
            "spearman_raw_views": round(spearman(score, views), 6),
            "spearman_channel_percentile": round(
                spearman(score, percentiles), 6
            ),
        }
        pass_vectors[pass_name] = vector

    pass_names = sorted(pass_vectors)
    pairwise_stability: dict[str, float] = {}
    ids = sorted(targets)
    for left_index, left in enumerate(pass_names):
        for right in pass_names[left_index + 1 :]:
            pairwise_stability[f"{left}__{right}"] = round(
                spearman(
                    np.array([pass_vectors[left][candidate_id] for candidate_id in ids]),
                    np.array([pass_vectors[right][candidate_id] for candidate_id in ids]),
                ),
                6,
            )

    repeat_std = np.array(
        [float(row["joint_package_score_1_10_std"]) for row in aggregate]
    )
    oof_rows = [dict(row) for row in predictions]
    locked_rows = [
        dict(row)
        for row in predictions
        if row["split"] == "locked_test"
    ]
    robustness = {
        "pass_metrics": pass_metrics,
        "pairwise_pass_spearman": pairwise_stability,
        "joint_score_repeat_std": {
            "mean": round(float(repeat_std.mean()), 6),
            "median": round(float(np.median(repeat_std)), 6),
            "max": round(float(repeat_std.max()), 6),
        },
        "incremental_value_over_channel_age": {
            "new_3pass_package_all94_group_oof": group_bootstrap_delta(
                oof_rows,
                "prediction_content_package_plus_channel_age",
                "prediction_channel_plus_age",
                args.bootstrap_iterations,
                20260729,
            ),
            "new_3pass_package_locked75_dev19_fit": group_bootstrap_delta(
                locked_rows,
                "lockedfit_prediction_content_package_plus_channel_age",
                "lockedfit_prediction_channel_plus_age",
                args.bootstrap_iterations,
                20260730,
            ),
            "best_fixed_50_25_25_all94_group_oof": group_bootstrap_delta(
                oof_rows,
                "prediction_legacy_balanced_plus_channel_age",
                "prediction_channel_plus_age",
                args.bootstrap_iterations,
                20260731,
            ),
            "best_fixed_50_25_25_locked75_dev19_fit": group_bootstrap_delta(
                locked_rows,
                "lockedfit_prediction_legacy_balanced_plus_channel_age",
                "lockedfit_prediction_channel_plus_age",
                args.bootstrap_iterations,
                20260801,
            ),
        },
        "interpretation": [
            "The three passes are rubric-preserving sensitivity rescoring passes, not independent model replicas.",
            "A narrow repeat-score spread measures formula sensitivity but does not replace human inter-rater reliability.",
            "The paired group bootstrap resamples longforms, preserving within-longform candidate dependence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(robustness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(robustness, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
