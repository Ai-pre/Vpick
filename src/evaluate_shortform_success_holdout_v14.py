from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate_shortform_success_holdout_v13 import read_prediction_rows
from train_performance_calibrator_v11 import (
    ROOT,
    finite_spearman,
    json_safe,
    performance_metrics,
    residualize,
)


DEFAULT_CONFIG = ROOT / "config" / "performance_judge_validation_v14.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a packaged success judge with the mid-sensitive v14 protocol."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--holdout-id", required=True)
    parser.add_argument("--confirm-fresh-untouched", action="store_true")
    return parser.parse_args()


def read_targets(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "candidate_id",
        "longform_id",
        "channel_name",
        "channel_performance_percentile_PRIVATE",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Target file is missing columns: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("Target file has duplicate candidate IDs.")

    percentile = pd.to_numeric(
        frame["channel_performance_percentile_PRIVATE"],
        errors="raise",
    )
    thresholds = config["label_thresholds"]
    derived = np.where(
        percentile <= float(thresholds["neg_max_percentile"]),
        "neg",
        np.where(
            percentile >= float(thresholds["pos_min_percentile"]),
            "pos",
            "mid",
        ),
    )
    if "performance_label_PRIVATE" in frame:
        supplied = frame["performance_label_PRIVATE"].fillna("").astype(str).str.lower()
        valid = supplied.isin({"neg", "mid", "pos"})
        frame["evaluation_label_PRIVATE"] = np.where(valid, supplied, derived)
    else:
        frame["evaluation_label_PRIVATE"] = derived
    return frame[
        [
            "candidate_id",
            "longform_id",
            "channel_name",
            "channel_performance_percentile_PRIVATE",
            "evaluation_label_PRIVATE",
        ]
    ].copy()


def metric_bundle(frame: pd.DataFrame) -> dict[str, float | int]:
    y = (
        pd.to_numeric(
            frame["channel_performance_percentile_PRIVATE"],
            errors="raise",
        ).to_numpy(dtype=float)
        / 100.0
    )
    scores = (
        pd.to_numeric(
            frame["shortform_success_potential_0_100"],
            errors="raise",
        ).to_numpy(dtype=float)
        / 100.0
    )
    channels = frame["channel_name"].astype(str).to_numpy()
    labels = frame["evaluation_label_PRIVATE"].astype(str).to_numpy()
    base = performance_metrics(
        y,
        scores,
        channels,
        np.full(len(y), "__holdout__", dtype=object),
    )

    mid = labels == "mid"
    extremes = np.isin(labels, ["neg", "pos"])
    extreme_labels = (labels[extremes] == "pos").astype(int)
    extreme_auc = (
        float(roc_auc_score(extreme_labels, scores[extremes]))
        if len(set(extreme_labels.tolist())) == 2
        else math.nan
    )
    return {
        "candidate_count": int(len(frame)),
        "mid_candidate_count": int(np.sum(mid)),
        "all_channel_centered_spearman": float(
            base["channel_centered_spearman"]
        ),
        "pooled_spearman": float(base["pooled_spearman"]),
        "same_channel_pairwise_accuracy": float(
            base["same_channel_pairwise_accuracy"]
        ),
        "same_channel_local_pairwise_accuracy": float(
            base["same_channel_local_pairwise_accuracy"]
        ),
        "top_quintile_precision": float(base["top_quintile_precision"]),
        "mid_only_pooled_spearman": finite_spearman(y[mid], scores[mid]),
        "mid_only_channel_centered_spearman": finite_spearman(
            residualize(y[mid], channels[mid]),
            residualize(scores[mid], channels[mid]),
        ),
        "extremes_pos_neg_auc": extreme_auc,
    }


def bootstrap_intervals(
    frame: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    group_values = frame["longform_id"].astype(str).to_numpy()
    unique_groups = np.array(sorted(set(group_values)), dtype=object)
    group_indices = {
        group: np.flatnonzero(group_values == group) for group in unique_groups
    }
    metric_names = [
        "mid_only_channel_centered_spearman",
        "same_channel_local_pairwise_accuracy",
        "extremes_pos_neg_auc",
    ]
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        metrics = metric_bundle(frame.iloc[indices])
        for name in metric_names:
            value = float(metrics[name])
            if math.isfinite(value):
                samples[name].append(value)

    output = {}
    for name in metric_names:
        values = samples[name]
        output[name] = {
            "lower_95": (
                float(np.quantile(values, 0.025)) if values else math.nan
            ),
            "median": float(np.quantile(values, 0.5)) if values else math.nan,
            "upper_95": (
                float(np.quantile(values, 0.975)) if values else math.nan
            ),
            "valid_repetitions": len(values),
        }
    return output


def evaluate(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    holdout_id: str,
    confirm_fresh_untouched: bool,
) -> dict[str, Any]:
    prediction_ids = set(predictions["candidate_id"].astype(str))
    target_ids = set(targets["candidate_id"].astype(str))
    if prediction_ids != target_ids:
        raise ValueError(
            "Prediction and target candidate IDs differ: "
            f"missing_predictions={sorted(target_ids - prediction_ids)[:5]}, "
            f"unexpected_predictions={sorted(prediction_ids - target_ids)[:5]}"
        )
    frame = targets.merge(predictions, on="candidate_id", validate="one_to_one")
    metrics = metric_bundle(frame)
    bootstrap = bootstrap_intervals(
        frame,
        int(config["bootstrap_repetitions"]),
        int(config["bootstrap_seed"]),
    )

    minimums = config["minimum_evaluation_size"]
    sample_gates = {
        "total_candidates": {
            "observed": int(metrics["candidate_count"]),
            "required_minimum": int(minimums["total_candidates"]),
        },
        "mid_candidates": {
            "observed": int(metrics["mid_candidate_count"]),
            "required_minimum": int(minimums["mid_candidates"]),
        },
    }
    for gate in sample_gates.values():
        gate["passed"] = gate["observed"] >= gate["required_minimum"]

    requirements = config["acceptance_gates"]
    statistical_gates = {
        "mid_only_channel_centered_spearman": {
            "observed": metrics["mid_only_channel_centered_spearman"],
            "required_minimum": requirements[
                "mid_only_channel_centered_spearman_min"
            ],
            "ci_lower": bootstrap[
                "mid_only_channel_centered_spearman"
            ]["lower_95"],
            "ci_lower_required_minimum": requirements[
                "mid_only_channel_centered_spearman_ci_lower_min"
            ],
        },
        "same_channel_local_pairwise_accuracy": {
            "observed": metrics["same_channel_local_pairwise_accuracy"],
            "required_minimum": requirements[
                "same_channel_local_pairwise_accuracy_min"
            ],
            "ci_lower": bootstrap[
                "same_channel_local_pairwise_accuracy"
            ]["lower_95"],
            "ci_lower_required_minimum": requirements[
                "same_channel_local_pairwise_accuracy_ci_lower_min"
            ],
        },
        "extremes_pos_neg_auc": {
            "observed": metrics["extremes_pos_neg_auc"],
            "required_minimum": requirements["extremes_pos_neg_auc_min"],
            "ci_lower": bootstrap["extremes_pos_neg_auc"]["lower_95"],
            "ci_lower_required_minimum": requirements[
                "extremes_pos_neg_auc_ci_lower_min"
            ],
        },
    }
    for gate in statistical_gates.values():
        gate["passed"] = bool(
            math.isfinite(float(gate["observed"]))
            and math.isfinite(float(gate["ci_lower"]))
            and float(gate["observed"]) >= float(gate["required_minimum"])
            and float(gate["ci_lower"])
            > float(gate["ci_lower_required_minimum"])
        )

    size_pass = all(bool(gate["passed"]) for gate in sample_gates.values())
    statistical_pass = all(
        bool(gate["passed"]) for gate in statistical_gates.values()
    )
    accepted = bool(
        confirm_fresh_untouched and size_pass and statistical_pass
    )
    return {
        "protocol_id": config["protocol_id"],
        "holdout_id": holdout_id,
        "confirmed_fresh_untouched": confirm_fresh_untouched,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "sample_size_gates": sample_gates,
        "statistical_gates": statistical_gates,
        "accepted_as_continuous_performance_judge": accepted,
        "acceptance_note": (
            "Acceptance requires a fresh untouched holdout, minimum total and mid "
            "sample sizes, and all pre-registered point-estimate and CI gates."
        ),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate(
        read_prediction_rows(args.predictions),
        read_targets(args.targets, config),
        config,
        args.holdout_id,
        args.confirm_fresh_untouched,
    )
    rendered = json.dumps(
        json_safe(result),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
