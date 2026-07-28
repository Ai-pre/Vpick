from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_performance_calibrator_v11 import (
    ROOT,
    acceptance_result,
    json_safe,
    performance_metrics,
)


DEFAULT_CONFIG = ROOT / "config" / "performance_calibrator_v13.json"
PUBLIC_METRICS = [
    "pooled_spearman",
    "channel_centered_spearman",
    "channel_macro_spearman",
    "same_channel_pairwise_accuracy",
    "same_channel_pair_count",
    "same_channel_local_pairwise_accuracy",
    "same_channel_local_pair_count",
    "top_quintile_precision",
    "channel_macro_ndcg",
    "robust_rank_score",
    "selection_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen v13 predictions on a fresh holdout."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--holdout-id", required=True)
    parser.add_argument("--confirm-fresh-untouched", action="store_true")
    return parser.parse_args()


def read_prediction_rows(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Prediction JSON must contain a results array.")
    frame = pd.DataFrame.from_records(rows)
    required = {"candidate_id", "shortform_success_potential_0_100"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("Prediction file has duplicate candidate IDs.")
    return frame[
        ["candidate_id", "shortform_success_potential_0_100"]
    ].copy()


def read_targets(path: Path) -> pd.DataFrame:
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
    return frame[list(required)].copy()


def metric_bundle(
    y: np.ndarray,
    scores: np.ndarray,
    channels: np.ndarray,
) -> dict[str, float | int]:
    metrics = performance_metrics(
        y,
        scores,
        channels,
        np.full(len(y), "__holdout__", dtype=object),
    )
    return {name: metrics[name] for name in PUBLIC_METRICS}


def bootstrap_intervals(
    y: np.ndarray,
    scores: np.ndarray,
    channels: np.ndarray,
    groups: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(groups)), dtype=object)
    group_indices = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }
    metric_names = [
        "channel_centered_spearman",
        "channel_macro_spearman",
        "same_channel_pairwise_accuracy",
        "same_channel_local_pairwise_accuracy",
        "robust_rank_score",
    ]
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )
        metrics = metric_bundle(
            y[indices],
            scores[indices],
            channels[indices],
        )
        for name in metric_names:
            value = float(metrics[name])
            if math.isfinite(value):
                samples[name].append(value)
    intervals = {}
    for name in metric_names:
        values = samples[name]
        intervals[name] = {
            "lower_95": (
                float(np.quantile(values, 0.025)) if values else math.nan
            ),
            "median": (
                float(np.quantile(values, 0.5)) if values else math.nan
            ),
            "upper_95": (
                float(np.quantile(values, 0.975)) if values else math.nan
            ),
            "valid_repetitions": len(values),
        }
    return intervals


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
    merged = targets.merge(
        predictions,
        on="candidate_id",
        validate="one_to_one",
    )
    y = (
        pd.to_numeric(
            merged["channel_performance_percentile_PRIVATE"],
            errors="raise",
        ).to_numpy(dtype=float)
        / 100.0
    )
    scores = (
        pd.to_numeric(
            merged["shortform_success_potential_0_100"],
            errors="raise",
        ).to_numpy(dtype=float)
        / 100.0
    )
    channels = merged["channel_name"].astype(str).to_numpy()
    groups = merged["longform_id"].astype(str).to_numpy()
    metrics = metric_bundle(y, scores, channels)
    bootstrap = bootstrap_intervals(
        y,
        scores,
        channels,
        groups,
        int(config["bootstrap_repetitions"]),
        int(config["random_seeds"][0]),
    )
    internal_pass, gates = acceptance_result(
        metrics,
        bootstrap,
        config["acceptance_gates"],
    )
    channels_with_fewer_than_three = sorted(
        name
        for name, count in merged["channel_name"].value_counts().items()
        if int(count) < 3
    )
    final_acceptance = bool(internal_pass and confirm_fresh_untouched)
    return {
        "protocol_id": config["protocol_id"],
        "holdout_id": holdout_id,
        "confirmed_fresh_untouched": confirm_fresh_untouched,
        "candidate_count": int(len(merged)),
        "longform_count": int(merged["longform_id"].nunique()),
        "channel_count": int(merged["channel_name"].nunique()),
        "channels_with_fewer_than_three_candidates": (
            channels_with_fewer_than_three
        ),
        "metrics": metrics,
        "bootstrap": bootstrap,
        "acceptance_gates": gates,
        "internal_gate_pass": internal_pass,
        "accepted_as_performance_judge": final_acceptance,
        "acceptance_note": (
            "Accepted only when all frozen gates pass and the caller confirms "
            "that this target file was untouched during model development."
        ),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate(
        read_prediction_rows(args.predictions),
        read_targets(args.targets),
        config,
        args.holdout_id,
        args.confirm_fresh_untouched,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        json_safe(result),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
