from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


STRUCTURE_FEATURES = [
    "duration_sec",
    "position_ratio",
    "scene_rate_per_min",
    "transcript_line_rate_per_min",
    "question_rate_per_min",
    "exclamation_rate_per_min",
    "speech_coverage_ratio",
    "start_boundary_distance_sec",
    "end_boundary_distance_sec",
]

CODEX_FEATURES = [
    "saliency_market_1_5",
    "check_hook_within_3s",
    "check_surprise_or_twist",
    "check_emotional_peak",
    "check_quotable_moment",
    "check_payoff_or_conclusion",
    "check_natural_start",
    "check_natural_end",
]

GEMINI_FEATURES = [
    "source_salience_score_0_4",
    "hook_score_0_4",
    "payoff_score_0_4",
    "self_contained_score_0_4",
    "density_score_0_4",
    "boundary_score_0_4",
]

# The six-dimension Highlight Quality schema is provider-independent.
RUBRIC_FEATURES = GEMINI_FEATURES

TIMED_CUE_RE = re.compile(r"\[([0-9]+(?::[0-9]+){1,2})-([0-9]+(?::[0-9]+){1,2})\]")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def to_float(value: Any) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.nan
    except (TypeError, ValueError):
        return math.nan


def timestamp_to_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"Unsupported timestamp: {value}")


def timed_cues(text: str) -> list[tuple[float, float]]:
    cues: list[tuple[float, float]] = []
    for match in TIMED_CUE_RE.finditer(text or ""):
        start = timestamp_to_seconds(match.group(1))
        end = timestamp_to_seconds(match.group(2))
        if end > start:
            cues.append((start, end))
    return cues


def interval_coverage(
    intervals: list[tuple[float, float]],
    start: float,
    end: float,
) -> float:
    clipped = sorted(
        (max(start, left), min(end, right))
        for left, right in intervals
        if right > start and left < end
    )
    if not clipped or end <= start:
        return 0.0
    merged: list[list[float]] = []
    for left, right in clipped:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    covered = sum(right - left for left, right in merged)
    return min(1.0, covered / (end - start))


def nearest_boundary_distance(
    boundary: float,
    intervals: list[tuple[float, float]],
) -> float:
    points = [point for interval in intervals for point in interval]
    return min((abs(boundary - point) for point in points), default=math.nan)


def extract_structure_features(candidate: dict[str, Any]) -> dict[str, float]:
    start = to_float(candidate.get("start_ms")) / 1000.0
    end = to_float(candidate.get("end_ms")) / 1000.0
    duration = max(end - start, 1e-6)
    transcript = str(candidate.get("transcript") or "")
    context = "\n".join(
        str(candidate.get(key) or "")
        for key in ("before_context", "transcript", "after_context")
    )
    transcript_cues = timed_cues(transcript)
    context_cues = timed_cues(context)
    transcript_lines = [
        line for line in transcript.splitlines() if line.strip()
    ]
    overview = candidate.get("longform_overview") or []
    longform_end_ms = max(
        (to_float(scene.get("end_ms")) for scene in overview),
        default=math.nan,
    )
    position_ratio = (
        start * 1000.0 / longform_end_ms
        if math.isfinite(longform_end_ms) and longform_end_ms > 0
        else math.nan
    )
    per_minute = 60.0 / duration
    return {
        "duration_sec": duration,
        "position_ratio": position_ratio,
        "scene_rate_per_min": len(candidate.get("scene_ids") or []) * per_minute,
        "transcript_line_rate_per_min": len(transcript_lines) * per_minute,
        "question_rate_per_min": transcript.count("?") * per_minute,
        "exclamation_rate_per_min": transcript.count("!") * per_minute,
        "speech_coverage_ratio": interval_coverage(
            transcript_cues,
            start,
            end,
        ),
        "start_boundary_distance_sec": nearest_boundary_distance(
            start,
            context_cues,
        ),
        "end_boundary_distance_sec": nearest_boundary_distance(
            end,
            context_cues,
        ),
    }


def feature_matrix(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> np.ndarray:
    return np.array(
        [
            [to_float(row.get(feature)) for feature in feature_names]
            for row in rows
        ],
        dtype=float,
    )


def preprocessing_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(x), x, medians)
    means = filled.mean(axis=0)
    stds = filled.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    return medians, means, stds


def transform_features(
    x: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, medians)
    return (filled - means) / stds


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 2.0,
    max_iter: int = 100,
) -> dict[str, np.ndarray | float]:
    medians, means, stds = preprocessing_stats(x)
    standardized = transform_features(x, medians, means, stds)
    design = np.column_stack([np.ones(len(standardized)), standardized])
    weights = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        probabilities = sigmoid(design @ weights)
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
        gradient = design.T @ (probabilities - y) + penalty @ weights
        hessian = design.T @ (design * variance[:, None]) + penalty
        step = np.linalg.pinv(hessian) @ gradient
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {
        "weights": weights,
        "medians": medians,
        "means": means,
        "stds": stds,
        "alpha": float(alpha),
    }


def predict_logistic(
    model: dict[str, np.ndarray | float],
    x: np.ndarray,
) -> np.ndarray:
    transformed = transform_features(
        x,
        np.asarray(model["medians"], dtype=float),
        np.asarray(model["means"], dtype=float),
        np.asarray(model["stds"], dtype=float),
    )
    design = np.column_stack([np.ones(len(transformed)), transformed])
    return sigmoid(design @ np.asarray(model["weights"], dtype=float))


def grouped_cv_predictions(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    group_key: str,
    alpha: float = 2.0,
) -> np.ndarray:
    x = feature_matrix(rows, feature_names)
    y = np.array([int(row["target"]) for row in rows], dtype=float)
    groups = np.array([str(row[group_key]) for row in rows])
    predictions = np.full(len(rows), math.nan, dtype=float)
    for group in sorted(set(groups)):
        test_mask = groups == group
        train_mask = ~test_mask
        if len(set(y[train_mask])) < 2:
            predictions[test_mask] = float(np.mean(y[train_mask]))
            continue
        model = fit_logistic(x[train_mask], y[train_mask], alpha=alpha)
        predictions[test_mask] = predict_logistic(model, x[test_mask])
    return predictions


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return math.nan
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return math.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return math.nan
    wins = 0.0
    for score in positive:
        wins += float(np.sum(score > negative))
        wins += 0.5 * float(np.sum(score == negative))
    return wins / (len(positive) * len(negative))


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    predictions = scores >= 0.5
    positive = labels == 1
    negative = labels == 0
    sensitivity = float(np.mean(predictions[positive])) if positive.any() else math.nan
    specificity = float(np.mean(~predictions[negative])) if negative.any() else math.nan
    return float(np.nanmean([sensitivity, specificity]))


def evaluate_scores(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, float]:
    labels = np.array([int(row["target"]) for row in rows])
    percentiles = np.array(
        [to_float(row["channel_performance_percentile"]) for row in rows]
    )
    channels = np.array([str(row["channel_name"]) for row in rows])
    channel_aucs: list[float] = []
    channel_correlations: list[float] = []
    for channel in sorted(set(channels)):
        mask = channels == channel
        auc = binary_auc(labels[mask], scores[mask])
        correlation = spearman(percentiles[mask], scores[mask])
        if math.isfinite(auc):
            channel_aucs.append(auc)
        if math.isfinite(correlation):
            channel_correlations.append(correlation)
    return {
        "pooled_auc": binary_auc(labels, scores),
        "balanced_accuracy_at_0_5": balanced_accuracy(labels, scores),
        "macro_channel_auc": (
            float(np.mean(channel_aucs)) if channel_aucs else math.nan
        ),
        "pooled_percentile_spearman": spearman(percentiles, scores),
        "macro_channel_percentile_spearman": (
            float(np.mean(channel_correlations))
            if channel_correlations
            else math.nan
        ),
    }


def bootstrap_group_auc(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    group_key: str = "longform_id",
    iterations: int = 1000,
    seed: int = 20260724,
) -> tuple[float, float]:
    labels = np.array([int(row["target"]) for row in rows])
    groups = np.array([str(row[group_key]) for row in rows])
    unique_groups = sorted(set(groups))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled_groups]
        )
        value = binary_auc(labels[sampled_indices], scores[sampled_indices])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return math.nan, math.nan
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def serializable_model(
    model: dict[str, np.ndarray | float],
    feature_names: list[str],
    model_name: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "performance_judge_v1",
        "model_name": model_name,
        "feature_names": feature_names,
        "alpha": float(model["alpha"]),
        "medians": np.asarray(model["medians"]).tolist(),
        "means": np.asarray(model["means"]).tolist(),
        "stds": np.asarray(model["stds"]).tolist(),
        "weights": np.asarray(model["weights"]).tolist(),
        "output_name": "high_performance_score_0_100",
        "output_semantics": (
            "Balanced extreme-cohort score for channel-relative high versus low "
            "performance. It is not an exact view forecast or a population-calibrated "
            "probability."
        ),
        "validation": validation,
    }


def model_from_artifact(artifact: dict[str, Any]) -> dict[str, np.ndarray | float]:
    return {
        "alpha": float(artifact["alpha"]),
        "medians": np.asarray(artifact["medians"], dtype=float),
        "means": np.asarray(artifact["means"], dtype=float),
        "stds": np.asarray(artifact["stds"], dtype=float),
        "weights": np.asarray(artifact["weights"], dtype=float),
    }
