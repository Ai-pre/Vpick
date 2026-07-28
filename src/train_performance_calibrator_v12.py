from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train_performance_calibrator_v11 import (
    DEFAULT_PRIVATE_DIR,
    QUALITY_COLUMNS,
    ROOT,
    STRUCTURE_COLUMNS,
    acceptance_result,
    grouped_splits,
    json_safe,
    load_bundle,
    markdown_table,
    performance_metrics,
    write_csv,
)


DEFAULT_CONFIG = ROOT / "config" / "performance_calibrator_v12.json"
DEFAULT_PUBLIC_DIR = ROOT / "results" / "performance_calibrator_v12"
DEFAULT_PRIVATE_OUTPUT = (
    DEFAULT_PRIVATE_DIR / "performance_calibrator_v12"
)
DEFAULT_REPORT = ROOT / "reports" / "performance_calibrator_v12_2026-07-28.md"

PUBLIC_METRIC_NAMES = [
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


@dataclass(frozen=True)
class RankerSpec:
    name: str
    representation: str
    score_calibration: str
    engineered_numeric: bool
    channel_balanced_pairs: bool
    min_gap: float
    local_boost: float
    reliability_weighting: bool
    cross_channel_weight: float = 0.0
    numeric_scale: float = 1.0
    char_ngram_min: int = 2
    char_ngram_max: int = 5
    char_max_features: int = 4000


SPECS = [
    RankerSpec(
        name="baseline_v11_raw",
        representation="concat_raw_char",
        score_calibration="raw",
        engineered_numeric=False,
        channel_balanced_pairs=False,
        min_gap=0.05,
        local_boost=1.0,
        reliability_weighting=False,
    ),
    RankerSpec(
        name="baseline_v11_ecdf",
        representation="concat_raw_char",
        score_calibration="train_ecdf",
        engineered_numeric=False,
        channel_balanced_pairs=False,
        min_gap=0.05,
        local_boost=1.0,
        reliability_weighting=False,
    ),
    RankerSpec(
        name="normalized_char_ecdf",
        representation="concat_normalized_char",
        score_calibration="train_ecdf",
        engineered_numeric=True,
        channel_balanced_pairs=False,
        min_gap=0.05,
        local_boost=1.0,
        reliability_weighting=False,
    ),
    RankerSpec(
        name="field_aware_ecdf",
        representation="field_aware_char_word",
        score_calibration="train_ecdf",
        engineered_numeric=True,
        channel_balanced_pairs=False,
        min_gap=0.05,
        local_boost=1.0,
        reliability_weighting=False,
    ),
    RankerSpec(
        name="field_channel_balanced",
        representation="field_aware_char_word",
        score_calibration="train_ecdf",
        engineered_numeric=True,
        channel_balanced_pairs=True,
        min_gap=0.05,
        local_boost=1.0,
        reliability_weighting=False,
    ),
    RankerSpec(
        name="registered_v12",
        representation="field_aware_char_word",
        score_calibration="train_ecdf",
        engineered_numeric=True,
        channel_balanced_pairs=True,
        min_gap=0.03,
        local_boost=1.5,
        reliability_weighting=True,
    ),
    RankerSpec(
        name="registered_v12_cross",
        representation="field_aware_char_word",
        score_calibration="train_ecdf",
        engineered_numeric=True,
        channel_balanced_pairs=True,
        min_gap=0.03,
        local_boost=1.5,
        reliability_weighting=True,
        cross_channel_weight=0.10,
    ),
]


@dataclass
class PreparedFold:
    train_matrix: sparse.csr_matrix
    test_matrix: sparse.csr_matrix
    pair_matrix: sparse.csr_matrix
    pair_labels: np.ndarray
    pair_weights: np.ndarray
    base_pair_count: int
    same_channel_pair_count: int
    cross_channel_pair_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fixed-family Vpick performance calibrator v12."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--private-output", type=Path, default=DEFAULT_PRIVATE_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_semantic_text(value: Any) -> str:
    text = safe_text(value)
    text = re.sub(
        r"\[(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\s*-\s*"
        r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\]",
        " ",
        text,
    )
    text = re.sub(
        r"(?m)(^|\s)(?:S\d+|S\?|화자\s*\d*|speaker\s*\d*)\s*:",
        r"\1화자:",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compose_normalized_text(row: Any, include_context: bool = True) -> str:
    parts = [
        "[DESCRIPTION]",
        normalize_semantic_text(row.description),
        "[TRANSCRIPT]",
        normalize_semantic_text(row.transcript),
    ]
    if include_context:
        parts.extend(
            [
                "[BEFORE]",
                normalize_semantic_text(row.before_context),
                "[AFTER]",
                normalize_semantic_text(row.after_context),
            ]
        )
    return "\n".join(parts)


def quality_interactions(quality: np.ndarray) -> np.ndarray:
    normalized = np.asarray(quality, dtype=float) / 4.0
    return np.column_stack(
        [
            np.mean(normalized, axis=1),
            np.min(normalized, axis=1),
            np.std(normalized, axis=1),
            np.mean(normalized[:, :3], axis=1),
            np.mean(normalized[:, 3:], axis=1),
            np.minimum(normalized[:, 3], normalized[:, 1]),
            normalized[:, 3] * normalized[:, 1],
            normalized[:, 0] * normalized[:, 2],
            normalized[:, 4] * normalized[:, 5],
            normalized[:, 5] * normalized[:, 6],
        ]
    )


def numeric_features(bundle: Any, engineered: bool) -> np.ndarray:
    base = np.asarray(bundle.quality_structure, dtype=float)
    if not engineered:
        return base
    return np.column_stack([base, quality_interactions(bundle.quality)])


def confidence_weight(value: Any) -> float:
    normalized = safe_text(value).lower()
    mapping = {
        "high": 1.0,
        "medium": 0.85,
        "mid": 0.85,
        "low": 0.65,
        "manual": 0.9,
    }
    return mapping.get(normalized, 0.8)


def candidate_reliability(bundle: Any) -> np.ndarray:
    values = []
    for row in bundle.frame.itertuples(index=False):
        mapping = confidence_weight(getattr(row, "mapping_confidence", ""))
        timestamp = confidence_weight(getattr(row, "timestamp_confidence", ""))
        values.append(math.sqrt(mapping * timestamp))
    return np.asarray(values, dtype=float)


def fit_vectorizer(
    train_texts: list[str],
    test_texts: list[str],
    **kwargs: Any,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(**kwargs)
    try:
        return (
            vectorizer.fit_transform(train_texts).tocsr(),
            vectorizer.transform(test_texts).tocsr(),
        )
    except ValueError as error:
        if "empty vocabulary" not in str(error).lower():
            raise
        return (
            sparse.csr_matrix((len(train_texts), 1), dtype=float),
            sparse.csr_matrix((len(test_texts), 1), dtype=float),
        )


def build_text_matrices(
    bundle: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    spec: RankerSpec,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    records = list(bundle.frame.itertuples(index=False))
    if spec.representation in {"concat_raw_char", "concat_raw_char_word"}:
        train_texts = [bundle.texts[index] for index in train_indices]
        test_texts = [bundle.texts[index] for index in test_indices]
        char_train, char_test = fit_vectorizer(
            train_texts,
            test_texts,
            analyzer="char_wb",
            ngram_range=(spec.char_ngram_min, spec.char_ngram_max),
            min_df=2,
            max_df=0.98,
            max_features=spec.char_max_features,
            sublinear_tf=True,
            norm="l2",
        )
        if spec.representation == "concat_raw_char":
            return char_train, char_test
        word_train, word_test = fit_vectorizer(
            train_texts,
            test_texts,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=3000,
            sublinear_tf=True,
            norm="l2",
            token_pattern=r"(?u)\b\w+\b",
        )
        return (
            sparse.hstack([char_train, word_train * 0.75], format="csr"),
            sparse.hstack([char_test, word_test * 0.75], format="csr"),
        )

    if spec.representation == "concat_normalized_char":
        train_texts = [
            compose_normalized_text(records[index]) for index in train_indices
        ]
        test_texts = [
            compose_normalized_text(records[index]) for index in test_indices
        ]
        return fit_vectorizer(
            train_texts,
            test_texts,
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_df=0.98,
            max_features=5000,
            sublinear_tf=True,
            norm="l2",
        )

    if spec.representation != "field_aware_char_word":
        raise ValueError(f"Unknown representation: {spec.representation}")

    def semantic(row: Any) -> str:
        return "\n".join(
            [
                "[DESCRIPTION]",
                normalize_semantic_text(row.description),
                "[TRANSCRIPT]",
                normalize_semantic_text(row.transcript),
            ]
        )

    def context(row: Any) -> str:
        return "\n".join(
            [
                "[BEFORE]",
                normalize_semantic_text(row.before_context),
                "[AFTER]",
                normalize_semantic_text(row.after_context),
            ]
        )

    train_semantic = [semantic(records[index]) for index in train_indices]
    test_semantic = [semantic(records[index]) for index in test_indices]
    train_context = [context(records[index]) for index in train_indices]
    test_context = [context(records[index]) for index in test_indices]
    semantic_char_train, semantic_char_test = fit_vectorizer(
        train_semantic,
        test_semantic,
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.98,
        max_features=5000,
        sublinear_tf=True,
        norm="l2",
    )
    semantic_word_train, semantic_word_test = fit_vectorizer(
        train_semantic,
        test_semantic,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=3000,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    context_char_train, context_char_test = fit_vectorizer(
        train_context,
        test_context,
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.98,
        max_features=1500,
        sublinear_tf=True,
        norm="l2",
    )
    return (
        sparse.hstack(
            [
                semantic_char_train,
                semantic_word_train * 0.75,
                context_char_train * 0.50,
            ],
            format="csr",
        ),
        sparse.hstack(
            [
                semantic_char_test,
                semantic_word_test * 0.75,
                context_char_test * 0.50,
            ],
            format="csr",
        ),
    )


def build_feature_matrices(
    bundle: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    spec: RankerSpec,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    text_train, text_test = build_text_matrices(
        bundle,
        train_indices,
        test_indices,
        spec,
    )
    numeric = numeric_features(bundle, spec.engineered_numeric)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    numeric_train = imputer.fit_transform(numeric[train_indices])
    numeric_test = imputer.transform(numeric[test_indices])
    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(numeric_train)
    numeric_test = scaler.transform(numeric_test)
    numeric_train *= spec.numeric_scale
    numeric_test *= spec.numeric_scale
    return (
        sparse.hstack(
            [text_train, sparse.csr_matrix(numeric_train)],
            format="csr",
        ),
        sparse.hstack(
            [text_test, sparse.csr_matrix(numeric_test)],
            format="csr",
        ),
    )


def pair_infos(
    y: np.ndarray,
    channels: np.ndarray,
    reliability: np.ndarray,
    spec: RankerSpec,
) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for channel in sorted(set(channels)):
        indices = np.flatnonzero(channels == channel)
        channel_infos = []
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                gap = abs(float(y[left] - y[right]))
                if gap < spec.min_gap:
                    continue
                weight = 1.0
                if 0.10 <= gap <= 0.40:
                    weight *= spec.local_boost
                if spec.reliability_weighting:
                    weight *= math.sqrt(
                        float(reliability[left]) * float(reliability[right])
                    )
                channel_infos.append(
                    {
                        "left": int(left),
                        "right": int(right),
                        "gap": gap,
                        "channel": str(channel),
                        "same_channel": True,
                        "weight": weight,
                    }
                )
        if spec.channel_balanced_pairs and channel_infos:
            total = sum(float(info["weight"]) for info in channel_infos)
            for info in channel_infos:
                info["weight"] = float(info["weight"]) / total
        infos.extend(channel_infos)

    same_total = sum(float(info["weight"]) for info in infos)
    if spec.cross_channel_weight > 0:
        cross_infos = []
        for left in range(len(y)):
            for right in range(left + 1, len(y)):
                if channels[left] == channels[right]:
                    continue
                gap = abs(float(y[left] - y[right]))
                if gap < spec.min_gap:
                    continue
                weight = 1.0
                if 0.10 <= gap <= 0.40:
                    weight *= spec.local_boost
                if spec.reliability_weighting:
                    weight *= math.sqrt(
                        float(reliability[left]) * float(reliability[right])
                    )
                cross_infos.append(
                    {
                        "left": left,
                        "right": right,
                        "gap": gap,
                        "channel": "__cross__",
                        "same_channel": False,
                        "weight": weight,
                    }
                )
        cross_total = sum(float(info["weight"]) for info in cross_infos)
        if cross_total > 0 and same_total > 0:
            scale = spec.cross_channel_weight * same_total / cross_total
            for info in cross_infos:
                info["weight"] = float(info["weight"]) * scale
        infos.extend(cross_infos)

    if not infos:
        return infos
    mean_weight = float(np.mean([float(info["weight"]) for info in infos]))
    for info in infos:
        info["weight"] = float(info["weight"]) / mean_weight
    return infos


def build_pair_matrix(
    train_matrix: sparse.csr_matrix,
    train_y: np.ndarray,
    train_channels: np.ndarray,
    train_reliability: np.ndarray,
    spec: RankerSpec,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    infos = pair_infos(
        train_y,
        train_channels,
        train_reliability,
        spec,
    )
    if not infos:
        raise ValueError(f"{spec.name}: no eligible training pairs")
    rows = []
    labels = []
    weights = []
    for info in infos:
        left = int(info["left"])
        right = int(info["right"])
        difference = train_matrix[left] - train_matrix[right]
        label = int(train_y[left] > train_y[right])
        rows.extend([difference, -difference])
        labels.extend([label, 1 - label])
        weights.extend([float(info["weight"]), float(info["weight"])])
    same_count = sum(bool(info["same_channel"]) for info in infos)
    cross_count = len(infos) - same_count
    return (
        sparse.vstack(rows, format="csr"),
        np.asarray(labels, dtype=int),
        np.asarray(weights, dtype=float),
        len(infos),
        same_count,
        cross_count,
    )


def prepare_fold(
    bundle: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    spec: RankerSpec,
    reliability: np.ndarray,
) -> PreparedFold:
    train_matrix, test_matrix = build_feature_matrices(
        bundle,
        train_indices,
        test_indices,
        spec,
    )
    (
        pair_matrix,
        pair_labels,
        pair_weights,
        pair_count,
        same_count,
        cross_count,
    ) = build_pair_matrix(
        train_matrix,
        bundle.y[train_indices],
        bundle.channels[train_indices],
        reliability[train_indices],
        spec,
    )
    return PreparedFold(
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        pair_matrix=pair_matrix,
        pair_labels=pair_labels,
        pair_weights=pair_weights,
        base_pair_count=pair_count,
        same_channel_pair_count=same_count,
        cross_channel_pair_count=cross_count,
    )


def fit_prepared(
    prepared: PreparedFold,
    c_value: float,
    score_calibration: str,
    seed: int,
) -> np.ndarray:
    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        max_iter=3000,
        random_state=seed,
    )
    model.fit(
        prepared.pair_matrix,
        prepared.pair_labels,
        sample_weight=prepared.pair_weights,
    )
    test_scores = np.asarray(
        model.decision_function(prepared.test_matrix),
        dtype=float,
    )
    if score_calibration == "raw":
        return test_scores
    train_raw_scores = np.asarray(
        model.decision_function(prepared.train_matrix),
        dtype=float,
    )
    if score_calibration == "train_zscore":
        scale = float(np.std(train_raw_scores))
        if scale < 1e-12:
            return np.zeros(len(test_scores), dtype=float)
        return (test_scores - float(np.mean(train_raw_scores))) / scale
    if score_calibration != "train_ecdf":
        raise ValueError(f"Unknown score calibration: {score_calibration}")
    train_scores = np.sort(train_raw_scores)
    ranks = np.searchsorted(train_scores, test_scores, side="right")
    return (ranks + 0.5) / (len(train_scores) + 1.0)


def public_metrics(
    bundle: Any,
    scores: np.ndarray,
) -> dict[str, float | int]:
    metrics = performance_metrics(
        bundle.y,
        scores,
        bundle.channels,
        bundle.sources,
    )
    return {name: metrics[name] for name in PUBLIC_METRIC_NAMES}


def select_c(
    bundle: Any,
    outer_train_indices: np.ndarray,
    spec: RankerSpec,
    reliability: np.ndarray,
    c_values: list[float],
    inner_splits: int,
    seed: int,
) -> tuple[float, dict[str, float]]:
    splits = grouped_splits(
        bundle.groups[outer_train_indices],
        inner_splits,
        seed,
    )
    predictions = {
        float(c_value): np.full(len(outer_train_indices), np.nan, dtype=float)
        for c_value in c_values
    }
    for fold_index, (inner_train_local, inner_test_local) in enumerate(splits):
        inner_train = outer_train_indices[inner_train_local]
        inner_test = outer_train_indices[inner_test_local]
        prepared = prepare_fold(
            bundle,
            inner_train,
            inner_test,
            spec,
            reliability,
        )
        for c_value in c_values:
            predictions[float(c_value)][inner_test_local] = fit_prepared(
                prepared,
                float(c_value),
                spec.score_calibration,
                seed + fold_index,
            )
    scores = {}
    for c_value, values in predictions.items():
        metrics = performance_metrics(
            bundle.y[outer_train_indices],
            values,
            bundle.channels[outer_train_indices],
            bundle.sources[outer_train_indices],
        )
        scores[str(c_value)] = float(metrics["selection_score"])
    selected = max(
        c_values,
        key=lambda value: (
            scores[str(float(value))]
            if math.isfinite(scores[str(float(value))])
            else -math.inf,
            -abs(math.log10(float(value))),
        ),
    )
    return float(selected), scores


def repeated_nested_oof(
    bundle: Any,
    spec: RankerSpec,
    reliability: np.ndarray,
    seeds: list[int],
    outer_splits: int,
    inner_splits: int,
    c_values: list[float],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_sum = np.zeros(len(bundle.y), dtype=float)
    prediction_count = np.zeros(len(bundle.y), dtype=int)
    tuning_log = []
    repeat_metrics = []
    for repeat_index, seed in enumerate(seeds):
        repeat_predictions = np.full(len(bundle.y), np.nan, dtype=float)
        for fold_index, (train_indices, test_indices) in enumerate(
            grouped_splits(bundle.groups, outer_splits, seed)
        ):
            selected_c, inner_scores = select_c(
                bundle,
                train_indices,
                spec,
                reliability,
                c_values,
                inner_splits,
                seed + 1000 + fold_index,
            )
            prepared = prepare_fold(
                bundle,
                train_indices,
                test_indices,
                spec,
                reliability,
            )
            fold_predictions = fit_prepared(
                prepared,
                selected_c,
                spec.score_calibration,
                seed + fold_index,
            )
            repeat_predictions[test_indices] = fold_predictions
            prediction_sum[test_indices] += fold_predictions
            prediction_count[test_indices] += 1
            tuning_log.append(
                {
                    "spec": spec.name,
                    "repeat_index": repeat_index,
                    "outer_fold": fold_index,
                    "selected_c": selected_c,
                    "inner_scores": inner_scores,
                    "train_count": int(len(train_indices)),
                    "test_count": int(len(test_indices)),
                    "test_longform_count": int(
                        len(set(bundle.groups[test_indices]))
                    ),
                    "training_pair_count": prepared.base_pair_count,
                    "same_channel_pair_count": (
                        prepared.same_channel_pair_count
                    ),
                    "cross_channel_pair_count": (
                        prepared.cross_channel_pair_count
                    ),
                }
            )
        repeat_metrics.append(
            {
                "repeat_index": repeat_index,
                **public_metrics(bundle, repeat_predictions),
            }
        )
    if np.any(prediction_count != len(seeds)):
        raise RuntimeError(
            f"{spec.name}: invalid OOF coverage "
            f"{Counter(prediction_count.tolist())}"
        )
    return prediction_sum / prediction_count, tuning_log, repeat_metrics


def leave_one_channel_out(
    bundle: Any,
    spec: RankerSpec,
    reliability: np.ndarray,
    c_values: list[float],
    inner_splits: int,
    seed: int,
) -> np.ndarray:
    predictions = np.full(len(bundle.y), np.nan, dtype=float)
    for channel_index, channel in enumerate(sorted(set(bundle.channels))):
        test_indices = np.flatnonzero(bundle.channels == channel)
        train_indices = np.flatnonzero(bundle.channels != channel)
        selected_c, _ = select_c(
            bundle,
            train_indices,
            spec,
            reliability,
            c_values,
            inner_splits,
            seed + channel_index,
        )
        prepared = prepare_fold(
            bundle,
            train_indices,
            test_indices,
            spec,
            reliability,
        )
        predictions[test_indices] = fit_prepared(
            prepared,
            selected_c,
            spec.score_calibration,
            seed + channel_index,
        )
    return predictions


def bootstrap_intervals(
    bundle: Any,
    scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(bundle.groups)), dtype=object)
    group_indices = {
        group: np.flatnonzero(bundle.groups == group)
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
        metrics = performance_metrics(
            bundle.y[indices],
            scores[indices],
            bundle.channels[indices],
            bundle.sources[indices],
        )
        for name in metric_names:
            value = float(metrics[name])
            if math.isfinite(value):
                samples[name].append(value)
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
            "valid_repetitions": len(values),
        }
        for name, values in samples.items()
    }


def candidate_input_audit(bundle: Any) -> dict[str, Any]:
    frame = bundle.frame

    def complete(column: str, minimum: int = 1) -> int:
        return int(
            frame[column]
            .map(lambda value: len(safe_text(value)) >= minimum)
            .sum()
        )

    return {
        "candidate_count": int(len(frame)),
        "longform_count": int(frame["longform_id"].nunique()),
        "channel_count": int(frame["channel_name"].nunique()),
        "complete_description_count": complete("description", 8),
        "complete_transcript_count": complete("transcript", 20),
        "complete_before_context_count": complete("before_context"),
        "complete_after_context_count": complete("after_context"),
        "nonempty_longform_overview_count": int(
            frame["longform_overview"].map(bool).sum()
        ),
        "nonempty_scene_ids_count": int(frame["scene_ids"].map(bool).sum()),
        "visual_evidence_candidate_count": int(
            frame["visual_evidence_available"].astype(bool).sum()
        ),
        "longform_candidate_multiplicity": dict(
            sorted(
                Counter(
                    frame.groupby("longform_id").size().astype(int).tolist()
                ).items()
            )
        ),
    }


def write_report(
    path: Path,
    config: dict[str, Any],
    input_audit: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    primary_metrics: dict[str, Any],
    best_development: dict[str, Any],
    loco_metrics: dict[str, Any],
    gates: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    accepted: bool,
) -> None:
    comparison = markdown_table(
        comparison_rows,
        [
            ("spec", "실험"),
            ("channel_centered_spearman", "채널 중심 rho"),
            ("channel_macro_spearman", "채널 Macro rho"),
            ("same_channel_pairwise_accuracy", "Pairwise"),
            ("same_channel_local_pairwise_accuracy", "Local Pairwise"),
            ("selection_score", "선택 점수"),
        ],
    )
    gate_table = markdown_table(
        gates,
        [
            ("gate", "게이트"),
            ("observed", "관측값"),
            ("required_minimum", "최소 기준"),
            ("passed", "통과"),
        ],
    )
    ci = bootstrap["channel_centered_spearman"]
    status = "채택" if accepted else "기각"
    text = f"""# Vpick 성과 보정기 v12 개선·검증 보고서

## 1. 목적

기존 최고 개발 모델인 Pairwise 문자 TF-IDF + 수치 특징 구조를 고정하고,
같은 94개 데이터에서 구현 가능한 개선을 사전 정의해 검증했다. Pos/Neg,
AUC, 채널명, 조회수, 좋아요, 성과 백분위, URL, 자막 출처는 모델 입력으로
사용하지 않았다.

## 2. 데이터 상태

- 후보: {input_audit['candidate_count']}개
- 롱폼: {input_audit['longform_count']}개
- 설명 완성: {input_audit['complete_description_count']}/{input_audit['candidate_count']}
- 후보 자막 완성: {input_audit['complete_transcript_count']}/{input_audit['candidate_count']}
- 앞·뒤 문맥 완성: {input_audit['complete_before_context_count']}/{input_audit['candidate_count']},
  {input_audit['complete_after_context_count']}/{input_audit['candidate_count']}
- 전체 롱폼 개요 보유: {input_audit['nonempty_longform_overview_count']}/{input_audit['candidate_count']}
- 시각 근거 보유: {input_audit['visual_evidence_candidate_count']}/{input_audit['candidate_count']}

후보 단위 텍스트는 완성되어 있다. 다만 전체 롱폼 개요와 시각 근거가 없으므로
텍스트 기반 성과 일치도 실험이며 프로덕션 동등 멀티모달 검증은 아니다.

## 3. 등록 개선안

1. 모델군 자동 선택을 중단하고 Pairwise 선형 ranker를 고정했다.
2. fold별 raw score를 학습 점수 ECDF로 교정했다.
3. 타임스탬프·화자 표기를 정규화했다.
4. 설명·자막과 경계 문맥에 필드별 문자·단어 TF-IDF를 적용했다.
5. Codex 7개 특징의 사전 정의 상호작용을 추가했다.
6. 채널별 학습 쌍의 총 가중치를 균등화했다.
7. 백분위 차이 10~40인 근접 쌍을 1.5배 강화했다.
8. 매핑·타임스탬프 신뢰도는 학습 가중치에만 사용했다.

외부 5-fold·내부 4-fold GroupKFold를 롱폼 ID로 분리해 3개 seed로 반복했다.
외부 fold 안에서는 정규화·벡터화·C 선택을 모두 다시 수행했다.

## 4. Ablation

{comparison}

등록 주 모델은 `{config['registered_primary_spec']}`이다. 사후 최고 개발 실험은
`{best_development['spec']}`이며, 이 값은 비교 후 선택된 낙관적 수치로만
보고한다.

## 5. 등록 모델 결과

- 채널 중심 Spearman: {primary_metrics['channel_centered_spearman']:.4f}
- 채널 Macro Spearman: {primary_metrics['channel_macro_spearman']:.4f}
- Pairwise 정확도: {primary_metrics['same_channel_pairwise_accuracy']:.4f}
- Local Pairwise 정확도: {primary_metrics['same_channel_local_pairwise_accuracy']:.4f}
- Leave-one-channel-out 채널 중심 Spearman:
  {loco_metrics['channel_centered_spearman']:.4f}
- 2,000회 롱폼 bootstrap 채널 중심 Spearman 95% CI:
  [{ci['lower_95']:.4f}, {ci['upper_95']:.4f}]

## 6. 검증 게이트

판정: **{status}**

{gate_table}

출처 관련 수치는 게이트에 포함하지 않았다.

## 7. 결론

현재 등록 개선안은 고정 LLM 품질 점수와 v11 raw-score 기준선보다 강한지
ablation으로 판단한다. 단, 94개 전체를 이미 개발 과정에서 반복 사용했으므로
어떤 개선 수치도 최종 일반화 성능으로 확정하지 않는다. 다음 최우선 데이터는
동일 롱폼에서 실제 공개된 숏폼이 3개 이상인 묶음과, 모델·가중치를 고정한 뒤
수집한 신규 미공개 holdout이다.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle = load_bundle(args.private_dir)
    reliability = candidate_reliability(bundle)
    seeds = [int(value) for value in config["random_seeds"]]
    c_values = [float(value) for value in config["c_values"]]
    args.public_dir.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, np.ndarray] = {}
    comparison_rows = []
    tuning_logs = {}
    repeat_metrics = {}
    for spec in SPECS:
        print(f"[v12] evaluating {spec.name}", flush=True)
        scores, tuning_log, repeats = repeated_nested_oof(
            bundle,
            spec,
            reliability,
            seeds,
            int(config["outer_splits"]),
            int(config["inner_splits"]),
            c_values,
        )
        predictions[spec.name] = scores
        metrics = public_metrics(bundle, scores)
        comparison_rows.append({"spec": spec.name, **metrics})
        tuning_logs[spec.name] = tuning_log
        repeat_metrics[spec.name] = repeats

    comparison_rows.sort(
        key=lambda row: float(row["selection_score"]),
        reverse=True,
    )
    primary_name = str(config["registered_primary_spec"])
    spec_by_name = {spec.name: spec for spec in SPECS}
    if primary_name not in spec_by_name:
        raise ValueError(f"Unknown registered primary spec: {primary_name}")
    primary_metrics = next(
        row for row in comparison_rows if row["spec"] == primary_name
    )
    best_development = comparison_rows[0]
    primary_scores = predictions[primary_name]
    bootstrap = bootstrap_intervals(
        bundle,
        primary_scores,
        int(config["bootstrap_repetitions"]),
        seeds[0],
    )
    accepted, gates = acceptance_result(
        primary_metrics,
        bootstrap,
        config["acceptance_gates"],
    )
    print("[v12] evaluating leave-one-channel-out", flush=True)
    loco_scores = leave_one_channel_out(
        bundle,
        spec_by_name[primary_name],
        reliability,
        c_values,
        int(config["inner_splits"]),
        seeds[0],
    )
    loco_metrics = public_metrics(bundle, loco_scores)
    input_audit = candidate_input_audit(bundle)

    oof = bundle.frame[
        [
            "candidate_id",
            "longform_id",
            "channel_name",
            "channel_performance_percentile_PRIVATE",
        ]
    ].copy()
    for name, values in predictions.items():
        oof[f"oof_{name}"] = values
    oof["loco_registered_v12"] = loco_scores
    oof.to_csv(
        args.private_output / "oof_predictions_PRIVATE.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.private_output / "tuning_log_PRIVATE.json").write_text(
        json.dumps(
            json_safe(
                {
                    "per_spec": tuning_logs,
                    "repeat_metrics": repeat_metrics,
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    write_csv(args.public_dir / "model_comparison_PUBLIC.csv", comparison_rows)
    summary = {
        "protocol_id": config["protocol_id"],
        "registered_primary_spec": primary_name,
        "accepted_as_performance_judge": accepted,
        "registered_primary_metrics": primary_metrics,
        "best_development_spec": best_development["spec"],
        "best_development_metrics": best_development,
        "leave_one_channel_out_metrics": loco_metrics,
        "acceptance_gates": gates,
        "bootstrap_registered_primary": bootstrap,
        "candidate_input_audit": input_audit,
        "specifications": [asdict(spec) for spec in SPECS],
        "status_note": config["status_note"],
    }
    (args.public_dir / "summary_PUBLIC.json").write_text(
        json.dumps(
            json_safe(summary),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    write_report(
        args.report,
        config,
        input_audit,
        comparison_rows,
        primary_metrics,
        best_development,
        loco_metrics,
        gates,
        bootstrap,
        accepted,
    )
    print(
        json.dumps(
            json_safe(
                {
                    "registered_primary_spec": primary_name,
                    "registered_primary_metrics": primary_metrics,
                    "best_development_spec": best_development["spec"],
                    "best_development_metrics": best_development,
                    "accepted": accepted,
                    "report": str(args.report),
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
