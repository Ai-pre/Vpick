from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE_DIR = ROOT / "data" / "private" / "judge_validation_94"
DEFAULT_CONFIG = ROOT / "config" / "performance_calibrator_v11.json"
DEFAULT_PUBLIC_DIR = ROOT / "results" / "performance_calibrator_v11"
DEFAULT_REPORT = ROOT / "reports" / "performance_calibrator_v11_2026-07-28.md"
DEFAULT_RAW_VPICK = ROOT / "data" / "raw" / "vpick"
DEFAULT_FALLBACK = ROOT / "data" / "raw" / "subtitle_fallback_scenes"

QUALITY_COLUMNS = [
    "self_contained_clarity",
    "progression_payoff",
    "boundary_integrity",
    "opening_pull",
    "change_or_surprise",
    "emotional_or_information_gain",
    "memorable_specificity",
]

EVIDENCE_COLUMNS = [
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
]

STRUCTURE_COLUMNS = [
    "duration_sec_feature",
    "transcript_char_count",
    "description_char_count",
    "before_context_char_count",
    "after_context_char_count",
    "transcript_timed_line_count",
    "transcript_line_count",
    "speaker_marker_count",
    "question_count",
    "exclamation_count",
    "bracket_count",
    "hangul_ratio",
    "transcript_chars_per_sec",
    "description_chars_per_sec",
    "visual_evidence_available_feature",
    "description_missing",
    "transcript_missing",
]

MODEL_PARAMS: dict[str, list[Any]] = {
    "fixed_v10": [None],
    "fixed_equal_quality": [None],
    "ridge_quality": [0.1, 1.0, 10.0, 100.0],
    "ridge_quality_structure": [0.1, 1.0, 10.0, 100.0],
    "pairwise_quality_structure": [0.01, 0.1, 1.0],
    "extra_trees_quality_structure": [2, 4, 8],
    "char_tfidf": [1.0, 10.0, 100.0],
    "char_tfidf_numeric": [1.0, 10.0, 100.0],
    "pairwise_char_tfidf": [0.01, 0.1, 1.0],
    "pairwise_char_tfidf_numeric": [0.01, 0.1, 1.0],
    "rank_ensemble_text_structure": [None],
    "stacked_text_structure": [None],
    "constant": [None],
    "random": [None],
    "source_presence_fixed": [None],
    "source_only": [0.1, 1.0, 10.0, 100.0],
}

FIXED_MODELS = {
    "fixed_v10",
    "fixed_equal_quality",
    "constant",
    "random",
}

DIAGNOSTIC_MODELS = {
    "constant",
    "random",
    "source_presence_fixed",
    "source_only",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass
class Bundle:
    frame: pd.DataFrame
    quality: np.ndarray
    quality_structure: np.ndarray
    source_features: np.ndarray
    texts: list[str]
    y: np.ndarray
    groups: np.ndarray
    channels: np.ndarray
    sources: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate the leakage-safe Vpick performance calibrator v11."
    )
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--raw-vpick", type=Path, default=DEFAULT_RAW_VPICK)
    parser.add_argument("--fallback-scenes", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--bootstrap-repetitions", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def nonspace_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def hangul_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    return sum("\uac00" <= char <= "\ud7a3" for char in visible) / len(visible)


def stable_random_score(candidate_id: str) -> float:
    digest = hashlib.sha256(f"v11-random-control:{candidate_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def validate_id_sets(named_frames: dict[str, pd.DataFrame]) -> list[str]:
    id_sets = {
        name: set(frame["candidate_id"].astype(str))
        for name, frame in named_frames.items()
    }
    names = list(id_sets)
    reference = id_sets[names[0]]
    errors = []
    for name in names[1:]:
        missing = sorted(reference - id_sets[name])
        extra = sorted(id_sets[name] - reference)
        if missing or extra:
            errors.append(
                f"{name}: missing={missing[:5]} ({len(missing)}), "
                f"extra={extra[:5]} ({len(extra)})"
            )
    duplicates = {
        name: frame.loc[frame["candidate_id"].duplicated(), "candidate_id"].astype(str).tolist()
        for name, frame in named_frames.items()
    }
    duplicates = {name: values for name, values in duplicates.items() if values}
    if duplicates:
        errors.append(f"duplicate candidate IDs: {duplicates}")
    if errors:
        raise ValueError("Candidate bundle mismatch: " + " | ".join(errors))
    return sorted(reference)


def build_structure_features(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in frame.to_dict("records"):
        transcript = safe_text(row.get("transcript"))
        description = safe_text(row.get("description"))
        before = safe_text(row.get("before_context"))
        after = safe_text(row.get("after_context"))
        duration_value = (
            row.get("duration_sec")
            or row.get("duration_sec_y")
            or row.get("duration_sec_x")
            or 0.0
        )
        duration = max(float(duration_value), 1.0)
        transcript_chars = nonspace_length(transcript)
        description_chars = nonspace_length(description)
        records.append(
            {
                "candidate_id": row["candidate_id"],
                "duration_sec_feature": duration,
                "transcript_char_count": math.log1p(transcript_chars),
                "description_char_count": math.log1p(description_chars),
                "before_context_char_count": math.log1p(nonspace_length(before)),
                "after_context_char_count": math.log1p(nonspace_length(after)),
                "transcript_timed_line_count": math.log1p(
                    len(re.findall(r"(?m)^\s*\[[0-9:.\s-]+\]", transcript))
                ),
                "transcript_line_count": math.log1p(
                    len([line for line in transcript.splitlines() if line.strip()])
                ),
                "speaker_marker_count": math.log1p(
                    len(re.findall(r"(?m)(?:^|\]\s*)(?:S\d+|[^:\n]{1,16}):", transcript))
                ),
                "question_count": math.log1p(transcript.count("?") + description.count("?")),
                "exclamation_count": math.log1p(
                    transcript.count("!") + description.count("!")
                ),
                "bracket_count": math.log1p(
                    transcript.count("[") + transcript.count("(") + transcript.count("「")
                ),
                "hangul_ratio": hangul_ratio(f"{description}\n{transcript}"),
                "transcript_chars_per_sec": math.log1p(transcript_chars / duration),
                "description_chars_per_sec": math.log1p(description_chars / duration),
                "visual_evidence_available_feature": float(
                    bool(row.get("visual_evidence_available"))
                ),
                "description_missing": float(description_chars < 8),
                "transcript_missing": float(transcript_chars < 20),
            }
        )
    return pd.DataFrame.from_records(records)


def compose_candidate_text(row: Any, include_context: bool = True) -> str:
    def value(name: str) -> Any:
        if isinstance(row, dict):
            return row.get(name)
        return getattr(row, name, "")

    parts = [
        "[DESCRIPTION]",
        safe_text(value("description")),
        "[TRANSCRIPT]",
        safe_text(value("transcript")),
    ]
    if include_context:
        parts.extend(
            [
                "[BEFORE]",
                safe_text(value("before_context")),
                "[AFTER]",
                safe_text(value("after_context")),
            ]
        )
    return "\n".join(parts)


def load_bundle(private_dir: Path) -> Bundle:
    targets = pd.read_csv(
        private_dir / "validation_targets_94_PRIVATE.csv",
        encoding="utf-8-sig",
    )
    dimensions = pd.read_csv(
        private_dir / "codex_direct_v10_dimensions.csv",
        encoding="utf-8-sig",
    )
    scores = pd.read_csv(
        private_dir / "codex_direct_v10_scores_94.csv",
        encoding="utf-8-sig",
    )
    candidates = pd.DataFrame.from_records(
        read_jsonl(private_dir / "candidates_blind_94.jsonl")
    )
    validate_id_sets(
        {
            "targets": targets,
            "dimensions": dimensions,
            "scores": scores,
            "candidates": candidates,
        }
    )

    frame = (
        targets.merge(dimensions, on="candidate_id", validate="one_to_one")
        .merge(
            scores[["candidate_id", "judge_score_100"]],
            on="candidate_id",
            validate="one_to_one",
        )
        .merge(candidates, on=["candidate_id", "longform_id"], validate="one_to_one")
    )
    structure = build_structure_features(frame)
    frame = frame.merge(structure, on="candidate_id", validate="one_to_one")

    for column in QUALITY_COLUMNS + EVIDENCE_COLUMNS + STRUCTURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    quality = frame[QUALITY_COLUMNS].to_numpy(dtype=float)
    quality_structure = frame[QUALITY_COLUMNS + STRUCTURE_COLUMNS].to_numpy(dtype=float)
    source_features = np.column_stack(
        [
            (frame["transcript_source"].astype(str) == "vpick_scene_api").astype(float),
            (
                frame["transcript_source"].astype(str)
                == "yt_dlp_transcript_fallback"
            ).astype(float),
        ]
    )
    texts = [
        compose_candidate_text(row, include_context=True)
        for row in frame.itertuples(index=False)
    ]
    y = (
        pd.to_numeric(
            frame["channel_performance_percentile_PRIVATE"],
            errors="raise",
        ).to_numpy(dtype=float)
        / 100.0
    )
    return Bundle(
        frame=frame,
        quality=quality,
        quality_structure=quality_structure,
        source_features=source_features,
        texts=texts,
        y=y,
        groups=frame["longform_id"].astype(str).to_numpy(),
        channels=frame["channel_name"].astype(str).to_numpy(),
        sources=frame["transcript_source"].astype(str).to_numpy(),
    )


def finite_spearman(y: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(scores)
    if mask.sum() < 3 or np.std(y[mask]) < 1e-12 or np.std(scores[mask]) < 1e-12:
        return math.nan
    value = spearmanr(y[mask], scores[mask]).statistic
    return float(value) if math.isfinite(value) else math.nan


def residualize(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for group in sorted(set(groups)):
        mask = groups == group
        result[mask] -= float(np.nanmean(result[mask]))
    return result


def grouped_macro_spearman(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
) -> float:
    values = []
    for group in sorted(set(groups)):
        mask = groups == group
        value = finite_spearman(y[mask], scores[mask])
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else math.nan


def pairwise_accuracy(
    y: np.ndarray,
    scores: np.ndarray,
    channels: np.ndarray,
    min_gap: float = 0.01,
    max_gap: float | None = None,
) -> tuple[float, int]:
    correct = 0.0
    total = 0
    for channel in sorted(set(channels)):
        indices = np.flatnonzero(channels == channel)
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                gap = abs(float(y[left] - y[right]))
                if gap < min_gap or (max_gap is not None and gap > max_gap):
                    continue
                target_direction = np.sign(y[left] - y[right])
                score_direction = np.sign(scores[left] - scores[right])
                correct += 1.0 if score_direction == target_direction else 0.5 if score_direction == 0 else 0.0
                total += 1
    return (correct / total if total else math.nan), total


def top_quintile_precision(
    y: np.ndarray,
    scores: np.ndarray,
    channels: np.ndarray,
) -> float:
    hits = 0
    selected = 0
    for channel in sorted(set(channels)):
        indices = np.flatnonzero(channels == channel)
        count = max(1, math.ceil(len(indices) * 0.2))
        predicted_top = indices[np.argsort(scores[indices], kind="mergesort")[-count:]]
        actual_top = set(indices[np.argsort(y[indices], kind="mergesort")[-count:]])
        hits += sum(index in actual_top for index in predicted_top)
        selected += count
    return hits / selected if selected else math.nan


def channel_ndcg(
    y: np.ndarray,
    scores: np.ndarray,
    channels: np.ndarray,
) -> float:
    values = []
    for channel in sorted(set(channels)):
        mask = channels == channel
        if mask.sum() < 2:
            continue
        values.append(float(ndcg_score(y[mask][None, :], scores[mask][None, :])))
    return float(np.mean(values)) if values else math.nan


def performance_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    channels: np.ndarray,
    sources: np.ndarray,
) -> dict[str, float | int]:
    channel_centered_y = residualize(y, channels)
    channel_centered_scores = residualize(scores, channels)
    source_centered_y = residualize(channel_centered_y, sources)
    source_centered_scores = residualize(channel_centered_scores, sources)
    pair_accuracy, pair_count = pairwise_accuracy(y, scores, channels)
    local_accuracy, local_count = pairwise_accuracy(
        y,
        scores,
        channels,
        min_gap=0.10,
        max_gap=0.40,
    )
    source_rho = finite_spearman(source_centered_y, source_centered_scores)
    centered_rho = finite_spearman(channel_centered_y, channel_centered_scores)
    macro_rho = grouped_macro_spearman(y, scores, channels)
    pair_skill = (2.0 * pair_accuracy - 1.0) if math.isfinite(pair_accuracy) else math.nan
    local_pair_skill = (
        2.0 * local_accuracy - 1.0 if math.isfinite(local_accuracy) else math.nan
    )
    robust_components = [centered_rho, macro_rho, pair_skill, local_pair_skill]
    robust_rank_score = (
        float(np.mean(robust_components))
        if all(math.isfinite(value) for value in robust_components)
        else math.nan
    )
    selection_components = [
        (0.4, centered_rho),
        (0.3, macro_rho),
        (0.15, pair_skill),
        (0.15, local_pair_skill),
    ]
    selection_score = (
        float(sum(weight * value for weight, value in selection_components))
        if all(math.isfinite(value) for _, value in selection_components)
        else math.nan
    )
    return {
        "pooled_spearman": finite_spearman(y, scores),
        "channel_centered_spearman": centered_rho,
        "channel_macro_spearman": macro_rho,
        "source_residual_spearman": source_rho,
        "same_channel_pairwise_accuracy": pair_accuracy,
        "same_channel_pair_count": pair_count,
        "same_channel_local_pairwise_accuracy": local_accuracy,
        "same_channel_local_pair_count": local_count,
        "top_quintile_precision": top_quintile_precision(y, scores, channels),
        "channel_macro_ndcg": channel_ndcg(y, scores, channels),
        "robust_rank_score": robust_rank_score,
        "selection_score": selection_score,
    }


def fit_numeric_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    pipeline.fit(train_x, train_y)
    return pipeline.predict(test_x)


def fit_extra_trees(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    min_samples_leaf: int,
    seed: int,
) -> np.ndarray:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=200,
                    min_samples_leaf=int(min_samples_leaf),
                    max_features=0.75,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(train_x, train_y)
    return pipeline.predict(test_x)


def fit_pairwise_ranker(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_channels: np.ndarray,
    test_x: np.ndarray,
    c_value: float,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    train = imputer.fit_transform(train_x)
    test = imputer.transform(test_x)
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    pair_rows: list[np.ndarray] = []
    pair_labels: list[int] = []
    for channel in sorted(set(train_channels)):
        indices = np.flatnonzero(train_channels == channel)
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if abs(float(train_y[left] - train_y[right])) < 0.05:
                    continue
                difference = train[left] - train[right]
                label = int(train_y[left] > train_y[right])
                pair_rows.append(difference)
                pair_labels.append(label)
                pair_rows.append(-difference)
                pair_labels.append(1 - label)
    if len(set(pair_labels)) < 2:
        return np.full(len(test), float(np.mean(train_y)))
    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        max_iter=2000,
        random_state=20260728,
    )
    model.fit(np.vstack(pair_rows), np.array(pair_labels))
    return model.decision_function(test)


def fit_text_ridge(
    train_texts: list[str],
    train_y: np.ndarray,
    test_texts: list[str],
    alpha: float,
    train_numeric: np.ndarray | None = None,
    test_numeric: np.ndarray | None = None,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.98,
        max_features=4000,
        sublinear_tf=True,
        norm="l2",
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    if train_numeric is not None and test_numeric is not None:
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        train_dense = imputer.fit_transform(train_numeric)
        test_dense = imputer.transform(test_numeric)
        scaler = StandardScaler()
        train_dense = scaler.fit_transform(train_dense)
        test_dense = scaler.transform(test_dense)
        train_matrix = sparse.hstack(
            [train_matrix, sparse.csr_matrix(train_dense)],
            format="csr",
        )
        test_matrix = sparse.hstack(
            [test_matrix, sparse.csr_matrix(test_dense)],
            format="csr",
        )
    model = Ridge(alpha=float(alpha), solver="lsqr")
    model.fit(train_matrix, train_y)
    return model.predict(test_matrix)


def fit_pairwise_text_ranker(
    train_texts: list[str],
    train_y: np.ndarray,
    train_channels: np.ndarray,
    test_texts: list[str],
    c_value: float,
    train_numeric: np.ndarray | None = None,
    test_numeric: np.ndarray | None = None,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.98,
        max_features=4000,
        sublinear_tf=True,
        norm="l2",
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    if train_numeric is not None and test_numeric is not None:
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        train_dense = imputer.fit_transform(train_numeric)
        test_dense = imputer.transform(test_numeric)
        scaler = StandardScaler()
        train_dense = scaler.fit_transform(train_dense)
        test_dense = scaler.transform(test_dense)
        train_matrix = sparse.hstack(
            [train_matrix, sparse.csr_matrix(train_dense)],
            format="csr",
        )
        test_matrix = sparse.hstack(
            [test_matrix, sparse.csr_matrix(test_dense)],
            format="csr",
        )

    pair_rows = []
    pair_labels = []
    for channel in sorted(set(train_channels)):
        indices = np.flatnonzero(train_channels == channel)
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if abs(float(train_y[left] - train_y[right])) < 0.05:
                    continue
                difference = train_matrix[left] - train_matrix[right]
                label = int(train_y[left] > train_y[right])
                pair_rows.extend([difference, -difference])
                pair_labels.extend([label, 1 - label])
    if len(set(pair_labels)) < 2:
        return np.full(len(test_texts), float(np.mean(train_y)))
    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        max_iter=3000,
        random_state=20260728,
    )
    model.fit(sparse.vstack(pair_rows, format="csr"), np.asarray(pair_labels))
    return model.decision_function(test_matrix)


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    if not len(ordered):
        return np.full(len(values), 0.5)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def fit_text_structure_ensemble(
    model_name: str,
    bundle: Bundle,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> np.ndarray:
    base_models = [
        "char_tfidf",
        "char_tfidf_numeric",
        "extra_trees_quality_structure",
    ]
    local_splits = grouped_splits(
        bundle.groups[train_indices],
        4,
        seed + 20000,
    )
    train_columns = []
    test_columns = []
    for base_offset, base_model in enumerate(base_models):
        parameter, _ = select_parameter(
            base_model,
            bundle,
            train_indices,
            4,
            seed + 21000 + base_offset,
        )
        local_oof = np.full(len(train_indices), np.nan)
        for fold_offset, (inner_train_local, inner_test_local) in enumerate(local_splits):
            inner_train = train_indices[inner_train_local]
            inner_test = train_indices[inner_test_local]
            local_oof[inner_test_local] = fit_predict(
                base_model,
                bundle,
                inner_train,
                inner_test,
                parameter,
                seed + 22000 + (base_offset * 100) + fold_offset,
            )
        outer_test = fit_predict(
            base_model,
            bundle,
            train_indices,
            test_indices,
            parameter,
            seed + 23000 + base_offset,
        )
        train_columns.append(local_oof)
        test_columns.append(outer_test)

    train_meta = np.column_stack(train_columns)
    test_meta = np.column_stack(test_columns)
    if model_name == "rank_ensemble_text_structure":
        train_ranked = np.column_stack(
            [
                empirical_percentile(train_meta[:, column], train_meta[:, column])
                for column in range(train_meta.shape[1])
            ]
        )
        test_ranked = np.column_stack(
            [
                empirical_percentile(train_meta[:, column], test_meta[:, column])
                for column in range(test_meta.shape[1])
            ]
        )
        return np.mean(test_ranked, axis=1)

    meta_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0, positive=True)),
        ]
    )
    meta_model.fit(train_meta, bundle.y[train_indices])
    return meta_model.predict(test_meta)


def fit_predict(
    model_name: str,
    bundle: Bundle,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    parameter: Any,
    seed: int,
) -> np.ndarray:
    if model_name == "fixed_v10":
        return (
            pd.to_numeric(
                bundle.frame.iloc[test_indices]["judge_score_100"],
                errors="coerce",
            ).to_numpy(dtype=float)
            / 100.0
        )
    if model_name == "fixed_equal_quality":
        return np.nanmean(bundle.quality[test_indices], axis=1) / 4.0
    if model_name == "constant":
        return np.full(len(test_indices), float(np.mean(bundle.y[train_indices])))
    if model_name == "random":
        return np.array(
            [
                stable_random_score(candidate_id)
                for candidate_id in bundle.frame.iloc[test_indices]["candidate_id"].astype(str)
            ],
            dtype=float,
        )
    if model_name == "source_presence_fixed":
        return bundle.source_features[test_indices, 0].copy()
    if model_name == "source_only":
        return fit_numeric_ridge(
            bundle.source_features[train_indices],
            bundle.y[train_indices],
            bundle.source_features[test_indices],
            float(parameter),
        )
    if model_name == "ridge_quality":
        return fit_numeric_ridge(
            bundle.quality[train_indices],
            bundle.y[train_indices],
            bundle.quality[test_indices],
            float(parameter),
        )
    if model_name == "ridge_quality_structure":
        return fit_numeric_ridge(
            bundle.quality_structure[train_indices],
            bundle.y[train_indices],
            bundle.quality_structure[test_indices],
            float(parameter),
        )
    if model_name == "pairwise_quality_structure":
        return fit_pairwise_ranker(
            bundle.quality_structure[train_indices],
            bundle.y[train_indices],
            bundle.channels[train_indices],
            bundle.quality_structure[test_indices],
            float(parameter),
        )
    if model_name == "extra_trees_quality_structure":
        return fit_extra_trees(
            bundle.quality_structure[train_indices],
            bundle.y[train_indices],
            bundle.quality_structure[test_indices],
            int(parameter),
            seed,
        )
    if model_name == "char_tfidf":
        return fit_text_ridge(
            [bundle.texts[index] for index in train_indices],
            bundle.y[train_indices],
            [bundle.texts[index] for index in test_indices],
            float(parameter),
        )
    if model_name == "char_tfidf_numeric":
        return fit_text_ridge(
            [bundle.texts[index] for index in train_indices],
            bundle.y[train_indices],
            [bundle.texts[index] for index in test_indices],
            float(parameter),
            bundle.quality_structure[train_indices],
            bundle.quality_structure[test_indices],
        )
    if model_name == "pairwise_char_tfidf":
        return fit_pairwise_text_ranker(
            [bundle.texts[index] for index in train_indices],
            bundle.y[train_indices],
            bundle.channels[train_indices],
            [bundle.texts[index] for index in test_indices],
            float(parameter),
        )
    if model_name == "pairwise_char_tfidf_numeric":
        return fit_pairwise_text_ranker(
            [bundle.texts[index] for index in train_indices],
            bundle.y[train_indices],
            bundle.channels[train_indices],
            [bundle.texts[index] for index in test_indices],
            float(parameter),
            bundle.quality_structure[train_indices],
            bundle.quality_structure[test_indices],
        )
    if model_name in {
        "rank_ensemble_text_structure",
        "stacked_text_structure",
    }:
        return fit_text_structure_ensemble(
            model_name,
            bundle,
            train_indices,
            test_indices,
            seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def grouped_splits(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_group_count = len(set(groups))
    splits = min(n_splits, unique_group_count)
    if splits < 2:
        raise ValueError("At least two unique longform groups are required.")
    splitter = GroupKFold(
        n_splits=splits,
        shuffle=True,
        random_state=seed,
    )
    placeholder = np.zeros(len(groups))
    return list(splitter.split(placeholder, groups=groups))


def select_parameter(
    model_name: str,
    bundle: Bundle,
    outer_train_indices: np.ndarray,
    inner_splits: int,
    seed: int,
) -> tuple[Any, dict[str, float]]:
    parameters = MODEL_PARAMS[model_name]
    if len(parameters) == 1:
        return parameters[0], {"fixed": math.nan}

    local_groups = bundle.groups[outer_train_indices]
    splits = grouped_splits(
        local_groups,
        inner_splits,
        seed,
    )
    scores: dict[str, float] = {}
    for parameter in parameters:
        predictions = np.full(len(outer_train_indices), np.nan)
        for fold_index, (inner_train_local, inner_test_local) in enumerate(splits):
            inner_train = outer_train_indices[inner_train_local]
            inner_test = outer_train_indices[inner_test_local]
            predictions[inner_test_local] = fit_predict(
                model_name,
                bundle,
                inner_train,
                inner_test,
                parameter,
                seed + fold_index,
            )
        metrics = performance_metrics(
            bundle.y[outer_train_indices],
            predictions,
            bundle.channels[outer_train_indices],
            bundle.sources[outer_train_indices],
        )
        value = float(metrics["selection_score"])
        scores[str(parameter)] = value
    best = max(
        parameters,
        key=lambda parameter: (
            scores[str(parameter)]
            if math.isfinite(scores[str(parameter)])
            else -math.inf
        ),
    )
    return best, scores


def repeated_nested_oof(
    model_name: str,
    bundle: Bundle,
    seeds: list[int],
    outer_splits: int,
    inner_splits: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_sum = np.zeros(len(bundle.y), dtype=float)
    prediction_count = np.zeros(len(bundle.y), dtype=int)
    tuning_log: list[dict[str, Any]] = []
    fold_outputs: list[dict[str, Any]] = []
    for repeat_index, seed in enumerate(seeds):
        splits = grouped_splits(
            bundle.groups,
            outer_splits,
            seed,
        )
        for fold_index, (train_indices, test_indices) in enumerate(splits):
            parameter, inner_scores = select_parameter(
                model_name,
                bundle,
                train_indices,
                inner_splits,
                seed + 1000 + fold_index,
            )
            fold_prediction = fit_predict(
                model_name,
                bundle,
                train_indices,
                test_indices,
                parameter,
                seed + fold_index,
            )
            prediction_sum[test_indices] += fold_prediction
            prediction_count[test_indices] += 1
            finite_inner_scores = [
                float(value)
                for value in inner_scores.values()
                if math.isfinite(float(value))
            ]
            inner_selection_score = (
                max(finite_inner_scores) if finite_inner_scores else math.nan
            )
            tuning_log.append(
                {
                    "model": model_name,
                    "repeat_index": repeat_index,
                    "outer_fold": fold_index,
                    "selected_parameter": parameter,
                    "inner_scores": inner_scores,
                    "inner_selection_score": inner_selection_score,
                    "train_count": int(len(train_indices)),
                    "test_count": int(len(test_indices)),
                    "test_longform_count": int(len(set(bundle.groups[test_indices]))),
                }
            )
            fold_outputs.append(
                {
                    "model": model_name,
                    "repeat_index": repeat_index,
                    "outer_fold": fold_index,
                    "test_indices": test_indices.copy(),
                    "predictions": np.asarray(fold_prediction, dtype=float).copy(),
                    "selected_parameter": parameter,
                    "inner_selection_score": inner_selection_score,
                }
            )
    if np.any(prediction_count != len(seeds)):
        raise RuntimeError(
            f"{model_name}: invalid OOF coverage {Counter(prediction_count.tolist())}"
        )
    return prediction_sum / prediction_count, tuning_log, fold_outputs


def assemble_fully_nested_selection(
    fold_outputs_by_model: dict[str, list[dict[str, Any]]],
    candidate_models: list[str],
    sample_count: int,
    repeat_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    indexed = {
        model: {
            (int(item["repeat_index"]), int(item["outer_fold"])): item
            for item in fold_outputs_by_model[model]
        }
        for model in candidate_models
    }
    fold_keys = sorted(set.intersection(*(set(items) for items in indexed.values())))
    prediction_sum = np.zeros(sample_count, dtype=float)
    prediction_count = np.zeros(sample_count, dtype=int)
    selection_log = []
    for repeat_index, outer_fold in fold_keys:
        candidates = []
        for model in candidate_models:
            item = indexed[model][(repeat_index, outer_fold)]
            score = float(item["inner_selection_score"])
            candidates.append((score if math.isfinite(score) else -math.inf, model, item))
        score, model, selected = max(candidates, key=lambda item: (item[0], item[1]))
        test_indices = np.asarray(selected["test_indices"], dtype=int)
        prediction_sum[test_indices] += np.asarray(selected["predictions"], dtype=float)
        prediction_count[test_indices] += 1
        selection_log.append(
            {
                "repeat_index": repeat_index,
                "outer_fold": outer_fold,
                "selected_model": model,
                "selected_parameter": selected["selected_parameter"],
                "inner_selection_score": score,
                "test_count": int(len(test_indices)),
            }
        )
    if np.any(prediction_count != repeat_count):
        raise RuntimeError(
            "Fully nested selector has invalid OOF coverage: "
            f"{Counter(prediction_count.tolist())}"
        )
    return prediction_sum / prediction_count, selection_log


def leave_one_channel_out(
    model_name: str,
    bundle: Bundle,
    inner_splits: int,
    seed: int,
) -> np.ndarray:
    predictions = np.full(len(bundle.y), np.nan)
    for channel_index, channel in enumerate(sorted(set(bundle.channels))):
        test_indices = np.flatnonzero(bundle.channels == channel)
        train_indices = np.flatnonzero(bundle.channels != channel)
        parameter, _ = select_parameter(
            model_name,
            bundle,
            train_indices,
            inner_splits,
            seed + channel_index,
        )
        predictions[test_indices] = fit_predict(
            model_name,
            bundle,
            train_indices,
            test_indices,
            parameter,
            seed + channel_index,
        )
    return predictions


def bootstrap_metric_intervals(
    bundle: Bundle,
    scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(bundle.groups)), dtype=object)
    group_indices = {
        group: np.flatnonzero(bundle.groups == group)
        for group in unique_groups
    }
    metric_names = [
        "channel_centered_spearman",
        "channel_macro_spearman",
        "source_residual_spearman",
        "same_channel_pairwise_accuracy",
        "same_channel_local_pairwise_accuracy",
        "robust_rank_score",
    ]
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
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
    intervals = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        intervals[name] = {
            "lower_95": float(np.quantile(array, 0.025)),
            "median": float(np.quantile(array, 0.5)),
            "upper_95": float(np.quantile(array, 0.975)),
            "valid_repetitions": int(len(array)),
        }
    return intervals


def audit_vpick_coverage(
    bundle: Bundle,
    raw_vpick: Path,
    fallback_scenes: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory: dict[str, list[dict[str, str]]] = defaultdict(list)
    accounts_dir = raw_vpick / "accounts"
    for inventory_path in accounts_dir.glob("*/inventory.csv"):
        account = inventory_path.parent.name
        with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                longform_id = str(row.get("long_video_id") or "").strip()
                if longform_id:
                    inventory[longform_id].append({"account": account, **row})

    audit_rows = []
    for row in bundle.frame.to_dict("records"):
        longform_id = str(row["longform_id"])
        canonical_path = raw_vpick / f"{longform_id}_scenes.json"
        fallback_path = fallback_scenes / f"{longform_id}_scenes.json"
        statuses = inventory.get(longform_id, [])
        audit_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "longform_id": longform_id,
                "target_transcript_source": row["transcript_source"],
                "canonical_vpick_scenes": int(canonical_path.exists()),
                "fallback_scene_file": int(fallback_path.exists()),
                "inventory_accounts": "|".join(
                    sorted({status["account"] for status in statuses})
                ),
                "inventory_statuses": "|".join(
                    sorted({str(status.get("status") or "") for status in statuses})
                ),
                "ready_asset_count": sum(
                    str(status.get("status") or "") == "READY" for status in statuses
                ),
                "failed_asset_count": sum(
                    str(status.get("status") or "") == "FAILED" for status in statuses
                ),
            }
        )

    audit = pd.DataFrame.from_records(audit_rows)
    unique_longforms = audit.drop_duplicates("longform_id")
    source_counts = (
        bundle.frame["transcript_source"].astype(str).value_counts().to_dict()
    )
    summary = {
        "candidate_count": int(len(audit)),
        "longform_count": int(audit["longform_id"].nunique()),
        "canonical_vpick_longforms": int(
            unique_longforms["canonical_vpick_scenes"].sum()
        ),
        "fallback_scene_longforms": int(unique_longforms["fallback_scene_file"].sum()),
        "longforms_without_scene_file": int(
            (
                (unique_longforms["canonical_vpick_scenes"] == 0)
                & (unique_longforms["fallback_scene_file"] == 0)
            ).sum()
        ),
        "ready_inventory_longforms": int(
            (unique_longforms["ready_asset_count"] > 0).sum()
        ),
        "failed_inventory_longforms": int(
            (unique_longforms["failed_asset_count"] > 0).sum()
        ),
        "candidate_count_by_transcript_source": source_counts,
        "source_shortcut_warning": (
            "Transcript source is forbidden in deployment models and retained only "
            "as a post-hoc shortcut control."
        ),
    }
    return audit_rows, summary


def acceptance_result(
    metrics: dict[str, Any],
    strongest_source_control_metrics: dict[str, Any],
    bootstrap: dict[str, dict[str, float]],
    gates: dict[str, float],
) -> tuple[bool, list[dict[str, Any]]]:
    checks = [
        (
            "channel_centered_spearman",
            float(metrics["channel_centered_spearman"]),
            float(gates["channel_centered_spearman_min"]),
        ),
        (
            "channel_macro_spearman",
            float(metrics["channel_macro_spearman"]),
            float(gates["channel_macro_spearman_min"]),
        ),
        (
            "source_residual_spearman",
            float(metrics["source_residual_spearman"]),
            float(gates["source_residual_spearman_min"]),
        ),
        (
            "same_channel_pairwise_accuracy",
            float(metrics["same_channel_pairwise_accuracy"]),
            float(gates["same_channel_pairwise_accuracy_min"]),
        ),
        (
            "same_channel_local_pairwise_accuracy",
            float(metrics["same_channel_local_pairwise_accuracy"]),
            float(gates["same_channel_local_pairwise_accuracy_min"]),
        ),
        (
            "channel_centered_gain_over_source_control",
            float(metrics["channel_centered_spearman"])
            - float(
                strongest_source_control_metrics["channel_centered_spearman"]
            ),
            float(
                gates["minimum_channel_centered_gain_over_source_control"]
            ),
        ),
        (
            "bootstrap_primary_ci_lower",
            float(
                bootstrap["channel_centered_spearman"]["lower_95"]
            ),
            float(gates["bootstrap_primary_ci_lower_min"]),
        ),
    ]
    details = [
        {
            "gate": name,
            "observed": observed,
            "required_minimum": required,
            "passed": bool(math.isfinite(observed) and observed >= required),
        }
        for name, observed, required in checks
    ]
    return all(item["passed"] for item in details), details


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}" if math.isfinite(value) else "NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    audit_summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    best_model: str,
    accepted: bool,
    gate_details: list[dict[str, Any]],
    bootstrap: dict[str, dict[str, float]],
) -> None:
    best = next(row for row in comparison_rows if row["model"] == best_model)
    source_controls = [
        row
        for row in comparison_rows
        if row["model"] in {"source_presence_fixed", "source_only"}
    ]
    source = max(
        source_controls,
        key=lambda row: float(row["channel_centered_spearman"]),
    )
    comparison = markdown_table(
        sorted(
            comparison_rows,
            key=lambda row: (
                0 if row["deployment_eligible"] else 1,
                -float(row["robust_rank_score"])
                if math.isfinite(float(row["robust_rank_score"]))
                else math.inf,
            ),
        ),
        [
            ("model", "모델"),
            ("deployment_eligible", "배포 후보"),
            ("pooled_spearman", "Pooled rho"),
            ("channel_centered_spearman", "채널 중심 rho"),
            ("channel_macro_spearman", "채널 Macro rho"),
            ("source_residual_spearman", "출처 제거 rho"),
            ("same_channel_pairwise_accuracy", "쌍 정확도"),
            ("robust_rank_score", "강건 점수"),
        ],
    )
    gates = markdown_table(
        gate_details,
        [
            ("gate", "게이트"),
            ("observed", "관측값"),
            ("required_minimum", "최소 기준"),
            ("passed", "통과"),
        ],
    )
    ci = bootstrap["channel_centered_spearman"]
    status = "채택" if accepted else "기각"
    text = f"""# Vpick 성과 예측 Judge v11 검증 보고서

## 1. 목적

Vpick 과제 PDF 10쪽의 `정답 일치도` 정의에 맞춰, 블라인드 Judge 점수가 실제
채널 내 Shorts 성과 백분위 순서를 복원하는지 검증했다. 기존 v10에서 변별력이
있는 7개 루브릭 축을 고정된 품질 특징 추출기로 사용하고, 성과 백분위는 모델
입력에서 제외했다.

## 2. 데이터 감사

- 후보: {audit_summary['candidate_count']}개
- 원본 롱폼: {audit_summary['longform_count']}개
- canonical Vpick 장면 파일이 있는 롱폼: {audit_summary['canonical_vpick_longforms']}개
- yt-dlp 대체 장면 파일이 있는 롱폼: {audit_summary['fallback_scene_longforms']}개
- 장면 파일이 전혀 없는 롱폼: {audit_summary['longforms_without_scene_file']}개
- Vpick READY inventory와 연결된 롱폼: {audit_summary['ready_inventory_longforms']}개
- Vpick FAILED inventory와 연결된 롱폼: {audit_summary['failed_inventory_longforms']}개

수집 방식이 예측 편법으로 작동하는지 확인하기 위해 `transcript_source`는
배포 후보 입력에서 제외하고, 출처 존재 여부만 쓰는 점수를 사후 대조군으로
두었다.

## 3. 검증 설계

- 목표값: 채널 내 연속 성과 백분위
- 외부 검증: 5-fold GroupKFold를 3개 seed로 반복
- 그룹 키: `longform_id` (동일 원본의 후보가 학습·검증에 동시에 들어가지 않음)
- 하이퍼파라미터: 각 외부 학습 폴드 안의 4-fold grouped CV에서만 선택
- 주 지표: 채널 중심화 Spearman
- 편법 방지 지표: 채널별 macro, 자막 출처 제거 상관,
  같은 채널 쌍 정확도, 연속 백분위 차이 10~40인 근접 쌍 정확도
- 불확실성: 롱폼 단위 bootstrap 95% CI

기존 locked test는 이미 여러 차례 열람했으므로, 이번 수치는 `exploratory nested
OOF`다. 최종 상용 주장에는 새로 수집한 미공개 holdout이 필요하다.

## 4. 모델 비교

{comparison}

## 5. 최종 게이트

- 주 검증 파이프라인: `{best_model}`
- 판정: **{status}**
- 채널 중심화 Spearman: {best['channel_centered_spearman']:.4f}
- 채널 중심화 Spearman 95% bootstrap CI:
  [{ci['lower_95']:.4f}, {ci['upper_95']:.4f}]
- 가장 강한 출처 대조군: `{source['model']}`
- 출처 대조군 채널 중심화 Spearman: {source['channel_centered_spearman']:.4f}
- 최상위 후보 강건 점수: {best['robust_rank_score']:.4f}

{gates}

## 6. 결론

"""
    if accepted:
        text += (
            "v11 후보는 사전 등록한 편법 방지 게이트를 모두 통과했다. 다만 모델군 "
            "선택까지 같은 94개 OOF 결과에서 수행했으므로, 별도 신규 채널·롱폼 "
            "holdout에서 한 번 더 재현된 뒤 성과 예측 Judge로 고정한다.\n"
        )
    else:
        text += (
            "현재 입력만으로는 PDF 10쪽에서 요구하는 높은 정답 일치도를 확보하지 "
            "못했다. 개발 단계의 개별 최고 모델보다 완전 중첩 모델 선택 "
            "파이프라인의 성능이 크게 낮아 안정적인 일반화 신호를 확인하지 못했다. "
            "v10은 편집·내용 "
            "품질 진단기로 유지하고, 성과 예측 Judge라는 명칭은 사용하지 않는다. "
            "다음 데이터 수집에서는 모든 후보에 동일한 Vpick 장면·자막 근거를 "
            "확보하고, 연속 성과 분포의 신규 미공개 holdout을 추가해야 한다.\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.bootstrap_repetitions is not None:
        config["bootstrap_repetitions"] = args.bootstrap_repetitions
    bundle = load_bundle(args.private_dir)

    private_output = args.private_dir / "performance_calibrator_v11"
    private_output.mkdir(parents=True, exist_ok=True)
    args.public_dir.mkdir(parents=True, exist_ok=True)

    audit_rows, audit_summary = audit_vpick_coverage(
        bundle,
        args.raw_vpick,
        args.fallback_scenes,
    )
    write_csv(private_output / "vpick_coverage_audit_PRIVATE.csv", audit_rows)
    (args.public_dir / "vpick_coverage_summary_PUBLIC.json").write_text(
        json.dumps(
            json_safe(audit_summary),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    seeds = [int(seed) for seed in config["random_seeds"]]
    model_names = list(config["deployment_eligible_models"]) + list(
        config["diagnostic_controls"]
    )
    predictions: dict[str, np.ndarray] = {}
    tuning_logs: dict[str, list[dict[str, Any]]] = {}
    fold_outputs_by_model: dict[str, list[dict[str, Any]]] = {}
    comparison_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        scores, tuning_log, fold_outputs = repeated_nested_oof(
            model_name,
            bundle,
            seeds,
            int(config["outer_splits"]),
            int(config["inner_splits"]),
        )
        predictions[model_name] = scores
        tuning_logs[model_name] = tuning_log
        fold_outputs_by_model[model_name] = fold_outputs
        metrics = performance_metrics(
            bundle.y,
            scores,
            bundle.channels,
            bundle.sources,
        )
        comparison_rows.append(
            {
                "model": model_name,
                "deployment_eligible": model_name not in DIAGNOSTIC_MODELS,
                **metrics,
                "loco_pooled_spearman": math.nan,
                "loco_channel_centered_spearman": math.nan,
                "loco_channel_macro_spearman": math.nan,
                "loco_source_residual_spearman": math.nan,
            }
        )

    nested_models = [str(model) for model in config["nested_selection_models"]]
    nested_predictions, nested_selection_log = assemble_fully_nested_selection(
        fold_outputs_by_model,
        nested_models,
        len(bundle.y),
        len(seeds),
    )
    nested_model_name = "nested_selected_pipeline"
    predictions[nested_model_name] = nested_predictions
    nested_metrics = performance_metrics(
        bundle.y,
        nested_predictions,
        bundle.channels,
        bundle.sources,
    )
    nested_row = {
        "model": nested_model_name,
        "deployment_eligible": True,
        **nested_metrics,
        "loco_pooled_spearman": math.nan,
        "loco_channel_centered_spearman": math.nan,
        "loco_channel_macro_spearman": math.nan,
        "loco_source_residual_spearman": math.nan,
    }
    comparison_rows.append(nested_row)

    eligible_rows = [
        row
        for row in comparison_rows
        if row["deployment_eligible"] and row["model"] != nested_model_name
    ]
    best_development_row = max(
        eligible_rows,
        key=lambda row: float(row["robust_rank_score"]),
    )
    best_development_model = str(best_development_row["model"])
    source_control_rows = [
        row
        for row in comparison_rows
        if row["model"] in {"source_presence_fixed", "source_only"}
    ]
    source_row = max(
        source_control_rows,
        key=lambda row: float(row["channel_centered_spearman"]),
    )

    bootstrap_best = bootstrap_metric_intervals(
        bundle,
        predictions[nested_model_name],
        int(config["bootstrap_repetitions"]),
        seeds[0] + 90000,
    )
    bootstrap_source = bootstrap_metric_intervals(
        bundle,
        predictions[str(source_row["model"])],
        int(config["bootstrap_repetitions"]),
        seeds[0] + 91000,
    )
    accepted, gate_details = acceptance_result(
        nested_row,
        source_row,
        bootstrap_best,
        config["acceptance_gates"],
    )

    oof_frame = bundle.frame[
        [
            "candidate_id",
            "longform_id",
            "channel_name",
            "channel_performance_percentile_PRIVATE",
            "transcript_source",
        ]
    ].copy()
    for model_name, values in predictions.items():
        oof_frame[f"oof_{model_name}"] = values
    oof_frame.to_csv(
        private_output / "oof_predictions_PRIVATE.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (private_output / "nested_tuning_log_PRIVATE.json").write_text(
        json.dumps(
            json_safe({
                "per_model": tuning_logs,
                "fully_nested_model_selection": nested_selection_log,
            }),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    write_csv(args.public_dir / "model_comparison_PUBLIC.csv", comparison_rows)
    summary = {
        "protocol_id": config["protocol_id"],
        "candidate_count": int(len(bundle.y)),
        "longform_count": int(len(set(bundle.groups))),
        "channel_count": int(len(set(bundle.channels))),
        "primary_validation_pipeline": nested_model_name,
        "best_development_candidate": best_development_model,
        "accepted_as_performance_judge": accepted,
        "primary_validation_metrics": nested_row,
        "best_development_metrics": best_development_row,
        "nested_selected_model_counts": dict(
            Counter(
                str(item["selected_model"])
                for item in nested_selection_log
            )
        ),
        "strongest_source_control_metrics": source_row,
        "acceptance_gates": gate_details,
        "bootstrap_best": bootstrap_best,
        "bootstrap_strongest_source_control": bootstrap_source,
        "data_audit": audit_summary,
        "status_note": (
            "Exploratory repeated nested grouped OOF. A fresh untouched holdout is "
            "required before production claims."
        ),
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
        audit_summary,
        comparison_rows,
        nested_model_name,
        accepted,
        gate_details,
        bootstrap_best,
    )
    print(
        json.dumps(
            json_safe({
                "primary_validation_pipeline": nested_model_name,
                "best_development_candidate": best_development_model,
                "accepted": accepted,
                "channel_centered_spearman": nested_row[
                    "channel_centered_spearman"
                ],
                "channel_macro_spearman": nested_row["channel_macro_spearman"],
                "source_residual_spearman": nested_row[
                    "source_residual_spearman"
                ],
                "strongest_source_control": source_row["model"],
                "source_control_channel_centered_spearman": source_row[
                    "channel_centered_spearman"
                ],
                "primary_robust_rank_score": nested_row["robust_rank_score"],
                "report": str(args.report),
            }),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
