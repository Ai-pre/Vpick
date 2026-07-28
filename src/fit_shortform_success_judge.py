from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train_performance_calibrator_v11 import (
    DEFAULT_PRIVATE_DIR,
    QUALITY_COLUMNS,
    STRUCTURE_COLUMNS,
    load_bundle,
    select_parameter,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_SUMMARY = (
    ROOT / "results" / "performance_calibrator_v11" / "summary_PUBLIC.json"
)
DEFAULT_ARTIFACT = (
    DEFAULT_PRIVATE_DIR
    / "performance_calibrator_v11"
    / "shortform_success_judge_v11_experimental.joblib"
)
DEFAULT_METADATA = (
    ROOT
    / "results"
    / "performance_calibrator_v11"
    / "deployment_artifact_METADATA.json"
)
MODEL_NAME = "pairwise_char_tfidf_numeric"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the experimental standalone Shortform Success Judge."
    )
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=DEFAULT_VALIDATION_SUMMARY,
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--allow-rejected", action="store_true")
    return parser.parse_args()


def make_pairwise_rows(
    matrix: sparse.csr_matrix,
    y: np.ndarray,
    channels: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, int]:
    rows = []
    labels = []
    pair_count = 0
    for channel in sorted(set(channels)):
        indices = np.flatnonzero(channels == channel)
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if abs(float(y[left] - y[right])) < 0.05:
                    continue
                difference = matrix[left] - matrix[right]
                label = int(y[left] > y[right])
                rows.extend([difference, -difference])
                labels.extend([label, 1 - label])
                pair_count += 1
    if not rows:
        raise ValueError("No same-channel continuous-percentile training pairs.")
    return (
        sparse.vstack(rows, format="csr"),
        np.asarray(labels, dtype=int),
        pair_count,
    )


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def public_artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "<private artifact outside repository>"


def main() -> None:
    args = parse_args()
    validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    accepted = bool(validation.get("accepted_as_performance_judge"))
    if not accepted and not args.allow_rejected:
        raise SystemExit(
            "Validation status is rejected. Re-run with --allow-rejected only to "
            "create an explicitly experimental artifact."
        )

    bundle = load_bundle(args.private_dir)
    all_indices = np.arange(len(bundle.y))
    selected_c, inner_scores = select_parameter(
        MODEL_NAME,
        bundle,
        all_indices,
        inner_splits=4,
        seed=20260728,
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.98,
        max_features=4000,
        sublinear_tf=True,
        norm="l2",
    )
    text_matrix = vectorizer.fit_transform(bundle.texts)
    numeric_imputer = SimpleImputer(strategy="median", add_indicator=True)
    numeric = numeric_imputer.fit_transform(bundle.quality_structure)
    numeric_scaler = StandardScaler()
    numeric = numeric_scaler.fit_transform(numeric)
    full_matrix = sparse.hstack(
        [text_matrix, sparse.csr_matrix(numeric)],
        format="csr",
    )
    pair_matrix, pair_labels, pair_count = make_pairwise_rows(
        full_matrix,
        bundle.y,
        bundle.channels,
    )
    model = LogisticRegression(
        C=float(selected_c),
        solver="liblinear",
        max_iter=3000,
        random_state=20260728,
    )
    model.fit(pair_matrix, pair_labels)
    raw_training_scores = model.decision_function(full_matrix)

    status = "validated" if accepted else "experimental_rejected"
    artifact = {
        "artifact_id": "shortform_success_judge_v11_continuous",
        "status": status,
        "model_name": MODEL_NAME,
        "target_definition": "continuous within-channel Shorts view percentile",
        "score_definition": (
            "Empirical percentile of the pairwise ranker score against the frozen "
            "94-candidate training reference distribution."
        ),
        "forbidden_inputs": [
            "channel_name",
            "views",
            "likes",
            "performance_label",
            "performance_percentile",
            "percentile_bucket",
            "short_url",
        ],
        "quality_columns": QUALITY_COLUMNS,
        "structure_columns": STRUCTURE_COLUMNS,
        "selected_c": float(selected_c),
        "inner_selection_scores": inner_scores,
        "training_candidate_count": int(len(bundle.y)),
        "training_longform_count": int(len(set(bundle.groups))),
        "training_pair_count": int(pair_count),
        "vectorizer": vectorizer,
        "numeric_imputer": numeric_imputer,
        "numeric_scaler": numeric_scaler,
        "pairwise_model": model,
        "score_reference_sorted": np.sort(raw_training_scores),
        "validation_summary": validation,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)

    metadata = finite_or_none(
        {
            "artifact_id": artifact["artifact_id"],
            "status": status,
            "model_name": MODEL_NAME,
            "target_definition": artifact["target_definition"],
            "selected_c": artifact["selected_c"],
            "training_candidate_count": artifact["training_candidate_count"],
            "training_longform_count": artifact["training_longform_count"],
            "training_pair_count": artifact["training_pair_count"],
            "quality_columns": QUALITY_COLUMNS,
            "structure_columns": STRUCTURE_COLUMNS,
            "forbidden_inputs": artifact["forbidden_inputs"],
            "validation": {
                "primary_validation_pipeline": validation.get(
                    "primary_validation_pipeline"
                ),
                "accepted_as_performance_judge": accepted,
                "primary_validation_metrics": validation.get(
                    "primary_validation_metrics"
                ),
            },
            "artifact_private_path": public_artifact_reference(args.artifact),
            "warning": (
                "This artifact is experimental and failed the registered validation "
                "gates. It may be used for research-only candidate reranking."
                if not accepted
                else ""
            ),
        }
    )
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
