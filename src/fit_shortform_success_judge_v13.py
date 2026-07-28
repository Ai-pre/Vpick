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
    ROOT,
    STRUCTURE_COLUMNS,
    load_bundle,
)
from train_performance_calibrator_v12 import (
    build_pair_matrix,
    candidate_reliability,
    select_c,
)
from train_performance_calibrator_v13 import MEMBER_SPECS, weighted_average


DEFAULT_CONFIG = ROOT / "config" / "performance_calibrator_v13.json"
DEFAULT_VALIDATION = (
    ROOT / "results" / "performance_calibrator_v13" / "summary_PUBLIC.json"
)
DEFAULT_ARTIFACT = (
    DEFAULT_PRIVATE_DIR
    / "performance_calibrator_v13"
    / "shortform_success_judge_v13_pending_holdout.joblib"
)
DEFAULT_METADATA = (
    ROOT
    / "results"
    / "performance_calibrator_v13"
    / "deployment_artifact_METADATA.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the frozen v13 development ensemble artifact."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--allow-pending-holdout", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def public_artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "<private artifact outside repository>"


def repeated_full_data_c_selection(
    bundle: Any,
    spec: Any,
    reliability: np.ndarray,
    c_values: list[float],
    inner_splits: int,
    seeds: list[int],
) -> tuple[float, dict[str, Any]]:
    all_indices = np.arange(len(bundle.y))
    per_seed = []
    aggregate = {float(c_value): [] for c_value in c_values}
    for seed in seeds:
        selected, scores = select_c(
            bundle,
            all_indices,
            spec,
            reliability,
            c_values,
            inner_splits,
            seed,
        )
        per_seed.append(
            {
                "seed": seed,
                "selected_c": selected,
                "scores": scores,
            }
        )
        for c_value in c_values:
            value = float(scores[str(float(c_value))])
            if math.isfinite(value):
                aggregate[float(c_value)].append(value)
    mean_scores = {
        c_value: (
            float(np.mean(values)) if values else -math.inf
        )
        for c_value, values in aggregate.items()
    }
    selected = max(
        c_values,
        key=lambda value: (
            mean_scores[float(value)],
            -abs(math.log10(float(value))),
        ),
    )
    return float(selected), {
        "per_seed": per_seed,
        "mean_selection_score_by_c": {
            str(key): value for key, value in mean_scores.items()
        },
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if (
        validation.get("status") == "development_frozen_pending_holdout"
        and not args.allow_pending_holdout
    ):
        raise SystemExit(
            "The v13 candidate is pending a fresh holdout. Re-run with "
            "--allow-pending-holdout only to create a research artifact."
        )

    bundle = load_bundle(args.private_dir)
    reliability = candidate_reliability(bundle)
    member_names = [str(value) for value in config["ensemble_members"]]
    weights = [float(value) for value in config["ensemble_weights"]]
    c_values = [float(value) for value in config["c_values"]]
    seeds = [int(value) for value in config["random_seeds"]]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.98,
        max_features=4000,
        sublinear_tf=True,
        norm="l2",
    )
    text_matrix = vectorizer.fit_transform(bundle.texts).tocsr()
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    numeric = imputer.fit_transform(bundle.quality_structure)
    scaler = StandardScaler()
    numeric = scaler.fit_transform(numeric)

    members = []
    member_training_scores = []
    for member_name in member_names:
        print(f"[fit-v13] selecting C for {member_name}", flush=True)
        spec = MEMBER_SPECS[member_name]
        selected_c, selection_log = repeated_full_data_c_selection(
            bundle,
            spec,
            reliability,
            c_values,
            int(config["inner_splits"]),
            seeds,
        )
        member_matrix = sparse.hstack(
            [
                text_matrix,
                sparse.csr_matrix(numeric * spec.numeric_scale),
            ],
            format="csr",
        )
        (
            pair_matrix,
            pair_labels,
            pair_weights,
            pair_count,
            same_count,
            cross_count,
        ) = build_pair_matrix(
            member_matrix,
            bundle.y,
            bundle.channels,
            reliability,
            spec,
        )
        model = LogisticRegression(
            C=selected_c,
            solver="liblinear",
            max_iter=3000,
            random_state=seeds[0],
        )
        model.fit(
            pair_matrix,
            pair_labels,
            sample_weight=pair_weights,
        )
        training_scores = np.asarray(
            model.decision_function(member_matrix),
            dtype=float,
        )
        member_training_scores.append(training_scores)
        members.append(
            {
                "name": member_name,
                "weight": weights[len(members)],
                "numeric_scale": spec.numeric_scale,
                "channel_balanced_pairs": spec.channel_balanced_pairs,
                "selected_c": selected_c,
                "c_selection": selection_log,
                "training_pair_count": pair_count,
                "same_channel_pair_count": same_count,
                "cross_channel_pair_count": cross_count,
                "model": model,
                "score_reference_sorted": np.sort(training_scores),
            }
        )

    ensemble_training_scores = weighted_average(
        member_training_scores,
        weights,
    )
    artifact = {
        "artifact_id": "shortform_success_judge_v13_frozen_ensemble",
        "status": "development_frozen_pending_holdout",
        "target_definition": "continuous within-channel Shorts view percentile",
        "score_definition": (
            "Empirical percentile of the frozen three-member pairwise ensemble "
            "against the 94-candidate development reference distribution."
        ),
        "quality_columns": QUALITY_COLUMNS,
        "structure_columns": STRUCTURE_COLUMNS,
        "forbidden_inputs": [
            "channel_name",
            "views",
            "likes",
            "view_count",
            "like_count",
            "performance_label",
            "performance_label_PRIVATE",
            "performance_percentile",
            "channel_performance_percentile_PRIVATE",
            "percentile_bucket",
            "dataset_role_v2",
            "dataset_role_v3",
            "short_url",
            "short_video_url",
            "short_video_id",
            "long_video_url",
            "transcript_source",
        ],
        "training_candidate_count": int(len(bundle.y)),
        "training_longform_count": int(len(set(bundle.groups))),
        "vectorizer": vectorizer,
        "numeric_imputer": imputer,
        "numeric_scaler": scaler,
        "members": members,
        "ensemble_weights": weights,
        "ensemble_score_reference_sorted": np.sort(ensemble_training_scores),
        "validation_summary": validation,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)

    metadata = {
        "artifact_id": artifact["artifact_id"],
        "status": artifact["status"],
        "target_definition": artifact["target_definition"],
        "training_candidate_count": artifact["training_candidate_count"],
        "training_longform_count": artifact["training_longform_count"],
        "quality_columns": QUALITY_COLUMNS,
        "structure_columns": STRUCTURE_COLUMNS,
        "forbidden_inputs": artifact["forbidden_inputs"],
        "members": [
            {
                key: member[key]
                for key in [
                    "name",
                    "weight",
                    "numeric_scale",
                    "channel_balanced_pairs",
                    "selected_c",
                    "training_pair_count",
                ]
            }
            for member in members
        ],
        "internal_acceptance_gates_passed": validation[
            "internal_acceptance_gates_passed"
        ],
        "accepted_as_performance_judge": False,
        "artifact_private_path": public_artifact_reference(args.artifact),
        "warning": (
            "The architecture passed internal OOF gates but was selected on the "
            "same development dataset. A fresh holdout is required."
        ),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(
            json_safe(metadata),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(metadata),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
