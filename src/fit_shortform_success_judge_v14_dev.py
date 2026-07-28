from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from train_performance_calibrator_v11 import (
    DEFAULT_PRIVATE_DIR,
    ROOT,
    json_safe,
    load_bundle,
)
from train_performance_calibrator_v12 import (
    build_pair_matrix,
    normalize_semantic_text,
)
from train_performance_calibrator_v14_dev import (
    candidate_specs,
    remove_proxy_features,
)


DEFAULT_CONFIG = ROOT / "config" / "performance_calibrator_v14_dev.json"
DEFAULT_ARTIFACT = (
    DEFAULT_PRIVATE_DIR
    / "performance_calibrator_v14_dev"
    / "shortform_success_judge_v14_dev.joblib"
)
DEFAULT_METADATA = (
    ROOT
    / "results"
    / "performance_calibrator_v14_dev"
    / "deployment_artifact_METADATA.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the frozen v14 development candidate on all 94 items."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def safe_text(record: Any, field: str) -> str:
    value = getattr(record, field, "")
    return "" if value is None else str(value)


def compose_fields(records: list[Any]) -> tuple[list[str], list[str]]:
    semantic = []
    context = []
    for record in records:
        semantic.append(
            "\n".join(
                [
                    "[DESCRIPTION]",
                    normalize_semantic_text(safe_text(record, "description")),
                    "[TRANSCRIPT]",
                    normalize_semantic_text(safe_text(record, "transcript")),
                ]
            )
        )
        context.append(
            "\n".join(
                [
                    "[BEFORE]",
                    normalize_semantic_text(safe_text(record, "before_context")),
                    "[AFTER]",
                    normalize_semantic_text(safe_text(record, "after_context")),
                ]
            )
        )
    return semantic, context


def fit_text_transformers(
    records: list[Any],
) -> tuple[dict[str, TfidfVectorizer], sparse.csr_matrix]:
    semantic, context = compose_fields(records)
    semantic_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.98,
        max_features=5000,
        sublinear_tf=True,
        norm="l2",
    )
    semantic_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=3000,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    context_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.98,
        max_features=1500,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = sparse.hstack(
        [
            semantic_char.fit_transform(semantic),
            semantic_word.fit_transform(semantic) * 0.75,
            context_char.fit_transform(context) * 0.50,
        ],
        format="csr",
    )
    return {
        "semantic_char": semantic_char,
        "semantic_word": semantic_word,
        "context_char": context_char,
    }, matrix


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_deployment_candidate"]
    raw_bundle = load_bundle(args.private_dir)
    bundle, kept_structure = remove_proxy_features(
        raw_bundle,
        [str(value) for value in config["excluded_proxy_features"]],
    )
    specs = candidate_specs(config)
    spec_name = str(frozen["spec"])
    if spec_name not in specs:
        raise ValueError(f"Unknown frozen deployment spec: {spec_name}")
    spec = specs[spec_name]
    if spec.numeric_scale != 0.0:
        raise ValueError(
            "The v14 artifact implementation is intentionally text-only."
        )

    records = list(bundle.frame.itertuples(index=False))
    transformers, candidate_matrix = fit_text_transformers(records)
    (
        pair_matrix,
        pair_labels,
        pair_weights,
        pair_count,
        same_channel_pair_count,
        cross_channel_pair_count,
    ) = build_pair_matrix(
        candidate_matrix,
        bundle.y,
        bundle.channels,
        np.ones(len(bundle.y), dtype=float),
        spec,
    )
    model = LogisticRegression(
        C=float(frozen["c_value"]),
        solver="liblinear",
        max_iter=3000,
        random_state=int(config["random_seeds"][0]),
    )
    model.fit(
        pair_matrix,
        pair_labels,
        sample_weight=pair_weights,
    )
    training_scores = np.asarray(
        model.decision_function(candidate_matrix),
        dtype=float,
    )
    artifact = {
        "artifact_id": "shortform_success_judge_v14_dev_text_only",
        "status": "development_only_not_validated",
        "protocol_id": config["protocol_id"],
        "score_definition": (
            "Empirical percentile of the fixed pairwise text ranker against "
            "the 94-candidate development reference score distribution."
        ),
        "required_input_fields": [
            "candidate_id",
            "description",
            "transcript",
            "before_context",
            "after_context",
        ],
        "forbidden_inputs": config["forbidden_model_inputs"],
        "normalization": "normalize_semantic_text",
        "spec": spec,
        "c_value": float(frozen["c_value"]),
        "transformers": transformers,
        "model": model,
        "training_score_reference_sorted": np.sort(training_scores),
        "training_candidate_count": int(len(bundle.y)),
        "training_longform_count": int(len(set(bundle.groups))),
        "training_pair_count": int(pair_count),
        "same_channel_pair_count": int(same_channel_pair_count),
        "cross_channel_pair_count": int(cross_channel_pair_count),
        "accepted_as_performance_judge": False,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)

    metadata = {
        key: artifact[key]
        for key in [
            "artifact_id",
            "status",
            "protocol_id",
            "score_definition",
            "required_input_fields",
            "forbidden_inputs",
            "normalization",
            "c_value",
            "training_candidate_count",
            "training_longform_count",
            "training_pair_count",
            "same_channel_pair_count",
            "cross_channel_pair_count",
            "accepted_as_performance_judge",
        ]
    }
    metadata["spec"] = {
        "name": spec.name,
        "representation": spec.representation,
        "score_calibration": spec.score_calibration,
        "mid_pair_boost": spec.mid_pair_boost,
        "local_pair_boost": spec.local_boost,
        "extreme_pair_weight": spec.extreme_pair_weight,
    }
    metadata["excluded_proxy_features"] = config[
        "excluded_proxy_features"
    ]
    metadata["kept_structure_features_for_audit_only"] = kept_structure
    metadata["artifact_private_path"] = str(
        args.artifact.relative_to(ROOT)
    ).replace("\\", "/")
    metadata["warning"] = (
        "This artifact is a development candidate. It must be scored once on "
        "a fresh mid-enriched holdout before any performance-judge claim."
    )
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(
            json_safe(metadata),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "metadata": str(args.metadata),
                "training_pair_count": pair_count,
                "spec": spec_name,
                "c_value": frozen["c_value"],
                "accepted_as_performance_judge": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
