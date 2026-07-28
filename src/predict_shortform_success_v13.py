from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from predict_shortform_success import read_records, validate_record
from train_performance_calibrator_v11 import (
    DEFAULT_PRIVATE_DIR,
    build_structure_features,
    compose_candidate_text,
)


DEFAULT_ARTIFACT = (
    DEFAULT_PRIVATE_DIR
    / "performance_calibrator_v13"
    / "shortform_success_judge_v13_pending_holdout.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score candidates with the frozen v13 development ensemble."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-pending-holdout", action="store_true")
    return parser.parse_args()


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.asarray(reference, dtype=float)
    return (
        np.searchsorted(ordered, values, side="right")
        / len(ordered)
        * 100.0
    )


def score_records(
    records: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    quality_rows = []
    normalized_records = []
    for record in records:
        quality_rows.append(
            validate_record(
                record,
                artifact["quality_columns"],
                artifact["forbidden_inputs"],
            )
        )
        normalized_records.append(
            {
                "candidate_id": str(record["candidate_id"]),
                "duration_sec": float(record.get("duration_sec") or 0.0),
                "description": str(record.get("description") or ""),
                "transcript": str(record.get("transcript") or ""),
                "before_context": str(record.get("before_context") or ""),
                "after_context": str(record.get("after_context") or ""),
                "visual_evidence_available": bool(
                    record.get("visual_evidence_available")
                ),
            }
        )

    frame = pd.DataFrame.from_records(normalized_records)
    structure = build_structure_features(frame)
    quality = np.asarray(
        [
            [row[column] for column in artifact["quality_columns"]]
            for row in quality_rows
        ],
        dtype=float,
    )
    structure_values = structure[
        artifact["structure_columns"]
    ].to_numpy(dtype=float)
    numeric = np.column_stack([quality, structure_values])
    numeric = artifact["numeric_imputer"].transform(numeric)
    numeric = artifact["numeric_scaler"].transform(numeric)
    texts = [
        compose_candidate_text(record, include_context=True)
        for record in normalized_records
    ]
    text_matrix = artifact["vectorizer"].transform(texts)

    raw_member_scores = []
    member_percentiles = []
    weights = []
    for member in artifact["members"]:
        matrix = sparse.hstack(
            [
                text_matrix,
                sparse.csr_matrix(
                    numeric * float(member["numeric_scale"])
                ),
            ],
            format="csr",
        )
        raw = np.asarray(
            member["model"].decision_function(matrix),
            dtype=float,
        )
        raw_member_scores.append(raw)
        member_percentiles.append(
            empirical_percentile(
                member["score_reference_sorted"],
                raw,
            )
        )
        weights.append(float(member["weight"]))

    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights /= float(np.sum(normalized_weights))
    ensemble_raw = np.average(
        np.vstack(raw_member_scores),
        axis=0,
        weights=normalized_weights,
    )
    ensemble_percentile = empirical_percentile(
        artifact["ensemble_score_reference_sorted"],
        ensemble_raw,
    )
    member_percentile_matrix = np.vstack(member_percentiles)
    disagreement = np.std(member_percentile_matrix, axis=0)
    lower_member = np.min(member_percentile_matrix, axis=0)
    upper_member = np.max(member_percentile_matrix, axis=0)
    order = np.argsort(-ensemble_percentile, kind="mergesort")
    ranks = np.empty(len(records), dtype=int)
    ranks[order] = np.arange(1, len(records) + 1)

    return [
        {
            "candidate_id": normalized_records[index]["candidate_id"],
            "shortform_success_potential_0_100": round(
                float(ensemble_percentile[index]),
                4,
            ),
            "batch_rank": int(ranks[index]),
            "ensemble_raw_score": round(float(ensemble_raw[index]), 6),
            "member_percentile_min": round(
                float(lower_member[index]),
                4,
            ),
            "member_percentile_max": round(
                float(upper_member[index]),
                4,
            ),
            "ensemble_disagreement_std": round(
                float(disagreement[index]),
                4,
            ),
            "judge_status": artifact["status"],
            "score_interpretation": (
                "Content-based expected relative performance percentile under "
                "comparable publishing conditions; not a view-count prediction."
            ),
            "warning": (
                "The model passed internal development gates but still requires "
                "a fresh untouched holdout."
            ),
        }
        for index in range(len(records))
    ]


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.artifact)
    if (
        artifact["status"] == "development_frozen_pending_holdout"
        and not args.allow_pending_holdout
    ):
        raise SystemExit(
            "The v13 artifact is pending a fresh holdout. Use "
            "--allow-pending-holdout for development diagnostics."
        )
    payload = {
        "artifact_id": artifact["artifact_id"],
        "status": artifact["status"],
        "results": score_records(read_records(args.input), artifact),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
