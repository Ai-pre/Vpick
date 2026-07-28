from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from train_performance_calibrator_v11 import (
    DEFAULT_PRIVATE_DIR,
    build_structure_features,
    compose_candidate_text,
)


DEFAULT_ARTIFACT = (
    DEFAULT_PRIVATE_DIR
    / "performance_calibrator_v11"
    / "shortform_success_judge_v11_experimental.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score new anonymous shortform candidates with the frozen success Judge."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-experimental", action="store_true")
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        return payload["candidates"]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Input must be one candidate, a candidate array, or JSONL.")


def validate_record(
    record: dict[str, Any],
    quality_columns: list[str],
    forbidden_inputs: list[str],
) -> dict[str, float]:
    leaked = [
        key
        for key in forbidden_inputs
        if key in record and record.get(key) not in (None, "", [])
    ]
    if leaked:
        raise ValueError(
            f"{record.get('candidate_id', '<unknown>')}: forbidden inputs {leaked}"
        )
    if not str(record.get("candidate_id") or "").strip():
        raise ValueError("candidate_id is required.")
    if not str(record.get("description") or "").strip():
        raise ValueError(f"{record['candidate_id']}: description is required.")
    if not str(record.get("transcript") or "").strip():
        raise ValueError(f"{record['candidate_id']}: transcript is required.")
    dimensions = record.get("codex_features")
    if not isinstance(dimensions, dict):
        dimensions = record.get("quality_features")
    if not isinstance(dimensions, dict):
        dimensions = record
    values = {}
    for column in quality_columns:
        if column not in dimensions:
            raise ValueError(f"{record['candidate_id']}: missing Codex feature {column}")
        value = float(dimensions[column])
        if not 0.0 <= value <= 4.0:
            raise ValueError(
                f"{record['candidate_id']}: {column} must be between 0 and 4"
            )
        values[column] = value
    return values


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
    quality_matrix = np.array(
        [
            [row[column] for column in artifact["quality_columns"]]
            for row in quality_rows
        ],
        dtype=float,
    )
    structure_matrix = structure[artifact["structure_columns"]].to_numpy(dtype=float)
    numeric = np.column_stack([quality_matrix, structure_matrix])
    numeric = artifact["numeric_imputer"].transform(numeric)
    numeric = artifact["numeric_scaler"].transform(numeric)
    texts = [
        compose_candidate_text(record, include_context=True)
        for record in normalized_records
    ]
    text_matrix = artifact["vectorizer"].transform(texts)
    matrix = sparse.hstack(
        [text_matrix, sparse.csr_matrix(numeric)],
        format="csr",
    )
    raw_scores = artifact["pairwise_model"].decision_function(matrix)
    reference = np.asarray(artifact["score_reference_sorted"], dtype=float)
    percentiles = (
        np.searchsorted(reference, raw_scores, side="right") / len(reference) * 100.0
    )
    batch_order = np.argsort(-percentiles, kind="mergesort")
    batch_ranks = np.empty(len(records), dtype=int)
    batch_ranks[batch_order] = np.arange(1, len(records) + 1)
    warning = (
        "Research-only score: the artifact failed registered validation gates."
        if artifact["status"] != "validated"
        else ""
    )
    return [
        {
            "candidate_id": normalized_records[index]["candidate_id"],
            "shortform_success_potential_0_100": round(
                float(percentiles[index]),
                4,
            ),
            "batch_rank": int(batch_ranks[index]),
            "raw_ranker_score": round(float(raw_scores[index]), 6),
            "judge_status": artifact["status"],
            "score_interpretation": (
                "Content-based expected relative performance percentile under "
                "comparable publishing conditions; not a view-count prediction."
            ),
            "warning": warning,
        }
        for index in range(len(records))
    ]


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.artifact)
    if artifact["status"] != "validated" and not args.allow_experimental:
        raise SystemExit(
            "Artifact status is experimental_rejected. Use --allow-experimental "
            "only for research diagnostics."
        )
    results = score_records(read_records(args.input), artifact)
    payload = {
        "artifact_id": artifact["artifact_id"],
        "status": artifact["status"],
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
