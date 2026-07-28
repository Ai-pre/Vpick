from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
from scipy import sparse

from fit_shortform_success_judge_v14_dev import compose_fields
from predict_shortform_success import read_records
from train_performance_calibrator_v11 import DEFAULT_PRIVATE_DIR


DEFAULT_ARTIFACT = (
    DEFAULT_PRIVATE_DIR
    / "performance_calibrator_v14_dev"
    / "shortform_success_judge_v14_dev.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score anonymous candidates with the v14 development artifact."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-development-candidate", action="store_true")
    return parser.parse_args()


def validate_record(
    record: dict[str, Any],
    forbidden_inputs: list[str],
) -> None:
    leaked = [
        key
        for key in forbidden_inputs
        if key in record and record.get(key) not in (None, "", [])
    ]
    if leaked:
        raise ValueError(
            f"{record.get('candidate_id', '<unknown>')}: forbidden inputs {leaked}"
        )
    candidate_id = str(record.get("candidate_id") or "").strip()
    if not candidate_id:
        raise ValueError("candidate_id is required.")
    if not str(record.get("description") or "").strip():
        raise ValueError(f"{candidate_id}: description is required.")
    if not str(record.get("transcript") or "").strip():
        raise ValueError(f"{candidate_id}: transcript is required.")


def score_records(
    records: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    for record in records:
        validate_record(record, artifact["forbidden_inputs"])
    normalized = [
        {
            "candidate_id": str(record["candidate_id"]),
            "description": str(record.get("description") or ""),
            "transcript": str(record.get("transcript") or ""),
            "before_context": str(record.get("before_context") or ""),
            "after_context": str(record.get("after_context") or ""),
        }
        for record in records
    ]
    objects = [SimpleNamespace(**record) for record in normalized]
    semantic, context = compose_fields(objects)
    transformers = artifact["transformers"]
    matrix = sparse.hstack(
        [
            transformers["semantic_char"].transform(semantic),
            transformers["semantic_word"].transform(semantic) * 0.75,
            transformers["context_char"].transform(context) * 0.50,
        ],
        format="csr",
    )
    raw_scores = np.asarray(
        artifact["model"].decision_function(matrix),
        dtype=float,
    )
    reference = np.asarray(
        artifact["training_score_reference_sorted"],
        dtype=float,
    )
    ranks = np.searchsorted(reference, raw_scores, side="right")
    percentiles = (ranks + 0.5) / (len(reference) + 1.0) * 100.0
    batch_order = np.argsort(-percentiles, kind="mergesort")
    batch_ranks = np.empty(len(records), dtype=int)
    batch_ranks[batch_order] = np.arange(1, len(records) + 1)
    return [
        {
            "candidate_id": normalized[index]["candidate_id"],
            "shortform_success_potential_0_100": round(
                float(percentiles[index]),
                4,
            ),
            "batch_rank": int(batch_ranks[index]),
            "raw_ranker_score": round(float(raw_scores[index]), 6),
            "judge_status": artifact["status"],
            "score_interpretation": (
                "Content-only relative performance potential against the "
                "94-item development reference; not a view-count prediction."
            ),
            "warning": (
                "Development candidate only. A fresh mid-enriched holdout has "
                "not validated this score."
            ),
        }
        for index in range(len(records))
    ]


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.artifact)
    if (
        artifact["status"] != "validated"
        and not args.allow_development_candidate
    ):
        raise SystemExit(
            "The v14 artifact is a development candidate. Use "
            "--allow-development-candidate only for holdout scoring or research."
        )
    payload = {
        "artifact_id": artifact["artifact_id"],
        "status": artifact["status"],
        "results": score_records(read_records(args.input), artifact),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
