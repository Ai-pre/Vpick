from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from performance_judge_v1 import (
    CODEX_FEATURES,
    RUBRIC_FEATURES,
    extract_structure_features,
    feature_matrix,
    model_from_artifact,
    predict_logistic,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def classify_score(
    score: float,
    low_max: float = 35.0,
    high_min: float = 65.0,
) -> str:
    if score >= high_min:
        return "high_signal"
    if score <= low_max:
        return "low_signal"
    return "uncertain"


def ensure_deployable(
    artifact: dict[str, Any],
    allow_unvalidated: bool,
) -> None:
    if (
        artifact.get("deployment_status") != "validated"
        and not allow_unvalidated
    ):
        raise ValueError(
            "This artifact is not validated for deployment: "
            + str(artifact.get("deployment_block_reason") or "unknown reason")
        )


def build_feature_row(
    candidate: dict[str, Any],
    judgment: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = extract_structure_features(candidate)
    row.update(judgment)
    for feature in RUBRIC_FEATURES:
        if feature in judgment:
            row[f"claude_{feature}"] = judgment[feature]
    for feature in CODEX_FEATURES:
        if feature in judgment:
            row[f"claude_v7_{feature}"] = judgment[feature]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one new Short candidate with Performance Judge v1."
    )
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--judge-json", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Permit diagnostic scoring with an artifact that failed validation.",
    )
    args = parser.parse_args()

    candidate = load_json(args.candidate_json)
    judgment = load_json(args.judge_json)
    artifact = load_json(args.model_artifact)
    ensure_deployable(artifact, args.allow_unvalidated)
    feature_names = list(artifact["feature_names"])
    row = build_feature_row(candidate, judgment)
    missing = [
        feature
        for feature in feature_names
        if row.get(feature, "") in ("", None)
    ]
    if missing:
        raise ValueError(
            "Required Judge features are missing: " + ", ".join(missing)
        )
    score = float(
        predict_logistic(
            model_from_artifact(artifact),
            feature_matrix([row], feature_names),
        )[0]
        * 100.0
    )
    policy = artifact.get("decision_policy") or {}
    low_max = float(policy.get("low_max_score_0_100", 35.0))
    high_min = float(policy.get("high_min_score_0_100", 65.0))
    result = {
        "candidate_id": candidate.get("candidate_id", ""),
        "model_name": artifact["model_name"],
        "high_performance_score_0_100": round(score, 4),
        "tier": classify_score(score, low_max=low_max, high_min=high_min),
        "decision_policy": policy,
        "output_semantics": artifact["output_semantics"],
        "validation": artifact.get("validation", {}),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
