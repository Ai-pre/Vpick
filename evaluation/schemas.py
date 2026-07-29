from __future__ import annotations

from typing import Any


STANDALONE_DIMENSIONS = (
    "hook",
    "engagement",
    "self_contained",
    "payoff",
    "density",
    "boundary",
)

SOURCE_DIMENSIONS = (
    "source_salience",
    "relative_competitiveness",
    "hook",
    "self_contained",
    "payoff",
    "density",
    "boundary",
)


def _score(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
        raise ValueError(f"{name} must be an integer from 0 to 4")
    return value


def _validate_axis(value: Any, dimensions: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(dimensions):
        raise ValueError(f"Axis must contain exactly: {', '.join(dimensions)}")
    result: dict[str, Any] = {}
    for dimension in dimensions:
        item = value[dimension]
        if not isinstance(item, dict):
            raise ValueError(f"{dimension} must be an object")
        score = _score(item.get("score"), dimension)
        reason = str(item.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"{dimension}.reason is required")
        result[dimension] = {"score": score, "reason": reason}
    return result


def validate_pointwise(value: Any, *, source_conditioned: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Judgment must be a JSON object")
    candidate_id = str(value.get("candidate_id", "")).strip()
    verdict = str(value.get("verdict", "")).strip()
    if not candidate_id:
        raise ValueError("candidate_id is required")
    if verdict not in {"score", "abstain"}:
        raise ValueError("verdict must be score or abstain")
    dimensions = SOURCE_DIMENSIONS if source_conditioned else STANDALONE_DIMENSIONS
    scores = None if verdict == "abstain" else _validate_axis(value.get("scores"), dimensions)
    confidence = value.get("confidence_1_5")
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 1 <= confidence <= 5:
        raise ValueError("confidence_1_5 must be an integer from 1 to 5")
    return {
        "candidate_id": candidate_id,
        "verdict": verdict,
        "scores": scores,
        "confidence_1_5": confidence,
        "failure_flags": [str(flag) for flag in value.get("failure_flags", [])],
        "reason": str(value.get("reason", "")).strip(),
    }


def pointwise_score_100(judgment: dict[str, Any]) -> float | None:
    scores = judgment.get("scores")
    if not isinstance(scores, dict):
        return None
    values = [float(item["score"]) for item in scores.values()]
    return 25.0 * sum(values) / len(values)


def validate_pairwise(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Pairwise judgment must be a JSON object")
    winner = str(value.get("winner", "")).strip()
    if winner not in {"A", "B", "tie", "invalid"}:
        raise ValueError("winner must be A, B, tie, or invalid")
    confidence = value.get("confidence_1_5")
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 1 <= confidence <= 5:
        raise ValueError("confidence_1_5 must be an integer from 1 to 5")
    comparison = value.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("comparison must be an object")
    required = {"source_salience", "standalone_quality", "boundary_integrity"}
    if set(comparison) != required:
        raise ValueError(f"comparison must contain exactly: {', '.join(sorted(required))}")
    for name, preference in comparison.items():
        if preference not in {"A", "B", "tie", "invalid"}:
            raise ValueError(f"Invalid comparison preference for {name}")
    return {
        "pair_id": str(value.get("pair_id", "")).strip(),
        "winner": winner,
        "comparison": comparison,
        "confidence_1_5": confidence,
        "reason": str(value.get("reason", "")).strip(),
    }
