from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


EVIDENCE_DIMENSIONS = (
    "overview_support",
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
)

EDITORIAL_DIMENSIONS = (
    "source_salience",
    "self_contained_clarity",
    "progression_payoff",
    "boundary_integrity",
)

ENGAGEMENT_DIMENSIONS = (
    "opening_pull",
    "change_or_surprise",
    "emotional_or_information_gain",
    "memorable_specificity",
)

FAILURE_FLAGS = {
    "weak_source_salience",
    "context_dependent",
    "weak_progression",
    "weak_payoff",
    "awkward_start",
    "awkward_end",
    "weak_opening",
    "no_change",
    "low_gain",
    "not_memorable",
    "visual_dependent",
    "asr_degraded",
    "insufficient_evidence",
}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    scale = config.get("score_scale") or {}
    if int(scale.get("minimum", -1)) != 0 or int(scale.get("maximum", -1)) != 4:
        raise ValueError("Shortform Judge v9 score range must be 0..4")
    validate_weights(
        config["dimension_weights"]["editorial"],
        EDITORIAL_DIMENSIONS,
        "editorial",
    )
    validate_weights(
        config["dimension_weights"]["engagement"],
        ENGAGEMENT_DIMENSIONS,
        "engagement",
    )
    validate_weights(
        config["axis_weights"],
        ("editorial", "engagement"),
        "axis",
    )
    return config


def validate_weights(
    weights: dict[str, Any],
    dimensions: tuple[str, ...],
    name: str,
) -> dict[str, float]:
    if set(weights) != set(dimensions):
        raise ValueError(f"{name} weights must exactly match {dimensions}")
    parsed = {dimension: float(weights[dimension]) for dimension in dimensions}
    if any(value < 0 for value in parsed.values()):
        raise ValueError(f"{name} weights must be non-negative")
    if not math.isclose(sum(parsed.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"{name} weights must sum to 1")
    return parsed


def bounded_int(value: Any, minimum: int, maximum: int, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be in [{minimum}, {maximum}]")
    return parsed


def weighted_score(
    scores: dict[str, int],
    weights: dict[str, Any],
    dimensions: tuple[str, ...],
) -> float:
    parsed_weights = validate_weights(weights, dimensions, "dimension")
    if set(scores) != set(dimensions):
        raise ValueError("Score dimensions do not match the configured rubric")
    return round(
        25.0
        * sum(
            parsed_weights[dimension]
            * bounded_int(scores[dimension], 0, 4, dimension)
            for dimension in dimensions
        ),
        4,
    )


def _normalize_axis(
    value: Any,
    dimensions: tuple[str, ...],
    axis: str,
) -> tuple[dict[str, int], dict[str, str]]:
    if not isinstance(value, dict) or set(value) != set(dimensions):
        raise ValueError(f"{axis} must contain exactly {dimensions}")
    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for dimension in dimensions:
        item = value[dimension]
        if not isinstance(item, dict):
            raise ValueError(f"{axis}.{dimension} must be an object")
        scores[dimension] = bounded_int(
            item.get("score"),
            0,
            4,
            f"{axis}.{dimension}.score",
        )
        reason = str(item.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"{axis}.{dimension}.reason must not be empty")
        reasons[dimension] = reason[:1000]
    return scores, reasons


def normalize_judgment(
    raw: dict[str, Any],
    candidate_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if str(raw.get("candidate_id", "")).strip() != candidate_id:
        raise ValueError("Unexpected candidate_id")
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in {"score", "abstain"}:
        raise ValueError("verdict must be score or abstain")

    evidence_raw = raw.get("evidence")
    if not isinstance(evidence_raw, dict) or set(evidence_raw) != set(
        EVIDENCE_DIMENSIONS
    ):
        raise ValueError("evidence dimensions do not match the rubric")
    evidence = {
        dimension: bounded_int(
            evidence_raw[dimension],
            1,
            5,
            f"evidence.{dimension}",
        )
        for dimension in EVIDENCE_DIMENSIONS
    }

    flags = list(
        dict.fromkeys(
            str(value).strip()
            for value in (raw.get("failure_flags") or [])
            if str(value).strip() in FAILURE_FLAGS
        )
    )
    if verdict == "abstain" and "insufficient_evidence" not in flags:
        flags.append("insufficient_evidence")

    output: dict[str, Any] = {
        "candidate_id": candidate_id,
        "verdict": verdict,
        **{f"evidence_{key}": value for key, value in evidence.items()},
        "confidence_1_5": bounded_int(
            raw.get("confidence_1_5"),
            1,
            5,
            "confidence_1_5",
        ),
        "failure_flags": "|".join(flags),
        "reason": str(raw.get("reason", "")).strip()[:1500],
    }
    if not output["reason"]:
        raise ValueError("reason must not be empty")

    if verdict == "abstain":
        if raw.get("editorial") is not None or raw.get("engagement") is not None:
            raise ValueError("abstain requires null editorial and engagement")
        output.update(
            {
                "editorial_score_100": "",
                "engagement_score_100": "",
                "judge_score_100": "",
            }
        )
        for axis, dimensions in (
            ("editorial", EDITORIAL_DIMENSIONS),
            ("engagement", ENGAGEMENT_DIMENSIONS),
        ):
            for dimension in dimensions:
                output[f"{axis}_{dimension}_score_0_4"] = ""
                output[f"{axis}_{dimension}_reason"] = ""
        return output

    editorial, editorial_reasons = _normalize_axis(
        raw.get("editorial"),
        EDITORIAL_DIMENSIONS,
        "editorial",
    )
    engagement, engagement_reasons = _normalize_axis(
        raw.get("engagement"),
        ENGAGEMENT_DIMENSIONS,
        "engagement",
    )
    editorial_score = weighted_score(
        editorial,
        config["dimension_weights"]["editorial"],
        EDITORIAL_DIMENSIONS,
    )
    engagement_score = weighted_score(
        engagement,
        config["dimension_weights"]["engagement"],
        ENGAGEMENT_DIMENSIONS,
    )
    axis_weights = validate_weights(
        config["axis_weights"],
        ("editorial", "engagement"),
        "axis",
    )
    output.update(
        {
            "editorial_score_100": editorial_score,
            "engagement_score_100": engagement_score,
            "judge_score_100": round(
                editorial_score * axis_weights["editorial"]
                + engagement_score * axis_weights["engagement"],
                4,
            ),
        }
    )
    for axis, dimensions, scores, reasons in (
        (
            "editorial",
            EDITORIAL_DIMENSIONS,
            editorial,
            editorial_reasons,
        ),
        (
            "engagement",
            ENGAGEMENT_DIMENSIONS,
            engagement,
            engagement_reasons,
        ),
    ):
        for dimension in dimensions:
            output[f"{axis}_{dimension}_score_0_4"] = scores[dimension]
            output[f"{axis}_{dimension}_reason"] = reasons[dimension]
    return output
