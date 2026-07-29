from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "source_salience",
    "hook",
    "payoff",
    "self_contained",
    "density",
    "boundary",
)

FATAL_FLAGS = {
    "missing_context",
    "abrupt_start",
    "abrupt_end",
    "no_payoff",
    "duplicate_content",
    "insufficient_information",
}

FATAL_FLAG_ALIASES = {
    "context_missing": "missing_context",
    "missing_context_at_start": "missing_context",
    "abrupt_beginning": "abrupt_start",
    "abrupt_finish": "abrupt_end",
    "payoff_missing": "no_payoff",
}

CANDIDATE_SOURCES = {
    "published_short",
    "vpick",
    "existing_model",
    "boundary_shift",
    "hard_negative",
    "random",
}

PAIRWISE_WINNERS = {"A", "B", "tie", "invalid"}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_weights(config.get("weights") or {})
    if int(config.get("score_min", -1)) != 0 or int(config.get("score_max", -1)) != 4:
        raise ValueError("Judge v1 score range must be 0..4")
    return config


def validate_weights(weights: dict[str, Any]) -> dict[str, float]:
    if set(weights) != set(DIMENSIONS):
        missing = sorted(set(DIMENSIONS) - set(weights))
        extra = sorted(set(weights) - set(DIMENSIONS))
        raise ValueError(f"Weight dimensions differ; missing={missing}, extra={extra}")
    normalized = {name: float(weights[name]) for name in DIMENSIONS}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("Weights must be non-negative")
    if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Weights must sum to 1")
    return normalized


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


def normalize_fatal_flags(values: Any) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    repairs: list[str] = []
    for raw_value in values or []:
        value = str(raw_value).strip()
        if not value:
            continue
        mapped = FATAL_FLAG_ALIASES.get(value, value)
        if mapped != value:
            repairs.append(f"{value}_to_{mapped}")
        if mapped not in FATAL_FLAGS:
            repairs.append(f"dropped_unsupported_fatal_flag:{value}")
            continue
        normalized.append(mapped)
    return list(dict.fromkeys(normalized)), repairs


def validate_candidate(candidate: dict[str, Any]) -> None:
    required = {
        "candidate_id",
        "longform_id",
        "start_ms",
        "end_ms",
        "scene_ids",
        "candidate_source",
        "is_published",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"Candidate missing fields: {missing}")
    if not str(candidate["candidate_id"]).strip() or not str(candidate["longform_id"]).strip():
        raise ValueError("candidate_id and longform_id must be non-empty")
    start_ms = int(candidate["start_ms"])
    end_ms = int(candidate["end_ms"])
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("Candidate interval is invalid")
    if candidate["candidate_source"] not in CANDIDATE_SOURCES:
        raise ValueError(f"Unsupported candidate_source: {candidate['candidate_source']}")
    if not isinstance(candidate["scene_ids"], list):
        raise ValueError("scene_ids must be a list")
    if not isinstance(candidate["is_published"], bool):
        raise ValueError("is_published must be boolean")


def validate_scene(scene: dict[str, Any]) -> None:
    required = {
        "scene_id",
        "start_ms",
        "end_ms",
        "scene_name",
        "description",
        "transcript",
        "speaker",
        "person_ids",
    }
    missing = sorted(required - set(scene))
    if missing:
        raise ValueError(f"Scene missing fields: {missing}")
    if not str(scene["scene_id"]).strip():
        raise ValueError("scene_id must be non-empty")
    start_ms = int(scene["start_ms"])
    end_ms = int(scene["end_ms"])
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("Scene interval is invalid")
    if not isinstance(scene["person_ids"], list):
        raise ValueError("person_ids must be a list")


def validate_longform(longform: dict[str, Any]) -> None:
    required = {
        "longform_id",
        "channel_id",
        "title",
        "duration_ms",
        "upload_date",
        "view_count",
        "scenes",
    }
    missing = sorted(required - set(longform))
    if missing:
        raise ValueError(f"Longform missing fields: {missing}")
    if not str(longform["longform_id"]).strip():
        raise ValueError("longform_id must be non-empty")
    if int(longform["duration_ms"]) <= 0:
        raise ValueError("duration_ms must be positive")
    if not isinstance(longform["scenes"], list) or not longform["scenes"]:
        raise ValueError("scenes must be a non-empty list")
    for scene in longform["scenes"]:
        if not isinstance(scene, dict):
            raise ValueError("Each scene must be an object")
        validate_scene(scene)


def validate_pairwise_annotation(annotation: dict[str, Any]) -> None:
    required = {
        "pair_id",
        "longform_id",
        "candidate_a_id",
        "candidate_b_id",
        "display_order",
        "annotator_id",
        "winner",
        "reason",
        "created_at",
    }
    missing = sorted(required - set(annotation))
    if missing:
        raise ValueError(f"Pairwise annotation missing fields: {missing}")
    if annotation["candidate_a_id"] == annotation["candidate_b_id"]:
        raise ValueError("Pairwise candidates must differ")
    if str(annotation["display_order"]) not in {"A-B", "B-A"}:
        raise ValueError("display_order must be A-B or B-A")
    if str(annotation["winner"]) not in PAIRWISE_WINNERS:
        raise ValueError("winner must be A, B, tie, or invalid")
    confidence = annotation.get("confidence_1_5")
    if confidence not in (None, ""):
        bounded_int(confidence, 1, 5, "confidence_1_5")


def weighted_total(scores: dict[str, int], weights: dict[str, Any]) -> float:
    validated_weights = validate_weights(weights)
    if set(scores) != set(DIMENSIONS):
        raise ValueError("Score dimensions do not match the rubric")
    values = {
        name: bounded_int(scores[name], 0, 4, f"dimensions.{name}.score")
        for name in DIMENSIONS
    }
    return round(
        100.0 * sum(validated_weights[name] * values[name] / 4.0 for name in DIMENSIONS),
        4,
    )


def normalize_pointwise(
    raw: dict[str, Any],
    candidate_id: str,
    weights: dict[str, Any],
) -> dict[str, Any]:
    if str(raw.get("candidate_id", "")).strip() != candidate_id:
        raise ValueError("Unexpected candidate_id")
    raw_verdict = str(raw.get("verdict", "")).strip().lower()
    schema_repairs: list[str] = []
    if raw_verdict in {"0", "1", "2", "3", "4"} and isinstance(
        raw.get("dimensions"), dict
    ):
        verdict = "score"
        schema_repairs.append("numeric_verdict_to_score")
    else:
        verdict = raw_verdict
    if verdict not in {"score", "invalid"}:
        raise ValueError("verdict must be score or invalid")
    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(DIMENSIONS):
        raise ValueError("dimensions must contain the six rubric dimensions")
    output: dict[str, Any] = {
        "candidate_id": candidate_id,
        "verdict": verdict,
    }
    scores: dict[str, int] = {}
    for name in DIMENSIONS:
        item = dimensions[name]
        if not isinstance(item, dict):
            raise ValueError(f"dimensions.{name} must be an object")
        value = bounded_int(item.get("score"), 0, 4, f"dimensions.{name}.score")
        reason = str(item.get("reason", "")).strip()
        scene_ids = item.get("scene_ids")
        insufficient = item.get("insufficient_information")
        if not reason or not isinstance(scene_ids, list) or not isinstance(insufficient, bool):
            raise ValueError(f"dimensions.{name} evidence fields are invalid")
        scores[name] = value
        output[f"{name}_score_0_4"] = value
        output[f"{name}_reason"] = reason[:1000]
        output[f"{name}_scene_ids"] = "|".join(str(scene_id) for scene_id in scene_ids)
        output[f"{name}_insufficient_information"] = int(insufficient)

    flags, flag_repairs = normalize_fatal_flags(raw.get("fatal_flags"))
    schema_repairs.extend(flag_repairs)
    confidence = bounded_int(raw.get("confidence_1_5"), 1, 5, "confidence_1_5")
    overall_reason = str(raw.get("overall_reason", "")).strip()
    if not overall_reason:
        raise ValueError("overall_reason must not be empty")
    if verdict == "invalid" and "insufficient_information" not in flags:
        raise ValueError("invalid verdict requires insufficient_information")
    output.update(
        {
            "highlight_quality_score_100": (
                "" if verdict == "invalid" else weighted_total(scores, weights)
            ),
            "fatal_flags": "|".join(dict.fromkeys(flags)),
            "confidence_1_5": confidence,
            "overall_reason": overall_reason[:1500],
            "schema_repairs": "|".join(schema_repairs),
        }
    )
    return output


def normalize_pairwise(raw: dict[str, Any], pair_id: str) -> dict[str, Any]:
    if str(raw.get("pair_id", "")).strip() != pair_id:
        raise ValueError("Unexpected pair_id")
    comparisons = raw.get("dimension_comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != set(DIMENSIONS):
        raise ValueError("dimension_comparisons must contain all dimensions")
    output: dict[str, Any] = {"pair_id": pair_id}
    for name in DIMENSIONS:
        item = comparisons[name]
        winner = str(item.get("winner", "")).strip()
        reason = str(item.get("reason", "")).strip()
        scene_ids = item.get("scene_ids")
        if winner not in {"A", "B", "tie"} or not reason or not isinstance(scene_ids, list):
            raise ValueError(f"Invalid pairwise evidence for {name}")
        output[f"{name}_winner"] = winner
        output[f"{name}_reason"] = reason[:1000]
        output[f"{name}_scene_ids"] = "|".join(str(value) for value in scene_ids)
    winner = str(raw.get("winner", "")).strip()
    if winner not in PAIRWISE_WINNERS:
        raise ValueError("winner must be A, B, tie, or invalid")
    flags_a, repairs_a = normalize_fatal_flags(raw.get("fatal_flags_a"))
    flags_b, repairs_b = normalize_fatal_flags(raw.get("fatal_flags_b"))
    output.update(
        {
            "winner": winner,
            "fatal_flags_a": "|".join(flags_a),
            "fatal_flags_b": "|".join(flags_b),
            "confidence_1_5": bounded_int(
                raw.get("confidence_1_5"), 1, 5, "confidence_1_5"
            ),
            "reason": str(raw.get("reason", "")).strip()[:1500],
            "schema_repairs": "|".join(repairs_a + repairs_b),
        }
    )
    if not output["reason"]:
        raise ValueError("Pairwise reason must not be empty")
    return output


def flip_pairwise_winner(value: str) -> str:
    return {"A": "B", "B": "A", "tie": "tie", "invalid": "invalid"}[value]
