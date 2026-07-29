from __future__ import annotations

from typing import Any


CHECK_DIMENSIONS = (
    "self_contained_context",
    "central_focus_clear",
    "opening_pull",
    "meaningful_progression",
    "payoff_or_conclusion",
    "distinctive_value",
    "memorable_specificity",
    "natural_start",
    "natural_end",
)

EVIDENCE_DIMENSIONS = (
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
)

CONTENT_MODES = {
    "entertainment",
    "informational",
    "narrative",
    "mixed",
    "unclear",
}

FAILURE_FLAGS = {
    "context_dependent",
    "weak_focus",
    "weak_opening",
    "no_progression",
    "weak_payoff",
    "low_distinctiveness",
    "not_memorable",
    "awkward_start",
    "awkward_end",
    "visual_dependent",
    "asr_degraded",
    "insufficient_evidence",
}


def bounded_integer(value: Any, minimum: int, maximum: int, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be in [{minimum}, {maximum}]")
    return parsed


def normalize_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [item.strip() for item in value.replace("|", ",").split(",")]
    elif isinstance(value, list):
        raw = [str(item).strip() for item in value]
    else:
        raise ValueError("failure_flags must be a list or delimited string")
    flags = [item for item in raw if item]
    invalid = sorted(set(flags) - FAILURE_FLAGS)
    if invalid:
        raise ValueError(f"Unsupported failure_flags: {invalid}")
    return list(dict.fromkeys(flags))


def normalize_judgment(item: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    verdict = str(item.get("verdict", "")).strip().lower()
    if verdict not in {"score", "abstain"}:
        raise ValueError("verdict must be score or abstain")

    evidence_raw = item.get("evidence") or {}
    if not isinstance(evidence_raw, dict):
        raise ValueError("evidence must be an object")
    evidence = {
        name: bounded_integer(evidence_raw.get(name), 1, 5, f"evidence.{name}")
        for name in EVIDENCE_DIMENSIONS
    }
    confidence = bounded_integer(item.get("confidence_1_5"), 1, 5, "confidence_1_5")
    flags = normalize_flags(item.get("failure_flags"))
    reason = str(item.get("reason", "")).strip()
    if not reason:
        raise ValueError("reason must not be empty")

    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "verdict": verdict,
        **{f"evidence_{name}": evidence[name] for name in EVIDENCE_DIMENSIONS},
        "content_mode": "",
        "editorial_quality_1_5": "",
        **{f"check_{name}": "" for name in CHECK_DIMENSIONS},
        "quality_score_100": "",
        "overall_editorial_suitable": "",
        "confidence_1_5": confidence,
        "failure_flags": "|".join(flags),
        "reason": reason,
    }

    if verdict == "abstain":
        if "insufficient_evidence" not in flags:
            raise ValueError("abstain requires insufficient_evidence")
        return row

    content_mode = str(item.get("content_mode", "")).strip().lower()
    if content_mode not in CONTENT_MODES:
        raise ValueError(f"content_mode must be one of {sorted(CONTENT_MODES)}")
    editorial_quality = bounded_integer(
        item.get("editorial_quality_1_5"),
        1,
        5,
        "editorial_quality_1_5",
    )
    checks_raw = item.get("checks")
    if not isinstance(checks_raw, dict):
        raise ValueError("checks must be an object for scored judgments")
    checks = {
        name: bounded_integer(checks_raw.get(name), 0, 1, f"checks.{name}")
        for name in CHECK_DIMENSIONS
    }
    suitable_raw = item.get("overall_editorial_suitable")
    if not isinstance(suitable_raw, bool):
        raise ValueError("overall_editorial_suitable must be boolean")

    row.update(
        {
            "content_mode": content_mode,
            "editorial_quality_1_5": editorial_quality,
            **{f"check_{name}": checks[name] for name in CHECK_DIMENSIONS},
            "quality_score_100": round(
                100 * sum(checks.values()) / len(CHECK_DIMENSIONS),
                4,
            ),
            "overall_editorial_suitable": int(suitable_raw),
        }
    )
    return row
