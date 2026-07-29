from __future__ import annotations

from typing import Any

from llm_client import LLMError


EVIDENCE_DIMENSIONS = (
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
)

CHECK_DIMENSIONS = (
    "hook_within_3s",
    "surprise_or_twist",
    "emotional_peak",
    "quotable_moment",
    "payoff_or_conclusion",
    "natural_start",
    "natural_end",
)

FAILURE_FLAGS = {
    "weak_hook",
    "no_surprise",
    "flat_emotion",
    "not_quotable",
    "weak_payoff",
    "awkward_start",
    "awkward_end",
    "visual_dependent",
    "asr_degraded",
    "insufficient_evidence",
}


def bounded_int(value: Any, low: int, high: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError) as exc:
        raise LLMError(f"Expected integer in [{low}, {high}], got {value!r}") from exc
    if not low <= parsed <= high:
        raise LLMError(f"Expected integer in [{low}, {high}], got {parsed}")
    return parsed


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise LLMError(f"Expected boolean, got {value!r}")


def normalize_judgment(item: dict[str, Any], expected_id: str) -> dict[str, Any]:
    candidate_id = str(item.get("candidate_id", "")).strip()
    if candidate_id != expected_id:
        raise LLMError(f"Expected candidate_id {expected_id!r}, got {candidate_id!r}")

    verdict = str(item.get("verdict", "")).strip().lower()
    if verdict not in {"score", "abstain"}:
        raise LLMError(f"Invalid verdict for {candidate_id}: {verdict!r}")

    evidence_raw = item.get("evidence")
    if not isinstance(evidence_raw, dict):
        raise LLMError(f"Missing evidence object for {candidate_id}")
    evidence = {
        name: bounded_int(evidence_raw.get(name), 1, 5)
        for name in EVIDENCE_DIMENSIONS
    }

    raw_flags = item.get("failure_flags") or []
    if not isinstance(raw_flags, list):
        raise LLMError(f"failure_flags must be a list for {candidate_id}")
    unknown_flags = {str(flag) for flag in raw_flags} - FAILURE_FLAGS
    if unknown_flags:
        raise LLMError(f"Unknown failure_flags for {candidate_id}: {sorted(unknown_flags)}")
    flags = [str(flag) for flag in raw_flags]

    if verdict == "abstain":
        if "insufficient_evidence" not in flags:
            flags.append("insufficient_evidence")
        saliency: int | str = ""
        checks: dict[str, int] = {}
        checklist_score: float | str = ""
        suitable: int | str = ""
    else:
        saliency = bounded_int(item.get("saliency_market_1_5"), 1, 5)
        checks_raw = item.get("checks")
        if not isinstance(checks_raw, dict):
            raise LLMError(f"Missing checks object for {candidate_id}")
        checks = {
            name: bounded_int(checks_raw.get(name), 0, 2)
            for name in CHECK_DIMENSIONS
        }
        checklist_score = 100.0 * sum(checks.values()) / (2 * len(CHECK_DIMENSIONS))
        suitable = int(boolean(item.get("overall_shortform_suitable")))

    return {
        "candidate_id": candidate_id,
        "verdict": verdict,
        **{f"evidence_{name}": value for name, value in evidence.items()},
        "saliency_market_1_5": saliency,
        **{
            f"check_{name}": checks.get(name, "")
            for name in CHECK_DIMENSIONS
        },
        "checklist_score_100": checklist_score,
        "overall_shortform_suitable": suitable,
        "confidence_1_5": bounded_int(item.get("confidence_1_5"), 1, 5),
        "failure_flags": "|".join(flags),
        "reason": str(item.get("reason", ""))[:1200],
    }
