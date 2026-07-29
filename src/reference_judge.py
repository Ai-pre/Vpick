from __future__ import annotations

from typing import Any

from llm_client import LLMError


EVIDENCE_DIMENSIONS = (
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
)

CHECKLIST_DIMENSIONS = (
    "central_focus_clear",
    "highlight_worthy",
    "important_or_representative",
    "context_sufficient",
    "meaningful_progression",
    "payoff_or_conclusion",
    "natural_start",
    "natural_end",
)


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


def normalize_judgments(response: dict[str, Any], expected_ids: set[str]) -> list[dict[str, Any]]:
    raw_items = response.get("judgments")
    if not isinstance(raw_items, list):
        raise LLMError("Reference Judge response must contain a judgments list")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id not in expected_ids or candidate_id in seen:
            continue

        verdict = str(item.get("verdict", "score")).strip().lower()
        if verdict not in {"score", "abstain"}:
            raise LLMError(f"Invalid verdict for {candidate_id}: {verdict}")

        evidence_raw = item.get("evidence") or {}
        evidence = {
            name: bounded_int(evidence_raw.get(name), 1, 5)
            for name in EVIDENCE_DIMENSIONS
        }
        flags = [str(value) for value in (item.get("failure_flags") or [])[:12]]

        if verdict == "abstain":
            if "insufficient_evidence" not in flags:
                flags.append("insufficient_evidence")
            saliency: int | str = ""
            checklist: dict[str, bool] = {}
            saliency_score: float | str = ""
            checklist_score: float | str = ""
            reference_score: float | str = ""
            suitable: int | str = ""
        else:
            saliency = bounded_int(item.get("highlight_saliency_1_5"), 1, 5)
            checklist_raw = item.get("checklist") or {}
            checklist = {
                name: boolean(checklist_raw.get(name))
                for name in CHECKLIST_DIMENSIONS
            }
            saliency_score = (saliency - 1) * 25.0
            checklist_score = 100.0 * sum(checklist.values()) / len(CHECKLIST_DIMENSIONS)
            # Equal aggregation is explicit and keeps both components independently reportable.
            reference_score = (saliency_score + checklist_score) / 2.0
            suitable = int(boolean(item.get("overall_shortform_suitable")))

        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "verdict": verdict,
            **{f"evidence_{name}": value for name, value in evidence.items()},
            "highlight_saliency_1_5": saliency,
            **{
                f"check_{name}": int(checklist[name]) if name in checklist else ""
                for name in CHECKLIST_DIMENSIONS
            },
            "saliency_score_100": saliency_score,
            "checklist_score_100": checklist_score,
            "reference_score_100": reference_score,
            "overall_shortform_suitable": suitable,
            "confidence": bounded_int(item.get("confidence", 1), 1, 5),
            "failure_flags": "|".join(flags),
            "reason": str(item.get("reason", ""))[:1200],
        }
        rows.append(row)
        seen.add(candidate_id)

    if seen != expected_ids:
        raise LLMError(f"Reference Judge omitted candidate IDs: {sorted(expected_ids - seen)}")
    return rows
