from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import ROOT, read_csv, read_jsonl, rounded, write_csv


REQUESTS = (
    ROOT / "results" / "evaluation_system_v1" / "requests" / "source_pointwise_requests.jsonl"
)
TARGETS = (
    ROOT / "results" / "evaluation_system_v1" / "prepared" / "targets_private.csv"
)
CASE2_JUDGMENTS = (
    ROOT / "results" / "evaluation_system_v1" / "case_2_codex_direct_judgments.csv"
)
OUTPUT = (
    ROOT
    / "results"
    / "evaluation_system_v1"
    / "reference_v7_codex_vpick_direct_scores.csv"
)


# saliency, hook, surprise, emotion, quotable, payoff, natural_start, natural_end
# Saliency uses 1-5. Checklist dimensions use 0-2.
DIRECT_SCORES: list[tuple[int, int, int, int, int, int, int, int]] = [
    (2, 1, 1, 1, 1, 1, 1, 1),
    (2, 1, 1, 1, 1, 1, 1, 1),
    (4, 1, 2, 2, 2, 2, 2, 1),
    (3, 1, 1, 0, 1, 1, 2, 2),
    (1, 0, 0, 0, 0, 0, 0, 0),
    (2, 1, 1, 1, 1, 1, 2, 2),
    (4, 1, 1, 2, 2, 2, 2, 2),
    (3, 1, 1, 2, 1, 1, 2, 1),
    (2, 1, 1, 1, 1, 1, 1, 0),
    (2, 1, 1, 1, 1, 1, 1, 0),
    (5, 2, 2, 2, 2, 2, 2, 1),
    (3, 1, 1, 1, 2, 1, 1, 0),
    (2, 1, 1, 1, 1, 1, 1, 2),
    (4, 2, 1, 2, 2, 2, 2, 1),
    (1, 0, 0, 0, 0, 0, 0, 0),
    (4, 1, 2, 1, 2, 2, 2, 0),
    (3, 1, 1, 1, 1, 1, 2, 1),
    (1, 0, 0, 0, 0, 0, 0, 0),
    (4, 2, 2, 2, 2, 2, 2, 1),
    (3, 1, 1, 1, 1, 1, 2, 1),
    (4, 2, 2, 2, 2, 2, 2, 2),
    (3, 1, 1, 1, 1, 1, 2, 0),
    (4, 2, 2, 2, 2, 2, 2, 1),
    (3, 1, 1, 1, 1, 1, 2, 0),
    (2, 1, 0, 1, 0, 1, 2, 2),
    (4, 2, 2, 2, 2, 2, 2, 1),
    (4, 1, 2, 2, 1, 2, 2, 2),
    (3, 1, 1, 2, 1, 1, 2, 0),
    (1, 0, 0, 0, 0, 0, 0, 0),
    (5, 2, 2, 2, 2, 2, 2, 2),
    (4, 1, 2, 2, 2, 2, 2, 1),
    (4, 1, 2, 2, 2, 2, 2, 1),
    (4, 2, 1, 2, 1, 2, 2, 1),
    (3, 1, 1, 1, 2, 2, 1, 2),
    (4, 2, 2, 2, 2, 2, 2, 2),
    (4, 1, 2, 2, 2, 2, 2, 0),
    (3, 1, 1, 1, 1, 2, 1, 1),
    (4, 1, 2, 2, 2, 2, 2, 1),
    (2, 1, 1, 1, 1, 1, 1, 0),
    (4, 1, 2, 2, 2, 2, 2, 1),
    (1, 1, 0, 0, 0, 0, 1, 0),
    (4, 2, 2, 2, 2, 2, 2, 2),
    (2, 1, 0, 1, 0, 1, 1, 0),
    (5, 1, 2, 2, 2, 2, 2, 1),
    (2, 1, 0, 1, 0, 1, 1, 0),
    (3, 1, 1, 1, 1, 1, 2, 1),
    (4, 2, 2, 2, 2, 2, 2, 2),
    (4, 1, 2, 2, 2, 2, 2, 2),
    (3, 1, 1, 2, 1, 1, 2, 0),
    (4, 1, 2, 2, 2, 2, 2, 1),
    (4, 1, 2, 2, 2, 2, 2, 0),
    (4, 2, 2, 2, 2, 2, 2, 2),
    (4, 2, 2, 2, 2, 2, 2, 1),
    (2, 1, 0, 0, 0, 1, 1, 0),
    (2, 1, 1, 1, 0, 1, 1, 0),
    (3, 1, 2, 1, 2, 2, 2, 1),
    (2, 1, 0, 1, 0, 1, 1, 0),
    (1, 0, 0, 0, 0, 0, 0, 0),
    (4, 1, 2, 2, 2, 2, 2, 1),
    (3, 1, 0, 2, 1, 1, 2, 1),
]


def _transcript_evidence(text: str) -> int:
    length = len(re.sub(r"\s+", "", text))
    if length < 20:
        return 2
    if length < 80:
        return 3
    return 4


def _failure_flags(
    checks: tuple[int, int, int, int, int, int, int],
    transcript_evidence: int,
) -> str:
    names = (
        "weak_hook",
        "no_surprise",
        "flat_emotion",
        "not_quotable",
        "weak_payoff",
        "awkward_start",
        "awkward_end",
    )
    flags = [name for name, value in zip(names, checks) if value == 0]
    if transcript_evidence <= 2:
        flags.append("asr_degraded")
    return "|".join(flags)


def compile_scores() -> list[dict[str, Any]]:
    requests = read_jsonl(REQUESTS)
    if len(requests) != len(DIRECT_SCORES):
        raise ValueError(f"Expected {len(DIRECT_SCORES)} requests, found {len(requests)}")
    target_by_current = {row["candidate_id"]: row for row in read_csv(TARGETS)}
    case2_by_current = {
        row["candidate_id"]: row for row in read_csv(CASE2_JUDGMENTS)
    }

    rows: list[dict[str, Any]] = []
    for request, direct in zip(requests, DIRECT_SCORES):
        current_id = request["candidate_id"]
        target = target_by_current[current_id]
        case2 = case2_by_current[current_id]
        saliency, *check_values = direct
        checks = tuple(check_values)
        transcript_evidence = _transcript_evidence(str(request.get("transcript") or ""))
        description_evidence = 4 if request.get("description") else 1
        context_count = int(bool(request.get("before_context"))) + int(
            bool(request.get("after_context"))
        )
        boundary_evidence = 3 + min(1, context_count)
        checklist = sum(checks) * 100.0 / 14.0
        suitable = int(
            saliency >= 4
            and checks[4] >= 1
            and checks[5] >= 1
            and checks[6] >= 1
        )
        rows.append(
            {
                "judge_run_id": "codex_direct_reference_v7_vpick_ablation",
                "provider": "openai_codex_session",
                "model": "codex_current_session",
                "prompt_id": "shortform_reference_judge_v7_ko",
                "input_modality": "vpick_scene_description_transcript_structurally_blind",
                "repeat_index": 1,
                "dry_run": False,
                "candidate_id": target["source_candidate_id"],
                "verdict": "score",
                "evidence_description_support": description_evidence,
                "evidence_transcript_intelligibility": transcript_evidence,
                "evidence_boundary_observability": boundary_evidence,
                "saliency_market_1_5": saliency,
                "check_hook_within_3s": checks[0],
                "check_surprise_or_twist": checks[1],
                "check_emotional_peak": checks[2],
                "check_quotable_moment": checks[3],
                "check_payoff_or_conclusion": checks[4],
                "check_natural_start": checks[5],
                "check_natural_end": checks[6],
                "checklist_score_100": rounded(checklist),
                "overall_shortform_suitable": suitable,
                "confidence_1_5": 2 if transcript_evidence <= 2 else 4,
                "failure_flags": _failure_flags(checks, transcript_evidence),
                "reason": case2.get("reason", ""),
            }
        )
    return rows


def main() -> int:
    rows = compile_scores()
    write_csv(OUTPUT, rows)
    print(f"Wrote {len(rows)} Codex Vpick-enriched v7 judgments to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
