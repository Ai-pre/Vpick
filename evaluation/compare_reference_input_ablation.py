from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

from .common import ROOT, as_float, read_csv, rounded, spearman, write_csv, write_json


TRANSCRIPT_SCORES = (
    ROOT / "deliverables" / "2026-07-23" / "vpick_llm_judge_v7_codex_scores_60.csv"
)
VPICK_SCORES = (
    ROOT
    / "results"
    / "evaluation_system_v1"
    / "reference_v7_codex_vpick_direct_scores.csv"
)
TARGETS = (
    ROOT / "results" / "evaluation_system_v1" / "prepared" / "targets_private.csv"
)
BEHAVIOR = (
    ROOT / "results" / "evaluation_system_v1" / "behavior_labels_private.csv"
)
OUTPUT_DIR = ROOT / "results" / "evaluation_system_v1" / "reference_input_ablation"


def _score(row: dict[str, str], field: str) -> float | None:
    if row.get("verdict") != "score":
        return None
    value = as_float(row.get(field))
    if value is None:
        return None
    if field == "saliency_market_1_5":
        return (value - 1.0) * 25.0
    return value


def _bootstrap_delta(
    records: list[dict[str, Any]],
    transcript_field: str,
    vpick_field: str,
    *,
    iterations: int = 3000,
) -> dict[str, Any]:
    by_longform: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_longform.setdefault(str(row["longform_id"]), []).append(row)
    groups = sorted(by_longform)
    randomizer = random.Random("reference-v7-input-ablation")
    values: list[float] = []
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for group in randomizer.choices(groups, k=len(groups)):
            sampled.extend(by_longform[group])
        transcript_rho = spearman(
            [float(row[transcript_field]) for row in sampled],
            [float(row["channel_view_percentile"]) for row in sampled],
        )
        vpick_rho = spearman(
            [float(row[vpick_field]) for row in sampled],
            [float(row["channel_view_percentile"]) for row in sampled],
        )
        if transcript_rho is not None and vpick_rho is not None:
            values.append(vpick_rho - transcript_rho)
    values.sort()
    if not values:
        return {"estimate": None, "ci_lower": None, "ci_upper": None, "iterations": 0}
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return {
        "estimate": rounded(
            spearman(
                [float(row[vpick_field]) for row in records],
                [float(row["channel_view_percentile"]) for row in records],
            )
            - spearman(
                [float(row[transcript_field]) for row in records],
                [float(row["channel_view_percentile"]) for row in records],
            )
        ),
        "ci_lower": rounded(lower),
        "ci_upper": rounded(upper),
        "iterations": len(values),
    }


def compare() -> dict[str, Any]:
    transcript_rows = {
        row["candidate_id"]: row for row in read_csv(TRANSCRIPT_SCORES)
    }
    vpick_rows = {row["candidate_id"]: row for row in read_csv(VPICK_SCORES)}
    targets = read_csv(TARGETS)
    behavior_by_current = {
        row["candidate_id"]: row for row in read_csv(BEHAVIOR)
    }
    current_by_source = {
        row["source_candidate_id"]: row["candidate_id"] for row in targets
    }

    records: list[dict[str, Any]] = []
    for source_id in sorted(set(transcript_rows) & set(vpick_rows)):
        current_id = current_by_source.get(source_id)
        behavior = behavior_by_current.get(str(current_id or ""))
        if not behavior:
            continue
        transcript_saliency = _score(
            transcript_rows[source_id], "saliency_market_1_5"
        )
        vpick_saliency = _score(vpick_rows[source_id], "saliency_market_1_5")
        transcript_checklist = _score(
            transcript_rows[source_id], "checklist_score_100"
        )
        vpick_checklist = _score(vpick_rows[source_id], "checklist_score_100")
        if None in (
            transcript_saliency,
            vpick_saliency,
            transcript_checklist,
            vpick_checklist,
        ):
            continue
        records.append(
            {
                "candidate_id": current_id,
                "source_candidate_id": source_id,
                "longform_id": behavior["longform_id"],
                "channel_name": behavior["channel_name"],
                "channel_view_percentile": as_float(
                    behavior["channel_view_percentile"]
                ),
                "relative_log_view_score": as_float(
                    behavior["relative_log_view_score"]
                ),
                "transcript_saliency_100": rounded(transcript_saliency),
                "vpick_saliency_100": rounded(vpick_saliency),
                "saliency_delta_vpick_minus_transcript": rounded(
                    vpick_saliency - transcript_saliency
                ),
                "transcript_checklist_100": rounded(transcript_checklist),
                "vpick_checklist_100": rounded(vpick_checklist),
                "checklist_delta_vpick_minus_transcript": rounded(
                    vpick_checklist - transcript_checklist
                ),
            }
        )

    metrics: list[dict[str, Any]] = []
    for name, transcript_field, vpick_field in (
        ("saliency_market", "transcript_saliency_100", "vpick_saliency_100"),
        ("reference_checklist", "transcript_checklist_100", "vpick_checklist_100"),
    ):
        transcript_values = [float(row[transcript_field]) for row in records]
        vpick_values = [float(row[vpick_field]) for row in records]
        targets_values = [float(row["channel_view_percentile"]) for row in records]
        deltas = [v - t for t, v in zip(transcript_values, vpick_values)]
        metrics.append(
            {
                "score_name": name,
                "paired_n": len(records),
                "transcript_percentile_spearman": rounded(
                    spearman(transcript_values, targets_values)
                ),
                "vpick_percentile_spearman": rounded(
                    spearman(vpick_values, targets_values)
                ),
                "vpick_minus_transcript_spearman": rounded(
                    spearman(vpick_values, targets_values)
                    - spearman(transcript_values, targets_values)
                ),
                "score_agreement_spearman": rounded(
                    spearman(transcript_values, vpick_values)
                ),
                "mean_score_delta_vpick_minus_transcript": rounded(
                    statistics.mean(deltas)
                ),
                "mean_absolute_score_delta": rounded(
                    statistics.mean(abs(value) for value in deltas)
                ),
                "spearman_delta_group_bootstrap": _bootstrap_delta(
                    records, transcript_field, vpick_field
                ),
            }
        )

    transcript_abstains = sum(
        row.get("verdict") != "score" for row in transcript_rows.values()
    )
    vpick_abstains = sum(row.get("verdict") != "score" for row in vpick_rows.values())
    summary = {
        "experiment": "same_codex_prompt_reference_v7_input_ablation",
        "controlled_factors": [
            "same 60 published-short candidates",
            "same shortform_reference_judge_v7_ko rubric",
            "same Codex direct-evaluation interface",
            "performance labels excluded during scoring",
        ],
        "changed_factor": (
            "transcript-only evidence versus Vpick scene description, transcript, "
            "and boundary context"
        ),
        "paired_candidate_count": len(records),
        "transcript_scored_count": len(transcript_rows) - transcript_abstains,
        "transcript_abstain_count": transcript_abstains,
        "vpick_scored_count": len(vpick_rows) - vpick_abstains,
        "vpick_abstain_count": vpick_abstains,
        "metrics": metrics,
        "limitation": (
            "Both are direct Codex judgments, but the provider does not expose a stable model "
            "snapshot ID. The Vpick pass was completed later in the same project context and is "
            "a single pass, so residual run/context effects remain."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "paired_candidate_deltas_PRIVATE.csv", records)
    write_json(OUTPUT_DIR / "ablation_summary.json", summary)
    return summary


def main() -> int:
    summary = compare()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
