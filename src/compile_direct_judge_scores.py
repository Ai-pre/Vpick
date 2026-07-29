from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


CHECK_FIELDS = [
    "check_central_focus_clear",
    "check_highlight_worthy",
    "check_important_or_representative",
    "check_context_sufficient",
    "check_meaningful_progression",
    "check_payoff_or_conclusion",
    "check_natural_start",
    "check_natural_end",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def rounded(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def run_scores(row: dict[str, str]) -> dict[str, Any]:
    if row.get("verdict") != "score":
        return {
            "saliency": None,
            "saliency_100": None,
            "checklist_100": None,
            "reference_100": None,
            "checks": {field: None for field in CHECK_FIELDS},
        }
    saliency = as_float(row.get("highlight_saliency_1_5"))
    checks = {field: as_float(row.get(field)) for field in CHECK_FIELDS}
    check_values = list(checks.values())
    if saliency is None or any(value is None for value in check_values):
        raise ValueError(f"Incomplete score row: {row.get('candidate_id')}")
    saliency_100 = (saliency - 1.0) * 25.0
    checklist_100 = sum(check_values) / len(check_values) * 100.0
    return {
        "saliency": saliency,
        "saliency_100": saliency_100,
        "checklist_100": checklist_100,
        "reference_100": (saliency_100 + checklist_100) / 2.0,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile two direct LLM Judge runs into one row per gold pair.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--run1", required=True)
    parser.add_argument("--run2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = read_csv(Path(args.dataset))
    candidates = {row["candidate_id"]: row for row in read_csv(Path(args.candidates))}
    sources = {row["candidate_id"]: row for row in read_csv(Path(args.sources))}
    run1 = {row["candidate_id"]: row for row in read_csv(Path(args.run1))}
    run2 = {row["candidate_id"]: row for row in read_csv(Path(args.run2))}
    expected = set(candidates)
    for name, rows in (("sources", sources), ("run1", run1), ("run2", run2)):
        if set(rows) != expected:
            raise ValueError(f"{name} candidate IDs do not match blind input")

    dataset_by_pair = {row["pair_id"]: row for row in dataset}
    output_rows: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates.items():
        source = sources[candidate_id]
        pair = dataset_by_pair[source["pair_id"]]
        first = run1[candidate_id]
        second = run2[candidate_id]
        score1 = run_scores(first)
        score2 = run_scores(second)
        reference_mean = mean([score1["reference_100"], score2["reference_100"]])
        suitable1 = as_float(first.get("overall_shortform_suitable"))
        suitable2 = as_float(second.get("overall_shortform_suitable"))
        row: dict[str, Any] = {
            "pair_id": pair["pair_id"],
            "performance_label": pair["performance_label"],
            "channel_name": pair["channel_name"],
            "channel_name_raw": pair.get("channel_name_raw", ""),
            "candidate_id": candidate_id,
            "long_video_id": pair["long_video_id"],
            "long_video_url": pair["long_video_url"],
            "short_video_id": pair["short_video_id"],
            "short_video_url": pair["short_video_url"],
            "start_sec": pair["start_sec"],
            "end_sec": pair["end_sec"],
            "start_time": pair["start_time"],
            "end_time": pair["end_time"],
            "duration_sec": pair["duration_sec"],
            "short_views": pair["short_views"],
            "short_likes": pair["short_likes"],
            "short_like_rate": pair["short_like_rate"],
            "channel_performance_percentile": pair["channel_performance_percentile"],
            "label_confidence": pair["label_confidence"],
            "mapping_confidence": pair["mapping_confidence"],
            "alignment_status": pair["alignment_status"],
            "evidence_source": candidate["evidence_source"],
            "evidence_provider": candidate["evidence_provider"],
            "transcript_language": candidate["language"],
            "verdict_run1": first["verdict"],
            "verdict_run2": second["verdict"],
            "highlight_saliency_run1_1_5": rounded(score1["saliency"]),
            "highlight_saliency_run2_1_5": rounded(score2["saliency"]),
            "highlight_saliency_mean_1_5": rounded(mean([score1["saliency"], score2["saliency"]])),
            "checklist_score_run1_100": rounded(score1["checklist_100"]),
            "checklist_score_run2_100": rounded(score2["checklist_100"]),
            "checklist_score_mean_100": rounded(mean([score1["checklist_100"], score2["checklist_100"]])),
            "reference_score_run1_100": rounded(score1["reference_100"]),
            "reference_score_run2_100": rounded(score2["reference_100"]),
            "reference_score_mean_100": rounded(reference_mean),
            "repeat_abs_diff_100": rounded(
                abs(score1["reference_100"] - score2["reference_100"])
                if score1["reference_100"] is not None and score2["reference_100"] is not None
                else None
            ),
            "overall_shortform_suitable_run1": first.get("overall_shortform_suitable", ""),
            "overall_shortform_suitable_run2": second.get("overall_shortform_suitable", ""),
            "overall_shortform_suitable_vote_rate": rounded(mean([suitable1, suitable2])),
            "confidence_run1_1_5": first.get("confidence", ""),
            "confidence_run2_1_5": second.get("confidence", ""),
            "confidence_mean_1_5": rounded(mean([as_float(first.get("confidence")), as_float(second.get("confidence"))])),
            "failure_flags_run1": first.get("failure_flags", ""),
            "failure_flags_run2": second.get("failure_flags", ""),
            "reason_run1": first.get("reason", ""),
            "reason_run2": second.get("reason", ""),
            "score_formula": "0.5*((saliency-1)*25)+0.5*(true_check_count/8*100)",
            "judge_model": "codex_current_gpt_direct",
            "judge_prompt_version": "shortform_reference_judge_v6_ko",
        }
        for field in CHECK_FIELDS:
            output_name = field.removeprefix("check_") + "_mean_0_1"
            row[output_name] = rounded(mean([score1["checks"][field], score2["checks"][field]]))
        output_rows.append(row)

    label_order = {"pos": 0, "neg": 1}
    output_rows.sort(
        key=lambda row: (
            label_order.get(str(row["performance_label"]), 9),
            str(row["channel_name"]),
            str(row["pair_id"]),
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
