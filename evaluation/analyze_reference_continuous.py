from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import (
    ROOT,
    as_float,
    bootstrap_group_metric,
    read_csv,
    rounded,
    spearman,
    write_csv,
    write_json,
)


DEFAULT_SCORES = (
    ROOT / "deliverables" / "2026-07-23" / "vpick_llm_judge_v7_codex_scores_60.csv"
)
DEFAULT_TARGETS = (
    ROOT / "results" / "evaluation_system_v1" / "prepared" / "targets_private.csv"
)
DEFAULT_BEHAVIOR = (
    ROOT / "results" / "evaluation_system_v1" / "behavior_labels_private.csv"
)
DEFAULT_OUTPUT = ROOT / "results" / "evaluation_system_v1"


def _channel_metrics(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["channel_name"])].append(row)

    details: list[dict[str, Any]] = []
    centered_scores: list[float] = []
    centered_targets: list[float] = []
    for channel, group in sorted(grouped.items()):
        scores = [float(row[score_field]) for row in group]
        targets = [float(row["channel_view_percentile"]) for row in group]
        value = spearman(scores, targets)
        details.append(
            {
                "score_name": score_field,
                "channel_name": channel,
                "n": len(group),
                "percentile_spearman": rounded(value),
            }
        )
        score_mean = statistics.mean(scores)
        target_mean = statistics.mean(targets)
        centered_scores.extend(score - score_mean for score in scores)
        centered_targets.extend(target - target_mean for target in targets)

    valid = [
        float(row["percentile_spearman"])
        for row in details
        if row["percentile_spearman"] is not None
    ]
    return {
        "within_channel_centered_spearman": rounded(
            spearman(centered_scores, centered_targets)
        ),
        "channel_macro_spearman": rounded(statistics.mean(valid) if valid else None),
        "details": details,
    }


def _same_longform(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["longform_id"])].append(row)
    eligible = 0
    correct = 0
    rank_correlations: list[float] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        eligible += 1
        scores = [float(row[score_field]) for row in group]
        targets = [float(row["channel_view_percentile"]) for row in group]
        best_score = max(scores)
        selected_targets = [
            target for score, target in zip(scores, targets) if score == best_score
        ]
        if max(selected_targets) == max(targets):
            correct += 1
        value = spearman(scores, targets)
        if value is not None:
            rank_correlations.append(value)
    return {
        "eligible_longform_count": eligible,
        "top1_accuracy": rounded(correct / eligible if eligible else None),
        "macro_rank_spearman": rounded(
            statistics.mean(rank_correlations) if rank_correlations else None
        ),
    }


def analyze(
    scores_path: Path,
    targets_path: Path,
    behavior_path: Path,
    output_dir: Path,
    output_stem: str,
) -> dict[str, Any]:
    score_rows = read_csv(scores_path)
    targets = read_csv(targets_path)
    behavior = read_csv(behavior_path)
    current_id_by_source = {
        row["source_candidate_id"]: row["candidate_id"] for row in targets
    }
    behavior_by_id = {row["candidate_id"]: row for row in behavior}

    records: list[dict[str, Any]] = []
    for row in score_rows:
        if row.get("verdict") != "score":
            continue
        current_id = current_id_by_source.get(str(row.get("candidate_id", "")))
        target = behavior_by_id.get(str(current_id or ""))
        saliency = as_float(row.get("saliency_market_1_5"))
        checklist = as_float(row.get("checklist_score_100"))
        relative = as_float(target.get("relative_log_view_score")) if target else None
        percentile = as_float(target.get("channel_view_percentile")) if target else None
        if None in (saliency, checklist, relative, percentile):
            continue
        records.append(
            {
                "candidate_id": current_id,
                "source_candidate_id": row["candidate_id"],
                "longform_id": target["longform_id"],
                "channel_name": target["channel_name"],
                "saliency_market_100": rounded((float(saliency) - 1.0) * 25.0),
                "reference_checklist_100": rounded(checklist),
                "channel_view_percentile": rounded(percentile),
                "relative_log_view_score": rounded(relative),
            }
        )

    score_summaries: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    for score_field in ("saliency_market_100", "reference_checklist_100"):
        channel = _channel_metrics(records, score_field)
        percentile_ci = bootstrap_group_metric(
            records,
            lambda sample: spearman(
                [float(row[score_field]) for row in sample],
                [float(row["channel_view_percentile"]) for row in sample],
            ),
            iterations=1000,
        )
        relative_ci = bootstrap_group_metric(
            records,
            lambda sample: spearman(
                [float(row[score_field]) for row in sample],
                [float(row["relative_log_view_score"]) for row in sample],
            ),
            iterations=1000,
        )
        score_summaries.append(
            {
                "score_name": score_field,
                "n": len(records),
                "percentile_spearman": rounded(
                    spearman(
                        [float(row[score_field]) for row in records],
                        [float(row["channel_view_percentile"]) for row in records],
                    )
                ),
                "relative_log_spearman": rounded(
                    spearman(
                        [float(row[score_field]) for row in records],
                        [float(row["relative_log_view_score"]) for row in records],
                    )
                ),
                "within_channel_centered_spearman": channel[
                    "within_channel_centered_spearman"
                ],
                "channel_macro_spearman": channel["channel_macro_spearman"],
                "percentile_spearman_group_bootstrap": {
                    key: rounded(value) if isinstance(value, float) else value
                    for key, value in percentile_ci.items()
                },
                "relative_log_spearman_group_bootstrap": {
                    key: rounded(value) if isinstance(value, float) else value
                    for key, value in relative_ci.items()
                },
                "same_longform_ranking": _same_longform(records, score_field),
            }
        )
        channel_rows.extend(channel["details"])

    summary = {
        "experiment": "reference_rubric_v7_without_pos_neg",
        "judge_input_performance_labels": "excluded",
        "validation_target": (
            "continuous channel-relative performance only; no pos/neg, no threshold, no AUC"
        ),
        "source_score_file": str(scores_path),
        "provider": score_rows[0].get("provider") if score_rows else None,
        "model": score_rows[0].get("model") if score_rows else None,
        "source_prompt_id": "shortform_reference_judge_v7_ko",
        "current_dataset_count": len(targets),
        "evaluated_candidate_count": len(records),
        "abstain_or_unmapped_count": len(targets) - len(records),
        "score_results": score_summaries,
        "interpretation_rule": (
            "A useful performance Judge should show a positive correlation whose grouped "
            "bootstrap interval does not broadly cross zero, plus consistent channel-level signs."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"{output_stem}_scores.csv", records)
    write_csv(output_dir / f"{output_stem}_channel_metrics.csv", channel_rows)
    write_json(output_dir / f"{output_stem}_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the blind v7 reference rubric without pos/neg labels."
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--behavior", type=Path, default=DEFAULT_BEHAVIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-stem", default="reference_v7_continuous")
    args = parser.parse_args()
    summary = analyze(
        args.scores,
        args.targets,
        args.behavior,
        args.output_dir,
        args.output_stem,
    )
    print(
        f"Evaluated {summary['evaluated_candidate_count']}/"
        f"{summary['current_dataset_count']} candidates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
