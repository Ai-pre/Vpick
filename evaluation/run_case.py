from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .build_behavior_labels import build as build_behavior_labels
from .common import (
    ROOT,
    as_float,
    average_precision,
    bootstrap_group_metric,
    load_config,
    macro_f1,
    read_csv,
    resolve_path,
    roc_auc,
    rounded,
    spearman,
    write_csv,
    write_json,
)
from .prepare_data import prepare


DEFAULT_CONFIG = ROOT / "configs" / "evaluation.yaml"
CASES = {
    "channel_baseline",
    "standalone_pointwise",
    "source_pointwise",
    "source_pairwise",
    "hybrid",
}


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _aggregate_repeats(
    rows: list[dict[str, str]],
    *,
    candidate_field: str,
    score_fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(candidate_field, ""))].append(row)
    output: list[dict[str, Any]] = []
    for candidate_id, group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "candidate_id": candidate_id,
            "repeat_count": len(group),
        }
        for field in score_fields:
            values = [
                value
                for row in group
                if (value := as_float(row.get(field))) is not None
            ]
            item[f"{field}_mean"] = rounded(_mean(values))
            item[f"{field}_std"] = rounded(
                statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
            )
        output.append(item)
    return output


def _repeat_reliability(
    rows: list[dict[str, str]],
    *,
    candidate_field: str,
    repeat_field: str,
    score_field: str,
) -> dict[str, Any]:
    repeats: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        score = as_float(row.get(score_field))
        repeat = as_float(row.get(repeat_field))
        if score is not None and repeat is not None:
            repeats[int(repeat)][str(row.get(candidate_field, ""))] = score
    correlations: list[float] = []
    absolute_errors: list[float] = []
    common_counts: list[int] = []
    for left, right in itertools.combinations(sorted(repeats), 2):
        common = sorted(set(repeats[left]) & set(repeats[right]))
        common_counts.append(len(common))
        left_values = [repeats[left][candidate] for candidate in common]
        right_values = [repeats[right][candidate] for candidate in common]
        value = spearman(left_values, right_values)
        if value is not None:
            correlations.append(value)
        absolute_errors.extend(abs(a - b) for a, b in zip(left_values, right_values))
    return {
        "repeat_count": len(repeats),
        "repeat_common_candidate_count": min(common_counts) if common_counts else 0,
        "repeat_spearman": rounded(_mean(correlations)),
        "repeat_mae": rounded(_mean(absolute_errors)),
    }


def _best_training_threshold(records: list[dict[str, Any]]) -> float | None:
    extremes = [row for row in records if row["performance_tier"] in {"top25", "bottom25"}]
    if not extremes:
        return None
    unique = sorted({float(row["score"]) for row in extremes})
    thresholds = [unique[0] - 1e-9, *[(a + b) / 2 for a, b in zip(unique, unique[1:])], unique[-1] + 1e-9]
    best: tuple[float, float] | None = None
    labels = [int(row["performance_tier"] == "top25") for row in extremes]
    for threshold in thresholds:
        predictions = [int(float(row["score"]) >= threshold) for row in extremes]
        value = macro_f1(labels, predictions)
        if value is None:
            continue
        candidate = (value, -threshold)
        if best is None or candidate > best:
            best = candidate
    return None if best is None else -best[1]


def _ndcg_top1(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["longform_id"])].append(row)
    ndcg_values: list[float] = []
    top1_values: list[float] = []
    eligible = 0
    for group in grouped.values():
        if len(group) < 2:
            continue
        eligible += 1
        ranked = sorted(group, key=lambda row: float(row["score"]), reverse=True)
        ideal = sorted(group, key=lambda row: float(row["channel_view_percentile"]), reverse=True)
        dcg = sum(
            (2 ** (float(row["channel_view_percentile"]) / 100.0) - 1)
            / math.log2(index + 2)
            for index, row in enumerate(ranked)
        )
        idcg = sum(
            (2 ** (float(row["channel_view_percentile"]) / 100.0) - 1)
            / math.log2(index + 2)
            for index, row in enumerate(ideal)
        )
        if idcg > 0:
            ndcg_values.append(dcg / idcg)
        best_performance = max(float(row["channel_view_percentile"]) for row in group)
        selected_performance = float(ranked[0]["channel_view_percentile"])
        top1_values.append(float(selected_performance == best_performance))
    return {
        "eligible_longform_count": eligible,
        "top1_accuracy": rounded(_mean(top1_values)),
        "ndcg": rounded(_mean(ndcg_values)),
    }


def evaluate_pointwise(
    score_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, str]],
    *,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    behavior_by_candidate = {row["candidate_id"]: row for row in behavior_rows}
    records: list[dict[str, Any]] = []
    for row in score_rows:
        score = as_float(row.get("score"))
        behavior = behavior_by_candidate.get(str(row.get("candidate_id", "")))
        if not behavior or score is None:
            continue
        relative = as_float(behavior.get("relative_log_view_score"))
        channel_percentile = as_float(behavior.get("channel_view_percentile"))
        if relative is None or channel_percentile is None:
            continue
        records.append(
            {
                **row,
                "score": score,
                "longform_id": behavior["longform_id"],
                "channel_name": behavior["channel_name"],
                "relative_log_view_score": relative,
                "channel_view_percentile": channel_percentile,
                "performance_tier": behavior["performance_tier"],
                "dataset_split": behavior["dataset_split"],
            }
        )

    scores = [float(row["score"]) for row in records]
    relative = [float(row["relative_log_view_score"]) for row in records]
    percentiles = [float(row["channel_view_percentile"]) for row in records]
    top_labels = [int(row["performance_tier"] == "top25") for row in records]

    extremes = [row for row in records if row["performance_tier"] in {"top25", "bottom25"}]
    threshold = statistics.median([float(row["score"]) for row in extremes]) if extremes else None
    extreme_labels = [int(row["performance_tier"] == "top25") for row in extremes]
    extreme_predictions = (
        [int(float(row["score"]) >= float(threshold)) for row in extremes]
        if threshold is not None
        else []
    )

    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_channel[row["channel_name"]].append(row)
    channel_metrics: list[dict[str, Any]] = []
    centered_scores: list[float] = []
    centered_performance: list[float] = []
    for channel, group in sorted(by_channel.items()):
        group_scores = [float(row["score"]) for row in group]
        group_performance = [float(row["channel_view_percentile"]) for row in group]
        channel_metrics.append(
            {
                "channel_name": channel,
                "n": len(group),
                "spearman": rounded(spearman(group_scores, group_performance)),
                "top25_auc": rounded(
                    roc_auc(
                        [int(row["performance_tier"] == "top25") for row in group],
                        group_scores,
                    )
                ),
            }
        )
        score_mean = statistics.mean(group_scores)
        performance_mean = statistics.mean(group_performance)
        centered_scores.extend(value - score_mean for value in group_scores)
        centered_performance.extend(value - performance_mean for value in group_performance)

    holdout_rows: list[dict[str, Any]] = []
    for held_channel in sorted(by_channel):
        train = [
            row
            for row in records
            if row["channel_name"] != held_channel
            and row["performance_tier"] in {"top25", "bottom25"}
        ]
        test = [
            row
            for row in records
            if row["channel_name"] == held_channel
            and row["performance_tier"] in {"top25", "bottom25"}
        ]
        learned_threshold = _best_training_threshold(train)
        if learned_threshold is None or not test:
            continue
        labels = [int(row["performance_tier"] == "top25") for row in test]
        predictions = [int(float(row["score"]) >= learned_threshold) for row in test]
        holdout_rows.append(
            {
                "channel_name": held_channel,
                "test_n": len(test),
                "training_threshold": rounded(learned_threshold),
                "macro_f1": rounded(macro_f1(labels, predictions)),
                "auc": rounded(roc_auc(labels, [float(row["score"]) for row in test])),
            }
        )

    split_metrics: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in records if row["dataset_split"] == split]
        split_metrics[split] = {
            "n": len(split_rows),
            "relative_log_spearman": rounded(
                spearman(
                    [float(row["score"]) for row in split_rows],
                    [float(row["relative_log_view_score"]) for row in split_rows],
                )
            ),
            "percentile_spearman": rounded(
                spearman(
                    [float(row["score"]) for row in split_rows],
                    [float(row["channel_view_percentile"]) for row in split_rows],
                )
            ),
        }

    relative_ci = bootstrap_group_metric(
        records,
        lambda sample: spearman(
            [float(row["score"]) for row in sample],
            [float(row["relative_log_view_score"]) for row in sample],
        ),
        iterations=bootstrap_iterations,
    )
    auc_ci = bootstrap_group_metric(
        records,
        lambda sample: roc_auc(
            [int(row["performance_tier"] == "top25") for row in sample],
            [float(row["score"]) for row in sample],
        ),
        iterations=bootstrap_iterations,
    )
    return {
        "evaluated_candidate_count": len(records),
        "score_vs_relative_log_spearman": rounded(spearman(scores, relative)),
        "score_vs_channel_percentile_spearman": rounded(spearman(scores, percentiles)),
        "top25_roc_auc": rounded(roc_auc(top_labels, scores)),
        "top25_pr_auc": rounded(average_precision(top_labels, scores)),
        "extreme_top_bottom_count": len(extremes),
        "extreme_macro_f1_at_score_median": rounded(
            macro_f1(extreme_labels, extreme_predictions)
        ),
        "within_channel_centered_spearman": rounded(
            spearman(centered_scores, centered_performance)
        ),
        "channel_macro_spearman": rounded(
            _mean(
                [
                    float(row["spearman"])
                    for row in channel_metrics
                    if row["spearman"] is not None
                ]
            )
        ),
        "channel_holdout_macro_f1": rounded(
            _mean(
                [
                    float(row["macro_f1"])
                    for row in holdout_rows
                    if row["macro_f1"] is not None
                ]
            )
        ),
        "relative_log_spearman_group_bootstrap": {
            key: rounded(value) if isinstance(value, float) else value
            for key, value in relative_ci.items()
        },
        "top25_auc_group_bootstrap": {
            key: rounded(value) if isinstance(value, float) else value
            for key, value in auc_ci.items()
        },
        "group_split_metrics": split_metrics,
        "same_longform_ranking": _ndcg_top1(records),
        "channel_metrics": channel_metrics,
        "channel_holdout_metrics": holdout_rows,
    }


def _run_standalone_historical(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = resolve_path(config["output_dir"])
    historical_path = resolve_path(config["historical_results"]["standalone_pointwise"])
    raw = read_csv(historical_path)
    historical_sources = read_csv(
        resolve_path(config["historical_results"]["standalone_pointwise_sources"])
    )
    current_targets = read_csv(output_dir / "prepared" / "targets_private.csv")
    behavior = read_csv(output_dir / "behavior_labels_private.csv")
    current_by_short = {
        row["short_video_id"]: row["candidate_id"]
        for row in current_targets
        if row.get("short_video_id")
    }
    historical_short_by_candidate = {
        row["candidate_id"]: row["short_video_id"]
        for row in historical_sources
        if row.get("candidate_id") and row.get("short_video_id")
    }
    alias = {
        historical_candidate: current_by_short[short_id]
        for historical_candidate, short_id in historical_short_by_candidate.items()
        if short_id in current_by_short
    }
    historical_ids = {row.get("candidate_id", "") for row in raw}
    direct_current = {row["candidate_id"] for row in current_targets}

    mapped_raw: list[dict[str, Any]] = []
    for row in raw:
        old_id = row.get("candidate_id", "")
        current_id = alias.get(old_id) or (old_id if old_id in direct_current else None)
        editorial = as_float(row.get("editorial_score"))
        performance = as_float(row.get("performance_score"))
        if not current_id or editorial is None or performance is None:
            continue
        mapped_raw.append(
            {
                **row,
                "historical_candidate_id": old_id,
                "candidate_id": current_id,
                "standalone_score": (editorial + performance) / 2.0,
            }
        )

    aggregates = _aggregate_repeats(
        mapped_raw,
        candidate_field="candidate_id",
        score_fields=("standalone_score",),
    )
    score_rows = [
        {
            "candidate_id": row["candidate_id"],
            "score": row["standalone_score_mean"],
            "repeat_count": row["repeat_count"],
            "score_std": row["standalone_score_std"],
        }
        for row in aggregates
        if row["standalone_score_mean"] is not None
    ]
    metrics = evaluate_pointwise(
        score_rows,
        behavior,
        bootstrap_iterations=int(config["performance"]["bootstrap_iterations"]),
    )
    reliability = _repeat_reliability(
        mapped_raw,
        candidate_field="candidate_id",
        repeat_field="repeat_index",
        score_field="standalone_score",
    )
    write_csv(output_dir / "case_2_standalone_scores.csv", score_rows)
    summary = {
        "case": "standalone_pointwise",
        "status": "historical_actual_proxy",
        "input_note": (
            "Actual GPT-5.6 Terra v4 outputs were mapped to the current 60-candidate IDs. "
            "The historical v4 rubric is close to, but not identical with, the new v1 rubric."
        ),
        "historical_candidate_count": len(historical_ids),
        "current_overlap_count": len(score_rows),
        "current_dataset_count": len(current_targets),
        "human_alignment": "N/A: no completed human labels for the current rubric",
        "repeat_reliability": reliability,
        "metrics": metrics,
    }
    write_json(output_dir / "case_2_summary.json", summary)
    return summary


def run_standalone(config: dict[str, Any]) -> dict[str, Any]:
    historical = _run_standalone_historical(config)
    direct_result = config.get("direct_results", {}).get("standalone_pointwise")
    if not direct_result:
        return historical
    direct_path = resolve_path(direct_result)
    if not direct_path.exists():
        return historical

    output_dir = resolve_path(config["output_dir"])
    behavior = read_csv(output_dir / "behavior_labels_private.csv")
    current_targets = read_csv(output_dir / "prepared" / "targets_private.csv")
    raw = read_csv(direct_path)
    dimensions = (
        "hook_0_4",
        "engagement_0_4",
        "self_contained_0_4",
        "payoff_0_4",
        "density_0_4",
        "boundary_0_4",
    )
    scored_raw: list[dict[str, Any]] = []
    for row in raw:
        values = [as_float(row.get(field)) for field in dimensions]
        if row.get("verdict") != "score" or any(value is None for value in values):
            continue
        score = sum(float(value) for value in values if value is not None) * 100.0 / 24.0
        scored_raw.append({**row, "standalone_score": score})

    aggregates = _aggregate_repeats(
        scored_raw,
        candidate_field="candidate_id",
        score_fields=("standalone_score", *dimensions),
    )
    score_rows = [
        {
            "candidate_id": row["candidate_id"],
            "score": row["standalone_score_mean"],
            "repeat_count": row["repeat_count"],
            "score_std": row["standalone_score_std"],
            **{
                field: row[f"{field}_mean"]
                for field in dimensions
            },
        }
        for row in aggregates
        if row["standalone_score_mean"] is not None
    ]
    metrics = evaluate_pointwise(
        score_rows,
        behavior,
        bootstrap_iterations=int(config["performance"]["bootstrap_iterations"]),
    )
    reliability = _repeat_reliability(
        scored_raw,
        candidate_field="candidate_id",
        repeat_field="repeat_index",
        score_field="standalone_score",
    )
    write_csv(output_dir / "case_2_standalone_scores.csv", score_rows)
    summary = {
        "case": "standalone_pointwise",
        "status": "actual_codex_direct_single_pass",
        "input_note": (
            "All current candidates were judged blind from description, transcript, and duration "
            "with the standalone v1 rubric. Channel, views, percentile, and performance labels "
            "were hidden until scoring was fixed."
        ),
        "provider": raw[0].get("provider") if raw else None,
        "model": raw[0].get("model") if raw else None,
        "prompt_id": raw[0].get("prompt_id") if raw else None,
        "current_dataset_count": len(current_targets),
        "current_overlap_count": len(score_rows),
        "scored_candidate_count": len(score_rows),
        "abstain_count": sum(row.get("verdict") == "abstain" for row in raw),
        "human_alignment": "N/A: no completed human labels for the current rubric",
        "repeat_reliability": reliability,
        "metrics": metrics,
        "historical_proxy_comparison": {
            "coverage_n": historical["metrics"]["evaluated_candidate_count"],
            "percentile_spearman": historical["metrics"][
                "score_vs_channel_percentile_spearman"
            ],
            "top25_roc_auc": historical["metrics"]["top25_roc_auc"],
            "repeat_spearman": historical["repeat_reliability"]["repeat_spearman"],
        },
        "limitation": (
            "This is one direct session pass, not an independent API repeat. It tests criterion "
            "validity against hidden performance labels, but does not establish repeat reliability."
        ),
    }
    write_json(output_dir / "case_2_summary.json", summary)
    return summary


def run_source_pointwise(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = resolve_path(config["output_dir"])
    raw = read_csv(resolve_path(config["historical_results"]["source_pointwise"]))
    behavior = read_csv(output_dir / "behavior_labels_private.csv")
    scored = [
        row
        for row in raw
        if row.get("verdict") == "score"
        and as_float(row.get("judge_score_100")) is not None
    ]
    aggregates = _aggregate_repeats(
        scored,
        candidate_field="candidate_id",
        score_fields=("judge_score_100", "editorial_score_100", "engagement_score_100"),
    )
    score_rows = [
        {
            "candidate_id": row["candidate_id"],
            "score": row["judge_score_100_mean"],
            "source_selection_score": row["editorial_score_100_mean"],
            "standalone_engagement_score": row["engagement_score_100_mean"],
            "repeat_count": row["repeat_count"],
            "score_std": row["judge_score_100_std"],
        }
        for row in aggregates
        if row["judge_score_100_mean"] is not None
    ]
    metrics = evaluate_pointwise(
        score_rows,
        behavior,
        bootstrap_iterations=int(config["performance"]["bootstrap_iterations"]),
    )
    reliability = _repeat_reliability(
        scored,
        candidate_field="candidate_id",
        repeat_field="repeat_index",
        score_field="judge_score_100",
    )
    current_count = len({row["candidate_id"] for row in behavior})
    write_csv(output_dir / "case_3_source_scores.csv", score_rows)
    summary = {
        "case": "source_pointwise",
        "status": "actual_incomplete_repeat_validation",
        "input_note": (
            "Claude Opus 4.8 v9 actual outputs: first pass 60/60, second pass 11/60. "
            "Three candidates abstained because transcript/context evidence was insufficient."
        ),
        "current_dataset_count": current_count,
        "scored_candidate_count": len(score_rows),
        "human_alignment": "N/A: current two-annotator template is not filled",
        "repeat_reliability": reliability,
        "metrics": metrics,
    }
    write_json(output_dir / "case_3_summary.json", summary)
    return summary


def run_source_pairwise(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = resolve_path(config["output_dir"])
    source_summary_path = resolve_path(config["historical_results"]["source_pairwise_summary"])
    with source_summary_path.open("r", encoding="utf-8") as handle:
        historical = json.load(handle)
    request_path = output_dir / "requests" / "source_pairwise_requests.jsonl"
    current_request_count = 0
    if request_path.exists():
        with request_path.open("r", encoding="utf-8") as handle:
            current_request_count = sum(1 for line in handle if line.strip())
    pairwise = historical.get("pairwise_overall", {})
    summary = {
        "case": "source_pairwise",
        "status": "actual_synthetic_alternative_validation",
        "historical_model": historical.get("model"),
        "historical_pair_count": pairwise.get("valid_count"),
        "historical_order_consistency_rate": pairwise.get("order_consistency_rate"),
        "historical_order_consistent_pair_count": pairwise.get("consistent_count"),
        "historical_order_abstain_count": pairwise.get("order_abstain_count"),
        "current_published_pair_request_count": current_request_count // 2,
        "current_published_pair_judgment_status": "N/A: blind requests built, no model calls made",
        "human_pairwise_accuracy": "N/A: current human response sheet is blank",
        "performance_pairwise_accuracy": (
            "N/A: historical alternatives are boundary shifts/random/hard negatives, "
            "not two independently published shorts with comparable performance snapshots"
        ),
        "bradley_terry": "N/A: current published-published comparison graph has no judgments",
        "top1_ndcg": "N/A: current published-published comparison graph has no judgments",
        "historical_breakdown": historical.get("pairwise_published_vs_alternative", []),
    }
    write_json(output_dir / "case_4_summary.json", summary)
    return summary


def run_hybrid(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = resolve_path(config["output_dir"])
    case_2_path = output_dir / "case_2_standalone_scores.csv"
    case_3_path = output_dir / "case_3_source_scores.csv"
    if not case_2_path.exists():
        run_standalone(config)
    if not case_3_path.exists():
        run_source_pointwise(config)
    standalone = {row["candidate_id"]: row for row in read_csv(case_2_path)}
    source = {row["candidate_id"]: row for row in read_csv(case_3_path)}
    behavior = read_csv(output_dir / "behavior_labels_private.csv")
    common = sorted(set(standalone) & set(source))
    alpha_summaries: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    for alpha_value in config["hybrid"]["alphas"]:
        alpha = float(alpha_value)
        rows: list[dict[str, Any]] = []
        for candidate_id in common:
            standalone_score = as_float(standalone[candidate_id].get("score"))
            source_score = as_float(source[candidate_id].get("source_selection_score"))
            if standalone_score is None or source_score is None:
                continue
            score = alpha * standalone_score + (1.0 - alpha) * source_score
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "alpha": alpha,
                    "standalone_score": rounded(standalone_score),
                    "source_selection_score": rounded(source_score),
                    "score": rounded(score),
                }
            )
        metrics = evaluate_pointwise(
            rows,
            behavior,
            bootstrap_iterations=int(config["performance"]["bootstrap_iterations"]),
        )
        alpha_summaries.append({"alpha": alpha, "metrics": metrics})
        all_scores.extend(rows)

    # Select only from the predeclared alphas and only on validation percentile correlation.
    usable = [
        item
        for item in alpha_summaries
        if item["metrics"]["group_split_metrics"]["validation"]["percentile_spearman"]
        is not None
    ]
    selected = (
        max(
            usable,
            key=lambda item: (
                item["metrics"]["group_split_metrics"]["validation"]["percentile_spearman"],
                -abs(float(item["alpha"]) - 0.5),
            ),
        )
        if usable
        else None
    )
    write_csv(output_dir / "case_5_hybrid_scores.csv", all_scores)
    summary = {
        "case": "hybrid",
        "status": "historical_proxy_overlap_only",
        "overlap_candidate_count": len(common),
        "alpha_results": alpha_summaries,
        "selection_rule": (
            "Choose among alpha=0.3/0.5/0.7 using validation-split percentile Spearman only; "
            "do not alter alpha after inspecting test metrics."
        ),
        "selected_alpha": None if selected is None else selected["alpha"],
        "selected_alpha_validation_percentile_spearman": (
            None
            if selected is None
            else selected["metrics"]["group_split_metrics"]["validation"]["percentile_spearman"]
        ),
        "selected_alpha_test_metrics": (
            None
            if selected is None
            else selected["metrics"]["group_split_metrics"]["test"]
        ),
        "human_alignment": "N/A: no completed current-rubric human labels",
        "limitation": (
            "Standalone input is a historical v4 proxy, so this is an exploratory hybrid analysis "
            "rather than final validation."
        ),
    }
    write_json(output_dir / "case_5_summary.json", summary)
    return summary


def ensure_common_inputs(config: dict[str, Any]) -> None:
    output_dir = resolve_path(config["output_dir"])
    if not (output_dir / "prepared" / "prepare_summary.json").exists():
        prepare(config)
    if not (output_dir / "behavior_labels_private.csv").exists():
        build_behavior_labels(config)


def run_case(case: str, config: dict[str, Any]) -> dict[str, Any]:
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")
    ensure_common_inputs(config)
    if case == "channel_baseline":
        return build_behavior_labels(config)
    if case == "standalone_pointwise":
        return run_standalone(config)
    if case == "source_pointwise":
        return run_source_pointwise(config)
    if case == "source_pairwise":
        return run_source_pairwise(config)
    return run_hybrid(config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or import one evaluation-system case.")
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run_case(args.case, load_config(args.config))
    print(f"{args.case}: {summary.get('status', 'completed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
