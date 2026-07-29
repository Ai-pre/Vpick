from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import (
    ROOT,
    as_float,
    load_config,
    percentile,
    read_csv,
    relative_log_view_score,
    resolve_path,
    rounded,
    spearman,
    write_csv,
    write_json,
)


DEFAULT_CONFIG = ROOT / "configs" / "evaluation.yaml"


def _normalize_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in read_csv(path):
        raw = row.get("channel_name_raw", "").strip()
        normalized = row.get("channel_name", "").strip()
        if raw and normalized:
            mapping[raw] = normalized
        if normalized:
            mapping[normalized] = normalized
    return mapping


def _bootstrap_percentile_interval(
    values: list[float],
    target: float,
    *,
    iterations: int,
    seed: str,
) -> tuple[float | None, float | None]:
    if len(values) < 2 or iterations <= 0:
        return None, None
    rng = random.Random(seed)
    samples = [
        percentile(rng.choices(values, k=len(values)), target)
        for _ in range(iterations)
    ]
    usable = sorted(value for value in samples if value is not None)
    if not usable:
        return None, None
    return (
        usable[int(0.025 * (len(usable) - 1))],
        usable[int(0.975 * (len(usable) - 1))],
    )


def build(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = resolve_path(config["output_dir"])
    prepared_targets = read_csv(output_dir / "prepared" / "targets_private.csv")
    channel_map = _normalize_map(resolve_path(config["dataset"]["channel_name_map"]))
    cohort_rows = read_csv(resolve_path(config["dataset"]["channel_cohorts"]))

    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cohort_rows:
        channel = channel_map.get(row.get("channel_name", ""), row.get("channel_name", ""))
        views = as_float(row.get("view_count"))
        if channel and views is not None:
            cohorts[channel].append(
                {
                    "video_id": row.get("video_id", ""),
                    "views": views,
                    "stats_as_of": row.get("stats_as_of", ""),
                }
            )

    iterations = int(config["performance"]["bootstrap_iterations"])
    top_cut = float(config["performance"]["top_percentile"])
    bottom_cut = float(config["performance"]["bottom_percentile"])
    behavior_rows: list[dict[str, Any]] = []

    for target in prepared_targets:
        channel = channel_map.get(target.get("channel_name", ""), target.get("channel_name", ""))
        views = as_float(target.get("short_views"))
        cohort = cohorts.get(channel, [])
        cohort_views = [float(row["views"]) for row in cohort]
        if views is None or not cohort_views:
            behavior_rows.append(
                {
                    **target,
                    "channel_name": channel,
                    "behavior_label_status": "N/A",
                    "behavior_label_reason": "missing target views or channel cohort",
                }
            )
            continue

        median_views = statistics.median(cohort_views)
        relative_score = relative_log_view_score(views, median_views)
        channel_percentile = percentile(cohort_views, views)
        matching_video = str(target.get("short_video_id", ""))
        leave_one_out = [
            float(row["views"])
            for row in cohort
            if str(row["video_id"]) != matching_video
        ]
        loo_median = statistics.median(leave_one_out) if leave_one_out else None
        loo_relative = (
            relative_log_view_score(views, loo_median)
            if loo_median is not None
            else None
        )
        loo_percentile = percentile(leave_one_out, views)

        trimmed = sorted(cohort_views)
        if len(trimmed) >= 5:
            trimmed = trimmed[1:-1]
        trimmed_median = statistics.median(trimmed)
        trimmed_relative = relative_log_view_score(views, trimmed_median)
        trimmed_percentile = percentile(trimmed, views)

        ci_lower, ci_upper = _bootstrap_percentile_interval(
            cohort_views,
            views,
            iterations=iterations,
            seed=f"{config['evaluation_id']}:{target['candidate_id']}",
        )
        tier = (
            "top25"
            if channel_percentile is not None and channel_percentile >= top_cut
            else "bottom25"
            if channel_percentile is not None and channel_percentile <= bottom_cut
            else "middle50"
        )
        behavior_rows.append(
            {
                **target,
                "channel_name": channel,
                "behavior_label_status": "exploratory",
                "behavior_label_reason": (
                    "cumulative views; 7-day/30-day fixed-window views unavailable"
                ),
                "channel_cohort_n": len(cohort_views),
                "channel_median_views": round(median_views, 4),
                "relative_log_view_score": round(relative_score, 6),
                "channel_view_percentile": round(float(channel_percentile), 4),
                "performance_tier": tier,
                "loo_relative_log_view_score": rounded(loo_relative, 6),
                "loo_channel_view_percentile": rounded(loo_percentile, 4),
                "trimmed_relative_log_view_score": round(trimmed_relative, 6),
                "trimmed_channel_view_percentile": rounded(trimmed_percentile, 4),
                "percentile_bootstrap_ci_lower": rounded(ci_lower, 4),
                "percentile_bootstrap_ci_upper": rounded(ci_upper, 4),
            }
        )

    labels_path = output_dir / "behavior_labels_private.csv"
    write_csv(labels_path, behavior_rows)

    valid = [
        row
        for row in behavior_rows
        if as_float(row.get("relative_log_view_score")) is not None
        and as_float(row.get("channel_view_percentile")) is not None
    ]
    channel_distribution: list[dict[str, Any]] = []
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_channel[row["channel_name"]].append(row)
    for channel, rows in sorted(by_channel.items()):
        relative_values = [float(row["relative_log_view_score"]) for row in rows]
        percentile_values = [float(row["channel_view_percentile"]) for row in rows]
        channel_distribution.append(
            {
                "channel_name": channel,
                "target_n": len(rows),
                "cohort_n": rows[0]["channel_cohort_n"],
                "relative_log_mean": rounded(statistics.mean(relative_values)),
                "relative_log_median": rounded(statistics.median(relative_values)),
                "relative_log_min": rounded(min(relative_values)),
                "relative_log_max": rounded(max(relative_values)),
                "percentile_mean": rounded(statistics.mean(percentile_values)),
                "percentile_median": rounded(statistics.median(percentile_values)),
                "top25_count": sum(row["performance_tier"] == "top25" for row in rows),
                "middle50_count": sum(row["performance_tier"] == "middle50" for row in rows),
                "bottom25_count": sum(row["performance_tier"] == "bottom25" for row in rows),
            }
        )
    write_csv(output_dir / "case_1_channel_distributions.csv", channel_distribution)

    relative_values = [float(row["relative_log_view_score"]) for row in valid]
    percentile_values = [float(row["channel_view_percentile"]) for row in valid]
    relative_deltas = [
        abs(float(row["relative_log_view_score"]) - float(row["loo_relative_log_view_score"]))
        for row in valid
        if as_float(row.get("loo_relative_log_view_score")) is not None
    ]
    percentile_deltas = [
        abs(float(row["channel_view_percentile"]) - float(row["loo_channel_view_percentile"]))
        for row in valid
        if as_float(row.get("loo_channel_view_percentile")) is not None
    ]
    summary = {
        "case": "channel_baseline",
        "status": "actual_exploratory",
        "candidate_count": len(behavior_rows),
        "valid_candidate_count": len(valid),
        "channel_count": len(by_channel),
        "observation_basis": "cumulative views at heterogeneous observation ages",
        "fixed_window_validation": "N/A: 7-day and 30-day view snapshots are absent",
        "relative_log_vs_percentile_spearman": rounded(
            spearman(relative_values, percentile_values)
        ),
        "leave_one_out_relative_log_mean_absolute_delta": rounded(
            statistics.mean(relative_deltas) if relative_deltas else None,
            6,
        ),
        "leave_one_out_percentile_mean_absolute_delta": rounded(
            statistics.mean(percentile_deltas) if percentile_deltas else None,
            4,
        ),
        "percentile_bootstrap_iterations": iterations,
        "label_distribution": {
            "top25": sum(row.get("performance_tier") == "top25" for row in valid),
            "middle50": sum(row.get("performance_tier") == "middle50" for row in valid),
            "bottom25": sum(row.get("performance_tier") == "bottom25" for row in valid),
        },
        "outputs": {
            "behavior_labels_private": str(labels_path),
            "channel_distributions": str(output_dir / "case_1_channel_distributions.csv"),
        },
    }
    write_json(output_dir / "case_1_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build channel-relative behavior targets.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = build(load_config(args.config))
    print(
        f"Built behavior labels for {summary['valid_candidate_count']}/"
        f"{summary['candidate_count']} candidates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
