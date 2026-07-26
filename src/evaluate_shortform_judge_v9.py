from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_channel_relative_validity import pearson, roc_auc, spearman
from shortform_judge_v9 import (
    EDITORIAL_DIMENSIONS,
    ENGAGEMENT_DIMENSIONS,
    load_config,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["judge_run_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def aggregate_scores(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["judge_run_id"], row["candidate_id"])].append(row)

    output: list[dict[str, Any]] = []
    for (run_id, candidate_id), group in sorted(grouped.items()):
        scored = [
            row
            for row in group
            if row.get("verdict") == "score"
            and number(row.get("judge_score_100")) is not None
        ]
        item: dict[str, Any] = {
            "judge_run_id": run_id,
            "candidate_id": candidate_id,
            "longform_id": group[0].get("longform_id", ""),
            "repeat_count": len(group),
            "scored_repeat_count": len(scored),
            "aggregate_status": "scored" if scored else "abstain",
        }
        for field in (
            "editorial_score_100",
            "engagement_score_100",
            "judge_score_100",
            "confidence_1_5",
        ):
            values = [
                value
                for row in scored
                if (value := number(row.get(field))) is not None
            ]
            item[f"{field}_mean"] = rounded(
                statistics.mean(values) if values else None
            )
            item[f"{field}_std"] = rounded(
                statistics.pstdev(values)
                if len(values) > 1
                else 0.0
                if values
                else None
            )
        output.append(item)
    return output


def repeat_metrics(
    rows: list[dict[str, str]],
    expected_candidate_count: int,
) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["judge_run_id"]].append(row)

    output: list[dict[str, Any]] = []
    for run_id, group in sorted(by_run.items()):
        by_repeat: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
        for row in group:
            values = {
                axis: number(row.get(f"{axis}_score_100"))
                for axis in ("editorial", "engagement", "judge")
            }
            if row.get("verdict") == "score" and all(
                value is not None for value in values.values()
            ):
                by_repeat[int(row["repeat_index"])][row["candidate_id"]] = {
                    axis: float(value)
                    for axis, value in values.items()
                    if value is not None
                }

        result: dict[str, Any] = {
            "judge_run_id": run_id,
            "expected_candidate_count": expected_candidate_count,
            "repeat_count": len(by_repeat),
        }
        scored_ids = {
            row["candidate_id"]
            for row in group
            if row.get("verdict") == "score"
            and number(row.get("judge_score_100")) is not None
        }
        result["candidate_scoring_coverage"] = rounded(
            len(scored_ids) / expected_candidate_count
            if expected_candidate_count
            else None
        )
        for axis in ("editorial", "engagement", "judge"):
            correlations: list[float] = []
            errors: list[float] = []
            common_counts: list[int] = []
            for left, right in itertools.combinations(sorted(by_repeat), 2):
                common = sorted(
                    set(by_repeat[left]) & set(by_repeat[right])
                )
                common_counts.append(len(common))
                left_values = [
                    by_repeat[left][candidate_id][axis]
                    for candidate_id in common
                ]
                right_values = [
                    by_repeat[right][candidate_id][axis]
                    for candidate_id in common
                ]
                correlation = spearman(left_values, right_values)
                if correlation is not None:
                    correlations.append(correlation)
                errors.extend(
                    abs(a - b) for a, b in zip(left_values, right_values)
                )
            result[f"{axis}_repeat_spearman"] = rounded(
                statistics.mean(correlations) if correlations else None
            )
            result[f"{axis}_repeat_mae"] = rounded(
                statistics.mean(errors) if errors else None
            )
            result[f"{axis}_repeat_common_count"] = (
                min(common_counts) if common_counts else 0
            )
        output.append(result)
    return output


def performance_metrics(
    aggregates: list[dict[str, Any]],
    targets: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_by_id = {row["candidate_id"]: row for row in targets}
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        target = target_by_id.get(row["candidate_id"])
        score = number(row.get("engagement_score_100_mean"))
        percentile = number(
            target.get("channel_performance_percentile")
            if target
            else None
        )
        if target and score is not None and percentile is not None:
            by_run[row["judge_run_id"]].append(
                {
                    "score": score,
                    "target": target,
                    "percentile": percentile,
                }
            )

    summaries: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    for run_id, records in sorted(by_run.items()):
        channels = sorted(
            {
                record["target"].get("channel_name", "")
                for record in records
                if record["target"].get("channel_name", "")
            }
        )
        run_channel_rows: list[dict[str, Any]] = []
        centered_scores: list[float] = []
        centered_percentiles: list[float] = []
        for channel in channels:
            group = [
                record
                for record in records
                if record["target"].get("channel_name") == channel
            ]
            pos = [
                record
                for record in group
                if record["target"].get("performance_label") == "pos"
            ]
            neg = [
                record
                for record in group
                if record["target"].get("performance_label") == "neg"
            ]
            if not pos or not neg:
                continue
            labels = [
                int(
                    record["target"].get("performance_label") == "pos"
                )
                for record in group
            ]
            scores = [record["score"] for record in group]
            score_mean = statistics.mean(scores)
            percentile_mean = statistics.mean(
                record["percentile"] for record in group
            )
            centered_scores.extend(score - score_mean for score in scores)
            centered_percentiles.extend(
                record["percentile"] - percentile_mean
                for record in group
            )
            channel_row = {
                "judge_run_id": run_id,
                "channel_name": channel,
                "pos_count": len(pos),
                "neg_count": len(neg),
                "stable_channel": len(pos) >= 3 and len(neg) >= 3,
                "pos_mean_engagement_score": rounded(
                    statistics.mean(record["score"] for record in pos)
                ),
                "neg_mean_engagement_score": rounded(
                    statistics.mean(record["score"] for record in neg)
                ),
                "channel_auc": rounded(roc_auc(labels, scores)),
                "channel_score_percentile_spearman": rounded(
                    spearman(
                        scores,
                        [record["percentile"] for record in group],
                    )
                ),
            }
            run_channel_rows.append(channel_row)
            channel_rows.append(channel_row)

        stable_aucs = [
            float(row["channel_auc"])
            for row in run_channel_rows
            if row["stable_channel"] and row["channel_auc"] is not None
        ]
        all_aucs = [
            float(row["channel_auc"])
            for row in run_channel_rows
            if row["channel_auc"] is not None
        ]
        pooled_labels = [
            int(record["target"].get("performance_label") == "pos")
            for record in records
        ]
        pooled_scores = [record["score"] for record in records]
        bootstrap_values: list[float] = []
        stable_channels = [
            row["channel_name"]
            for row in run_channel_rows
            if row["stable_channel"]
        ]
        rng = random.Random(f"{run_id}:stable-channel-bootstrap")
        for _ in range(2000):
            sampled_channel_aucs: list[float] = []
            for channel in stable_channels:
                group = [
                    record
                    for record in records
                    if record["target"].get("channel_name") == channel
                ]
                pos_scores = [
                    record["score"]
                    for record in group
                    if record["target"].get("performance_label") == "pos"
                ]
                neg_scores = [
                    record["score"]
                    for record in group
                    if record["target"].get("performance_label") == "neg"
                ]
                sampled_pos = rng.choices(pos_scores, k=len(pos_scores))
                sampled_neg = rng.choices(neg_scores, k=len(neg_scores))
                sampled_labels = [1] * len(sampled_pos) + [0] * len(
                    sampled_neg
                )
                sampled_scores = sampled_pos + sampled_neg
                sampled_auc = roc_auc(sampled_labels, sampled_scores)
                if sampled_auc is not None:
                    sampled_channel_aucs.append(sampled_auc)
            if sampled_channel_aucs:
                bootstrap_values.append(
                    statistics.mean(sampled_channel_aucs)
                )
        bootstrap_values.sort()
        ci_lower = (
            bootstrap_values[int(0.025 * (len(bootstrap_values) - 1))]
            if bootstrap_values
            else None
        )
        ci_upper = (
            bootstrap_values[int(0.975 * (len(bootstrap_values) - 1))]
            if bootstrap_values
            else None
        )
        summaries.append(
            {
                "judge_run_id": run_id,
                "candidate_count": len(records),
                "eligible_channel_count": len(run_channel_rows),
                "stable_channel_count": len(stable_aucs),
                "macro_channel_auc_all": rounded(
                    statistics.mean(all_aucs) if all_aucs else None
                ),
                "macro_channel_auc_stable": rounded(
                    statistics.mean(stable_aucs)
                    if stable_aucs
                    else None
                ),
                "stable_macro_auc_bootstrap_ci_lower": rounded(ci_lower),
                "stable_macro_auc_bootstrap_ci_upper": rounded(ci_upper),
                "within_channel_centered_correlation": rounded(
                    pearson(centered_scores, centered_percentiles)
                ),
                "pooled_auc_supplementary": rounded(
                    roc_auc(pooled_labels, pooled_scores)
                ),
            }
        )
    return summaries, channel_rows


def _human_axis_score(
    row: dict[str, str],
    axis: str,
    dimensions: tuple[str, ...],
) -> float | None:
    values = [
        number(row.get(f"{axis}_{dimension}_score_0_4"))
        for dimension in dimensions
    ]
    if any(value is None for value in values):
        return None
    return 25.0 * statistics.mean(
        float(value) for value in values if value is not None
    )


def human_metrics(
    aggregates: list[dict[str, Any]],
    human_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in human_rows:
        editorial = _human_axis_score(
            row,
            "editorial",
            EDITORIAL_DIMENSIONS,
        )
        engagement = _human_axis_score(
            row,
            "engagement",
            ENGAGEMENT_DIMENSIONS,
        )
        if editorial is not None and engagement is not None:
            valid.append(
                {
                    **row,
                    "editorial_score": editorial,
                    "engagement_score": engagement,
                }
            )

    by_annotator: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_annotator[row["annotator_id"]][row["candidate_id"]] = row
        by_candidate[row["candidate_id"]].append(row)

    inter_rater: dict[str, list[float]] = {
        "editorial": [],
        "engagement": [],
    }
    for left, right in itertools.combinations(sorted(by_annotator), 2):
        common = sorted(
            set(by_annotator[left]) & set(by_annotator[right])
        )
        for axis in inter_rater:
            correlation = spearman(
                [
                    by_annotator[left][candidate_id][f"{axis}_score"]
                    for candidate_id in common
                ],
                [
                    by_annotator[right][candidate_id][f"{axis}_score"]
                    for candidate_id in common
                ],
            )
            if correlation is not None:
                inter_rater[axis].append(correlation)

    human_means = {
        candidate_id: {
            axis: statistics.mean(
                row[f"{axis}_score"] for row in rows
            )
            for axis in ("editorial", "engagement")
        }
        for candidate_id, rows in by_candidate.items()
    }
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        if row["candidate_id"] in human_means:
            by_run[row["judge_run_id"]].append(row)

    alignment: list[dict[str, Any]] = []
    for run_id, rows in sorted(by_run.items()):
        alignment.append(
            {
                "judge_run_id": run_id,
                "candidate_count": len(rows),
                "editorial_human_spearman": rounded(
                    spearman(
                        [
                            float(row["editorial_score_100_mean"])
                            for row in rows
                        ],
                        [
                            human_means[row["candidate_id"]]["editorial"]
                            for row in rows
                        ],
                    )
                ),
                "engagement_human_spearman": rounded(
                    spearman(
                        [
                            float(row["engagement_score_100_mean"])
                            for row in rows
                        ],
                        [
                            human_means[row["candidate_id"]]["engagement"]
                            for row in rows
                        ],
                    )
                ),
            }
        )
    summary = {
        "completed_rating_count": len(valid),
        "candidate_count": len(human_means),
        "annotator_count": len(by_annotator),
        "editorial_inter_rater_spearman": rounded(
            statistics.mean(inter_rater["editorial"])
            if inter_rater["editorial"]
            else None
        ),
        "engagement_inter_rater_spearman": rounded(
            statistics.mean(inter_rater["engagement"])
            if inter_rater["engagement"]
            else None
        ),
    }
    return alignment, summary


def gate_status(
    reliability: list[dict[str, Any]],
    performance: list[dict[str, Any]],
    human_alignment: list[dict[str, Any]],
    human_summary: dict[str, Any],
    gates: dict[str, Any],
    dataset_role: str,
) -> list[dict[str, Any]]:
    performance_by_run = {
        row["judge_run_id"]: row for row in performance
    }
    human_by_run = {
        row["judge_run_id"]: row for row in human_alignment
    }
    output: list[dict[str, Any]] = []
    for row in reliability:
        run_id = row["judge_run_id"]
        stable = (
            (row.get("candidate_scoring_coverage") or 0)
            >= float(gates["min_candidate_coverage"])
            and (
                (row.get("editorial_repeat_common_count") or 0)
                / max(1, int(row.get("expected_candidate_count") or 0))
            )
            >= float(gates["min_repeat_pair_coverage"])
            and (row.get("editorial_repeat_spearman") or 0)
            >= float(gates["min_repeat_spearman_each_axis"])
            and (row.get("engagement_repeat_spearman") or 0)
            >= float(gates["min_repeat_spearman_each_axis"])
            and (row.get("editorial_repeat_mae") or 999)
            <= float(gates["max_repeat_mae_each_axis"])
            and (row.get("engagement_repeat_mae") or 999)
            <= float(gates["max_repeat_mae_each_axis"])
        )
        human = human_by_run.get(run_id, {})
        human_reliable = (
            (human_summary.get("candidate_count") or 0)
            >= int(gates["min_human_anchor_candidates"])
            and (human_summary.get("annotator_count") or 0)
            >= int(gates["min_human_annotators"])
            and
            (human_summary.get("editorial_inter_rater_spearman") or 0)
            >= float(gates["min_human_inter_rater_spearman"])
            and (human_summary.get("engagement_inter_rater_spearman") or 0)
            >= float(gates["min_human_inter_rater_spearman"])
        )
        editorial_valid = (
            stable
            and human_reliable
            and (human.get("editorial_human_spearman") or 0)
            >= float(gates["min_human_editorial_spearman"])
        )
        performance_row = performance_by_run.get(run_id, {})
        locked_ci_pass = (
            dataset_role != "locked_test"
            or (
                performance_row.get(
                    "stable_macro_auc_bootstrap_ci_lower"
                )
                or 0
            )
            >= float(gates["final_min_bootstrap_auc_ci_lower"])
        )
        engagement_valid = (
            stable
            and human_reliable
            and (human.get("engagement_human_spearman") or 0)
            >= float(gates["min_human_engagement_spearman"])
            and (performance_row.get("macro_channel_auc_stable") or 0)
            >= float(gates["development_min_stable_channel_macro_auc"])
            and (
                performance_row.get(
                    "within_channel_centered_correlation"
                )
                or 0
            )
            >= float(
                gates[
                    "development_min_within_channel_centered_correlation"
                ]
            )
            and locked_ci_pass
        )
        repeat_complete = (
            (row.get("editorial_repeat_common_count") or 0)
            / max(1, int(row.get("expected_candidate_count") or 0))
        ) >= float(gates["min_repeat_pair_coverage"])
        if not repeat_complete:
            status = "incomplete_repeat_run"
        elif not human_alignment:
            status = "pending_human_anchor_scores"
        elif not stable:
            status = "needs_revision_reliability"
        elif not editorial_valid:
            status = "needs_revision_editorial_validity"
        elif not engagement_valid:
            status = "editorial_only_engagement_unvalidated"
        elif dataset_role == "locked_test":
            status = "validated"
        else:
            status = "development_pass_pending_locked_test"
        output.append(
            {
                "judge_run_id": run_id,
                "reliability_gate_pass": stable,
                "editorial_validity_gate_pass": editorial_valid,
                "engagement_validity_gate_pass": engagement_valid,
                "locked_test_ci_gate_pass": locked_ci_pass,
                "dataset_role": dataset_role,
                "judge_status": status,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Shortform Judge v9."
    )
    parser.add_argument("--scores", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "config"
            / "shortform_judge_v9_opus.json"
        ),
    )
    parser.add_argument("--human-scores")
    parser.add_argument(
        "--dataset-role",
        choices=("development", "locked_test"),
        default="development",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    scores = read_csv(Path(args.scores))
    targets = read_csv(Path(args.targets))
    human_rows = (
        read_csv(Path(args.human_scores))
        if args.human_scores and Path(args.human_scores).exists()
        else []
    )
    aggregates = aggregate_scores(scores)
    reliability = repeat_metrics(scores, len(targets))
    performance, channels = performance_metrics(aggregates, targets)
    human_alignment, human_summary = human_metrics(
        aggregates,
        human_rows,
    )
    gates = gate_status(
        reliability,
        performance,
        human_alignment,
        human_summary,
        config["validation_gates"],
        args.dataset_role,
    )
    summary = {
        "protocol": "shortform_judge_v9",
        "dataset_role": args.dataset_role,
        "counts": {
            "target_candidate_count": len(targets),
            "score_row_count": len(scores),
            "aggregated_candidate_count": len(aggregates),
        },
        "reliability": reliability,
        "human": human_summary,
        "human_alignment": human_alignment,
        "performance_external_validity": performance,
        "gates": gates,
        "interpretation": {
            "editorial_axis": (
                "Validated against human reference judgments."
            ),
            "engagement_axis": (
                "Validated against both human judgments and within-channel "
                "relative performance."
            ),
            "pos_neg": (
                "Relative performance signal, not absolute good/bad content."
            ),
            "ours_vpick": (
                "Excluded until the Judge passes all required gates."
            ),
        },
    }
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "shortform_judge_v9_aggregated.csv", aggregates)
    write_csv(out_dir / "shortform_judge_v9_reliability.csv", reliability)
    write_csv(out_dir / "shortform_judge_v9_channel_metrics.csv", channels)
    write_csv(
        out_dir / "shortform_judge_v9_human_alignment.csv",
        human_alignment,
    )
    write_csv(out_dir / "shortform_judge_v9_gates.csv", gates)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shortform_judge_v9_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
