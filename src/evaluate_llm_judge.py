from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


CANDIDATE_DIMENSIONS = (
    "opening_strength",
    "standalone",
    "completeness",
    "engagement_value",
    "boundary_naturalness",
    "titleability",
)
EVIDENCE_DIMENSIONS = (
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
)
SET_DIMENSIONS = (
    "redundancy_control",
    "event_diversity",
    "timeline_coverage",
    "portfolio_quality",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 3 or len(x_values) != len(y_values):
        return None
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_scale == 0 or y_scale == 0:
        return None
    return numerator / (x_scale * y_scale)


def spearman(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 3 or len(x_values) != len(y_values):
        return None
    return pearson(average_ranks(x_values), average_ranks(y_values))


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def score_lookup_from_aggregates(aggregates: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    for row in aggregates:
        score = as_float(row.get("overall_score_mean"))
        if score is not None:
            output[(row["judge_run_id"], row["candidate_id"])] = score
    return output


def aggregate_candidate_scores(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["judge_run_id"], row["candidate_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (judge_run_id, candidate_id), group in sorted(grouped.items()):
        scored = [row for row in group if as_float(row.get("overall_score")) is not None]
        overall = [float(row["overall_score"]) for row in scored]
        evidence_values = {
            name: [value for row in group if (value := as_float(row.get(name))) is not None]
            for name in EVIDENCE_DIMENSIONS
        }
        confidence_values = [value for row in group if (value := as_float(row.get("confidence"))) is not None]
        output.append(
            {
                "judge_run_id": judge_run_id,
                "provider": group[0].get("provider", ""),
                "model": group[0].get("model", ""),
                "candidate_id": candidate_id,
                "long_video_id": group[0].get("long_video_id", ""),
                "repeat_count": len(group),
                "scored_repeat_count": len(scored),
                "abstain_repeat_count": len(group) - len(scored),
                "aggregate_status": "scored" if scored else "abstain",
                **{
                    f"{name}_mean": round_or_none(mean_or_none(values))
                    for name, values in evidence_values.items()
                },
                **{
                    f"{name}_mean": round_or_none(
                        mean_or_none([float(row[name]) for row in scored if as_float(row.get(name)) is not None])
                    )
                    for name in CANDIDATE_DIMENSIONS
                },
                "confidence_mean": round_or_none(mean_or_none(confidence_values)),
                "overall_score_mean": round_or_none(mean_or_none(overall)),
                "overall_score_std": round(pstdev(overall), 4) if len(overall) > 1 else (0.0 if overall else None),
            }
        )
    return output


def repeat_reliability(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_run_repeat: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        score = as_float(row.get("overall_score"))
        if score is not None:
            by_run_repeat[row["judge_run_id"]][int(row["repeat_index"])][row["candidate_id"]] = score
    output: list[dict[str, Any]] = []
    for judge_run_id, by_repeat in sorted(by_run_repeat.items()):
        correlations: list[float] = []
        pair_count = 0
        repeats = sorted(by_repeat)
        for left_repeat, right_repeat in itertools.combinations(repeats, 2):
            common = sorted(set(by_repeat[left_repeat]) & set(by_repeat[right_repeat]))
            correlation = spearman(
                [by_repeat[left_repeat][candidate_id] for candidate_id in common],
                [by_repeat[right_repeat][candidate_id] for candidate_id in common],
            )
            if correlation is not None:
                correlations.append(correlation)
                pair_count += 1
        output.append(
            {
                "judge_run_id": judge_run_id,
                "repeat_count": len(repeats),
                "repeat_pair_count": pair_count,
                "mean_repeat_spearman": round_or_none(mean_or_none(correlations)),
            }
        )
    return output


def evidence_coverage(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["judge_run_id"]].append(row)
    output: list[dict[str, Any]] = []
    for judge_run_id, group in sorted(by_run.items()):
        by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            by_candidate[row["candidate_id"]].append(row)
        scored_rows = [row for row in group if as_float(row.get("overall_score")) is not None]
        scored_candidates = {
            candidate_id
            for candidate_id, candidate_rows in by_candidate.items()
            if any(as_float(row.get("overall_score")) is not None for row in candidate_rows)
        }
        mixed_candidates = {
            candidate_id
            for candidate_id, candidate_rows in by_candidate.items()
            if 0 < sum(as_float(row.get("overall_score")) is not None for row in candidate_rows) < len(candidate_rows)
        }
        evidence_means = {}
        for name in EVIDENCE_DIMENSIONS:
            values = [value for row in group if (value := as_float(row.get(name))) is not None]
            evidence_means[f"mean_{name}"] = round_or_none(mean_or_none(values))
        confidence_values = [value for row in group if (value := as_float(row.get("confidence"))) is not None]
        output.append(
            {
                "judge_run_id": judge_run_id,
                "score_row_count": len(group),
                "scored_row_count": len(scored_rows),
                "abstain_row_count": len(group) - len(scored_rows),
                "row_scoring_coverage": round_or_none(len(scored_rows) / len(group) if group else None),
                "candidate_count": len(by_candidate),
                "scored_candidate_count": len(scored_candidates),
                "candidate_scoring_coverage": round_or_none(
                    len(scored_candidates) / len(by_candidate) if by_candidate else None
                ),
                "mixed_verdict_candidate_count": len(mixed_candidates),
                **evidence_means,
                "mean_confidence": round_or_none(mean_or_none(confidence_values)),
            }
        )
    return output


def inter_model_reliability(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, float]] = defaultdict(dict)
    for row in aggregates:
        score = as_float(row.get("overall_score_mean"))
        if score is not None:
            by_run[row["judge_run_id"]][row["candidate_id"]] = score
    output: list[dict[str, Any]] = []
    for left_run, right_run in itertools.combinations(sorted(by_run), 2):
        common = sorted(set(by_run[left_run]) & set(by_run[right_run]))
        correlation = spearman(
            [by_run[left_run][candidate_id] for candidate_id in common],
            [by_run[right_run][candidate_id] for candidate_id in common],
        )
        output.append(
            {
                "left_judge_run_id": left_run,
                "right_judge_run_id": right_run,
                "candidate_count": len(common),
                "spearman": round_or_none(correlation),
            }
        )
    return output


def source_system_summary(
    aggregates: list[dict[str, Any]], sources: list[dict[str, str]], set_scores: list[dict[str, str]]
) -> list[dict[str, Any]]:
    score_lookup = score_lookup_from_aggregates(aggregates)
    source_candidates: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in sources:
        key = (row.get("source_system", ""), row.get("long_video_id", ""))
        current = source_candidates[key].get(row["candidate_id"])
        if current is None or float(row.get("source_rank") or 999) < float(current.get("source_rank") or 999):
            source_candidates[key][row["candidate_id"]] = row

    set_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in set_scores:
        set_grouped[(row["judge_run_id"], row.get("source_system", ""))].append(float(row["overall_set_score"]))

    output: list[dict[str, Any]] = []
    judge_runs = sorted({row["judge_run_id"] for row in aggregates})
    systems = sorted({row.get("source_system", "") for row in sources})
    for judge_run_id in judge_runs:
        for system in systems:
            all_scores: list[float] = []
            top1_scores: list[float] = []
            long_count = 0
            for (source_system, _long_video_id), candidates_by_id in source_candidates.items():
                if source_system != system:
                    continue
                ranked = sorted(candidates_by_id.values(), key=lambda row: float(row.get("source_rank") or 999))
                scores = [score_lookup[(judge_run_id, row["candidate_id"])] for row in ranked if (judge_run_id, row["candidate_id"]) in score_lookup]
                if not scores:
                    continue
                long_count += 1
                all_scores.extend(scores)
                top1_scores.append(scores[0])
            if not all_scores:
                continue
            set_values = set_grouped.get((judge_run_id, system), [])
            output.append(
                {
                    "judge_run_id": judge_run_id,
                    "source_system": system,
                    "long_video_count": long_count,
                    "candidate_count": len(all_scores),
                    "mean_candidate_score": round(mean(all_scores), 4),
                    "mean_top1_score": round(mean(top1_scores), 4),
                    "mean_set_score": round_or_none(mean_or_none(set_values)),
                }
            )
    return output


def gold_alignment(
    aggregates: list[dict[str, Any]], sources: list[dict[str, str]], tie_margin: float
) -> list[dict[str, Any]]:
    score_lookup = score_lookup_from_aggregates(aggregates)
    by_long_system: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in sources:
        if row.get("source_system") == "gold" and row.get("evaluation_role", "gold") == "control":
            continue
        by_long_system[(row.get("long_video_id", ""), row.get("source_system", ""))].add(row["candidate_id"])
    judge_runs = sorted({row["judge_run_id"] for row in aggregates})
    systems = sorted({row.get("source_system", "") for row in sources if row.get("source_system") != "gold"})
    output: list[dict[str, Any]] = []
    for judge_run_id in judge_runs:
        for system in systems:
            wins = ties = losses = comparisons = 0
            for (long_video_id, source_name), gold_ids in by_long_system.items():
                if source_name != "gold":
                    continue
                system_ids = by_long_system.get((long_video_id, system), set())
                for gold_id in gold_ids:
                    for candidate_id in system_ids:
                        if gold_id == candidate_id:
                            continue
                        gold_score = score_lookup.get((judge_run_id, gold_id))
                        system_score = score_lookup.get((judge_run_id, candidate_id))
                        if gold_score is None or system_score is None:
                            continue
                        comparisons += 1
                        difference = gold_score - system_score
                        if difference > tie_margin:
                            wins += 1
                        elif difference < -tie_margin:
                            losses += 1
                        else:
                            ties += 1
            output.append(
                {
                    "judge_run_id": judge_run_id,
                    "compared_system": system,
                    "comparison_count": comparisons,
                    "gold_win_count": wins,
                    "tie_count": ties,
                    "gold_loss_count": losses,
                    "gold_nonloss_rate": round((wins + ties) / comparisons, 4) if comparisons else None,
                    "gold_strict_win_rate": round(wins / comparisons, 4) if comparisons else None,
                }
            )
    return output


def parse_channel_percentile(notes: str) -> float | None:
    match = re.search(r"(?:채널내백분위|channel_percentile)\s*=\s*([0-9]+(?:\.[0-9]+)?)", str(notes or ""))
    return float(match.group(1)) if match else None


def performance_alignment(
    aggregates: list[dict[str, Any]], sources: list[dict[str, str]]
) -> list[dict[str, Any]]:
    score_lookup = score_lookup_from_aggregates(aggregates)
    gold_rows = [row for row in sources if row.get("source_system") == "gold"]
    output: list[dict[str, Any]] = []
    for judge_run_id in sorted({row["judge_run_id"] for row in aggregates}):
        records = []
        for row in gold_rows:
            score = score_lookup.get((judge_run_id, row["candidate_id"]))
            views = as_float(row.get("short_views"))
            likes = as_float(row.get("short_likes"))
            if score is None or views is None or likes is None or views <= 0:
                continue
            records.append(
                {
                    "score": score,
                    "views": views,
                    "likes": likes,
                    "like_rate": likes / views,
                    "channel_percentile": parse_channel_percentile(row.get("source_notes", "")),
                }
            )
        for metric in ("views", "likes", "like_rate", "channel_percentile"):
            filtered = [record for record in records if record[metric] is not None]
            output.append(
                {
                    "judge_run_id": judge_run_id,
                    "performance_metric": metric,
                    "pair_count": len(filtered),
                    "spearman": round_or_none(
                        spearman(
                            [float(record["score"]) for record in filtered],
                            [float(record[metric]) for record in filtered],
                        )
                    ),
                }
            )
    return output


def performance_group_alignment(
    aggregates: list[dict[str, Any]], sources: list[dict[str, str]]
) -> list[dict[str, Any]]:
    score_lookup = score_lookup_from_aggregates(aggregates)
    output: list[dict[str, Any]] = []
    for judge_run_id in sorted({row["judge_run_id"] for row in aggregates}):
        groups: dict[str, list[float]] = {"pos": [], "neg": []}
        totals: Counter[str] = Counter()
        for row in sources:
            label = row.get("performance_label", "").strip().lower()
            if not label:
                label = {"main": "pos", "control": "neg"}.get(row.get("dataset_split", ""), "")
            if row.get("source_system") != "gold" or label not in groups:
                continue
            totals[label] += 1
            score = score_lookup.get((judge_run_id, row["candidate_id"]))
            if score is not None:
                groups[label].append(score)
        high_scores = groups["pos"]
        low_scores = groups["neg"]
        comparisons = len(high_scores) * len(low_scores)
        wins = sum(high > low for high in high_scores for low in low_scores)
        ties = sum(high == low for high in high_scores for low in low_scores)
        auc = (wins + (0.5 * ties)) / comparisons if comparisons else None
        output.append(
            {
                "judge_run_id": judge_run_id,
                "pos_count": len(high_scores),
                "pos_total_count": totals["pos"],
                "pos_scoring_coverage": round_or_none(len(high_scores) / totals["pos"] if totals["pos"] else None),
                "pos_mean_score": round_or_none(mean_or_none(high_scores)),
                "neg_count": len(low_scores),
                "neg_total_count": totals["neg"],
                "neg_scoring_coverage": round_or_none(len(low_scores) / totals["neg"] if totals["neg"] else None),
                "neg_mean_score": round_or_none(mean_or_none(low_scores)),
                "mean_score_gap": round_or_none(
                    mean(high_scores) - mean(low_scores) if high_scores and low_scores else None
                ),
                "pos_over_neg_auc": round_or_none(auc),
            }
        )
    return output


def fleiss_kappa(labels_by_item: dict[str, list[str]]) -> float | None:
    categories = ("left", "right", "tie")
    usable = [labels for labels in labels_by_item.values() if len(labels) >= 2]
    if not usable:
        return None
    total_labels = sum(len(labels) for labels in usable)
    category_totals = Counter(label for labels in usable for label in labels)
    expected = sum((category_totals[category] / total_labels) ** 2 for category in categories)
    observed_values = []
    for labels in usable:
        counts = Counter(labels)
        n = len(labels)
        observed_values.append(sum(counts[category] * (counts[category] - 1) for category in categories) / (n * (n - 1)))
    observed = mean(observed_values)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def human_alignment(
    aggregates: list[dict[str, Any]], human_rows: list[dict[str, str]], tie_margin: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_rows = []
    for row in human_rows:
        preference = row.get("preference", "").strip().lower()
        aliases = {"l": "left", "r": "right", "t": "tie", "왼쪽": "left", "오른쪽": "right", "동점": "tie"}
        preference = aliases.get(preference, preference)
        if preference in {"left", "right", "tie"}:
            valid_rows.append({**row, "normalized_preference": preference})

    by_comparison: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in valid_rows:
        by_comparison[row["comparison_id"]].append(row)
    labels_for_kappa = {
        comparison_id: [row["normalized_preference"] for row in rows]
        for comparison_id, rows in by_comparison.items()
    }
    score_lookup = score_lookup_from_aggregates(aggregates)
    result_rows: list[dict[str, Any]] = []
    for judge_run_id in sorted({row["judge_run_id"] for row in aggregates}):
        correct = total = 0
        for comparison_id, rows in by_comparison.items():
            counts = Counter(row["normalized_preference"] for row in rows)
            top_count = max(counts.values())
            winners = [label for label, count in counts.items() if count == top_count]
            if len(winners) != 1:
                continue
            human_choice = winners[0]
            first = rows[0]
            left_score = score_lookup.get((judge_run_id, first["left_candidate_id"]))
            right_score = score_lookup.get((judge_run_id, first["right_candidate_id"]))
            if left_score is None or right_score is None:
                continue
            difference = left_score - right_score
            judge_choice = "left" if difference > tie_margin else "right" if difference < -tie_margin else "tie"
            total += 1
            correct += int(judge_choice == human_choice)
        result_rows.append(
            {
                "judge_run_id": judge_run_id,
                "labeled_comparison_count": total,
                "human_preference_accuracy": round(correct / total, 4) if total else None,
            }
        )
    summary = {
        "label_row_count": len(valid_rows),
        "comparison_count": len(by_comparison),
        "fleiss_kappa": round_or_none(fleiss_kappa(labels_for_kappa)),
    }
    return result_rows, summary


def render_report(summary: dict[str, Any]) -> str:
    status = summary["validation_status"]
    lines = [
        "# LLM-as-a-Judge Validation Report",
        "",
        f"- validation status: `{status}`",
        f"- candidate score rows: {summary['counts']['candidate_score_rows']}",
        f"- unique candidates: {summary['counts']['unique_candidates']}",
        f"- judge runs: {summary['counts']['judge_runs']}",
        "",
        "## Interpretation",
        "",
    ]
    if status == "pending_human_labels":
        lines.append("The scoring pipeline is operational, but it is not yet a validated judge because human preference labels are empty.")
    elif status == "validated":
        lines.append("The judge met the configured human-agreement and repeatability gates.")
    else:
        lines.append("The judge ran successfully but did not meet all validation gates. Review the JSON metrics before using it as a final evaluator.")
    lines.extend(
        [
            "",
            "## Repeat Reliability",
            "",
            *[
                f"- {row['judge_run_id']}: Spearman {row['mean_repeat_spearman']}"
                for row in summary["repeat_reliability"]
            ],
            "",
            "## Evidence Coverage",
            "",
            *[
                (
                    f"- {row['judge_run_id']}: candidate coverage {row['candidate_scoring_coverage']}, "
                    f"row coverage {row['row_scoring_coverage']}, mixed verdicts {row['mixed_verdict_candidate_count']}"
                )
                for row in summary["evidence_coverage"]
            ],
            "",
            "## Pos vs Neg",
            "",
            *[
                (
                    f"- {row['judge_run_id']}: Pos mean {row['pos_mean_score']} "
                    f"({row['pos_count']}/{row['pos_total_count']} scored), Neg mean "
                    f"{row['neg_mean_score']} ({row['neg_count']}/{row['neg_total_count']} scored), "
                    f"AUC {row['pos_over_neg_auc']}"
                )
                for row in summary["performance_group_alignment"]
            ],
            "",
            "## Guardrails",
            "",
            "- Pos/Neg labels, views, likes, pair IDs, and published Shorts IDs were excluded from LLM inputs.",
            "- Candidate intervals were fixed; the judge could not select or trim them.",
            "- Performance correlations are supporting evidence only because publishing context affects views and likes.",
            "- Gold overlap metrics remain separate from content-quality Judge scores.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LLM-as-a-Judge scores against repeats, models, Gold, performance, and humans.")
    parser.add_argument("--candidate-scores", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--set-scores")
    parser.add_argument("--human-labels")
    parser.add_argument("--tie-margin", type=float, default=2.5)
    parser.add_argument("--min-human-accuracy", type=float, default=0.70)
    parser.add_argument("--min-repeat-spearman", type=float, default=0.80)
    args = parser.parse_args()

    candidate_scores = read_csv(Path(args.candidate_scores))
    sources = read_csv(Path(args.sources))
    set_scores = read_csv(Path(args.set_scores)) if args.set_scores and Path(args.set_scores).exists() else []
    human_rows = read_csv(Path(args.human_labels)) if args.human_labels and Path(args.human_labels).exists() else []
    out_dir = Path(args.out_dir)

    aggregates = aggregate_candidate_scores(candidate_scores)
    repeat_rows = repeat_reliability(candidate_scores)
    evidence_rows = evidence_coverage(candidate_scores)
    inter_model_rows = inter_model_reliability(aggregates)
    system_rows = source_system_summary(aggregates, sources, set_scores)
    gold_rows = gold_alignment(aggregates, sources, args.tie_margin)
    performance_rows = performance_alignment(aggregates, sources)
    performance_group_rows = performance_group_alignment(aggregates, sources)
    human_rows_out, human_summary = human_alignment(aggregates, human_rows, args.tie_margin)

    write_csv(
        out_dir / "candidate_scores_aggregated.csv",
        aggregates,
        [
            "judge_run_id", "provider", "model", "candidate_id", "long_video_id", "repeat_count",
            "scored_repeat_count", "abstain_repeat_count", "aggregate_status",
            *[f"{name}_mean" for name in EVIDENCE_DIMENSIONS],
            *[f"{name}_mean" for name in CANDIDATE_DIMENSIONS],
            "confidence_mean", "overall_score_mean", "overall_score_std",
        ],
    )
    write_csv(out_dir / "repeat_reliability.csv", repeat_rows, ["judge_run_id", "repeat_count", "repeat_pair_count", "mean_repeat_spearman"])
    write_csv(
        out_dir / "evidence_coverage.csv",
        evidence_rows,
        [
            "judge_run_id", "score_row_count", "scored_row_count", "abstain_row_count", "row_scoring_coverage",
            "candidate_count", "scored_candidate_count", "candidate_scoring_coverage", "mixed_verdict_candidate_count",
            *[f"mean_{name}" for name in EVIDENCE_DIMENSIONS], "mean_confidence",
        ],
    )
    write_csv(out_dir / "inter_model_reliability.csv", inter_model_rows, ["left_judge_run_id", "right_judge_run_id", "candidate_count", "spearman"])
    write_csv(
        out_dir / "system_judge_summary.csv",
        system_rows,
        ["judge_run_id", "source_system", "long_video_count", "candidate_count", "mean_candidate_score", "mean_top1_score", "mean_set_score"],
    )
    write_csv(
        out_dir / "gold_editorial_alignment.csv",
        gold_rows,
        [
            "judge_run_id", "compared_system", "comparison_count", "gold_win_count", "tie_count", "gold_loss_count",
            "gold_nonloss_rate", "gold_strict_win_rate",
        ],
    )
    write_csv(out_dir / "performance_alignment.csv", performance_rows, ["judge_run_id", "performance_metric", "pair_count", "spearman"])
    write_csv(
        out_dir / "performance_group_alignment.csv",
        performance_group_rows,
        [
            "judge_run_id", "pos_count", "pos_total_count", "pos_scoring_coverage", "pos_mean_score",
            "neg_count", "neg_total_count", "neg_scoring_coverage", "neg_mean_score",
            "mean_score_gap", "pos_over_neg_auc",
        ],
    )
    write_csv(out_dir / "human_alignment.csv", human_rows_out, ["judge_run_id", "labeled_comparison_count", "human_preference_accuracy"])

    repeat_values = [row["mean_repeat_spearman"] for row in repeat_rows if row["mean_repeat_spearman"] is not None]
    human_values = [row["human_preference_accuracy"] for row in human_rows_out if row["human_preference_accuracy"] is not None]
    if not human_values:
        validation_status = "pending_human_labels"
    elif repeat_values and min(human_values) >= args.min_human_accuracy and min(repeat_values) >= args.min_repeat_spearman:
        validation_status = "validated"
    else:
        validation_status = "needs_revision"

    summary = {
        "validation_status": validation_status,
        "gates": {
            "min_human_accuracy": args.min_human_accuracy,
            "min_repeat_spearman": args.min_repeat_spearman,
            "tie_margin": args.tie_margin,
        },
        "counts": {
            "candidate_score_rows": len(candidate_scores),
            "unique_candidates": len({row["candidate_id"] for row in candidate_scores}),
            "judge_runs": len({row["judge_run_id"] for row in candidate_scores}),
            "set_score_rows": len(set_scores),
        },
        "human": human_summary,
        "repeat_reliability": repeat_rows,
        "evidence_coverage": evidence_rows,
        "inter_model_reliability": inter_model_rows,
        "gold_editorial_alignment": gold_rows,
        "performance_alignment": performance_rows,
        "performance_group_alignment": performance_group_rows,
        "limitations": [
            "Transcript and Vpick scene descriptions do not expose final editing, subtitles, audio emphasis, or thumbnail quality.",
            "Views and likes are affected by channel size, upload timing, title, thumbnail, and distribution.",
            "The judge must not be called validated until human preference labels are completed.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "judge_validation_report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
