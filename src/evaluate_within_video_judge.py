from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def random_hit_at_k(candidate_count: int, gold_count: int, k: int) -> float:
    k = min(k, candidate_count)
    if gold_count <= 0:
        return 0.0
    if candidate_count - gold_count < k:
        return 1.0
    return 1.0 - (
        math.comb(candidate_count - gold_count, k)
        / math.comb(candidate_count, k)
    )


def expected_random_mrr(candidate_count: int, gold_count: int) -> float:
    positions = range(1, candidate_count + 1)
    combinations = list(itertools.combinations(positions, gold_count))
    return mean([1.0 / min(choice) for choice in combinations]) or 0.0


def tie_aware_metrics(
    scores: list[tuple[float, bool]],
    *,
    k_values: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    if not scores or not any(is_gold for _, is_gold in scores):
        raise ValueError("A pool must contain at least one gold-equivalent candidate")
    best_gold_score = max(score for score, is_gold in scores if is_gold)
    better = sum(score > best_gold_score for score, _ in scores)
    tied = [(score, is_gold) for score, is_gold in scores if score == best_gold_score]
    tied_gold = sum(is_gold for _, is_gold in tied)
    tied_non_gold = len(tied) - tied_gold

    output: dict[str, float] = {}
    for k in k_values:
        slots = max(0, min(len(tied), k - better))
        if slots <= 0:
            credit = 0.0
        elif slots >= len(tied):
            credit = 1.0
        else:
            credit = 1.0 - (
                math.comb(tied_non_gold, slots) / math.comb(len(tied), slots)
                if tied_non_gold >= slots
                else 0.0
            )
        output[f"hit_at_{k}"] = credit

    reciprocal_ranks: list[float] = []
    for gold_positions in itertools.combinations(range(1, len(tied) + 1), tied_gold):
        reciprocal_ranks.append(1.0 / (better + min(gold_positions)))
    output["mrr"] = mean(reciprocal_ranks) or 0.0
    output["best_gold_rank_min"] = float(better + 1)
    output["best_gold_rank_max"] = float(
        better + len(tied) - tied_gold + 1
    )
    return output


def aggregate_scores(
    score_rows: list[dict[str, str]],
    score_field: str,
) -> tuple[dict[str, float], set[str]]:
    values: dict[str, list[float]] = defaultdict(list)
    abstained: set[str] = set()
    for row in score_rows:
        candidate_id = row["candidate_id"]
        if row.get("verdict", "score") == "abstain":
            abstained.add(candidate_id)
            continue
        parsed = number(row.get(score_field))
        if parsed is not None:
            values[candidate_id].append(parsed)
    return (
        {
            candidate_id: statistics.mean(candidate_scores)
            for candidate_id, candidate_scores in values.items()
        },
        abstained,
    )


def summarize_pool_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_pool"]]
    return {
        "pool_count": len(rows),
        "valid_pool_count": len(valid),
        "invalid_pool_count": len(rows) - len(valid),
        "judge_hit_at_1": mean([row["judge_hit_at_1"] for row in valid]),
        "judge_hit_at_3": mean([row["judge_hit_at_3"] for row in valid]),
        "judge_mrr": mean([row["judge_mrr"] for row in valid]),
        "vpick_baseline_hit_at_1": mean(
            [row["vpick_baseline_hit_at_1"] for row in valid]
        ),
        "vpick_baseline_hit_at_3": mean(
            [row["vpick_baseline_hit_at_3"] for row in valid]
        ),
        "vpick_baseline_mrr": mean(
            [row["vpick_baseline_mrr"] for row in valid]
        ),
        "random_hit_at_1": mean([row["random_hit_at_1"] for row in valid]),
        "random_hit_at_3": mean([row["random_hit_at_3"] for row in valid]),
        "random_mrr": mean([row["random_mrr"] for row in valid]),
    }


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = 20260728,
    iterations: int = 20_000,
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(iterations)
    )
    lower = means[int(0.025 * iterations)]
    upper = means[min(iterations - 1, int(0.975 * iterations))]
    return [lower, upper]


def exact_random_position_p_value(
    pools: list[list[tuple[float, bool]]],
    observed_values: list[float],
    metric: str,
) -> float:
    distribution: dict[Fraction, float] = {Fraction(0): 1.0}
    for pool in pools:
        candidate_values: list[Fraction] = []
        for candidate_index in range(len(pool)):
            hypothetical = [
                (score, index == candidate_index)
                for index, (score, _is_gold) in enumerate(pool)
            ]
            value = tie_aware_metrics(hypothetical)[metric]
            candidate_values.append(Fraction(value).limit_denominator(10_000))
        probability = 1.0 / len(candidate_values)
        updated: dict[Fraction, float] = defaultdict(float)
        for running_sum, running_probability in distribution.items():
            for value in candidate_values:
                updated[running_sum + value] += running_probability * probability
        distribution = dict(updated)

    observed = Fraction(sum(observed_values)).limit_denominator(10_000)
    return sum(
        probability
        for total, probability in distribution.items()
        if total >= observed
    )


def evaluate(
    targets: list[dict[str, str]],
    scores: list[dict[str, str]],
    *,
    score_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores_by_id, abstained = aggregate_scores(scores, score_field)
    by_pool: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in targets:
        by_pool[row["pool_id"]].append(row)

    pool_rows: list[dict[str, Any]] = []
    null_pools: list[list[tuple[float, bool]]] = []
    for pool_id, members in sorted(by_pool.items()):
        missing = [
            row["candidate_id"]
            for row in members
            if row["candidate_id"] not in scores_by_id
        ]
        gold_count = sum(
            int(row["is_gold_equivalent_PRIVATE"]) for row in members
        )
        valid = not missing and gold_count > 0
        base = {
            "pool_id": pool_id,
            "source_candidate_id_PRIVATE": members[0][
                "source_candidate_id_PRIVATE"
            ],
            "channel_name_PRIVATE": members[0]["channel_name_PRIVATE"],
            "performance_label_PRIVATE": members[0][
                "performance_label_PRIVATE"
            ],
            "longform_id": members[0]["longform_id"],
            "candidate_count": len(members),
            "gold_equivalent_count": gold_count,
            "missing_or_abstained_candidate_count": len(missing),
            "valid_pool": int(valid),
        }
        if not valid:
            pool_rows.append(
                {
                    **base,
                    "invalid_reason": (
                        "missing_or_abstained_scores"
                        if missing
                        else "no_gold_equivalent"
                    ),
                }
            )
            continue

        pool_scores = [
            (
                scores_by_id[row["candidate_id"]],
                bool(int(row["is_gold_equivalent_PRIVATE"])),
            )
            for row in members
        ]
        judge = tie_aware_metrics(pool_scores)
        null_pools.append(pool_scores)
        baseline = tie_aware_metrics(
            [
                (
                    -float(row["baseline_rank"]),
                    bool(int(row["is_gold_equivalent_PRIVATE"])),
                )
                for row in members
            ]
        )
        pool_rows.append(
            {
                **base,
                "invalid_reason": "",
                "judge_hit_at_1": judge["hit_at_1"],
                "judge_hit_at_3": judge["hit_at_3"],
                "judge_mrr": judge["mrr"],
                "judge_best_gold_rank_min": judge["best_gold_rank_min"],
                "judge_best_gold_rank_max": judge["best_gold_rank_max"],
                "vpick_baseline_hit_at_1": baseline["hit_at_1"],
                "vpick_baseline_hit_at_3": baseline["hit_at_3"],
                "vpick_baseline_mrr": baseline["mrr"],
                "random_hit_at_1": random_hit_at_k(
                    len(members),
                    gold_count,
                    1,
                ),
                "random_hit_at_3": random_hit_at_k(
                    len(members),
                    gold_count,
                    3,
                ),
                "random_mrr": expected_random_mrr(
                    len(members),
                    gold_count,
                ),
            }
        )

    valid_rows = [row for row in pool_rows if row["valid_pool"]]
    uncertainty = {
        metric: {
            "bootstrap_95_ci": bootstrap_mean_ci(
                [float(row[f"judge_{metric}"]) for row in valid_rows]
            ),
            "one_sided_exact_random_position_p": exact_random_position_p_value(
                null_pools,
                [float(row[f"judge_{metric}"]) for row in valid_rows],
                metric,
            ),
        }
        for metric in ("hit_at_1", "hit_at_3", "mrr")
    }
    summary = {
        "experiment": "exp2_within_video_segment_alignment",
        "score_field": score_field,
        "overall": summarize_pool_rows(pool_rows),
        "by_performance_bucket_PRIVATE": {
            label: summarize_pool_rows(
                [
                    row
                    for row in pool_rows
                    if row["performance_label_PRIVATE"] == label
                ]
            )
            for label in ("pos", "mid", "neg")
        },
        "tie_policy": (
            "Expected credit under a random ordering inside the best-gold score "
            "tie group; no candidate_id tie-break is credited as model skill."
        ),
        "statistical_uncertainty": uncertainty,
        "interpretation": {
            "primary": ["judge_hit_at_1", "judge_hit_at_3", "judge_mrr"],
            "required_controls": [
                "vpick_baseline",
                "exact_random_baseline_by_pool_size",
            ],
            "asymmetry": (
                "High alignment is strong evidence. Low alignment is weak "
                "counter-evidence because unselected Vpick candidates can still be good."
            ),
            "vpick_baseline_limitation": (
                "All evaluated pools are Vpick-miss cases in which no stored Vpick "
                "candidate reached the gold IoU threshold and the gold interval was "
                "injected after the Vpick candidates. The zero Vpick Hit@K is therefore "
                "descriptive of this selected subset, not an unbiased head-to-head "
                "estimate."
            ),
        },
        "score_candidate_count": len(scores_by_id),
        "abstained_candidate_count": len(abstained),
    }
    return pool_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(summary: dict[str, Any]) -> str:
    overall = summary["overall"]

    def pct(value: Any) -> str:
        return "N/A" if value is None else f"{float(value) * 100:.1f}%"

    return "\n".join(
        [
            "# Judge 구간 정합성",
            "",
            "같은 롱폼의 Vpick 자동 후보와 실제 채택 구간을 동일한 방식으로 "
            "조립하고, 후보별 독립 pointwise 점수를 정렬했다.",
            "",
            f"- 유효 풀: {overall['valid_pool_count']}/{overall['pool_count']}",
            f"- Judge Hit@1: {pct(overall['judge_hit_at_1'])}",
            f"- Judge Hit@3: {pct(overall['judge_hit_at_3'])}",
            f"- Judge MRR: {overall['judge_mrr']}",
            (
                "- Judge Hit@3 95% bootstrap CI: "
                f"{summary['statistical_uncertainty']['hit_at_3']['bootstrap_95_ci']}"
            ),
            (
                "- Judge Hit@3 exact random-position p: "
                f"{summary['statistical_uncertainty']['hit_at_3']['one_sided_exact_random_position_p']}"
            ),
            f"- Vpick 원래 순서 Hit@1: {pct(overall['vpick_baseline_hit_at_1'])}",
            f"- Vpick 원래 순서 Hit@3: {pct(overall['vpick_baseline_hit_at_3'])}",
            f"- 풀 크기별 정확 우연 Hit@1: {pct(overall['random_hit_at_1'])}",
            f"- 풀 크기별 정확 우연 Hit@3: {pct(overall['random_hit_at_3'])}",
            "",
            "높은 일치도는 강한 증거지만, 낮은 일치도는 약한 반증이다. "
            "Vpick 후보는 오답이 아니라 편집자가 선택하지 않은 후보이기 때문이다.",
            "",
            "이번 16개 풀은 모두 Vpick 후보가 gold IoU 기준을 통과하지 못해 "
            "gold를 별도로 주입한 Vpick-miss 표본이다. 따라서 Vpick 0%는 "
            "비편향 성능 비교값이 아니라 이 표본의 구성 특성이다.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate within-video segment alignment with exact controls."
    )
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-field", default="judge_score_100")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool_rows, summary = evaluate(
        read_csv(args.targets),
        read_csv(args.scores),
        score_field=args.score_field,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "within_video_pool_metrics_PRIVATE.csv", pool_rows)
    (args.output_dir / "within_video_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "within_video_report.md").write_text(
        report(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
