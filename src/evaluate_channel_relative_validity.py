from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    return pearson(average_ranks(left), average_ranks(right))


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def rounded(value: float | None, digits: int = 4) -> float | str:
    return "" if value is None else round(value, digits)


def quality_band(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 50:
        return "adequate"
    return "weak"


def alignment_case(tier: str, band: str) -> str:
    if tier == "relative_high" and band == "strong":
        return "aligned_high_quality"
    if tier == "relative_low" and band == "weak":
        return "aligned_low_quality"
    if tier == "relative_low" and band == "strong":
        return "high_quality_relative_underperformer"
    if tier == "relative_high" and band == "weak":
        return "low_quality_relative_outperformer"
    return "mixed_or_nondiagnostic"


def render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 채널 상대 성과 정합성 진단",
            "",
            "## 해석 원칙",
            "",
            "- Judge 점수는 숏폼의 내재적 편집 품질이며 실제 성과 라벨이 아니다.",
            "- relative_high/relative_low는 같은 채널 내 상대 성과이며 좋은/나쁜 영상의 절대 정답이 아니다.",
            "- 주 외부 진단은 채널별 AUC와 그 Macro 평균이다. 전체 pooled AUC는 참고값이다.",
            "- 성과 정합성은 Judge의 예측 타당성 보조 신호이며 콘텐츠 타당성의 단독 합격 기준이 아니다.",
            "",
            "## 결과",
            "",
            f"- scored candidates: {summary['scored_candidate_count']}",
            f"- eligible channels: {summary['eligible_channel_count']}",
            f"- stable channels (각 tier 3개 이상): {summary['stable_channel_count']}",
            f"- macro channel AUC: {summary['macro_channel_auc_all']}",
            f"- stable-channel macro AUC: {summary['macro_channel_auc_stable']}",
            f"- micro within-channel pairwise half-credit: {summary['micro_pairwise_half_credit']}",
            f"- within-channel centered Pearson: {summary['within_channel_centered_pearson']}",
            f"- pooled AUC (supplementary): {summary['pooled_auc_supplementary']}",
            "",
            "## 판정",
            "",
            "이 보고서만으로 Judge를 validated 또는 invalidated로 선언하지 않는다. "
            "반복 실행·교차 모델 일치도와 소규모 인간 기준 정합성을 별도로 확인한다.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate intrinsic Judge scores against within-channel relative performance."
    )
    parser.add_argument("--scores", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-field", default="quality_score_100")
    parser.add_argument("--score-name", default="intrinsic_quality")
    parser.add_argument(
        "--allow-missing-dataset-ids",
        action="store_true",
        help="Keep score rows that cannot be joined out of the analysis.",
    )
    args = parser.parse_args()

    score_rows = read_csv(Path(args.scores))
    score_ids = [row["candidate_id"] for row in score_rows]
    if len(score_ids) != len(set(score_ids)):
        raise ValueError("Score file contains duplicate candidate_id values")
    dataset_rows = read_csv(Path(args.dataset))
    dataset_ids = [row["candidate_id"] for row in dataset_rows]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Dataset contains duplicate candidate_id values")
    dataset = {row["candidate_id"]: row for row in dataset_rows}

    joined: list[dict[str, Any]] = []
    missing_dataset_ids: list[str] = []
    for score_row in score_rows:
        candidate_id = score_row.get("candidate_id", "")
        metadata = dataset.get(candidate_id)
        if metadata is None:
            missing_dataset_ids.append(candidate_id)
            continue
        score = number(score_row.get(args.score_field))
        if score is None or score_row.get("verdict", "score") == "abstain":
            continue
        tier_source = (
            metadata.get("relative_performance_tier")
            or metadata.get("performance_label", "")
        )
        tier = {
            "pos": "relative_high",
            "neg": "relative_low",
            "relative_high": "relative_high",
            "relative_low": "relative_low",
        }.get(tier_source, "unclassified")
        percentile = number(metadata.get("channel_performance_percentile"))
        band = quality_band(score)
        joined.append(
            {
                "candidate_id": candidate_id,
                "channel_name": metadata.get("channel_name", ""),
                "relative_performance_tier": tier,
                "channel_performance_percentile": (
                    rounded(percentile, 3) if percentile is not None else ""
                ),
                "judge_score_name": args.score_name,
                "judge_score_100": rounded(score),
                "quality_band_provisional": band,
                "alignment_case": alignment_case(tier, band),
                "short_video_id": metadata.get("short_video_id", ""),
                "short_views": metadata.get("short_views", ""),
                "short_likes": metadata.get("short_likes", ""),
            }
        )

    if missing_dataset_ids and not args.allow_missing_dataset_ids:
        preview = ", ".join(missing_dataset_ids[:10])
        raise ValueError(
            f"{len(missing_dataset_ids)} score candidate IDs are missing from "
            f"the dataset: {preview}"
        )

    eligible = [
        row
        for row in joined
        if row["relative_performance_tier"] in {"relative_high", "relative_low"}
    ]
    channels = sorted({row["channel_name"] for row in eligible})
    channel_rows: list[dict[str, Any]] = []
    all_pair_correct = all_pair_tied = all_pair_wrong = 0
    centered_scores: list[float] = []
    centered_percentiles: list[float] = []

    for channel in channels:
        group = [row for row in eligible if row["channel_name"] == channel]
        high = [
            row for row in group
            if row["relative_performance_tier"] == "relative_high"
        ]
        low = [
            row for row in group
            if row["relative_performance_tier"] == "relative_low"
        ]
        if not high or not low:
            continue
        labels = [
            int(row["relative_performance_tier"] == "relative_high")
            for row in group
        ]
        scores = [float(row["judge_score_100"]) for row in group]
        percentile_pairs = [
            (
                float(row["judge_score_100"]),
                number(row["channel_performance_percentile"]),
            )
            for row in group
            if number(row["channel_performance_percentile"]) is not None
        ]
        channel_score_mean = statistics.mean(score for score, _ in percentile_pairs)
        channel_percentile_mean = statistics.mean(
            float(percentile) for _, percentile in percentile_pairs
        )
        centered_scores.extend(
            score - channel_score_mean for score, _ in percentile_pairs
        )
        centered_percentiles.extend(
            float(percentile) - channel_percentile_mean
            for _, percentile in percentile_pairs
        )

        correct = tied = wrong = 0
        for high_row in high:
            for low_row in low:
                high_score = float(high_row["judge_score_100"])
                low_score = float(low_row["judge_score_100"])
                if high_score > low_score:
                    correct += 1
                elif high_score == low_score:
                    tied += 1
                else:
                    wrong += 1
        pair_count = correct + tied + wrong
        all_pair_correct += correct
        all_pair_tied += tied
        all_pair_wrong += wrong
        high_mean = statistics.mean(float(row["judge_score_100"]) for row in high)
        low_mean = statistics.mean(float(row["judge_score_100"]) for row in low)
        channel_rows.append(
            {
                "channel_name": channel,
                "relative_high_count": len(high),
                "relative_low_count": len(low),
                "stable_channel": len(high) >= 3 and len(low) >= 3,
                "relative_high_mean_100": rounded(high_mean),
                "relative_low_mean_100": rounded(low_mean),
                "high_minus_low_100": rounded(high_mean - low_mean),
                "channel_auc": rounded(roc_auc(labels, scores)),
                "channel_spearman_score_percentile": rounded(
                    spearman(
                        [item[0] for item in percentile_pairs],
                        [float(item[1]) for item in percentile_pairs],
                    )
                ),
                "pair_count": pair_count,
                "pair_correct": correct,
                "pair_tied": tied,
                "pair_wrong": wrong,
                "pairwise_strict_accuracy": rounded(correct / pair_count),
                "pairwise_half_credit": rounded((correct + 0.5 * tied) / pair_count),
            }
        )

    auc_all = [
        float(row["channel_auc"])
        for row in channel_rows
        if row["channel_auc"] != ""
    ]
    auc_stable = [
        float(row["channel_auc"])
        for row in channel_rows
        if row["stable_channel"] and row["channel_auc"] != ""
    ]
    pooled_labels = [
        int(row["relative_performance_tier"] == "relative_high")
        for row in eligible
    ]
    pooled_scores = [float(row["judge_score_100"]) for row in eligible]
    total_pairs = all_pair_correct + all_pair_tied + all_pair_wrong
    summary = {
        "score_name": args.score_name,
        "score_field": args.score_field,
        "input_score_row_count": len(score_rows),
        "scored_candidate_count": len(eligible),
        "missing_dataset_candidate_ids": missing_dataset_ids,
        "eligible_channel_count": len(channel_rows),
        "stable_channel_count": sum(
            bool(row["stable_channel"]) for row in channel_rows
        ),
        "macro_channel_auc_all": rounded(
            statistics.mean(auc_all) if auc_all else None
        ),
        "macro_channel_auc_stable": rounded(
            statistics.mean(auc_stable) if auc_stable else None
        ),
        "micro_pair_count": total_pairs,
        "micro_pairwise_strict_accuracy": rounded(
            all_pair_correct / total_pairs if total_pairs else None
        ),
        "micro_pairwise_half_credit": rounded(
            (all_pair_correct + 0.5 * all_pair_tied) / total_pairs
            if total_pairs
            else None
        ),
        "within_channel_centered_pearson": rounded(
            pearson(centered_scores, centered_percentiles)
        ),
        "pooled_auc_supplementary": rounded(
            roc_auc(pooled_labels, pooled_scores)
        ),
        "interpretation": {
            "primary_external_diagnostic": "macro_channel_auc_all",
            "pooled_auc": "supplementary_only",
            "relative_low": (
                "channel-relative underperformance, not bad-content ground truth"
            ),
            "judge_validity": (
                "requires reliability and human/cross-model content alignment separately"
            ),
        },
    }

    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "channel_relative_metrics.csv",
        channel_rows,
        [
            "channel_name",
            "relative_high_count",
            "relative_low_count",
            "stable_channel",
            "relative_high_mean_100",
            "relative_low_mean_100",
            "high_minus_low_100",
            "channel_auc",
            "channel_spearman_score_percentile",
            "pair_count",
            "pair_correct",
            "pair_tied",
            "pair_wrong",
            "pairwise_strict_accuracy",
            "pairwise_half_credit",
        ],
    )
    write_csv(
        out_dir / "candidate_alignment_diagnostics.csv",
        joined,
        [
            "candidate_id",
            "channel_name",
            "relative_performance_tier",
            "channel_performance_percentile",
            "judge_score_name",
            "judge_score_100",
            "quality_band_provisional",
            "alignment_case",
            "short_video_id",
            "short_views",
            "short_likes",
        ],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "channel_relative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "channel_relative_report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
