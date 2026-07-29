from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE = (
    ROOT
    / "deliverables"
    / "2026-07-24"
    / "hard_negative_eval"
    / "hard_negative_pairs_PRIVATE.csv"
)
DEFAULT_SCORES = (
    ROOT
    / "results"
    / "same_longform_hard_negative_v8"
    / "direct_codex"
    / "intrinsic_judge_v8_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "same_longform_hard_negative_v8" / "direct_codex" / "validity"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def to_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def exact_two_sided_binomial_p(successes: int, trials: int, probability: float = 0.5) -> float | None:
    if trials <= 0:
        return None
    observed_probability = (
        math.comb(trials, successes)
        * probability**successes
        * (1 - probability) ** (trials - successes)
    )
    p_value = 0.0
    for count in range(trials + 1):
        outcome_probability = (
            math.comb(trials, count)
            * probability**count
            * (1 - probability) ** (trials - count)
        )
        if outcome_probability <= observed_probability + 1e-12:
            p_value += outcome_probability
    return min(1.0, p_value)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def build_results(
    private_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    private_by_id = {row["candidate_id"]: row for row in private_rows}
    scores_by_id = {row["candidate_id"]: row for row in score_rows}
    if set(private_by_id) != set(scores_by_id):
        missing = sorted(set(private_by_id) - set(scores_by_id))
        extra = sorted(set(scores_by_id) - set(private_by_id))
        raise ValueError(f"Score/private IDs differ; missing={missing}, extra={extra}")

    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in private_rows:
        by_pair[row["eval_pair_id"]].append(row)

    pair_results: list[dict[str, Any]] = []
    for pair_id, members in sorted(by_pair.items()):
        if len(members) != 2:
            raise ValueError(f"{pair_id} must contain exactly two candidates")
        positive = next(
            (row for row in members if row["reference_role"] == "positive"), None
        )
        negative = next(
            (row for row in members if row["reference_role"] == "hard_negative"), None
        )
        if positive is None or negative is None:
            raise ValueError(f"{pair_id} has invalid role composition")
        positive_score_row = scores_by_id[positive["candidate_id"]]
        negative_score_row = scores_by_id[negative["candidate_id"]]
        positive_score = (
            to_float(positive_score_row.get("quality_score_100"))
            if positive_score_row.get("verdict") == "score"
            else None
        )
        negative_score = (
            to_float(negative_score_row.get("quality_score_100"))
            if negative_score_row.get("verdict") == "score"
            else None
        )
        valid = positive_score is not None and negative_score is not None
        margin = positive_score - negative_score if valid else None
        evidence_adequate = all(
            score_row.get("verdict") == "score"
            and int(float(score_row.get("evidence_transcript_intelligibility") or 0)) >= 3
            and int(float(score_row.get("evidence_boundary_observability") or 0)) >= 3
            for score_row in (positive_score_row, negative_score_row)
        )
        pair_results.append(
            {
                "eval_pair_id": pair_id,
                "channel_name": positive["channel_name"],
                "long_video_id": positive["long_video_id"],
                "positive_candidate_id": positive["candidate_id"],
                "hard_negative_candidate_id": negative["candidate_id"],
                "positive_score": positive_score,
                "hard_negative_score": negative_score,
                "score_margin": margin,
                "strict_correct": int(margin > 0) if margin is not None else "",
                "tie_aware_credit": (
                    1.0 if margin is not None and margin > 0
                    else 0.5 if margin == 0
                    else 0.0 if margin is not None
                    else ""
                ),
                "both_candidates_evidence_adequate": int(evidence_adequate),
                "valid_pair": int(valid),
            }
        )

    valid_pairs = [row for row in pair_results if row["valid_pair"] == 1]
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_pairs:
        by_channel[row["channel_name"]].append(row)
    channel_metrics = []
    for channel_name, rows in sorted(by_channel.items()):
        channel_metrics.append(
            {
                "channel_name": channel_name,
                "pair_count": len(rows),
                "strict_pairwise_accuracy": mean(
                    [float(row["strict_correct"]) for row in rows]
                ),
                "tie_aware_pairwise_accuracy": mean(
                    [float(row["tie_aware_credit"]) for row in rows]
                ),
                "mean_score_margin": mean([float(row["score_margin"]) for row in rows]),
            }
        )

    wins = sum(float(row["score_margin"]) > 0 for row in valid_pairs)
    ties = sum(float(row["score_margin"]) == 0 for row in valid_pairs)
    losses = sum(float(row["score_margin"]) < 0 for row in valid_pairs)
    decisive_count = wins + losses
    evidence_adequate_pairs = [
        row for row in valid_pairs if row["both_candidates_evidence_adequate"] == 1
    ]
    summary = {
        "metric_definition": (
            "A pair is correct when the blinded v8 Judge gives the high-performing "
            "published Short source interval a higher quality_score_100 than the "
            "same-longform, similar-duration, known-Short-nonoverlap hard negative."
        ),
        "total_pair_count": len(pair_results),
        "valid_pair_count": len(valid_pairs),
        "abstained_pair_count": len(pair_results) - len(valid_pairs),
        "win_tie_loss": {"win": wins, "tie": ties, "loss": losses},
        "micro_strict_pairwise_accuracy": mean(
            [float(row["strict_correct"]) for row in valid_pairs]
        ),
        "micro_tie_aware_pairwise_accuracy": mean(
            [float(row["tie_aware_credit"]) for row in valid_pairs]
        ),
        "mean_score_margin": mean([float(row["score_margin"]) for row in valid_pairs]),
        "decisive_pair_count": decisive_count,
        "decisive_pairwise_accuracy": wins / decisive_count if decisive_count else None,
        "decisive_accuracy_wilson_95ci": wilson_interval(wins, decisive_count),
        "exact_binomial_two_sided_p_vs_chance": exact_two_sided_binomial_p(
            wins, decisive_count
        ),
        "evidence_adequate_pair_count": len(evidence_adequate_pairs),
        "evidence_adequate_tie_aware_accuracy": mean(
            [float(row["tie_aware_credit"]) for row in evidence_adequate_pairs]
        ),
        "channel_macro_strict_pairwise_accuracy": mean(
            [float(row["strict_pairwise_accuracy"]) for row in channel_metrics]
        ),
        "channel_macro_tie_aware_pairwise_accuracy": mean(
            [float(row["tie_aware_pairwise_accuracy"]) for row in channel_metrics]
        ),
        "channel_metrics": channel_metrics,
        "interpretation": {
            "primary": "micro_tie_aware_pairwise_accuracy",
            "secondary": [
                "micro_strict_pairwise_accuracy",
                "channel_macro_tie_aware_pairwise_accuracy",
                "mean_score_margin",
            ],
            "chance_level": 0.5,
        },
    }
    return pair_results, summary


def report_markdown(summary: dict[str, Any]) -> str:
    def percent(value: Any) -> str:
        return "N/A" if value is None else f"{float(value) * 100:.1f}%"

    lines = [
        "# Same-longform hard-negative Judge validity",
        "",
        "## 설계",
        "",
        "- 정답: 채널 내 고성과 실제 Shorts의 원본 구간",
        "- 대조: 같은 롱폼, 유사 길이, 승인된 60개 데이터의 알려진 Shorts 구간과 비중첩인 Vpick 후보",
        "- 통제: 양쪽 모두 Vpick 장면 설명과 Vpick ASR만 Judge에 제공",
        "- 블라인드: 채널, 조회수, 성과 라벨, 정답 역할은 채점 입력에서 제외",
        "",
        "## 결과",
        "",
        f"- 전체 쌍: {summary['total_pair_count']}",
        f"- 유효 쌍: {summary['valid_pair_count']}",
        f"- abstain 포함 쌍: {summary['abstained_pair_count']}",
        f"- Pairwise accuracy (tie 0.5): {percent(summary['micro_tie_aware_pairwise_accuracy'])}",
        f"- Strict pairwise accuracy: {percent(summary['micro_strict_pairwise_accuracy'])}",
        f"- 승/동률/패: {summary['win_tie_loss']['win']}/{summary['win_tie_loss']['tie']}/{summary['win_tie_loss']['loss']}",
        f"- 동률 제외 정확도: {percent(summary['decisive_pairwise_accuracy'])}",
        f"- 우연 50% 대비 exact binomial p-value: {summary['exact_binomial_two_sided_p_vs_chance']:.4f}",
        f"- 근거 충분 쌍 정확도(tie 0.5): {percent(summary['evidence_adequate_tie_aware_accuracy'])} ({summary['evidence_adequate_pair_count']}쌍)",
        f"- 채널 macro pairwise accuracy: {percent(summary['channel_macro_tie_aware_pairwise_accuracy'])}",
        f"- 평균 점수 차이(pos - hard negative): {summary['mean_score_margin']}",
        "",
        "## 채널별",
        "",
        "| 채널 | 쌍 수 | tie-aware | strict | 평균 margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["channel_metrics"]:
        lines.append(
            f"| {row['channel_name']} | {row['pair_count']} | "
            f"{percent(row['tie_aware_pairwise_accuracy'])} | "
            f"{percent(row['strict_pairwise_accuracy'])} | "
            f"{float(row['mean_score_margin']):.2f} |"
        )
    lines.extend(
        [
            "",
            "이 결과는 기존 저성과 실제 Shorts를 `neg`로 둔 채널 백분위 AUC와 별개다. "
            "여기서는 같은 원본 안에서 실제 고성과 선택 구간을 비선택 후보보다 높게 "
            "평가하는지를 직접 검증한다.",
            "",
            "현재 결과가 우연 수준을 유의하게 넘지 못하므로 v8 Judge를 성과 정답의 대리 평가자로 "
            "확정할 수 없다. 또한 hard negative는 미게시 구간이 아니라 현재 데이터셋의 알려진 "
            "Shorts와 비중첩인 후보이므로, 실제로 좋은 미발견 하이라이트일 가능성을 남긴다.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate same-longform positive vs hard-negative Judge validity."
    )
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_results, summary = build_results(
        read_csv(args.private),
        read_csv(args.scores),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pair_results_PRIVATE.csv", pair_results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        report_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
