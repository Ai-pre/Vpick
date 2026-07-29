from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "deliverables" / "2026-07-24" / "highlight_quality_v1"
DEFAULT_RESULTS = ROOT / "results" / "highlight_quality_judge_v1_full"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def source_score_summary(
    judgments: list[dict[str, Any]],
    candidate_by_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    values: dict[str, list[float]] = defaultdict(list)
    for judgment in judgments:
        score = numeric(judgment.get("highlight_quality_score_100"))
        private = candidate_by_id.get(str(judgment.get("candidate_id")), {})
        source = private.get("candidate_source", "unknown")
        if score is not None:
            values[source].append(score)
    rows = []
    for source, scores in sorted(values.items()):
        rows.append(
            {
                "candidate_source": source,
                "count": len(scores),
                "mean_score": round(statistics.fmean(scores), 4),
                "median_score": round(statistics.median(scores), 4),
                "min_score": min(scores),
                "max_score": max(scores),
            }
        )
    return rows


def pointwise_pair_rows(
    judgments: list[dict[str, Any]],
    private_pairs: list[dict[str, str]],
    candidate_by_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    score_by_id = {
        str(row.get("candidate_id")): numeric(row.get("highlight_quality_score_100"))
        for row in judgments
    }
    output = []
    for pair in private_pairs:
        anchor_id = pair["published_anchor_id"]
        a_id = pair["candidate_a_id"]
        b_id = pair["candidate_b_id"]
        alternative_id = b_id if a_id == anchor_id else a_id
        anchor_score = score_by_id.get(anchor_id)
        alternative_score = score_by_id.get(alternative_id)
        if anchor_score is None or alternative_score is None:
            continue
        delta = round(anchor_score - alternative_score, 4)
        output.append(
            {
                "pair_id": pair["pair_id"],
                "longform_id": pair["longform_id"],
                "published_candidate_id": anchor_id,
                "alternative_candidate_id": alternative_id,
                "alternative_source": candidate_by_id.get(alternative_id, {}).get(
                    "candidate_source",
                    "unknown",
                ),
                "published_score": anchor_score,
                "alternative_score": alternative_score,
                "score_delta": delta,
                "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie",
            }
        )
    return output


def comparison_summary(
    rows: list[dict[str, Any]],
    source_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(source_key) or "unknown")].append(row)
    output = []
    for source, group in sorted(grouped.items()):
        wins = sum(row["outcome"] == "win" for row in group)
        ties = sum(row["outcome"] == "tie" for row in group)
        losses = sum(row["outcome"] == "loss" for row in group)
        deltas = [
            float(row["score_delta"])
            for row in group
            if numeric(row.get("score_delta")) is not None
        ]
        output.append(
            {
                "alternative_source": source,
                "count": len(group),
                "published_wins": wins,
                "ties": ties,
                "published_losses": losses,
                "strict_win_rate": rate(wins, len(group)),
                "non_loss_rate": rate(wins + ties, len(group)),
                "mean_score_delta": (
                    round(statistics.fmean(deltas), 4) if deltas else None
                ),
            }
        )
    return output


def pairwise_preference_rows(
    judgments: list[dict[str, Any]],
    private_pair_by_id: dict[str, dict[str, str]],
    candidate_by_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for judgment in judgments:
        pair = private_pair_by_id.get(str(judgment.get("pair_id")))
        if not pair:
            continue
        anchor_id = pair["published_anchor_id"]
        a_id = pair["candidate_a_id"]
        b_id = pair["candidate_b_id"]
        alternative_id = b_id if a_id == anchor_id else a_id
        winner = str(judgment.get("winner") or "")
        if winner == "tie":
            outcome = "tie"
        elif winner == "A":
            outcome = "win" if a_id == anchor_id else "loss"
        elif winner == "B":
            outcome = "win" if b_id == anchor_id else "loss"
        else:
            outcome = "invalid"
        output.append(
            {
                "pair_id": pair["pair_id"],
                "longform_id": pair["longform_id"],
                "published_candidate_id": anchor_id,
                "alternative_candidate_id": alternative_id,
                "alternative_source": candidate_by_id.get(alternative_id, {}).get(
                    "candidate_source",
                    "unknown",
                ),
                "winner": winner,
                "outcome": outcome,
                "order_inconsistent": bool(judgment.get("order_inconsistent")),
                "confidence_1_5": judgment.get("confidence_1_5", ""),
            }
        )
    return output


def consensus_judgment_rows(
    judgments: list[dict[str, Any]],
    private_pair_by_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for judgment in judgments:
        pair = private_pair_by_id.get(str(judgment.get("pair_id")))
        if not pair:
            continue
        winner_ab = str(judgment.get("winner") or "")
        winner_ba = str(judgment.get("swapped_winner_restored") or "")
        accepted = bool(winner_ba) and winner_ab == winner_ba
        consensus_winner = winner_ab if accepted else "abstain"
        selected_candidate_id = ""
        if consensus_winner == "A":
            selected_candidate_id = pair["candidate_a_id"]
        elif consensus_winner == "B":
            selected_candidate_id = pair["candidate_b_id"]
        output.append(
            {
                "pair_id": pair["pair_id"],
                "longform_id": pair["longform_id"],
                "candidate_a_id": pair["candidate_a_id"],
                "candidate_b_id": pair["candidate_b_id"],
                "winner_ab": winner_ab,
                "winner_ba_restored": winner_ba,
                "consensus_status": (
                    "accepted" if accepted else "abstain_order_inconsistent"
                ),
                "consensus_winner": consensus_winner,
                "selected_candidate_id": selected_candidate_id,
                "confidence_ab_1_5": judgment.get("confidence_1_5", ""),
            }
        )
    return output


def pairwise_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["alternative_source"]].append(row)
    output = []
    for source, group in sorted(grouped.items()):
        valid = [row for row in group if row["outcome"] != "invalid"]
        consistent = [row for row in valid if not row["order_inconsistent"]]
        wins = sum(row["outcome"] == "win" for row in consistent)
        ties = sum(row["outcome"] == "tie" for row in consistent)
        losses = sum(row["outcome"] == "loss" for row in consistent)
        inconsistent = sum(bool(row["order_inconsistent"]) for row in valid)
        output.append(
            {
                "alternative_source": source,
                "count": len(group),
                "valid_count": len(valid),
                "consistent_count": len(consistent),
                "order_abstain_count": inconsistent,
                "published_wins": wins,
                "ties": ties,
                "published_losses": losses,
                "strict_win_rate": rate(wins, len(consistent)),
                "non_loss_rate": rate(wins + ties, len(consistent)),
                "order_consistency_rate": rate(len(valid) - inconsistent, len(valid)),
            }
        )
    return output


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_결과 없음_"
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Highlight Quality v1 pointwise and pairwise judgments."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    args = parser.parse_args()

    suffix = args.model.replace("/", "_")
    pointwise = read_jsonl(
        args.results_dir / f"pointwise_{suffix}_judgments.jsonl"
    )
    pairwise = read_jsonl(
        args.results_dir / f"pairwise_{suffix}_judgments.jsonl"
    )
    private_candidates = read_csv(args.eval_dir / "candidate_sources_PRIVATE.csv")
    private_pairs = read_csv(args.eval_dir / "pair_sources_PRIVATE.csv")
    candidate_by_id = {row["candidate_id"]: row for row in private_candidates}
    private_pair_by_id = {row["pair_id"]: row for row in private_pairs}

    source_scores = source_score_summary(pointwise, candidate_by_id)
    pointwise_pairs = pointwise_pair_rows(
        pointwise,
        private_pairs,
        candidate_by_id,
    )
    pointwise_comparison = comparison_summary(
        pointwise_pairs,
        "alternative_source",
    )
    pairwise_rows = pairwise_preference_rows(
        pairwise,
        private_pair_by_id,
        candidate_by_id,
    )
    consensus_rows = consensus_judgment_rows(pairwise, private_pair_by_id)
    pairwise_comparison = pairwise_summary(pairwise_rows)
    pairwise_valid = [row for row in pairwise_rows if row["outcome"] != "invalid"]
    pairwise_consistent = [
        row for row in pairwise_valid if not row["order_inconsistent"]
    ]
    overall_pairwise = {
        "valid_count": len(pairwise_valid),
        "consistent_count": len(pairwise_consistent),
        "order_abstain_count": len(pairwise_valid) - len(pairwise_consistent),
        "order_consistency_rate": rate(
            len(pairwise_consistent),
            len(pairwise_valid),
        ),
    }

    write_csv(args.results_dir / "pointwise_scored_PRIVATE.csv", pointwise_pairs)
    write_csv(args.results_dir / "pairwise_scored_PRIVATE.csv", pairwise_rows)
    write_jsonl(args.results_dir / "pairwise_consensus.jsonl", consensus_rows)
    summary = {
        "model": args.model,
        "pointwise_judgment_count": len(pointwise),
        "pairwise_judgment_count": len(pairwise),
        "source_score_summary": source_scores,
        "pointwise_published_vs_alternative": pointwise_comparison,
        "pairwise_overall": overall_pairwise,
        "pairwise_published_vs_alternative": pairwise_comparison,
    }
    (args.results_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = "\n".join(
        [
            "# Highlight Quality Judge v1 평가 결과",
            "",
            f"- 모델: `{args.model}`",
            f"- Pointwise 완료: {len(pointwise)}",
            f"- Pairwise 완료: {len(pairwise)}",
            "",
            "## 후보 출처별 Pointwise 점수",
            "",
            markdown_table(source_scores),
            "",
            "## Pointwise: 실제 숏폼 대 대안 후보",
            "",
            markdown_table(pointwise_comparison),
            "",
            "## Pairwise: 실제 숏폼 대 대안 후보",
            "",
            markdown_table(pairwise_comparison),
            "",
            "## 판정",
            "",
            f"- Pairwise A/B 순서 일치: {overall_pairwise['consistent_count']}/"
            f"{overall_pairwise['valid_count']} "
            f"({overall_pairwise['order_consistency_rate']})",
            f"- 순서 불일치로 abstain: {overall_pairwise['order_abstain_count']}",
            "- Pointwise는 random 후보를 어느 정도 구별했지만 boundary shift와 hard negative "
            "분리는 약했다.",
            "- Pairwise는 boundary shift 구별이 개선됐지만 순서 일치율이 낮아 현재 모델을 "
            "주 Judge로 확정할 수 없다.",
            "- hard negative는 비게시·비중첩 구간일 뿐 실제 저품질 정답이 아니므로 해당 "
            "승패를 정확도로 해석하지 않는다.",
            "- Vpick 및 existing model은 각각 2개뿐이므로 시스템 우열을 결론 내리지 않는다.",
            "",
            "성과가 검증된 published short는 신뢰할 수 있는 정답 신호이지만 완전한 정답은 "
            "아니다. 따라서 승률은 Judge 진단 지표로 해석하며 강제 정답 정확도로 부르지 않는다.",
            "",
        ]
    )
    (args.results_dir / "EVALUATION_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
