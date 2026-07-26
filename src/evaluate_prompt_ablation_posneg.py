from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CHECK_NAMES = (
    "hook_within_3s",
    "surprise_or_twist",
    "emotional_peak",
    "quotable_moment",
    "payoff_or_conclusion",
    "natural_start",
    "natural_end",
)


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return statistics.mean(valid) if valid else None


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    x_var = sum((value - x_mean) ** 2 for value in xs)
    y_var = sum((value - y_mean) ** 2 for value in ys)
    if x_var == 0 or y_var == 0:
        return None
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / math.sqrt(x_var * y_var)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def pairwise_concordance(rows: list[dict[str, Any]]) -> float | None:
    wins = 0.0
    pairs = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            target_diff = left["percentile"] - right["percentile"]
            if target_diff == 0:
                continue
            score_diff = left["score"] - right["score"]
            pairs += 1
            if score_diff == 0:
                wins += 0.5
            elif target_diff * score_diff > 0:
                wins += 1.0
    return wins / pairs if pairs else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def primary_score(version: str, judgment: dict[str, Any]) -> dict[str, float] | None:
    if judgment.get("verdict") != "score":
        return None
    if version == "v1":
        saliency = number(judgment.get("saliency_market_1_5"))
        checks = judgment.get("checks") or {}
        check_score = mean(number(checks.get(name)) for name in CHECK_NAMES)
        if saliency is None or check_score is None:
            return None
        saliency_100 = (saliency - 1.0) / 4.0 * 100.0
        checks_100 = check_score / 2.0 * 100.0
        return {
            "primary_score": 0.5 * saliency_100 + 0.5 * checks_100,
            "saliency_score_100": saliency_100,
            "checks_score_100": checks_100,
        }
    if version in {"v2", "v3"}:
        saliency_100 = number(judgment.get("saliency_market_0_100"))
        checks = judgment.get("checks") or {}
        check_score = mean(number(checks.get(name)) for name in CHECK_NAMES)
        if saliency_100 is None or check_score is None:
            return None
        checks_100 = check_score / 4.0 * 100.0
        return {
            "primary_score": 0.5 * saliency_100 + 0.5 * checks_100,
            "saliency_score_100": saliency_100,
            "checks_score_100": checks_100,
        }
    if version == "v4":
        direct = number(judgment.get("channel_percentile_0_100"))
        if direct is None:
            return None
        return {"primary_score": direct}
    if version == "v5":
        stop = number(judgment.get("p_stop"))
        watch = number(judgment.get("p_watch"))
        share = number(judgment.get("p_share"))
        if stop is None or watch is None or share is None:
            return None
        return {
            "primary_score": stop * watch * share / 10000.0,
            "p_stop": stop,
            "p_watch": watch,
            "p_share": share,
        }
    raise ValueError(f"unsupported version: {version}")


def rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt ablations using private Pos/Neg labels."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--run", action="append", required=True, help="VERSION=JSONL")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with Path(args.labels).open("r", encoding="utf-8-sig", newline="") as handle:
        label_rows = list(csv.DictReader(handle))
    labels = {
        row["source_candidate_id"]: {
            "label": row["performance_label_PRIVATE"].strip().lower(),
            "channel": row["channel_name"],
            "percentile": float(row["channel_performance_percentile_PRIVATE"]),
        }
        for row in label_rows
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    channel_metrics: list[dict[str, Any]] = []

    for spec in args.run:
        version, path_text = spec.split("=", 1)
        judgments = read_jsonl(Path(path_text))
        scored: list[dict[str, Any]] = []
        abstain_count = 0
        invalid_count = 0
        for judgment in judgments:
            candidate_id = str(judgment.get("candidate_id", ""))
            target = labels.get(candidate_id)
            if target is None:
                invalid_count += 1
                continue
            score_parts = primary_score(version, judgment)
            if score_parts is None:
                abstain_count += 1
                continue
            row = {
                "version": version,
                "candidate_id": candidate_id,
                "channel": target["channel"],
                "label": target["label"],
                "percentile": target["percentile"],
                "score": score_parts["primary_score"],
                **score_parts,
            }
            scored.append(row)
            diagnostics.append(row)

        scores = [row["score"] for row in scored]
        binary = [1 if row["label"] == "pos" else 0 for row in scored]
        counts = Counter(scores)
        by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            by_channel[row["channel"]].append(row)
            by_cell[(row["channel"], row["label"])].append(row)

        per_channel_auc: list[float] = []
        stable_channel_auc: list[float] = []
        per_channel_spearman: list[float] = []
        for channel, rows in sorted(by_channel.items()):
            channel_labels = [1 if row["label"] == "pos" else 0 for row in rows]
            channel_scores = [row["score"] for row in rows]
            channel_auc = auc(channel_labels, channel_scores)
            channel_rho = spearman(
                channel_scores, [row["percentile"] for row in rows]
            )
            pos_count = sum(channel_labels)
            neg_count = len(channel_labels) - pos_count
            if channel_auc is not None:
                per_channel_auc.append(channel_auc)
                if pos_count >= 3 and neg_count >= 3:
                    stable_channel_auc.append(channel_auc)
            if channel_rho is not None:
                per_channel_spearman.append(channel_rho)
            channel_metrics.append(
                {
                    "version": version,
                    "channel": channel,
                    "n": len(rows),
                    "pos": pos_count,
                    "neg": neg_count,
                    "auc": rounded(channel_auc),
                    "spearman_percentile": rounded(channel_rho),
                }
            )

        cell_agreements = [
            value
            for rows in by_cell.values()
            if len(rows) >= 2
            if (value := pairwise_concordance(rows)) is not None
        ]
        cell_spearmans = [
            value
            for rows in by_cell.values()
            if len(rows) >= 3
            if (
                value := spearman(
                    [row["score"] for row in rows],
                    [row["percentile"] for row in rows],
                )
            )
            is not None
        ]
        summaries.append(
            {
                "version": version,
                "judgment_rows": len(judgments),
                "scored_rows": len(scored),
                "abstain_or_unscored": abstain_count,
                "unmatched_candidate_ids": invalid_count,
                "unique_scores": len(counts),
                "largest_tie_group": max(counts.values(), default=0),
                "pooled_auc": rounded(auc(binary, scores)),
                "channel_macro_auc": rounded(mean(per_channel_auc)),
                "stable_channel_macro_auc": rounded(mean(stable_channel_auc)),
                "cell_pairwise_accuracy_macro": rounded(mean(cell_agreements)),
                "cell_spearman_macro": rounded(mean(cell_spearmans)),
                "channel_macro_spearman_percentile": rounded(
                    mean(per_channel_spearman)
                ),
                "overall_spearman_percentile": rounded(
                    spearman(scores, [row["percentile"] for row in scored])
                ),
            }
        )

    write_csv(out_dir / "version_comparison.csv", summaries)
    write_csv(out_dir / "candidate_diagnostics_PRIVATE.csv", diagnostics)
    write_csv(out_dir / "channel_metrics.csv", channel_metrics)
    (out_dir / "version_comparison.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
