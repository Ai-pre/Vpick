"""Score a judge run against the gold label with significance attached.

Takes the split gold files (label table + judge scores) and reports, for every
score axis: Spearman against the channel percentile, AUC against the binary
label, a permutation p-value, a Holm-corrected verdict across axes, and the same
figures stratified by `transcript_source`. Also reports the sample size a given
effect would need, so a null result can be read as "underpowered" or "absent"
rather than left ambiguous.

Nothing here re-derives the metrics the existing scripts compute; it adds the
uncertainty they omit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from significance import (
    auc_with_p,
    describe,
    holm_bonferroni,
    min_n_for_spearman,
    spearman_with_p,
    stratified_spearman,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate judge scores against the gold label, with p-values."
    )
    parser.add_argument("--labels", required=True, help="gold label table (PRIVATE)")
    parser.add_argument("--scores", required=True, help="judge score CSV keyed by candidate_id")
    parser.add_argument("--score-column", action="append", default=[],
                        help="Score columns to evaluate; default is every numeric column.")
    parser.add_argument("--positive-label", default="pos")
    parser.add_argument("--negative-label", default="neg")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = {r["candidate_id"]: r for r in read_csv(Path(args.labels))}
    scores = read_csv(Path(args.scores))
    joined = [r for r in scores if r["candidate_id"] in labels]

    if not joined:
        raise SystemExit("no candidate_id overlap between labels and scores")

    columns = args.score_column or [
        c
        for c in joined[0]
        if c != "candidate_id" and sum(1 for r in joined if number(r.get(c)) is not None) >= len(joined) * 0.8
    ]

    report: dict[str, Any] = {
        "label_rows": len(labels),
        "score_rows": len(scores),
        "joined_rows": len(joined),
        "unmatched_score_rows": [r["candidate_id"] for r in scores if r["candidate_id"] not in labels],
        "evaluated_columns": columns,
        "iterations": args.iterations,
        "axes": {},
    }

    p_for_holm: dict[str, float | None] = {}
    for column in columns:
        usable = [r for r in joined if number(r.get(column)) is not None]
        values = [number(r[column]) for r in usable]
        percentiles = [
            number(labels[r["candidate_id"]].get("channel_performance_percentile_PRIVATE"))
            for r in usable
        ]
        pairs = [(v, p) for v, p in zip(values, percentiles) if p is not None]
        rho = p_rho = None
        n_rho = 0
        if len(pairs) >= 3:
            rho, p_rho, n_rho = spearman_with_p(
                [v for v, _ in pairs], [p for _, p in pairs],
                iterations=args.iterations, seed=args.seed,
            )

        binary = [
            (v, 1 if labels[r["candidate_id"]]["performance_label_PRIVATE"] == args.positive_label else 0)
            for v, r in zip(values, usable)
            if labels[r["candidate_id"]]["performance_label_PRIVATE"]
            in {args.positive_label, args.negative_label}
        ]
        auc = p_auc = None
        n_auc = 0
        if binary:
            auc, p_auc, n_auc = auc_with_p(
                [l for _, l in binary], [v for v, _ in binary],
                iterations=args.iterations, seed=args.seed,
            )

        strata = {}
        by_source: dict[str, list[tuple[float, float]]] = {}
        for value, row in zip(values, usable):
            meta = labels[row["candidate_id"]]
            percentile = number(meta.get("channel_performance_percentile_PRIVATE"))
            if percentile is None:
                continue
            by_source.setdefault(meta.get("transcript_source", "unknown"), []).append(
                (value, percentile)
            )
        if len(by_source) > 1:
            strata = stratified_spearman(
                [(name, [v for v, _ in items], [p for _, p in items])
                 for name, items in sorted(by_source.items()) if len(items) >= 3],
                iterations=args.iterations,
                seed=args.seed,
            )

        report["axes"][column] = {
            "spearman_percentile": None if rho is None else round(rho, 4),
            "spearman_p": None if p_rho is None else round(p_rho, 5),
            "spearman_n": n_rho,
            "auc_pos_neg": None if auc is None else round(auc, 4),
            "auc_p": None if p_auc is None else round(p_auc, 5),
            "auc_n": n_auc,
            "n_needed_for_observed_rho": (
                min_n_for_spearman(rho) if rho not in (None, 0) else None
            ),
            "underpowered": (
                bool(rho is not None and rho != 0 and min_n_for_spearman(rho) > n_rho)
            ),
            "stratified_by_transcript_source": strata,
        }
        p_for_holm[column] = p_rho

    report["holm_across_axes"] = holm_bonferroni(p_for_holm)
    report["any_axis_significant"] = any(
        v.get("significant") for v in report["holm_across_axes"].values()
    )

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"joined {len(joined)} rows | 축 {len(columns)}개 | permutation {args.iterations}회")
    print()
    for column, axis in report["axes"].items():
        print(describe(f"{column:44} rho", axis["spearman_percentile"], axis["spearman_p"], axis["spearman_n"]))
        if axis["auc_pos_neg"] is not None:
            print(describe(f"{'':44} auc", axis["auc_pos_neg"], axis["auc_p"], axis["auc_n"]))
        if axis["underpowered"]:
            print(f"{'':46}검정력 부족: 이 크기의 rho 를 잡으려면 n≥{axis['n_needed_for_observed_rho']}")
        strata = axis["stratified_by_transcript_source"]
        if strata:
            parts = [
                f"{name}={info['rho']}(n={info['n']})"
                for name, info in strata.items()
                if not name.startswith("_")
            ]
            print(f"{'':46}층화: {'  '.join(parts)}  macro={strata['_macro_weighted']['rho']}")
    print()
    print(f"Holm 보정 후 유의한 축: {'있음' if report['any_axis_significant'] else '없음'}")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
