from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_prompt_ablation_posneg import average_ranks, spearman


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare mR3 and Codex prompt-ablation rankings."
    )
    parser.add_argument("--mr3-summary", required=True)
    parser.add_argument("--mr3-diagnostics", required=True)
    parser.add_argument("--codex-summary", required=True)
    parser.add_argument("--codex-diagnostics", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    combined_summary: list[dict[str, Any]] = []
    for provider, path in (
        ("mr3", Path(args.mr3_summary)),
        ("codex_rescore", Path(args.codex_summary)),
    ):
        for row in read_csv(path):
            combined_summary.append({"provider": provider, **row})

    diagnostics: dict[str, dict[str, dict[str, float]]] = {}
    for provider, path in (
        ("mr3", Path(args.mr3_diagnostics)),
        ("codex_rescore", Path(args.codex_diagnostics)),
    ):
        by_version: dict[str, dict[str, float]] = {}
        for row in read_csv(path):
            by_version.setdefault(row["version"], {})[row["candidate_id"]] = float(
                row["score"]
            )
        diagnostics[provider] = by_version

    agreements: list[dict[str, Any]] = []
    for version in ("v1", "v2", "v3", "v4", "v5"):
        mr3 = diagnostics["mr3"].get(version, {})
        codex = diagnostics["codex_rescore"].get(version, {})
        common = sorted(set(mr3) & set(codex))
        mr3_scores = [mr3[candidate_id] for candidate_id in common]
        codex_scores = [codex[candidate_id] for candidate_id in common]
        mr3_top = {
            candidate_id
            for candidate_id, _ in sorted(
                mr3.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        }
        codex_top = {
            candidate_id
            for candidate_id, _ in sorted(
                codex.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        }
        mr3_ranks = average_ranks(mr3_scores)
        codex_ranks = average_ranks(codex_scores)
        mean_rank_gap = (
            sum(abs(left - right) for left, right in zip(mr3_ranks, codex_ranks))
            / len(common)
            if common
            else None
        )
        agreements.append(
            {
                "version": version,
                "common_candidates": len(common),
                "score_spearman": rounded(spearman(mr3_scores, codex_scores)),
                "top10_overlap_count": len(mr3_top & codex_top),
                "top10_jaccard": rounded(
                    len(mr3_top & codex_top) / len(mr3_top | codex_top)
                    if mr3_top | codex_top
                    else None
                ),
                "mean_absolute_rank_gap": rounded(mean_rank_gap),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "combined_version_comparison.csv", combined_summary)
    write_csv(out_dir / "cross_model_agreement.csv", agreements)
    (out_dir / "cross_model_agreement.json").write_text(
        json.dumps(agreements, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(agreements, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
