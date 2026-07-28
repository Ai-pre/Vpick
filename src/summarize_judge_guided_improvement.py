from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    "deterministic_baseline",
    "pointwise_only",
    "v14_only",
    "hybrid_50_50",
)

METRICS = (
    "top1_core_hit_rate",
    "top1_tight_hit_rate",
    "hit_at_3_core_rate",
    "hit_at_3_tight_rate",
    "hit_at_5_core_rate",
    "hit_at_5_tight_rate",
    "best_iou_at_5_mean",
)


def load_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs") or []
    if len(runs) != 1:
        raise ValueError(f"{path} must contain exactly one run")
    return runs[0]


def summarize(evaluation_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        run = load_run(evaluation_dir / variant / "summary.json")
        row = {
            "variant": variant,
            "pair_count": int(run["pair_count"]),
            "prediction_count": int(run["prediction_count"]),
        }
        row.update({metric: float(run[metric]) for metric in METRICS})
        rows.append(row)

    by_variant = {row["variant"]: row for row in rows}

    def contrast(left: str, right: str) -> dict[str, float]:
        return {
            metric: round(
                by_variant[left][metric] - by_variant[right][metric],
                8,
            )
            for metric in METRICS
        }

    return {
        "variants": rows,
        "contrasts": {
            "pointwise_minus_deterministic": contrast(
                "pointwise_only",
                "deterministic_baseline",
            ),
            "hybrid_minus_v14": contrast(
                "hybrid_50_50",
                "v14_only",
            ),
            "hybrid_minus_deterministic": contrast(
                "hybrid_50_50",
                "deterministic_baseline",
            ),
        },
        "claim_rule": (
            "Judge value requires positive held-out improvement; "
            "Judge scores are never the validation target."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the four frozen Judge-guided selector variants."
    )
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(args.evaluation_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "comparison.csv", summary["variants"])
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "comparison_csv": str(args.output_dir / "comparison.csv"),
                "comparison_json": str(args.output_dir / "comparison.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
