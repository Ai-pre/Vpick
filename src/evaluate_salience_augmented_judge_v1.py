from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_package_and_context_v1 import (
    choose_alpha,
    fit_ridge,
    metrics,
    nested_group_oof,
    predict_ridge,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]
FORMULAS = {
    "baseline_change50_title25_thumbnail25": {
        "salience": 0.00,
        "change": 0.50,
        "title": 0.25,
        "thumbnail": 0.25,
    },
    "salience10_change40_title25_thumbnail25": {
        "salience": 0.10,
        "change": 0.40,
        "title": 0.25,
        "thumbnail": 0.25,
    },
    "salience20_change30_title25_thumbnail25": {
        "salience": 0.20,
        "change": 0.30,
        "title": 0.25,
        "thumbnail": 0.25,
    },
    "salience25_change25_title25_thumbnail25": {
        "salience": 0.25,
        "change": 0.25,
        "title": 0.25,
        "thumbnail": 0.25,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["candidate_id"]: row for row in rows}


def formula_score(
    row: dict[str, str], salience: float, weights: dict[str, float]
) -> float:
    values = {
        "salience": salience / 4.0,
        "change": float(row["content_success_0_1"]),
        "title": float(row["title_success_0_1"]),
        "thumbnail": float(row["thumbnail_success_0_1"]),
    }
    return sum(values[name] * weight for name, weight in weights.items())


def evaluate_universe(
    rows: list[dict[str, str]],
    salience: dict[str, dict[str, str]],
    universe_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_views = np.array([float(row["raw_views"]) for row in rows])
    percentile = np.array([float(row["channel_percentile"]) for row in rows])
    target = np.log1p(raw_views)
    groups = [row["longform_id"] for row in rows]
    split = np.array([row["split"] for row in rows])
    dev = np.flatnonzero(split == "dev")
    locked = np.flatnonzero(split == "locked_test")
    context = np.array(
        [
            [
                math.log1p(float(row["channel_prior_median_views"])),
                math.log1p(float(row["upload_age_days"])),
            ]
            for row in rows
        ]
    )

    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = [
        {
            "candidate_id": row["candidate_id"],
            "universe": universe_name,
            "split": row["split"],
            "overview_status": salience[row["candidate_id"]]["overview_status"],
            "source_salience_0_4": salience[row["candidate_id"]][
                "source_salience_0_4"
            ],
        }
        for row in rows
    ]
    for formula_name, weights in FORMULAS.items():
        judge_score = np.array(
            [
                formula_score(
                    row,
                    float(salience[row["candidate_id"]]["source_salience_0_4"]),
                    weights,
                )
                for row in rows
            ]
        )
        direct_result = metrics(judge_score, raw_views, percentile)
        summary_rows.append(
            {
                "universe": universe_name,
                "formula": formula_name,
                "stage": "pointwise_judge_only",
                "n": len(rows),
                **direct_result,
                "alpha_or_note": "not_applicable",
            }
        )

        calibrated_features = np.column_stack([judge_score, context])
        oof_prediction, alphas = nested_group_oof(
            calibrated_features,
            target,
            groups,
            "package-context-v1-common-folds",
        )
        oof_result = metrics(oof_prediction, raw_views, percentile)
        summary_rows.append(
            {
                "universe": universe_name,
                "formula": formula_name,
                "stage": "judge_plus_channel_age_ridge_group_oof",
                "n": len(rows),
                **oof_result,
                "alpha_or_note": json.dumps(alphas),
            }
        )

        dev_groups = [groups[index] for index in dev]
        alpha = choose_alpha(
            calibrated_features[dev],
            target[dev],
            dev_groups,
            "package-context-v1-common-locked-inner-folds",
        )
        model = fit_ridge(
            calibrated_features[dev], target[dev], alpha
        )
        locked_prediction = predict_ridge(
            model, calibrated_features[locked]
        )
        locked_result = metrics(
            locked_prediction, raw_views[locked], percentile[locked]
        )
        summary_rows.append(
            {
                "universe": universe_name,
                "formula": formula_name,
                "stage": "judge_plus_channel_age_ridge_locked_test",
                "n": len(locked),
                **locked_result,
                "alpha_or_note": alpha,
            }
        )
        for index, candidate in enumerate(candidate_rows):
            candidate[f"judge_score_{formula_name}"] = round(
                float(judge_score[index]), 8
            )
            candidate[f"oof_prediction_{formula_name}"] = round(
                float(oof_prediction[index]), 8
            )
    return summary_rows, candidate_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "results/package_context_performance_v1_2026-07-29/candidate_predictions_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--salience",
        type=Path,
        default=ROOT
        / "data/private/judge_validation_94/codex_direct_source_salience_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/salience_augmented_judge_v1_2026-07-29",
    )
    args = parser.parse_args()

    rows = read_csv(args.input)
    salience = by_id(read_csv(args.salience))
    if {row["candidate_id"] for row in rows} != set(salience):
        raise RuntimeError("Candidate coverage mismatch")

    universes = {
        "all94_neutral_for_3_missing": rows,
        "available_overview_91": [
            row
            for row in rows
            if salience[row["candidate_id"]]["overview_status"]
            != "missing_neutral"
        ],
    }
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for universe_name, universe_rows in universes.items():
        summary, candidates = evaluate_universe(
            universe_rows, salience, universe_name
        )
        summary_rows.extend(summary)
        candidate_rows.extend(candidates)

    source_scores = np.array(
        [float(salience[row["candidate_id"]]["source_salience_0_4"]) for row in rows]
    )
    changes = np.array([float(row["content_success_0_1"]) * 4 for row in rows])
    result_summary = {
        "candidate_count": len(rows),
        "overview_available_count": len(universes["available_overview_91"]),
        "missing_overview_count": len(rows)
        - len(universes["available_overview_91"]),
        "source_salience_distribution": {
            str(score): int(np.sum(source_scores == score))
            for score in sorted(set(source_scores))
        },
        "source_salience_change_spearman": round(
            spearman(source_scores, changes), 6
        ),
        "predeclared_primary_formula": (
            "salience25_change25_title25_thumbnail25"
        ),
        "formulas": FORMULAS,
        "warning": (
            "The 10% and 20% formulas are sensitivity analyses. "
            "The 25% formula was declared before outcome evaluation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "salience_formula_comparison_PRIVATE.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "salience_candidate_scores_PRIVATE.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    (args.output_dir / "salience_experiment_summary_PRIVATE.json").write_text(
        json.dumps(result_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
