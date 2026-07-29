from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
TARGET_FIELDS = (
    "content_success_0_1",
    "title_success_0_1",
    "thumbnail_success_0_1",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evidence_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("description", "")),
        str(row.get("transcript", "")),
        str(row.get("before_context", "")),
        str(row.get("after_context", "")),
        str(row.get("context_transcript", "")),
    ]
    return "\n".join(part.strip() for part in parts if part.strip())


def rho(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return 0.0 if np.isnan(value) else float(value)


def clipped_prediction(
    x_train: Any,
    y_train: np.ndarray,
    x_test: Any,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    return np.clip(model.predict(x_test), 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Distill existing fixed-Judge scores into a text-only LOLO "
            "surrogate for scalable candidate reranking."
        )
    )
    parser.add_argument(
        "--teacher-input",
        type=Path,
        default=ROOT
        / "data/private/judge_validation_94"
        / "candidates_blind_94_with_overview_PRIVATE.jsonl",
    )
    parser.add_argument(
        "--teacher-scores",
        type=Path,
        default=ROOT
        / "results/package_context_performance_v1_2026-07-29"
        / "candidate_predictions_94_PRIVATE.csv",
    )
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max-features", type=int, default=24000)
    args = parser.parse_args()

    teacher_input = read_jsonl(args.teacher_input)
    teacher_scores = {
        row["candidate_id"]: row for row in read_csv(args.teacher_scores)
    }
    teacher = [
        row for row in teacher_input if row["candidate_id"] in teacher_scores
    ]
    if len(teacher) != len(teacher_scores):
        raise RuntimeError(
            f"Teacher coverage mismatch: input={len(teacher)}, "
            f"scores={len(teacher_scores)}"
        )
    candidate_pool = read_csv(args.candidate_pool)
    train_text = [evidence_text(row) for row in teacher]
    candidate_text = [evidence_text(row) for row in candidate_pool]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_text)
    x_candidate = vectorizer.transform(candidate_text)
    groups = np.array([str(row["longform_id"]) for row in teacher])
    unique_groups = sorted(set(groups))
    folds = GroupKFold(n_splits=min(5, len(unique_groups)))

    targets = {
        field: np.array(
            [
                float(teacher_scores[row["candidate_id"]][field])
                for row in teacher
            ],
            dtype=float,
        )
        for field in TARGET_FIELDS
    }
    oof = {
        field: np.full(len(teacher), np.nan, dtype=float)
        for field in TARGET_FIELDS
    }
    for train_indices, validation_indices in folds.split(
        x_train, groups=groups
    ):
        for field in TARGET_FIELDS:
            oof[field][validation_indices] = clipped_prediction(
                x_train[train_indices],
                targets[field][train_indices],
                x_train[validation_indices],
                args.alpha,
            )

    candidate_groups: dict[str, list[int]] = {}
    for index, row in enumerate(candidate_pool):
        candidate_groups.setdefault(str(row["longform_id"]), []).append(index)
    candidate_predictions = {
        field: np.full(len(candidate_pool), np.nan, dtype=float)
        for field in TARGET_FIELDS
    }
    for longform_id, indices in candidate_groups.items():
        train_indices = np.flatnonzero(groups != longform_id)
        if len(train_indices) < 10:
            raise RuntimeError(
                f"Too few LOLO teachers for longform {longform_id}"
            )
        for field in TARGET_FIELDS:
            candidate_predictions[field][indices] = clipped_prediction(
                x_train[train_indices],
                targets[field][train_indices],
                x_candidate[indices],
                args.alpha,
            )

    teacher_fixed = (
        0.40 * targets["content_success_0_1"]
        + 0.15 * targets["title_success_0_1"]
        + 0.45 * targets["thumbnail_success_0_1"]
    )
    oof_fixed = (
        0.40 * oof["content_success_0_1"]
        + 0.15 * oof["title_success_0_1"]
        + 0.45 * oof["thumbnail_success_0_1"]
    )
    teacher_text_selection = (
        0.40 * targets["content_success_0_1"]
        + 0.15 * targets["title_success_0_1"]
    ) / 0.55
    oof_text_selection = (
        0.40 * oof["content_success_0_1"]
        + 0.15 * oof["title_success_0_1"]
    ) / 0.55
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_pool):
        content = candidate_predictions["content_success_0_1"][index]
        title = candidate_predictions["title_success_0_1"][index]
        thumbnail = candidate_predictions["thumbnail_success_0_1"][index]
        output_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "longform_id": row["longform_id"],
                "predicted_change_0_1": round(float(content), 8),
                "predicted_title_0_1": round(float(title), 8),
                "predicted_thumbnail_0_1": round(float(thumbnail), 8),
                "judge_score_0_1": round(
                    float(
                        0.40 * content + 0.15 * title + 0.45 * thumbnail
                    ),
                    8,
                ),
                "selection_score_text_0_1": round(
                    float((0.40 * content + 0.15 * title) / 0.55),
                    8,
                ),
                "score_source": "text_distilled_fixed_judge_lolo",
            }
        )

    summary = {
        "teacher_candidate_count": len(teacher),
        "teacher_longform_count": len(unique_groups),
        "target_candidate_count": len(candidate_pool),
        "target_longform_count": len(candidate_groups),
        "alpha": args.alpha,
        "feature_count": len(vectorizer.vocabulary_),
        "teacher_group_oof_spearman": {
            field: round(rho(oof[field], targets[field]), 6)
            for field in TARGET_FIELDS
        },
        "teacher_fixed_formula_group_oof_spearman": round(
            rho(oof_fixed, teacher_fixed), 6
        ),
        "teacher_text_selection_group_oof_spearman": round(
            rho(oof_text_selection, teacher_text_selection), 6
        ),
        "formula": (
            "0.40*predicted_change + 0.15*predicted_title + "
            "0.45*predicted_thumbnail"
        ),
        "leakage_control": (
            "For each target longform, all teacher candidates from the same "
            "longform are excluded before fitting."
        ),
        "warning": (
            "This is a text-only distilled surrogate of the fixed Judge. "
            "The full fixed formula is invalid when thumbnail evidence is absent. "
            "Use selection_score_text_0_1 only as a candidate-selection proxy; "
            "it does not replace direct LLM and image-based scoring."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_scores.csv", output_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
