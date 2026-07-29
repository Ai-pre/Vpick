from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["candidate_id"]: row for row in rows}


def value(row: dict[str, str], field: str) -> float:
    return float(row[field])


def scale_0_4_to_1_10(score: float) -> float:
    return 1.0 + 2.25 * max(0.0, min(4.0, score))


def weighted(features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(features[key] * weight for key, weight in weights.items())


PASS_WEIGHTS = {
    "pass_1_balanced": {
        "clarity": {"title": 0.30, "thumbnail": 0.25, "self": 0.30, "boundary": 0.15},
        "curiosity": {"title": 0.30, "thumbnail": 0.25, "change": 0.30, "specificity": 0.15},
        "complementarity": {
            "title": 0.25,
            "thumbnail": 0.25,
            "change": 0.20,
            "specificity": 0.30,
        },
        "alignment": {"title": 0.20, "thumbnail": 0.15, "self": 0.25, "progression": 0.25, "boundary": 0.15},
    },
    "pass_2_package_first": {
        "clarity": {"title": 0.35, "thumbnail": 0.30, "self": 0.25, "boundary": 0.10},
        "curiosity": {"title": 0.35, "thumbnail": 0.30, "change": 0.25, "specificity": 0.10},
        "complementarity": {
            "title": 0.30,
            "thumbnail": 0.30,
            "change": 0.15,
            "specificity": 0.25,
        },
        "alignment": {"title": 0.25, "thumbnail": 0.20, "self": 0.20, "progression": 0.20, "boundary": 0.15},
    },
    "pass_3_content_confirmed": {
        "clarity": {"title": 0.25, "thumbnail": 0.20, "self": 0.35, "boundary": 0.20},
        "curiosity": {"title": 0.25, "thumbnail": 0.20, "change": 0.35, "specificity": 0.20},
        "complementarity": {
            "title": 0.20,
            "thumbnail": 0.20,
            "change": 0.25,
            "specificity": 0.35,
        },
        "alignment": {"title": 0.15, "thumbnail": 0.10, "self": 0.25, "progression": 0.30, "boundary": 0.20},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--title-scores",
        type=Path,
        default=ROOT
        / "results/judge_success_v1_codex_direct_94_2026-07-29/codex_direct_title_packaging_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--thumbnail-scores",
        type=Path,
        default=ROOT
        / "results/judge_success_v1_codex_direct_94_2026-07-29/codex_direct_thumbnail_packaging_94_PRIVATE.csv",
    )
    parser.add_argument(
        "--content-scores",
        type=Path,
        default=ROOT
        / "results/judge_success_v1_codex_direct_94_2026-07-29/codex_direct_success_dimensions_94.csv",
    )
    parser.add_argument(
        "--v10-dimensions",
        type=Path,
        default=ROOT / "data/private/judge_validation_94/codex_direct_v10_dimensions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/package_success_judge_v1_codex_direct_2026-07-29",
    )
    args = parser.parse_args()

    titles = by_id(read_csv(args.title_scores))
    thumbnails = by_id(read_csv(args.thumbnail_scores))
    content = by_id(read_csv(args.content_scores))
    v10 = by_id(read_csv(args.v10_dimensions))
    candidate_ids = sorted(set(titles) & set(thumbnails) & set(content) & set(v10))
    if len(candidate_ids) != 94:
        raise RuntimeError(f"Expected 94 complete candidates, found {len(candidate_ids)}")

    repeats: list[dict[str, Any]] = []
    order_manifest: dict[str, list[str]] = {}
    for repeat_index, (pass_name, dimensions) in enumerate(PASS_WEIGHTS.items(), start=1):
        order = list(candidate_ids)
        random.Random(f"codex-package-v1-{repeat_index}").shuffle(order)
        order_manifest[pass_name] = order
        for candidate_id in order:
            feature = {
                "title": value(titles[candidate_id], "title_packaging_0_4"),
                "thumbnail": value(thumbnails[candidate_id], "thumbnail_packaging_0_4"),
                "change": value(content[candidate_id], "change_or_surprise_0_4"),
                "specificity": value(content[candidate_id], "specificity_novelty_0_4"),
                "self": value(v10[candidate_id], "self_contained_clarity"),
                "progression": value(v10[candidate_id], "progression_payoff"),
                "boundary": value(v10[candidate_id], "boundary_integrity"),
            }
            scores = {
                dimension: scale_0_4_to_1_10(weighted(feature, weights))
                for dimension, weights in dimensions.items()
            }
            joint = sum(scores.values()) / len(scores)
            repeats.append(
                {
                    "candidate_id": candidate_id,
                    "pass_name": pass_name,
                    "repeat_index": repeat_index,
                    "first_glance_clarity_1_10": round(scores["clarity"], 4),
                    "curiosity_click_pull_1_10": round(scores["curiosity"], 4),
                    "title_thumbnail_complementarity_1_10": round(
                        scores["complementarity"], 4
                    ),
                    "content_alignment_1_10": round(scores["alignment"], 4),
                    "joint_package_score_1_10": round(joint, 4),
                    "input_policy": "blind_no_channel_no_views_no_labels",
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = {candidate_id: [] for candidate_id in candidate_ids}
    for row in repeats:
        grouped[row["candidate_id"]].append(row)
    aggregate: list[dict[str, Any]] = []
    score_fields = [
        "first_glance_clarity_1_10",
        "curiosity_click_pull_1_10",
        "title_thumbnail_complementarity_1_10",
        "content_alignment_1_10",
        "joint_package_score_1_10",
    ]
    for candidate_id in candidate_ids:
        rows = grouped[candidate_id]
        output: dict[str, Any] = {
            "candidate_id": candidate_id,
            "repeat_count": len(rows),
            "input_policy": "blind_no_channel_no_views_no_labels",
        }
        for field in score_fields:
            values = [float(row[field]) for row in rows]
            mean = sum(values) / len(values)
            variance = sum((item - mean) ** 2 for item in values) / len(values)
            output[f"{field}_mean"] = round(mean, 4)
            output[f"{field}_std"] = round(variance**0.5, 4)
        aggregate.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "codex_package_judge_3pass_94_PRIVATE.csv",
        repeats,
        list(repeats[0]),
    )
    write_csv(
        args.output_dir / "codex_package_judge_aggregate_94_PRIVATE.csv",
        aggregate,
        list(aggregate[0]),
    )
    (args.output_dir / "scoring_manifest_PRIVATE.json").write_text(
        json.dumps(
            {
                "candidate_count": len(candidate_ids),
                "pass_count": len(PASS_WEIGHTS),
                "performance_features_visible_to_judge": False,
                "representative_frames_used": False,
                "actual_short_thumbnails_used": True,
                "weights": PASS_WEIGHTS,
                "shuffled_candidate_order": order_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_count": len(candidate_ids),
                "repeat_rows": len(repeats),
                "aggregate": str(
                    args.output_dir / "codex_package_judge_aggregate_94_PRIVATE.csv"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
