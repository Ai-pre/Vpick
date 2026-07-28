from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_judge_candidates import source_rows  # noqa: E402
from judge_scores_to_predictions import main as predictions_main  # noqa: E402
from rerank_candidates_with_judges import (  # noqa: E402
    average_rank_percentiles,
    rank_within_groups,
)
from summarize_judge_guided_improvement import summarize  # noqa: E402


def test_average_rank_percentiles_preserve_ties() -> None:
    values = {"a": 1.0, "b": 2.0, "c": 2.0, "d": 4.0}
    assert average_rank_percentiles(values) == {
        "a": 0.0,
        "b": 50.0,
        "c": 50.0,
        "d": 100.0,
    }


def test_rank_normalization_is_within_candidate_group() -> None:
    scores = {"a": 1.0, "b": 2.0, "c": 10.0, "d": 20.0}
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g2"}
    assert rank_within_groups(scores, groups) == {
        "a": 0.0,
        "b": 100.0,
        "c": 0.0,
        "d": 100.0,
    }


def test_source_rows_accepts_slate_start_and_end_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "slate.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "long_video_id",
                "rank",
                "start_sec",
                "end_sec",
                "source_run_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "long_video_id": "long-a",
                "rank": 1,
                "start_sec": 12.5,
                "end_sec": 45.0,
                "source_run_id": "slate-v1",
            }
        )
    rows = source_rows(path, "slate", top_k=5)
    assert rows[0]["start_sec"] == 12.5
    assert rows[0]["end_sec"] == 45.0
    assert rows[0]["source_run_id"] == "slate-v1"


def test_score_predictions_are_ranked_and_joined_to_gold_pairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "candidate_id": "a",
                        "longform_id": "long-a",
                        "start_ms": 10000,
                        "end_ms": 30000,
                        "scene_ids": ["s1"],
                    }
                ),
                json.dumps(
                    {
                        "candidate_id": "b",
                        "longform_id": "long-a",
                        "start_ms": 40000,
                        "end_ms": 70000,
                        "scene_ids": ["s2"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "candidate_id,verdict,score\n"
        "a,score,20\n"
        "b,score,80\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "gold.csv"
    dataset.write_text(
        "pair_id,long_video_id,short_video_id,gold_start_sec,gold_end_sec\n"
        "p1,long-a,short-a,42,68\n",
        encoding="utf-8",
    )
    output = tmp_path / "predictions.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_scores_to_predictions.py",
            "--dataset",
            str(dataset),
            "--candidates",
            str(candidates),
            "--scores",
            str(scores),
            "--output",
            str(output),
            "--run-id",
            "test",
        ],
    )
    predictions_main()
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig")))
    assert rows[0]["pair_id"] == "p1"
    assert rows[0]["rank"] == "1"
    assert rows[0]["pred_start_sec"] == "40.0"
    assert rows[0]["pred_end_sec"] == "70.0"


def test_score_predictions_can_rank_lower_source_rank_first(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "candidate_id": "a",
                        "longform_id": "long-a",
                        "start_ms": 10000,
                        "end_ms": 30000,
                    }
                ),
                json.dumps(
                    {
                        "candidate_id": "b",
                        "longform_id": "long-a",
                        "start_ms": 40000,
                        "end_ms": 70000,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scores = tmp_path / "source_scores.csv"
    scores.write_text(
        "candidate_id,source_rank\n"
        "a,1\n"
        "b,2\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "gold.csv"
    dataset.write_text(
        "pair_id,long_video_id,short_video_id,gold_start_sec,gold_end_sec\n"
        "p1,long-a,short-a,12,28\n",
        encoding="utf-8",
    )
    output = tmp_path / "predictions.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_scores_to_predictions.py",
            "--dataset",
            str(dataset),
            "--candidates",
            str(candidates),
            "--scores",
            str(scores),
            "--score-field",
            "source_rank",
            "--lower-is-better",
            "--output",
            str(output),
            "--run-id",
            "test",
        ],
    )
    predictions_main()
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig")))
    assert rows[0]["rank"] == "1"
    assert rows[0]["pred_start_sec"] == "10.0"


def test_improvement_summary_reports_preregistered_contrasts(
    tmp_path: Path,
) -> None:
    values = {
        "deterministic_baseline": 0.25,
        "pointwise_only": 0.5,
        "v14_only": 0.25,
        "hybrid_50_50": 0.75,
    }
    for variant, hit_at_5 in values.items():
        path = tmp_path / variant
        path.mkdir()
        run = {
            "pair_count": 4,
            "prediction_count": 20,
            "top1_core_hit_rate": 0.0,
            "top1_tight_hit_rate": 0.0,
            "hit_at_3_core_rate": 0.0,
            "hit_at_3_tight_rate": 0.0,
            "hit_at_5_core_rate": hit_at_5,
            "hit_at_5_tight_rate": 0.0,
            "best_iou_at_5_mean": 0.0,
        }
        (path / "summary.json").write_text(
            json.dumps({"runs": [run]}),
            encoding="utf-8",
        )
    summary = summarize(tmp_path)
    assert (
        summary["contrasts"]["pointwise_minus_deterministic"][
            "hit_at_5_core_rate"
        ]
        == 0.25
    )
    assert (
        summary["contrasts"]["hybrid_minus_v14"]["hit_at_5_core_rate"]
        == 0.5
    )
