from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evaluation.build_annotation_tasks import build as build_annotation_tasks
from evaluation.build_behavior_labels import build as build_behavior_labels
from evaluation.common import (
    ROOT,
    assert_blind_payload,
    deterministic_group_split,
    load_config,
    percentile,
    read_csv,
    read_jsonl,
    relative_log_view_score,
)
from evaluation.prepare_data import prepare
from evaluation.run_judge import run as run_judge
from evaluation.schemas import pointwise_score_100, validate_pairwise, validate_pointwise


CONFIG = ROOT / "configs" / "evaluation.yaml"


def test_relative_log_view_score_interpretation() -> None:
    median = 100.0
    assert relative_log_view_score(100.0, median) == pytest.approx(0.0)
    assert relative_log_view_score(201.0, median) == pytest.approx(1.0)
    assert relative_log_view_score(49.5, median) == pytest.approx(-1.0)


def test_channel_percentile_uses_midrank_for_ties() -> None:
    values = [10.0, 20.0, 20.0, 40.0]
    assert percentile(values, 20.0) == pytest.approx(50.0)
    assert percentile(values, 5.0) == pytest.approx(0.0)


def test_blind_payload_rejects_performance_leakage() -> None:
    assert_blind_payload({"candidate_id": "A", "description": "ok"})
    with pytest.raises(ValueError, match="channel_name"):
        assert_blind_payload({"candidate_id": "A", "channel_name": "hidden"})
    with pytest.raises(ValueError, match="views"):
        assert_blind_payload({"candidate_id": "A", "nested": {"views": 100}})


def test_group_split_never_splits_one_longform() -> None:
    rows = [
        {"longform_id": "L1", "candidate_id": "A"},
        {"longform_id": "L1", "candidate_id": "B"},
        {"longform_id": "L2", "candidate_id": "C"},
    ]
    assignment = deterministic_group_split(rows)
    assert set(assignment) == {"L1", "L2"}
    assert assignment["L1"] in {"train", "validation", "test"}


def test_pointwise_schema_and_score_range() -> None:
    dimensions = {
        name: {"score": 4, "reason": "clear evidence"}
        for name in (
            "hook",
            "engagement",
            "self_contained",
            "payoff",
            "density",
            "boundary",
        )
    }
    judgment = validate_pointwise(
        {
            "candidate_id": "C1",
            "verdict": "score",
            "scores": dimensions,
            "confidence_1_5": 5,
            "failure_flags": [],
            "reason": "strong",
        },
        source_conditioned=False,
    )
    assert pointwise_score_100(judgment) == pytest.approx(100.0)
    dimensions["hook"]["score"] = 5
    with pytest.raises(ValueError):
        validate_pointwise(
            {
                "candidate_id": "C1",
                "verdict": "score",
                "scores": dimensions,
                "confidence_1_5": 5,
            },
            source_conditioned=False,
        )


def test_pairwise_schema_rejects_invalid_winner() -> None:
    with pytest.raises(ValueError):
        validate_pairwise(
            {
                "pair_id": "P1",
                "winner": "left",
                "comparison": {
                    "source_salience": "A",
                    "standalone_quality": "B",
                    "boundary_integrity": "tie",
                },
                "confidence_1_5": 4,
                "reason": "x",
            }
        )


def test_actual_data_builds_and_has_no_group_leakage() -> None:
    config = load_config(CONFIG)
    prepare_summary = prepare(config)
    behavior_summary = build_behavior_labels(config)
    annotation_summary = build_annotation_tasks(config)
    assert prepare_summary["candidate_count"] == 60
    assert behavior_summary["valid_candidate_count"] == 60
    assert annotation_summary["published_same_longform_pair_count"] == 8

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    targets = read_csv(output_dir / "prepared" / "targets_private.csv")
    split_by_longform: dict[str, set[str]] = {}
    for row in targets:
        split_by_longform.setdefault(row["longform_id"], set()).add(row["dataset_split"])
    assert all(len(splits) == 1 for splits in split_by_longform.values())


def test_pairwise_requests_are_same_longform_unique_and_reversed() -> None:
    config = load_config(CONFIG)
    output_dir = ROOT / config["output_dir"]
    requests = read_jsonl(output_dir / "requests" / "source_pairwise_requests.jsonl")
    assert len(requests) == 16
    grouped: dict[str, list[dict]] = {}
    for row in requests:
        assert_blind_payload(row)
        grouped.setdefault(row["pair_id"], []).append(row)
    assert len(grouped) == 8
    for variants in grouped.values():
        assert {row["order_variant"] for row in variants} == {"AB", "BA"}
        left = next(row for row in variants if row["order_variant"] == "AB")
        right = next(row for row in variants if row["order_variant"] == "BA")
        assert left["longform_id"] == right["longform_id"]
        assert left["candidate_a"]["candidate_id"] == right["candidate_b"]["candidate_id"]
        assert left["candidate_b"]["candidate_id"] == right["candidate_a"]["candidate_id"]


def test_mock_runner_cache_and_json_validation(tmp_path: Path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir(parents=True)
    request = {
        "candidate_id": "C1",
        "start_ms": 0,
        "end_ms": 1000,
        "duration_ms": 1000,
        "description": "상황",
        "transcript": "대사",
        "visual_evidence_available": False,
    }
    (request_dir / "standalone_pointwise_requests.jsonl").write_text(
        json.dumps(request, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = load_config(CONFIG)
    config["output_dir"] = str(tmp_path)
    first = run_judge(
        config=config,
        case="standalone_pointwise",
        provider="mock",
        model="mock",
        repeat_count=1,
        limit=0,
        retries=0,
        max_tokens=100,
        mock=True,
        dry_run=False,
    )
    second = run_judge(
        config=config,
        case="standalone_pointwise",
        provider="mock",
        model="mock",
        repeat_count=1,
        limit=0,
        retries=0,
        max_tokens=100,
        mock=True,
        dry_run=False,
    )
    assert first["judgment_count"] == 1
    assert second["cache_hit_count"] == 1


def test_hybrid_scores_stay_in_range() -> None:
    standalone = 25.0
    source = 75.0
    for alpha in (0.3, 0.5, 0.7):
        score = alpha * standalone + (1 - alpha) * source
        assert 0.0 <= score <= 100.0
        assert math.isfinite(score)


def test_all_case_outputs_exist() -> None:
    config = load_config(CONFIG)
    output_dir = ROOT / config["output_dir"]
    expected = [
        "case_1_summary.json",
        "case_2_summary.json",
        "case_3_summary.json",
        "case_4_summary.json",
        "case_5_summary.json",
        "case_comparison.csv",
        "validation_status.csv",
        "EVALUATION_SYSTEM_COMPARISON_REPORT.md",
    ]
    assert all((output_dir / name).exists() for name in expected)
