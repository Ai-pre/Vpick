from __future__ import annotations

import argparse
import itertools
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import (
    ROOT,
    assert_blind_payload,
    load_config,
    read_csv,
    read_jsonl,
    resolve_path,
    stable_id,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_CONFIG = ROOT / "configs" / "evaluation.yaml"


def standalone_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate["candidate_id"],
        "start_ms": candidate.get("start_ms"),
        "end_ms": candidate.get("end_ms"),
        "duration_ms": candidate.get("duration_ms"),
        "description": candidate.get("description", ""),
        "transcript": candidate.get("transcript", ""),
        "visual_evidence_available": bool(candidate.get("visual_evidence_available", False)),
    }
    assert_blind_payload(payload)
    return payload


def source_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = {
        **standalone_payload(candidate),
        "longform_overview": candidate.get("longform_overview", []),
        "before_context": candidate.get("before_context", ""),
        "after_context": candidate.get("after_context", ""),
    }
    assert_blind_payload(payload)
    return payload


def _candidate_side(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "start_ms": candidate.get("start_ms"),
        "end_ms": candidate.get("end_ms"),
        "duration_ms": candidate.get("duration_ms"),
        "description": candidate.get("description", ""),
        "transcript": candidate.get("transcript", ""),
        "before_context": candidate.get("before_context", ""),
        "after_context": candidate.get("after_context", ""),
    }


def build(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = resolve_path(config["output_dir"])
    annotation_dir = resolve_path(config["annotation_dir"])
    candidates = read_jsonl(output_dir / "prepared" / "candidates_blind.jsonl")
    targets = read_csv(output_dir / "prepared" / "targets_private.csv")
    target_by_candidate = {row["candidate_id"]: row for row in targets}

    standalone = [standalone_payload(candidate) for candidate in candidates]
    source = [source_payload(candidate) for candidate in candidates]
    write_jsonl(output_dir / "requests" / "standalone_pointwise_requests.jsonl", standalone)
    write_jsonl(output_dir / "requests" / "source_pointwise_requests.jsonl", source)

    human_pointwise_tasks = [
        {
            "task_id": stable_id("HPT", candidate["candidate_id"]),
            **source_payload(candidate),
        }
        for candidate in candidates
    ]
    write_jsonl(annotation_dir / "human_source_pointwise_tasks.jsonl", human_pointwise_tasks)

    pointwise_response_rows: list[dict[str, Any]] = []
    shuffled = list(human_pointwise_tasks)
    random.Random(config["evaluation_id"]).shuffle(shuffled)
    for annotator_id in ("H1", "H2"):
        for display_order, task in enumerate(shuffled, 1):
            pointwise_response_rows.append(
                {
                    "task_id": task["task_id"],
                    "candidate_id": task["candidate_id"],
                    "annotator_id": annotator_id,
                    "display_order": display_order,
                    "source_salience_0_4": "",
                    "relative_competitiveness_0_4": "",
                    "hook_0_4": "",
                    "self_contained_0_4": "",
                    "payoff_0_4": "",
                    "density_0_4": "",
                    "boundary_0_4": "",
                    "overall_source_highlight_0_4": "",
                    "insufficient_evidence": "",
                    "notes": "",
                }
            )
    write_csv(annotation_dir / "human_source_pointwise_responses.csv", pointwise_response_rows)

    by_longform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        target = target_by_candidate.get(str(candidate["candidate_id"]), {})
        longform_id = str(target.get("longform_id", ""))
        if longform_id:
            by_longform[longform_id].append(candidate)

    canonical_pairs: list[dict[str, Any]] = []
    machine_pair_requests: list[dict[str, Any]] = []
    for longform_id, group in sorted(by_longform.items()):
        if len(group) < 2:
            continue
        overview = group[0].get("longform_overview", [])
        for left, right in itertools.combinations(sorted(group, key=lambda row: row["candidate_id"]), 2):
            pair_id = stable_id("EVALPAIR", longform_id, left["candidate_id"], right["candidate_id"])
            swap = int(pair_id[-1], 16) % 2 == 1
            first, second = (right, left) if swap else (left, right)
            canonical = {
                "pair_id": pair_id,
                "longform_id": longform_id,
                "longform_overview": overview,
                "candidate_a": _candidate_side(first),
                "candidate_b": _candidate_side(second),
            }
            assert_blind_payload(canonical)
            canonical_pairs.append(canonical)
            machine_pair_requests.extend(
                [
                    {**canonical, "order_variant": "AB"},
                    {
                        **canonical,
                        "candidate_a": canonical["candidate_b"],
                        "candidate_b": canonical["candidate_a"],
                        "order_variant": "BA",
                    },
                ]
            )

    write_jsonl(output_dir / "requests" / "source_pairwise_requests.jsonl", machine_pair_requests)
    write_jsonl(annotation_dir / "human_source_pairwise_tasks.jsonl", canonical_pairs)

    human_pairwise_responses: list[dict[str, Any]] = []
    for annotator_id in ("H1", "H2"):
        order = list(canonical_pairs)
        random.Random(f"{config['evaluation_id']}:{annotator_id}").shuffle(order)
        for display_order, task in enumerate(order, 1):
            human_pairwise_responses.append(
                {
                    "pair_id": task["pair_id"],
                    "annotator_id": annotator_id,
                    "display_order": display_order,
                    "winner": "",
                    "confidence_1_5": "",
                    "reason": "",
                    "invalid_reason": "",
                }
            )
    write_csv(annotation_dir / "human_source_pairwise_responses.csv", human_pairwise_responses)

    summary = {
        "candidate_count": len(candidates),
        "standalone_pointwise_request_count": len(standalone),
        "source_pointwise_request_count": len(source),
        "published_same_longform_pair_count": len(canonical_pairs),
        "pairwise_order_test_request_count": len(machine_pair_requests),
        "human_pointwise_response_rows": len(pointwise_response_rows),
        "human_pairwise_response_rows": len(human_pairwise_responses),
        "human_label_status": "blank_templates_only",
        "performance_fields_in_judge_payload": 0,
    }
    write_json(annotation_dir / "annotation_task_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build blind human and model evaluation tasks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = build(load_config(args.config))
    print(
        f"Built {summary['source_pointwise_request_count']} pointwise tasks and "
        f"{summary['published_same_longform_pair_count']} same-longform pairs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
