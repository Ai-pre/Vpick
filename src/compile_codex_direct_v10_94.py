from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from compile_codex_judge_v10 import DIMENSION_TEMPLATES
from shortform_judge_v9 import (
    EDITORIAL_DIMENSIONS,
    ENGAGEMENT_DIMENSIONS,
    load_config,
    normalize_judgment,
)


EVIDENCE_FIELDS = (
    "overview_support",
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def integer(row: dict[str, str], field: str, minimum: int, maximum: int) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{row.get('candidate_id', '<unknown>')} has invalid {field}"
        ) from exc
    if value < minimum or value > maximum:
        raise ValueError(
            f"{row.get('candidate_id', '<unknown>')} {field}={value} "
            f"is outside [{minimum}, {maximum}]"
        )
    return value


def failure_flags(row: dict[str, str]) -> list[str]:
    flags: list[str] = []
    if integer(row, "self_contained_clarity", 0, 4) <= 1:
        flags.append("context_dependent")
    if integer(row, "progression_payoff", 0, 4) <= 1:
        flags.extend(("weak_progression", "weak_payoff"))
    if integer(row, "boundary_integrity", 0, 4) <= 1:
        flags.append("awkward_end")
    if integer(row, "opening_pull", 0, 4) <= 1:
        flags.append("weak_opening")
    if integer(row, "change_or_surprise", 0, 4) <= 1:
        flags.append("no_change")
    if integer(row, "emotional_or_information_gain", 0, 4) <= 1:
        flags.append("low_gain")
    if integer(row, "memorable_specificity", 0, 4) <= 1:
        flags.append("not_memorable")
    if integer(row, "transcript_intelligibility", 1, 5) <= 2:
        flags.append("asr_degraded")
    return flags


def dimension_item(dimension: str, score: int) -> dict[str, Any]:
    if dimension == "source_salience":
        reason = (
            "원본 전체 개요가 없어 상대 중요도를 직접 판단하지 않고 "
            "사전 고정한 중립값을 적용했다."
        )
    else:
        reason = DIMENSION_TEMPLATES[dimension][score]
    return {"reason": reason, "score": score}


def build_raw(row: dict[str, str]) -> dict[str, Any]:
    editorial = {
        dimension: dimension_item(
            dimension,
            integer(row, dimension, 0, 4),
        )
        for dimension in EDITORIAL_DIMENSIONS
    }
    engagement = {
        dimension: dimension_item(
            dimension,
            integer(row, dimension, 0, 4),
        )
        for dimension in ENGAGEMENT_DIMENSIONS
    }
    return {
        "candidate_id": row["candidate_id"],
        "reason": row["reason"].strip(),
        "verdict": "score",
        "evidence": {
            field: integer(row, field, 1, 5)
            for field in EVIDENCE_FIELDS
        },
        "editorial": editorial,
        "engagement": engagement,
        "confidence_1_5": integer(row, "confidence_1_5", 1, 5),
        "failure_flags": failure_flags(row),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the frozen Codex direct v10 assessments for 94 candidates."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--dimensions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_jsonl(args.candidates)
    dimensions = read_csv(args.dimensions)
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    dimension_by_id = {str(row["candidate_id"]): row for row in dimensions}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("Duplicate candidate_id in blind input")
    if len(dimension_by_id) != len(dimensions):
        raise ValueError("Duplicate candidate_id in frozen dimensions")
    if set(candidate_by_id) != set(dimension_by_id):
        raise ValueError(
            "Assessment coverage mismatch: "
            f"missing={sorted(set(candidate_by_id) - set(dimension_by_id))}, "
            f"extra={sorted(set(dimension_by_id) - set(candidate_by_id))}"
        )

    config = load_config(args.config)
    raw_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        raw = build_raw(dimension_by_id[candidate_id])
        normalized = normalize_judgment(raw, candidate_id, config)
        raw_rows.append(raw)
        score_rows.append(
            {
                "judge_run_id": "codex_direct_shortform_judge_v10_94",
                "provider": "openai_codex",
                "model": "codex_direct",
                "prompt_id": "shortform_judge_v10_ko",
                "repeat_index": 1,
                "longform_id": candidate.get("longform_id", ""),
                "source_salience_policy": "fixed_neutral_2_no_longform_overview",
                **normalized,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "codex_direct_v10_judgments_94.jsonl",
        raw_rows,
    )
    write_csv(
        args.output_dir / "codex_direct_v10_scores_94.csv",
        score_rows,
    )

    scores = [float(row["judge_score_100"]) for row in score_rows]
    frequencies = Counter(scores)
    summary = {
        "judge_run_id": "codex_direct_shortform_judge_v10_94",
        "candidate_count": len(score_rows),
        "scored_count": len(score_rows),
        "abstain_count": 0,
        "label_blind": True,
        "gpu_used": False,
        "external_api_used": False,
        "prompt_id": "shortform_judge_v10_ko",
        "score_formula": (
            "0.5 * editorial_score_100 + 0.5 * engagement_score_100"
        ),
        "source_salience_policy": "fixed_neutral_2_no_longform_overview",
        "production_equivalent": False,
        "production_equivalent_limitation": (
            "All 94 candidates lack a compact longform_overview, so source_salience "
            "was fixed at neutral 2. The run validates transcript-context pointwise "
            "scoring, not full source-relative salience."
        ),
        "judge_score_mean": round(statistics.mean(scores), 4),
        "judge_score_min": min(scores),
        "judge_score_max": max(scores),
        "unique_judge_scores": len(frequencies),
        "largest_tie_group": max(frequencies.values()),
    }
    (args.output_dir / "codex_direct_v10_summary_94.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
