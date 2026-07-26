from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from shortform_judge_v9 import (
    EDITORIAL_DIMENSIONS,
    ENGAGEMENT_DIMENSIONS,
    load_config,
    normalize_judgment,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "judge_evaluation_v2_2026-07-27"
    / "short_candidate_descriptions_codex"
    / "candidates_blind_short_description_60.jsonl"
)
DEFAULT_ASSESSMENTS = (
    ROOT
    / "results"
    / "judge_evaluation_v2_2026-07-27"
    / "codex_judge_v10_blind"
    / "manual_blind_assessments.json"
)
DEFAULT_CONFIG = ROOT / "config" / "shortform_judge_v10_opus.json"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "judge_evaluation_v2_2026-07-27"
    / "codex_judge_v10_blind"
)

EVIDENCE_DIMENSIONS = (
    "overview_support",
    "description_support",
    "transcript_intelligibility",
    "boundary_observability",
)

DIMENSION_TEMPLATES: dict[str, dict[int, str]] = {
    "source_salience": {
        0: "원본 흐름에서 의미 있는 선택 단위로 보기 어렵다.",
        1: "원본의 연결 장면에 가까워 대표성이 낮다.",
        2: "독립 주제는 있으나 주변 장면으로 대체 가능한 보통 구간이다.",
        3: "원본의 한 주제를 담당하는 의미 있는 구간이다.",
        4: "원본의 핵심 사건·답변·통찰을 대표하는 구간이다.",
    },
    "self_contained_clarity": {
        0: "원본 앞뒤 없이는 중심 상황을 복원할 수 없다.",
        1: "설명과 대사를 함께 봐도 중요한 전제가 빠져 있다.",
        2: "대체로 이해되지만 인물·상황·논점의 일부가 불명확하다.",
        3: "후보 안의 설명과 대사만으로 중심 상황을 이해할 수 있다.",
        4: "첫 의미 단위에서 상황을 잡고 원본 없이도 정확히 이해된다.",
    },
    "progression_payoff": {
        0: "서로 연결되는 진행이나 도착점이 없다.",
        1: "진행 단서는 있으나 결과·답변·반응으로 회수되지 않는다.",
        2: "진행 또는 결론 중 하나는 있으나 회수가 약하다.",
        3: "상황이 의미 있게 진행되어 반응이나 결론에 도달한다.",
        4: "setup과 변화가 강한 결과·답변·반응으로 완전히 회수된다.",
    },
    "boundary_integrity": {
        0: "핵심 발화 중간에서 시작하거나 결론 전에 종료된다.",
        1: "시작과 종료 모두 핵심 발화 또는 인접 장면에 크게 걸친다.",
        2: "이해는 되지만 도입이나 종료 중 적어도 하나가 부자연스럽다.",
        3: "필요한 맥락을 포함하며 대체로 자연스럽게 끝난다.",
        4: "필요한 최소 맥락에서 시작해 도착점 직후 정확히 끝난다.",
    },
    "opening_pull": {
        0: "도입에서 중심 상황이나 시청 이유를 확인할 수 없다.",
        1: "도입이 준비·이동·잡담에 가까워 주목성이 낮다.",
        2: "중심 주제는 보이지만 즉각적인 궁금증이나 긴장은 보통이다.",
        3: "첫 의미 단위에서 갈등·질문·웃음·유용한 약속이 제시된다.",
        4: "첫 의미 단위부터 강한 갈등·질문·웃음·약속이 즉시 작동한다.",
    },
    "change_or_surprise": {
        0: "후보 안에서 변화나 발견이 없다.",
        1: "화제는 이어지지만 예상 변화나 관점 전환이 거의 없다.",
        2: "작은 변화나 반응은 있으나 전환 강도는 보통이다.",
        3: "발견·반응·관점 전환이 분명하게 일어난다.",
        4: "예상을 뒤집는 반전이나 변화가 장면의 중심을 이룬다.",
    },
    "emotional_or_information_gain": {
        0: "감정 또는 정보 이득을 확인할 수 없다.",
        1: "반응이나 정보가 약해 정점에 도달하지 않는다.",
        2: "일부 웃음·공감·정보는 있으나 강도는 보통이다.",
        3: "분명한 웃음·공감·긴장 또는 유용한 정보가 있다.",
        4: "강한 감정 반응이나 새롭고 유용한 정보가 명확한 정점에 도달한다.",
    },
    "memorable_specificity": {
        0: "후보를 구체적으로 요약하거나 인용할 핵심이 없다.",
        1: "상황이 일반적이어서 기억할 구체성이 약하다.",
        2: "요약 가능한 주제는 있으나 고유한 한 문장이나 상황은 약하다.",
        3: "제목·요약으로 뽑을 수 있는 구체적 상황이나 문장이 있다.",
        4: "왜곡 없이 바로 인용·제목화할 수 있는 고유한 순간이 강하다.",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def axis(
    dimensions: tuple[str, ...],
    scores: list[int],
) -> dict[str, dict[str, Any]]:
    if len(scores) != len(dimensions):
        raise ValueError(f"Expected {len(dimensions)} scores, got {scores}")
    return {
        dimension: {
            "reason": DIMENSION_TEMPLATES[dimension][int(score)],
            "score": int(score),
        }
        for dimension, score in zip(dimensions, scores)
    }


def build_raw(
    candidate_id: str,
    assessment: dict[str, Any],
) -> dict[str, Any]:
    evidence = assessment["evidence"]
    if len(evidence) != len(EVIDENCE_DIMENSIONS):
        raise ValueError(f"Invalid evidence for {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "reason": str(assessment["reason"]),
        "verdict": "score",
        "evidence": {
            dimension: int(score)
            for dimension, score in zip(EVIDENCE_DIMENSIONS, evidence)
        },
        "editorial": axis(
            EDITORIAL_DIMENSIONS,
            assessment["editorial"],
        ),
        "engagement": axis(
            ENGAGEMENT_DIMENSIONS,
            assessment["engagement"],
        ),
        "confidence_1_5": int(assessment["confidence"]),
        "failure_flags": list(assessment.get("flags") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--assessments",
        type=Path,
        default=DEFAULT_ASSESSMENTS,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    candidates = read_jsonl(args.input)
    candidate_by_id = {
        str(row["candidate_id"]): row
        for row in candidates
    }
    assessments = json.loads(args.assessments.read_text(encoding="utf-8"))
    if set(candidate_by_id) != set(assessments):
        raise ValueError(
            "Assessment coverage mismatch: "
            f"missing={sorted(set(candidate_by_id) - set(assessments))}, "
            f"extra={sorted(set(assessments) - set(candidate_by_id))}"
        )

    config = load_config(args.config)
    raw_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        raw = build_raw(candidate_id, assessments[candidate_id])
        normalized = normalize_judgment(raw, candidate_id, config)
        raw_rows.append(raw)
        score_rows.append(
            {
                "judge_run_id": "codex_direct_shortform_judge_v10",
                "provider": "openai_codex",
                "model": "codex",
                "prompt_id": "shortform_judge_v10_ko",
                "repeat_index": 1,
                "longform_id": candidate.get("longform_id", ""),
                **normalized,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "codex_judge_v10_judgments.jsonl",
        raw_rows,
    )
    write_csv(
        args.output_dir / "codex_judge_v10_scores.csv",
        score_rows,
    )
    scores = [float(row["judge_score_100"]) for row in score_rows]
    unique_scores = len(set(scores))
    frequencies = {score: scores.count(score) for score in set(scores)}
    summary = {
        "judge_run_id": "codex_direct_shortform_judge_v10",
        "candidate_count": len(score_rows),
        "scored_count": len(score_rows),
        "abstain_count": 0,
        "prompt_id": "shortform_judge_v10_ko",
        "label_blind": True,
        "score_formula": (
            "0.5 * editorial_score_100 + 0.5 * engagement_score_100"
        ),
        "judge_score_mean": round(statistics.mean(scores), 4),
        "judge_score_min": min(scores),
        "judge_score_max": max(scores),
        "unique_judge_scores": unique_scores,
        "largest_tie_group": max(frequencies.values()),
    }
    (args.output_dir / "codex_judge_v10_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
