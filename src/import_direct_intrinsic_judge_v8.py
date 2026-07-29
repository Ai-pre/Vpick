from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from intrinsic_judge_v8 import (
    CHECK_DIMENSIONS,
    EVIDENCE_DIMENSIONS,
    normalize_judgment,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import v8 intrinsic Judge JSONL.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input-modality",
        default="vpick_description_conditional_asr_structurally_blind",
    )
    parser.add_argument("--repeat-index", type=int, default=1)
    args = parser.parse_args()

    candidates = read_csv(Path(args.candidates))
    expected_ids = [row["candidate_id"] for row in candidates]
    items: dict[str, dict[str, Any]] = {}
    with Path(args.judgments).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            candidate_id = str(item.get("candidate_id", "")).strip()
            if candidate_id in items:
                raise ValueError(f"Duplicate candidate_id at line {line_number}: {candidate_id}")
            items[candidate_id] = normalize_judgment(item, candidate_id)

    if set(items) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(items))
        extra = sorted(set(items) - set(expected_ids))
        raise ValueError(f"Judgment IDs differ from blind input; missing={missing}, extra={extra}")

    rows = [
        {
            "judge_run_id": args.run_id,
            "provider": args.provider,
            "model": args.model,
            "prompt_id": "shortform_intrinsic_judge_v8_ko",
            "input_modality": args.input_modality,
            "repeat_index": args.repeat_index,
            **items[candidate_id],
        }
        for candidate_id in expected_ids
    ]
    fields = [
        "judge_run_id",
        "provider",
        "model",
        "prompt_id",
        "input_modality",
        "repeat_index",
        "candidate_id",
        "verdict",
    ]
    fields.extend(f"evidence_{name}" for name in EVIDENCE_DIMENSIONS)
    fields.extend(["content_mode", "editorial_quality_1_5"])
    fields.extend(f"check_{name}" for name in CHECK_DIMENSIONS)
    fields.extend(
        [
            "quality_score_100",
            "overall_editorial_suitable",
            "confidence_1_5",
            "failure_flags",
            "reason",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
