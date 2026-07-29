from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from reference_judge import CHECKLIST_DIMENSIONS, EVIDENCE_DIMENSIONS, normalize_judgments


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expand(record: dict[str, Any]) -> dict[str, Any]:
    evidence_values = list(record["e"])
    if len(evidence_values) != len(EVIDENCE_DIMENSIONS):
        raise ValueError(f"Invalid evidence vector for {record['id']}")
    verdict = str(record["v"])
    checklist_bits = str(record.get("c", ""))
    if verdict == "score" and (len(checklist_bits) != len(CHECKLIST_DIMENSIONS) or set(checklist_bits) - {"0", "1"}):
        raise ValueError(f"Invalid checklist bits for {record['id']}: {checklist_bits}")
    return {
        "candidate_id": record["id"],
        "verdict": verdict,
        "evidence": dict(zip(EVIDENCE_DIMENSIONS, evidence_values)),
        "highlight_saliency_1_5": record.get("s"),
        "checklist": (
            dict(zip(CHECKLIST_DIMENSIONS, (bit == "1" for bit in checklist_bits)))
            if verdict == "score"
            else None
        ),
        "overall_shortform_suitable": record.get("o"),
        "confidence": record["q"],
        "failure_flags": record.get("f", []),
        "reason": record["r"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Import compact manually generated Reference Judge judgments.")
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    candidates = read_csv(args.candidates)
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    compact = read_jsonl(args.judgments)
    supplied_ids = [str(record["id"]) for record in compact]
    expected_ids = set(candidate_by_id)
    if len(supplied_ids) != len(set(supplied_ids)):
        raise ValueError("Duplicate candidate_id in compact judgments")
    if set(supplied_ids) != expected_ids:
        raise ValueError(
            f"Candidate mismatch: missing={sorted(expected_ids - set(supplied_ids))}, "
            f"extra={sorted(set(supplied_ids) - expected_ids)}"
        )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    run = config["runs"][0]
    expanded = [expand(record) for record in compact]
    normalized = normalize_judgments({"judgments": expanded}, expected_ids)
    rows = [
        {
            "judge_run_id": run["run_id"],
            "provider": run["provider"],
            "model": run["model"],
            "prompt_id": config["prompt_id"],
            "input_modality": config["input_modality"],
            "batch_id": "CODEX_DIRECT_SINGLE",
            "repeat_index": 1,
            "dry_run": False,
            "candidate_id": row["candidate_id"],
            "long_video_id": candidate_by_id[row["candidate_id"]]["long_video_id"],
            **row,
        }
        for row in normalized
    ]
    rows.sort(key=lambda row: row["candidate_id"])
    fields = [
        "judge_run_id", "provider", "model", "prompt_id", "input_modality", "batch_id",
        "repeat_index", "dry_run", "candidate_id", "long_video_id", "verdict",
    ]
    fields.extend(f"evidence_{name}" for name in EVIDENCE_DIMENSIONS)
    fields.append("highlight_saliency_1_5")
    fields.extend(f"check_{name}" for name in CHECKLIST_DIMENSIONS)
    fields.extend(
        [
            "saliency_score_100", "checklist_score_100", "reference_score_100",
            "overall_shortform_suitable", "confidence", "failure_flags", "reason",
        ]
    )
    write_csv(args.out_dir / "reference_judge_scores.csv", rows, fields)
    summary = {
        "run_id": run["run_id"],
        "candidate_count": len(candidates),
        "repeat_count": 1,
        "score_row_count": len(rows),
        "scored_count": sum(row["verdict"] == "score" for row in rows),
        "abstain_count": sum(row["verdict"] == "abstain" for row in rows),
        "origin": "direct_codex_session_not_openai_api",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
