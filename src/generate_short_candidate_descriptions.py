#!/usr/bin/env python3
"""Generate label-blind, candidate-level shortform descriptions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from llm_client import call_llm


BLIND_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)
LEAKAGE_PATTERN = re.compile(
    r"(?i)(?:\bpos\b|\bneg\b|조회수|좋아요\s*수|백분위|상위\s*25|하위\s*25|"
    r"성과\s*라벨|골드\s*라벨)"
)

SYSTEM_PROMPT = """당신은 롱폼에서 선택된 한 구간을 독립적인 숏폼 후보로 설명하는
데이터 작성자입니다. 점수를 매기거나 성과를 예측하지 마십시오.

입력으로 candidate_id, duration_sec, source_scene_description, transcript만
제공됩니다. YouTube 제목ㆍ설명ㆍ채널ㆍ조회수ㆍ좋아요ㆍ성과 라벨은 알 수 없으며
추측해서도 안 됩니다.

작성 규칙:
1. 후보 내부에서 실제로 관찰되는 중심 상황, 행동이나 발화, 변화ㆍ반응ㆍ결과를
   자연스러운 한국어 1~2문장으로 작성하십시오.
2. transcript를 발화의 주 근거로 사용하고 source_scene_description은 화면 행동과
   인물 반응을 보완하는 근거로만 사용하십시오.
3. 후보 밖의 전후 문맥, 제목처럼 보이는 문구, 조회수 잠재력, 재미있다ㆍ바이럴하다
   같은 품질 판단을 넣지 마십시오.
4. 같은 장면 설명을 나열하지 말고 하나의 숏폼 후보 설명으로 통합하십시오.
5. 고유명사와 혼합 언어는 근거가 분명할 때만 유지하고, 불명확한 내용은 만들지
   마십시오.
6. 공백 제외 45~150자로 작성하십시오.

정확히 다음 JSON 객체만 출력하십시오.
{"items":[{"candidate_id":"...","short_description":"..."}]}"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BLIND_FIELDS:
            raise ValueError(
                f"Candidate columns must be exactly {BLIND_FIELDS}, "
                f"got {reader.fieldnames}"
            )
        rows = list(reader)
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("Candidate IDs must be unique")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def description_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def validate_batch(
    payload: dict[str, Any],
    expected_ids: set[str],
) -> dict[str, str]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Response must contain an items array")
    output: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each description item must be an object")
        candidate_id = str(item.get("candidate_id") or "").strip()
        description = compact(str(item.get("short_description") or ""))
        if not candidate_id or candidate_id in output:
            raise ValueError(f"Missing or duplicate candidate_id: {candidate_id!r}")
        length = description_length(description)
        if not 45 <= length <= 150:
            raise ValueError(
                f"{candidate_id} description length must be 45-150, got {length}"
            )
        if LEAKAGE_PATTERN.search(description):
            raise ValueError(f"{candidate_id} description contains leakage terms")
        output[candidate_id] = description
    if set(output) != expected_ids:
        raise ValueError(
            f"Description IDs mismatch: expected={sorted(expected_ids)} "
            f"actual={sorted(output)}"
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    candidates = read_csv(Path(args.candidates))
    if len(candidates) != 60:
        raise ValueError(f"Expected 60 candidates, found {len(candidates)}")
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}
    usage_rows: list[dict[str, Any]] = []

    batch_size = max(1, args.batch_size)
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        request_items = [
            {
                "candidate_id": row["candidate_id"],
                "duration_sec": row["duration_sec"],
                "source_scene_description": compact(row["description"])[:1800],
                "transcript": row["transcript"][:7000],
            }
            for row in batch
        ]
        expected_ids = {row["candidate_id"] for row in batch}
        digest = hashlib.sha256(
            json.dumps(
                {
                    "provider": args.provider,
                    "model": args.model,
                    "prompt": SYSTEM_PROMPT,
                    "items": request_items,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:20]
        cache_path = raw_dir / f"batch_{start // batch_size + 1:02d}_{digest}.json"

        if cache_path.exists():
            result = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            last_error: Exception | None = None
            for attempt in range(1, max(1, args.attempts) + 1):
                try:
                    result = call_llm(
                        args.provider,
                        args.model,
                        SYSTEM_PROMPT,
                        json.dumps({"items": request_items}, ensure_ascii=False),
                        max_tokens=args.max_tokens,
                    )
                    validate_batch(result["json"], expected_ids)
                    break
                except Exception as error:
                    last_error = error
                    if attempt >= max(1, args.attempts):
                        raise
                    time.sleep(min(30, 2**attempt))
            else:
                raise RuntimeError(last_error)
            cache_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        batch_output = validate_batch(result["json"], expected_ids)
        generated.update(batch_output)
        usage_rows.append(
            {
                "batch_index": start // batch_size + 1,
                "candidate_count": len(batch),
                "provider": args.provider,
                "model": args.model,
                "usage_json": json.dumps(result.get("usage", {}), ensure_ascii=False),
                "cache_file": str(cache_path),
            }
        )
        print(
            json.dumps(
                {
                    "event": "description_batch_complete",
                    "completed": len(generated),
                    "target": len(candidates),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    output_rows = []
    audit_rows = []
    for row in candidates:
        candidate_id = row["candidate_id"]
        new_description = generated[candidate_id]
        output = dict(row)
        output["description"] = new_description
        output_rows.append(output)
        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "old_description_chars": description_length(row["description"]),
                "new_description_chars": description_length(new_description),
                "old_description": row["description"],
                "short_candidate_description": new_description,
                "generation_input": "candidate_transcript+vpick_scene_description",
                "metadata_or_label_used": False,
                "provider": args.provider,
                "model": args.model,
            }
        )

    write_csv(out_dir / "candidates_short_description_final_60.csv", output_rows)
    write_csv(out_dir / "description_generation_audit_60.csv", audit_rows)
    write_csv(out_dir / "description_generation_usage.csv", usage_rows)
    lengths = [description_length(text) for text in generated.values()]
    summary = {
        "candidate_count": len(candidates),
        "description_nonempty_count": sum(bool(text) for text in generated.values()),
        "description_min_chars": min(lengths),
        "description_max_chars": max(lengths),
        "description_mean_chars": round(sum(lengths) / len(lengths), 2),
        "metadata_or_label_used": False,
        "generation_input": "candidate_transcript+vpick_scene_description",
        "provider": args.provider,
        "model": args.model,
    }
    (out_dir / "description_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
