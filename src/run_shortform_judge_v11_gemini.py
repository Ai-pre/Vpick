"""Run the v11 short-form judge over a blind evidence CSV using the Gemini REST API.

Emits the same flat score schema as the existing `codex_judge_v10_scores.csv`, so
`evaluate_judge_validity.py` consumes the output unchanged. Scores are recomputed
from the eight sub-axes rather than trusted from the model, because the prompt asks
for a fixed formula and a model that arithmetically slips would otherwise poison
the correlation silently.

Responses are cached per (prompt, candidate, model, repeat) so a rate-limited run
can be resumed without paying for completed rows again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

RANK_AXES = ("opening_pull", "body_strength", "boundary_integrity")
EVIDENCE_AXES = ("transcript_intelligibility", "boundary_observability")

FIELDS = (
    ["judge_run_id", "provider", "model", "prompt_id", "repeat_index", "candidate_id",
     "verdict", "observation", "confidence_1_5", "failure_flags"]
    + list(EVIDENCE_AXES)
    + list(RANK_AXES)
    + ["judge_score_100", "parse_status", "total_tokens"]
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_json(text: str) -> dict[str, Any] | None:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL) or [text]
    decoder = json.JSONDecoder()
    for block in blocks:
        block = block.strip()
        for match in re.finditer(r"\{", block):
            try:
                parsed, _ = decoder.raw_decode(block[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "candidate_id" in parsed:
                return parsed
    return None


def axis_score(parsed: dict[str, Any], axis: str, hi: float) -> float | None:
    """v11 puts each ranked axis at the top level as {reason, score}."""
    entry = parsed.get(axis)
    if isinstance(entry, dict):
        entry = entry.get("score")
    try:
        value = float(entry)
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= hi else None


def call_gemini(
    api_key: str, model: str, system: str, payload: str, attempts: int, timeout: int
) -> tuple[str, dict[str, Any], str]:
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": f"{system}\n\n[입력]\n{payload}"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 4096},
        }
    ).encode("utf-8")
    last = ""
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            f"{API_ROOT}/{model}:generateContent",
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            candidate = (data.get("candidates") or [{}])[0]
            text = "".join(
                part.get("text", "")
                for part in (candidate.get("content") or {}).get("parts", [])
            )
            return text, data.get("usageMetadata", {}), ""
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            last = f"HTTP {error.code} {detail[:160]}"
            retry = re.search(r"retry in\s+([0-9.]+)s", detail, flags=re.IGNORECASE)
            wait = float(retry.group(1)) + 2.0 if retry else min(90, 6 * 2**attempt)
        except Exception as error:
            last = f"{type(error).__name__}: {error}"
            wait = min(90, 6 * 2**attempt)
        if attempt < attempts:
            print(
                json.dumps({"event": "retry", "attempt": attempt, "wait_sec": round(wait, 1),
                            "error": last[:130]}, ensure_ascii=False),
                flush=True,
            )
            time.sleep(wait)
    return "", {}, last


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the v10 judge with Gemini.")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep-sec", type=float, default=6.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set.")

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    system = Path(args.prompt).read_text(encoding="utf-8")
    prompt_id = Path(args.prompt).stem
    run_id = f"gemini_{args.model.replace('.', '_')}_{prompt_id}"

    rows = read_csv(Path(args.evidence))
    if args.limit:
        rows = rows[: args.limit]

    results: list[dict[str, Any]] = []
    for repeat in range(1, args.repeat_count + 1):
        for index, row in enumerate(rows, start=1):
            payload = json.dumps(
                {k: row.get(k, "") for k in
                 ("candidate_id", "duration_sec", "description",
                  "transcript", "before_context", "after_context")},
                ensure_ascii=False,
            )
            digest = hashlib.sha256(
                f"{args.model}|{prompt_id}|{repeat}|{payload}".encode("utf-8")
            ).hexdigest()[:24]
            cache = cache_dir / f"{digest}.json"
            if cache.exists():
                blob = json.loads(cache.read_text(encoding="utf-8"))
                text, usage, error = blob["text"], blob.get("usage", {}), ""
            else:
                text, usage, error = call_gemini(
                    api_key, args.model, system, payload, args.attempts, args.timeout
                )
                cache.write_text(
                    json.dumps({"text": text, "usage": usage}, ensure_ascii=False),
                    encoding="utf-8",
                )
                time.sleep(args.sleep_sec)

            record: dict[str, Any] = {f: "" for f in FIELDS}
            record.update(
                {
                    "judge_run_id": run_id, "provider": "gemini", "model": args.model,
                    "prompt_id": prompt_id, "repeat_index": repeat,
                    "candidate_id": row["candidate_id"],
                    "total_tokens": usage.get("totalTokenCount", ""),
                }
            )
            parsed = extract_json(text) if text else None
            if parsed is None:
                record["parse_status"] = "parse_error" if text else f"api_error:{error[:60]}"
                results.append(record)
                print(json.dumps({"event": "row", "index": index, "repeat": repeat,
                                  "candidate_id": row["candidate_id"],
                                  "status": record["parse_status"]}, ensure_ascii=False), flush=True)
                continue

            ranked = [axis_score(parsed, a, 100.0) for a in RANK_AXES]
            record["verdict"] = str(parsed.get("verdict", ""))
            record["observation"] = str(parsed.get("observation", ""))[:600]
            record["confidence_1_5"] = parsed.get("confidence_1_5", "")
            flags = parsed.get("failure_flags") or []
            record["failure_flags"] = "|".join(str(f) for f in flags)
            for axis in EVIDENCE_AXES:
                record[axis] = axis_score(parsed.get("evidence") or {}, axis, 5.0) or ""
            for axis, value in zip(RANK_AXES, ranked):
                record[axis] = "" if value is None else value

            if all(v is not None for v in ranked):
                # Mean of the three ranked axes, recomputed rather than read back.
                record["judge_score_100"] = round(sum(ranked) / len(ranked), 4)
                record["parse_status"] = "score"
            else:
                record["parse_status"] = "abstain" if record["verdict"] == "abstain" else "incomplete_axes"

            results.append(record)
            print(
                json.dumps({"event": "row", "index": index, "repeat": repeat,
                            "candidate_id": row["candidate_id"],
                            "status": record["parse_status"],
                            "judge_score_100": record["judge_score_100"]}, ensure_ascii=False),
                flush=True,
            )

    scores_path = out_dir / "judge_v10_gemini_scores.csv"
    with scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(results)

    scored = [r for r in results if r["parse_status"] == "score"]
    values = [float(r["judge_score_100"]) for r in scored]
    summary = {
        "evidence_rows": len(rows),
        "repeat_count": args.repeat_count,
        "result_rows": len(results),
        "scored": len(scored),
        "parse_status_counts": {
            s: sum(1 for r in results if r["parse_status"] == s)
            for s in sorted({r["parse_status"] for r in results})
        },
        "judge_score_mean": round(sum(values) / len(values), 4) if values else None,
        "judge_score_min": min(values) if values else None,
        "judge_score_max": max(values) if values else None,
        "unique_judge_scores": len(set(values)),
        "largest_tie_group": max(
            (sum(1 for v in values if v == u) for u in set(values)), default=0
        ),
        "model": args.model,
        "prompt_id": prompt_id,
        "scores_path": str(scores_path),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
