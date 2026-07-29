from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from .common import (
    ROOT,
    assert_blind_payload,
    load_config,
    read_jsonl,
    resolve_path,
    write_json,
    write_jsonl,
)
from .schemas import (
    SOURCE_DIMENSIONS,
    STANDALONE_DIMENSIONS,
    pointwise_score_100,
    validate_pairwise,
    validate_pointwise,
)


sys.path.insert(0, str(ROOT / "src"))
from llm_client import LLMError, call_llm  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "evaluation.yaml"
PROMPTS = {
    "standalone_pointwise": ROOT / "prompts" / "evaluation_standalone_pointwise_v1_ko.md",
    "source_pointwise": ROOT / "prompts" / "evaluation_source_pointwise_v1_ko.md",
    "source_pairwise": ROOT / "prompts" / "evaluation_source_pairwise_v1_ko.md",
}


def _request_path(output_dir: Path, case: str) -> Path:
    return output_dir / "requests" / f"{case}_requests.jsonl"


def _request_hash(
    *,
    case: str,
    provider: str,
    model: str,
    repeat_index: int,
    prompt: str,
    payload: dict[str, Any],
) -> str:
    material = {
        "case": case,
        "provider": provider,
        "model": model,
        "repeat_index": repeat_index,
        "prompt": prompt,
        "payload": payload,
    }
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _mock_pointwise(candidate_id: str, source_conditioned: bool) -> dict[str, Any]:
    dimensions = SOURCE_DIMENSIONS if source_conditioned else STANDALONE_DIMENSIONS
    return {
        "candidate_id": candidate_id,
        "verdict": "score",
        "scores": {
            dimension: {"score": 2, "reason": "mock schema validation"}
            for dimension in dimensions
        },
        "confidence_1_5": 3,
        "failure_flags": [],
        "reason": "mock schema validation",
    }


def _mock_pairwise(pair_id: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "winner": "tie",
        "comparison": {
            "source_salience": "tie",
            "standalone_quality": "tie",
            "boundary_integrity": "tie",
        },
        "confidence_1_5": 3,
        "reason": "mock schema validation",
    }


def run(
    *,
    config: dict[str, Any],
    case: str,
    provider: str,
    model: str,
    repeat_count: int,
    limit: int,
    retries: int,
    max_tokens: int,
    mock: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if case not in PROMPTS:
        raise ValueError(f"Unsupported judge case: {case}")
    output_dir = resolve_path(config["output_dir"])
    request_path = _request_path(output_dir, case)
    requests = read_jsonl(request_path)
    if limit:
        requests = requests[:limit]
    for payload in requests:
        assert_blind_payload(payload)

    prompt = PROMPTS[case].read_text(encoding="utf-8")
    run_dir = output_dir / "model_runs" / f"{case}_{provider}_{model.replace('/', '_')}"
    if dry_run:
        summary = {
            "case": case,
            "request_count": len(requests),
            "provider": provider,
            "model": model,
            "blind_payload_check": "passed",
            "status": "dry_run_no_model_calls",
        }
        write_json(run_dir / "run_summary.json", summary)
        return summary

    tasks: list[tuple[int, dict[str, Any]]] = []
    effective_repeats = 1 if case == "source_pairwise" else max(1, repeat_count)
    for repeat_index in range(1, effective_repeats + 1):
        shuffled = list(requests)
        random.Random(f"{config['evaluation_id']}:{case}:{repeat_index}").shuffle(shuffled)
        tasks.extend((repeat_index, payload) for payload in shuffled)

    judgments: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for repeat_index, payload in tasks:
        digest = _request_hash(
            case=case,
            provider=provider,
            model=model,
            repeat_index=repeat_index,
            prompt=prompt,
            payload=payload,
        )
        cache_path = cache_dir / f"{digest}.json"
        response: dict[str, Any]
        cache_hit = cache_path.exists()
        if cache_hit:
            response = json.loads(cache_path.read_text(encoding="utf-8"))
        elif mock:
            value = (
                _mock_pairwise(str(payload["pair_id"]))
                if case == "source_pairwise"
                else _mock_pointwise(
                    str(payload["candidate_id"]),
                    source_conditioned=case == "source_pointwise",
                )
            )
            response = {
                "provider": "mock",
                "model": "mock",
                "json": value,
                "usage": {},
            }
            cache_path.write_text(
                json.dumps(response, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            last_error: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    response = call_llm(
                        provider,
                        model,
                        prompt,
                        json.dumps(payload, ensure_ascii=False),
                        max_tokens=max_tokens,
                    )
                    cache_path.write_text(
                        json.dumps(response, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    break
                except (LLMError, KeyError, TypeError, ValueError) as exc:
                    last_error = exc
                    if attempt == retries:
                        raise RuntimeError(f"Judge request failed: {last_error}") from exc
                    time.sleep(min(16.0, 2.0**attempt))

        raw_value = response.get("json")
        if case == "source_pairwise":
            normalized = validate_pairwise(raw_value)
        else:
            normalized = validate_pointwise(
                raw_value,
                source_conditioned=case == "source_pointwise",
            )
            normalized["score_100"] = pointwise_score_100(normalized)
        normalized.update(
            {
                "case": case,
                "provider": response.get("provider", provider),
                "model": response.get("model", model),
                "repeat_index": repeat_index,
                "order_variant": payload.get("order_variant", ""),
                "cache_hit": cache_hit,
                "request_hash": digest,
            }
        )
        judgments.append(normalized)
        usage_rows.append(
            {
                "request_hash": digest,
                "case": case,
                "repeat_index": repeat_index,
                "usage": response.get("usage", {}),
            }
        )

    write_jsonl(run_dir / "judgments.jsonl", judgments)
    write_jsonl(run_dir / "usage.jsonl", usage_rows)
    summary = {
        "case": case,
        "provider": "mock" if mock else provider,
        "model": "mock" if mock else model,
        "input_request_count": len(requests),
        "effective_repeat_count": effective_repeats,
        "judgment_count": len(judgments),
        "cache_hit_count": sum(bool(row["cache_hit"]) for row in judgments),
        "status": "mock_not_for_validation" if mock else "actual_model_run",
        "prompt": str(PROMPTS[case]),
        "input": str(request_path),
    }
    write_json(run_dir / "run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a blind pointwise or pairwise LLM Judge.")
    parser.add_argument(
        "--case",
        required=True,
        choices=sorted(PROMPTS),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--repeat-count", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run(
        config=config,
        case=args.case,
        provider=args.provider or config["judge"]["provider"],
        model=args.model or config["judge"]["model"],
        repeat_count=args.repeat_count or int(config["judge"]["repeat_count"]),
        limit=args.limit,
        retries=args.retries,
        max_tokens=args.max_tokens,
        mock=args.mock,
        dry_run=args.dry_run,
    )
    print(f"{summary['case']}: {summary['status']} ({summary.get('judgment_count', 0)} judgments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
