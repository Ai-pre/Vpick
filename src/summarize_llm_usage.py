from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from performance_judge_v1 import read_csv, read_jsonl, write_json


def read_usage_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".csv":
        return read_jsonl(path)
    rows: list[dict[str, Any]] = []
    for source in read_csv(path):
        rows.append(
            {
                "item_id": source["candidate_id"],
                "usage": json.loads(source.get("usage_json") or "{}"),
            }
        )
    return rows


def summarize(
    rows: list[dict[str, Any]],
    input_price_per_million: float,
    output_price_per_million: float,
) -> dict[str, Any]:
    item_ids = [
        str(
            row.get("item_id")
            or (
                f"{row.get('candidate_id')}#R{row.get('repeat_index')}"
                if row.get("candidate_id") is not None
                else ""
            )
        )
        for row in rows
    ]
    if any(not item_id for item_id in item_ids):
        raise ValueError("Each usage row requires item_id or candidate_id/repeat_index")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Usage files contain duplicate item_id values")
    input_tokens = sum(int(row.get("usage", {}).get("input_tokens", 0)) for row in rows)
    output_tokens = sum(
        int(row.get("usage", {}).get("output_tokens", 0)) for row in rows
    )
    thinking_tokens = sum(
        int(
            row.get("usage", {})
            .get("output_tokens_details", {})
            .get("thinking_tokens", 0)
        )
        for row in rows
    )
    input_cost = input_tokens / 1_000_000 * input_price_per_million
    output_cost = output_tokens / 1_000_000 * output_price_per_million
    return {
        "request_count": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens_in_output": thinking_tokens,
        "pricing_usd_per_million_tokens": {
            "input": input_price_per_million,
            "output": output_price_per_million,
        },
        "estimated_input_cost_usd": round(input_cost, 6),
        "estimated_output_cost_usd": round(output_cost, 6),
        "estimated_total_cost_usd": round(input_cost + output_cost, 6),
        "note": "Estimate excludes taxes, credits, caching, and provider-specific adjustments.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate JSONL LLM usage files.")
    parser.add_argument("usage_files", type=Path, nargs="+")
    parser.add_argument("--input-price", type=float, required=True)
    parser.add_argument("--output-price", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        row
        for path in args.usage_files
        for row in read_usage_file(path)
    ]
    summary = summarize(rows, args.input_price, args.output_price)
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
