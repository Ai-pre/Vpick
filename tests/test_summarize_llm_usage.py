from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_llm_usage import read_usage_file, summarize  # noqa: E402


class SummarizeLlmUsageTest(unittest.TestCase):
    def test_usage_and_cost_are_aggregated(self) -> None:
        rows = [
            {
                "item_id": "a",
                "usage": {
                    "input_tokens": 100_000,
                    "output_tokens": 10_000,
                    "output_tokens_details": {"thinking_tokens": 2_000},
                },
            },
            {
                "item_id": "b",
                "usage": {
                    "input_tokens": 200_000,
                    "output_tokens": 20_000,
                },
            },
        ]
        result = summarize(rows, 5.0, 25.0)
        self.assertEqual(result["request_count"], 2)
        self.assertEqual(result["input_tokens"], 300_000)
        self.assertEqual(result["output_tokens"], 30_000)
        self.assertEqual(result["thinking_tokens_in_output"], 2_000)
        self.assertAlmostEqual(result["estimated_total_cost_usd"], 2.25)

    def test_csv_usage_json_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.csv"
            path.write_text(
                'candidate_id,usage_json\n'
                'a,"{""input_tokens"": 10, ""output_tokens"": 2}"\n',
                encoding="utf-8",
            )
            rows = read_usage_file(path)
        self.assertEqual(rows[0]["item_id"], "a")
        self.assertEqual(rows[0]["usage"]["input_tokens"], 10)

    def test_repeated_candidate_ids_use_repeat_index(self) -> None:
        rows = [
            {
                "candidate_id": "a",
                "repeat_index": 1,
                "usage": {"input_tokens": 10},
            },
            {
                "candidate_id": "a",
                "repeat_index": 2,
                "usage": {"input_tokens": 20},
            },
        ]
        result = summarize(rows, 5.0, 25.0)
        self.assertEqual(2, result["request_count"])
        self.assertEqual(30, result["input_tokens"])


if __name__ == "__main__":
    unittest.main()
