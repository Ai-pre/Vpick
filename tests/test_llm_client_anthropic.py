from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_client import call_anthropic  # noqa: E402


class AnthropicClientTest(unittest.TestCase):
    def test_opus_48_uses_adaptive_thinking_without_temperature(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json(url, payload, headers):
            captured.update(payload)
            return {
                "content": [{"type": "text", "text": '{"ok": true}'}],
                "usage": {},
            }

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            with patch("llm_client.post_json", side_effect=fake_post_json):
                result = call_anthropic(
                    "claude-opus-4-8",
                    "system",
                    "user",
                    max_tokens=8000,
                )

        self.assertEqual(captured["thinking"], {"type": "adaptive"})
        self.assertEqual(captured["output_config"], {"effort": "high"})
        self.assertNotIn("temperature", captured)
        self.assertEqual(result["json"], {"ok": True})


if __name__ == "__main__":
    unittest.main()
