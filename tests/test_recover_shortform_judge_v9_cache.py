from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recover_shortform_judge_v9_cache import recover  # noqa: E402
from run_shortform_judge_v9 import candidate_payload, request_hash  # noqa: E402
from shortform_judge_v9 import (  # noqa: E402
    EDITORIAL_DIMENSIONS,
    ENGAGEMENT_DIMENSIONS,
    EVIDENCE_DIMENSIONS,
    load_config,
)


class RecoverShortformJudgeV9CacheTests(unittest.TestCase):
    def test_recovers_only_existing_repeat_cache(self) -> None:
        config = load_config(
            ROOT / "config" / "shortform_judge_v9_opus.json"
        )
        prompt = "prompt"
        candidate = {
            "candidate_id": "candidate",
            "longform_id": "long",
            "start_ms": 0,
            "end_ms": 1000,
            "longform_overview": [],
            "scene_ids": [],
            "description": "description",
            "transcript": "transcript",
            "before_context": "",
            "after_context": "",
            "visual_evidence_available": False,
        }
        raw = {
            "candidate_id": "candidate",
            "verdict": "score",
            "evidence": {
                dimension: 3 for dimension in EVIDENCE_DIMENSIONS
            },
            "editorial": {
                dimension: {"score": 2, "reason": "evidence"}
                for dimension in EDITORIAL_DIMENSIONS
            },
            "engagement": {
                dimension: {"score": 2, "reason": "evidence"}
                for dimension in ENGAGEMENT_DIMENSIONS
            },
            "confidence_1_5": 3,
            "failure_flags": [],
            "reason": "evidence",
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory)
            run_id = config["run"]["run_id"]
            key = request_hash(
                run_id,
                1,
                prompt,
                candidate_payload(candidate),
            )
            path = cache_root / run_id / f"{key}.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps({"json": raw, "usage": {}}),
                encoding="utf-8",
            )
            scores, usage, missing = recover(
                [candidate],
                config,
                prompt,
                cache_root,
                2,
            )
        self.assertEqual(1, len(scores))
        self.assertEqual(1, len(usage))
        self.assertEqual(
            [{"candidate_id": "candidate", "repeat_index": 2}],
            missing,
        )


if __name__ == "__main__":
    unittest.main()
