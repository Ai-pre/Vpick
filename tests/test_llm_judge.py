from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_gold_pairwise_eval import preserve_human_responses  # noqa: E402
from build_judge_candidates import blind_id, gold_source_rows, load_subtitle_cues, source_rows, subtitle_evidence  # noqa: E402
from evaluate_llm_judge import aggregate_candidate_scores, evidence_coverage, fleiss_kappa, performance_group_alignment, spearman  # noqa: E402
from evaluate_pairwise_judge import aggregate_scores as aggregate_pairwise_scores, human_metrics  # noqa: E402
from evaluate_reference_judge import cohen_kappa as reference_cohen_kappa  # noqa: E402
from llm_client import (  # noqa: E402
    call_gemini,
    call_gemini_text_batch,
    call_gemini_video_batch,
    call_gemini_video_pair,
    call_gemini_video_pointwise_batch,
    call_openrouter,
)
from merge_gold_datasets import performance_label  # noqa: E402
from run_llm_judge import CANDIDATE_DIMENSIONS, EVIDENCE_DIMENSIONS, normalize_candidate_judgments, weighted_score  # noqa: E402
from run_pairwise_judge import (  # noqa: E402
    EDITORIAL_DIMENSIONS,
    EVIDENCE_DIMENSIONS as PAIRWISE_EVIDENCE_DIMENSIONS,
    PERFORMANCE_DIMENSIONS,
    normalize_response as normalize_pairwise_response,
)
from run_pointwise_judge import normalize_judgments as normalize_pointwise_judgments  # noqa: E402
from reference_judge import CHECKLIST_DIMENSIONS as REFERENCE_CHECKLIST_DIMENSIONS  # noqa: E402
from reference_judge import normalize_judgments as normalize_reference_judgments  # noqa: E402
from reference_judge_v7 import CHECK_DIMENSIONS as REFERENCE_V7_CHECK_DIMENSIONS  # noqa: E402
from reference_judge_v7 import normalize_judgment as normalize_reference_v7_judgment  # noqa: E402
from intrinsic_judge_v8 import normalize_judgment as normalize_intrinsic_v8_judgment  # noqa: E402


class JudgeEvaluationTests(unittest.TestCase):
    def test_reference_v7_uses_seven_ternary_checks(self) -> None:
        item = {
            "candidate_id": "C_1",
            "verdict": "score",
            "evidence": {
                "description_support": 1,
                "transcript_intelligibility": 4,
                "boundary_observability": 5,
            },
            "saliency_market_1_5": 3,
            "checks": {
                name: value
                for name, value in zip(
                    REFERENCE_V7_CHECK_DIMENSIONS,
                    (0, 1, 2, 0, 1, 2, 2),
                )
            },
            "overall_shortform_suitable": True,
            "confidence_1_5": 4,
            "failure_flags": [],
            "reason": "structured",
        }
        row = normalize_reference_v7_judgment(item, "C_1")
        self.assertAlmostEqual(row["checklist_score_100"], 8 / 14 * 100)
        self.assertEqual(row["check_hook_within_3s"], 0)
        self.assertEqual(row["check_natural_end"], 2)

    def test_intrinsic_v8_uses_binary_auditable_checks(self) -> None:
        item = {
            "candidate_id": "C_1",
            "verdict": "score",
            "evidence": {
                "description_support": 1,
                "transcript_intelligibility": 4,
                "boundary_observability": 3,
            },
            "content_mode": "mixed",
            "editorial_quality_1_5": 4,
            "checks": {
                "self_contained_context": 1,
                "central_focus_clear": 1,
                "opening_pull": 0,
                "meaningful_progression": 1,
                "payoff_or_conclusion": 1,
                "distinctive_value": 1,
                "memorable_specificity": 0,
                "natural_start": 1,
                "natural_end": 1,
            },
            "overall_editorial_suitable": True,
            "confidence_1_5": 4,
            "failure_flags": ["weak_opening", "not_memorable"],
            "reason": "중심 내용과 도착점은 있으나 도입과 기억성은 약하다.",
        }
        row = normalize_intrinsic_v8_judgment(item, "C_1")
        self.assertEqual(row["quality_score_100"], round(7 / 9 * 100, 4))
        self.assertEqual(row["content_mode"], "mixed")

    def test_intrinsic_v8_abstain_requires_insufficient_evidence(self) -> None:
        item = {
            "candidate_id": "C_1",
            "verdict": "abstain",
            "evidence": {
                "description_support": 1,
                "transcript_intelligibility": 1,
                "boundary_observability": 1,
            },
            "content_mode": None,
            "editorial_quality_1_5": None,
            "checks": None,
            "overall_editorial_suitable": None,
            "confidence_1_5": 1,
            "failure_flags": ["insufficient_evidence"],
            "reason": "의미를 복원할 수 없다.",
        }
        row = normalize_intrinsic_v8_judgment(item, "C_1")
        self.assertEqual(row["verdict"], "abstain")
        self.assertEqual(row["quality_score_100"], "")

    def test_reference_two_rater_cohen_kappa(self) -> None:
        labels = {
            "C_1": [("H1", "1"), ("H2", "1")],
            "C_2": [("H1", "0"), ("H2", "0")],
            "C_3": [("H1", "1"), ("H2", "1")],
        }
        self.assertAlmostEqual(reference_cohen_kappa(labels, ("0", "1")) or 0.0, 1.0)

    def test_reference_judge_uses_saliency_and_boolean_checklist(self) -> None:
        response = {
            "judgments": [
                {
                    "candidate_id": "C_1",
                    "verdict": "score",
                    "evidence": {
                        "description_support": 4,
                        "transcript_intelligibility": 4,
                        "boundary_observability": 4,
                    },
                    "highlight_saliency_1_5": 4,
                    "checklist": {
                        name: index < 6
                        for index, name in enumerate(REFERENCE_CHECKLIST_DIMENSIONS)
                    },
                    "overall_shortform_suitable": True,
                    "confidence": 4,
                    "failure_flags": [],
                    "reason": "reference grounded",
                }
            ]
        }
        row = normalize_reference_judgments(response, {"C_1"})[0]
        self.assertEqual(row["saliency_score_100"], 75.0)
        self.assertEqual(row["checklist_score_100"], 75.0)
        self.assertEqual(row["reference_score_100"], 75.0)
        self.assertEqual(row["overall_shortform_suitable"], 1)

    def test_reference_judge_abstention_has_no_quality_score(self) -> None:
        response = {
            "judgments": [
                {
                    "candidate_id": "C_1",
                    "verdict": "abstain",
                    "evidence": {
                        "description_support": 1,
                        "transcript_intelligibility": 1,
                        "boundary_observability": 1,
                    },
                    "highlight_saliency_1_5": None,
                    "checklist": None,
                    "overall_shortform_suitable": None,
                    "confidence": 2,
                    "failure_flags": ["insufficient_evidence"],
                    "reason": "not enough evidence",
                }
            ]
        }
        row = normalize_reference_judgments(response, {"C_1"})[0]
        self.assertEqual(row["reference_score_100"], "")
        self.assertIn("insufficient_evidence", row["failure_flags"])

    def test_pairwise_swapped_presentation_is_restored(self) -> None:
        raw = {
            "comparison_id": "PW_1",
            "verdict": "score",
            "left": {
                "evidence": {name: 5 for name in PAIRWISE_EVIDENCE_DIMENSIONS},
                "editorial": {name: 5 for name in EDITORIAL_DIMENSIONS},
                "performance": {name: 5 for name in PERFORMANCE_DIMENSIONS},
            },
            "right": {
                "evidence": {name: 1 for name in PAIRWISE_EVIDENCE_DIMENSIONS},
                "editorial": {name: 1 for name in EDITORIAL_DIMENSIONS},
                "performance": {name: 1 for name in PERFORMANCE_DIMENSIONS},
            },
            "editorial_preference": "left",
            "performance_preference": "left",
            "confidence": 5,
            "failure_flags": [],
            "reason": "presented left is stronger",
        }
        editorial_weights = {name: 1 for name in EDITORIAL_DIMENSIONS}
        performance_weights = {name: 1 for name in PERFORMANCE_DIMENSIONS}
        row = normalize_pairwise_response(raw, "PW_1", editorial_weights, performance_weights, swapped=True)
        self.assertEqual(row["editorial_preference"], "right")
        self.assertEqual(row["performance_preference"], "right")
        self.assertEqual(row["left_editorial_score"], 0.0)
        self.assertEqual(row["right_editorial_score"], 100.0)

    def test_pairwise_repeat_agreement(self) -> None:
        rows = [
            {
                "judge_run_id": "judge",
                "provider": "provider",
                "model": "model",
                "comparison_id": "PW_1",
                "repeat_index": str(index),
                "verdict": "score",
                "editorial_preference": "left",
                "performance_preference": "right",
                "left_editorial_score": "60",
                "right_editorial_score": "40",
                "left_performance_score": "45",
                "right_performance_score": "55",
                "confidence": "4",
            }
            for index in (1, 2)
        ]
        aggregate = aggregate_pairwise_scores(rows)[0]
        self.assertEqual(aggregate["editorial_consensus"], "left")
        self.assertEqual(aggregate["performance_consensus"], "right")
        self.assertTrue(aggregate["performance_repeat_agreement"])

    def test_pairwise_abstain_accepts_null_evidence(self) -> None:
        raw = {
            "comparison_id": "PW_1",
            "verdict": "abstain",
            "left": {"evidence": None, "editorial": None, "performance": None},
            "right": {"evidence": None, "editorial": None, "performance": None},
            "editorial_preference": "tie",
            "performance_preference": "tie",
            "confidence": 2,
            "failure_flags": ["insufficient_evidence"],
            "reason": "insufficient evidence",
        }
        row = normalize_pairwise_response(
            raw,
            "PW_1",
            {name: 1 for name in EDITORIAL_DIMENSIONS},
            {name: 1 for name in PERFORMANCE_DIMENSIONS},
            swapped=False,
        )
        self.assertEqual(row["verdict"], "abstain")
        self.assertEqual(row["left_evidence_description_support"], "")
        self.assertEqual(row["left_editorial_score"], "")

    def test_partial_human_pairwise_labels_do_not_count_as_complete(self) -> None:
        aggregates = [{
            "judge_run_id": "judge", "comparison_id": "PW_1", "aggregate_status": "scored",
            "editorial_consensus": "left", "performance_consensus": "right",
        }]
        human_rows = [{
            "comparison_id": "PW_1", "annotator_id": "A01", "editorial_preference": "left",
            "performance_preference": "right", "confidence_1_to_5": "4", "insufficient_evidence": "",
        }]
        _, summary = human_metrics(aggregates, human_rows, required_annotators=3)
        self.assertEqual(summary["completed_label_row_count"], 1)
        self.assertEqual(summary["comparison_count"], 0)
        self.assertEqual(summary["comparison_coverage"], 0.0)

    def test_human_insufficient_evidence_is_an_abstention(self) -> None:
        aggregates = [{
            "judge_run_id": "judge", "comparison_id": "PW_1", "aggregate_status": "abstain",
            "editorial_consensus": "abstain", "performance_consensus": "abstain",
        }]
        human_rows = [
            {
                "comparison_id": "PW_1", "annotator_id": f"A0{index}", "editorial_preference": "",
                "performance_preference": "", "confidence_1_to_5": "2", "insufficient_evidence": "yes",
            }
            for index in range(1, 4)
        ]
        alignment, summary = human_metrics(aggregates, human_rows, required_annotators=3)
        self.assertEqual(summary["comparison_count"], 1)
        self.assertEqual(summary["comparison_coverage"], 1.0)
        self.assertEqual(alignment[0]["editorial_human_agreement"], 1.0)

    def test_pairwise_rebuild_preserves_matching_human_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "human.csv"
            path.write_text(
                "comparison_id,annotator_id,left_candidate_id,right_candidate_id,editorial_preference,"
                "performance_preference,confidence_1_to_5,insufficient_evidence,notes\n"
                "PW_1,A01,C_L,C_R,left,right,4,false,kept\n",
                encoding="utf-8",
            )
            rows = [{
                "comparison_id": "PW_1", "annotator_id": "A01", "left_candidate_id": "C_L",
                "right_candidate_id": "C_R", "editorial_preference": "", "performance_preference": "",
                "confidence_1_to_5": "", "insufficient_evidence": "", "notes": "",
            }]
            preserved = preserve_human_responses(path, rows)
        self.assertEqual(preserved, 1)
        self.assertEqual(rows[0]["editorial_preference"], "left")
        self.assertEqual(rows[0]["notes"], "kept")

    @patch("llm_client.post_json")
    def test_gemini_json_request_and_response(self, post_json_mock) -> None:
        post_json_mock.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"judgments": []}'}]}}],
            "usageMetadata": {"promptTokenCount": 10},
        }
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            response = call_gemini("gemini-3.5-flash", "system", "user", max_tokens=800)

        self.assertEqual(response["json"], {"judgments": []})
        url, payload, headers = post_json_mock.call_args.args
        self.assertIn("gemini-3.5-flash:generateContent", url)
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(headers["x-goog-api-key"], "test-key")

    @patch("llm_client.post_json")
    def test_openrouter_json_request_and_response(self, post_json_mock) -> None:
        post_json_mock.return_value = {
            "choices": [{"message": {"content": '{"judgments": []}'}}],
            "usage": {"prompt_tokens": 10},
        }
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            response = call_openrouter(
                "qwen/qwen3.7-plus", "system", "user", max_tokens=800
            )

        self.assertEqual(response["json"], {"judgments": []})
        url, payload, headers = post_json_mock.call_args.args[:3]
        self.assertEqual(url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(headers["Authorization"], "Bearer test-key")

    @patch("llm_client.post_json")
    def test_gemini_36_uses_high_thinking_without_temperature(self, post_json_mock) -> None:
        post_json_mock.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"judgments": []}'}]}}],
            "usageMetadata": {"promptTokenCount": 10},
        }
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            call_gemini("gemini-3.6-flash", "system", "user", max_tokens=800)

        _, payload, _ = post_json_mock.call_args.args
        config = payload["generationConfig"]
        self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "high"})
        self.assertNotIn("temperature", config)

    @patch("llm_client.post_json")
    def test_gemini_video_pair_uses_blind_clipped_youtube_inputs(self, post_json_mock) -> None:
        post_json_mock.return_value = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({"comparison_id": "PW_1"})}]}}],
            "usageMetadata": {"promptTokenCount": 10},
        }
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            response = call_gemini_video_pair(
                "gemini-3.5-flash",
                "system",
                "blind comparison",
                "https://www.youtube.com/watch?v=LEFT",
                10,
                40,
                "https://www.youtube.com/watch?v=RIGHT",
                60,
                100,
                max_tokens=800,
                fps=2.0,
            )

        self.assertEqual(response["json"], {"comparison_id": "PW_1"})
        _, payload, _ = post_json_mock.call_args.args
        parts = payload["contents"][0]["parts"]
        self.assertEqual(parts[1]["fileData"]["fileUri"], "https://www.youtube.com/watch?v=LEFT")
        self.assertEqual(parts[1]["videoMetadata"]["startOffset"], "10s")
        self.assertEqual(parts[1]["videoMetadata"]["endOffset"], "40s")
        self.assertEqual(parts[1]["videoMetadata"]["fps"], 2.0)
        self.assertEqual(parts[3]["fileData"]["fileUri"], "https://www.youtube.com/watch?v=RIGHT")

    @patch("llm_client.post_json")
    def test_gemini_video_batch_uses_ten_or_fewer_clips(self, post_json_mock) -> None:
        post_json_mock.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"judgments": []}'}]}}],
        }
        comparisons = [
            {
                "comparison_id": f"PW_{index}",
                "left": {"url": f"https://www.youtube.com/watch?v=L{index}", "start_sec": 1, "end_sec": 11},
                "right": {"url": f"https://www.youtube.com/watch?v=R{index}", "start_sec": 2, "end_sec": 12},
            }
            for index in range(5)
        ]
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            call_gemini_video_batch(
                "gemini-2.5-flash", "system", "batch", comparisons, max_tokens=12000, fps=2.0
            )

        _, payload, _ = post_json_mock.call_args.args
        parts = payload["contents"][0]["parts"]
        video_parts = [part for part in parts if "fileData" in part]
        self.assertEqual(len(video_parts), 10)
        schema = payload["generationConfig"]["responseJsonSchema"]
        self.assertEqual(schema["properties"]["judgments"]["minItems"], 5)

    @patch("llm_client.post_json")
    def test_gemini_text_batch_uses_the_same_structured_schema(self, post_json_mock) -> None:
        post_json_mock.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"judgments": []}'}]}}],
        }
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            call_gemini_text_batch(
                "gemini-3.1-flash-lite", "system", "comparisons", comparison_count=4, max_tokens=12000
            )

        _, payload, _ = post_json_mock.call_args.args
        self.assertEqual(payload["generationConfig"]["temperature"], 0)
        self.assertEqual(payload["contents"][0]["parts"], [{"text": "comparisons"}])
        schema = payload["generationConfig"]["responseJsonSchema"]
        self.assertEqual(schema["properties"]["judgments"]["minItems"], 4)

    @patch("llm_client.post_json")
    def test_gemini_video_pointwise_batch_attaches_one_clip_per_candidate(self, post_json_mock) -> None:
        post_json_mock.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"judgments": []}'}]}}],
        }
        candidates = [
            {
                "candidate_id": f"C_{index}",
                "url": f"https://www.youtube.com/watch?v=V{index}",
                "start_sec": index,
                "end_sec": index + 20,
            }
            for index in range(5)
        ]
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            call_gemini_video_pointwise_batch(
                "gemini-3.1-flash-lite", "system", "candidates", candidates, max_tokens=12000, fps=2.0
            )

        _, payload, _ = post_json_mock.call_args.args
        video_parts = [part for part in payload["contents"][0]["parts"] if "fileData" in part]
        self.assertEqual(len(video_parts), 5)
        schema = payload["generationConfig"]["responseJsonSchema"]
        self.assertEqual(schema["properties"]["judgments"]["minItems"], 5)

    def test_pointwise_normalization_keeps_editorial_and_performance_separate(self) -> None:
        response = {
            "judgments": [
                {
                    "candidate_id": "C_1",
                    "verdict": "score",
                    "evidence": {name: 4 for name in ("description_support", "transcript_intelligibility", "boundary_observability")},
                    "editorial": {name: 5 for name in EDITORIAL_DIMENSIONS},
                    "performance": {name: 1 for name in PERFORMANCE_DIMENSIONS},
                    "confidence": 4,
                    "failure_flags": [],
                    "reason": "complete but not engaging",
                }
            ]
        }
        rows = normalize_pointwise_judgments(
            response,
            {"C_1"},
            {name: 1 for name in EDITORIAL_DIMENSIONS},
            {name: 1 for name in PERFORMANCE_DIMENSIONS},
        )
        self.assertEqual(rows[0]["editorial_score"], 100.0)
        self.assertEqual(rows[0]["performance_score"], 0.0)

    def test_weighted_score_anchors(self) -> None:
        weights = {name: 1.0 for name in CANDIDATE_DIMENSIONS}
        self.assertEqual(weighted_score({name: 1 for name in CANDIDATE_DIMENSIONS}, weights, CANDIDATE_DIMENSIONS), 0.0)
        self.assertEqual(weighted_score({name: 5 for name in CANDIDATE_DIMENSIONS}, weights, CANDIDATE_DIMENSIONS), 100.0)

    def test_candidate_response_requires_every_id(self) -> None:
        weights = {name: 1.0 for name in CANDIDATE_DIMENSIONS}
        response = {
            "judgments": [
                {
                    "candidate_id": "C_1",
                    "scores": {name: 4 for name in CANDIDATE_DIMENSIONS},
                    "failure_flags": [],
                    "reason": "complete",
                }
            ]
        }
        rows = normalize_candidate_judgments(response, {"C_1"}, weights)
        self.assertEqual(rows[0]["overall_score"], 75.0)
        self.assertEqual(rows[0]["verdict"], "score")

    def test_abstention_does_not_create_a_zero_quality_score(self) -> None:
        weights = {name: 1.0 for name in CANDIDATE_DIMENSIONS}
        response = {
            "judgments": [
                {
                    "candidate_id": "C_1",
                    "verdict": "abstain",
                    "evidence": {name: 1 for name in EVIDENCE_DIMENSIONS},
                    "scores": None,
                    "confidence": 4,
                    "failure_flags": ["asr_degraded"],
                    "reason": "insufficient evidence",
                }
            ]
        }
        row = normalize_candidate_judgments(response, {"C_1"}, weights)[0]
        self.assertEqual(row["overall_score"], "")
        self.assertIn("insufficient_evidence", row["failure_flags"])

        score_rows = [
            {
                "judge_run_id": "gpt",
                "provider": "openai",
                "model": "model",
                "repeat_index": "1",
                "candidate_id": "C_1",
                "long_video_id": "L1",
                **row,
            }
        ]
        aggregate = aggregate_candidate_scores(score_rows)[0]
        self.assertEqual(aggregate["aggregate_status"], "abstain")
        self.assertIsNone(aggregate["overall_score_mean"])
        coverage = evidence_coverage(score_rows)[0]
        self.assertEqual(coverage["candidate_scoring_coverage"], 0.0)

    def test_spearman_and_fleiss(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]) or 0.0, 1.0)
        kappa = fleiss_kappa({"a": ["left", "left", "left"], "b": ["right", "right", "right"]})
        self.assertGreater(kappa or 0.0, 0.9)

    def test_prediction_source_deduplicates_pair_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            path.write_text(
                "long_video_id,rank,pred_start_sec,pred_end_sec,run_id,notes\n"
                "L1,1,10,40,r1,\n"
                "L1,1,10,40,r1,\n"
                "L1,2,50,80,r1,\n",
                encoding="utf-8",
            )
            rows = source_rows(path, "ours", top_k=5)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_system"], "ours")
        self.assertEqual(blind_id("L1", 10, 40, "salt"), blind_id("L1", 10, 40, "salt"))

    def test_gold_source_rows_accepts_balanced_dataset_columns(self) -> None:
        rows = gold_source_rows(
            [
                {
                    "long_video_id": "L1",
                    "start_sec": "12.5",
                    "end_sec": "42.5",
                    "pair_id": "P1",
                    "short_video_id": "S1",
                    "performance_label": "neg",
                    "source_notes": "balanced",
                    "performance_evidence_status": "verified",
                    "_dataset_split": "control",
                }
            ]
        )
        self.assertEqual(rows[0]["start_sec"], 12.5)
        self.assertEqual(rows[0]["end_sec"], 42.5)
        self.assertEqual(rows[0]["source_notes"], "balanced")
        self.assertEqual(rows[0]["performance_evidence_status"], "verified")

    def test_subtitle_evidence_prefers_short_transcript_and_uses_long_context(self) -> None:
        short = [{"start_sec": 0.0, "end_sec": 2.0, "text": "short exact"}]
        long = [
            {"start_sec": 8.0, "end_sec": 10.0, "text": "before"},
            {"start_sec": 10.0, "end_sec": 20.0, "text": "within"},
            {"start_sec": 20.0, "end_sec": 22.0, "text": "after"},
        ]
        evidence = subtitle_evidence(short, long, 10.0, 20.0, 5.0)
        self.assertIn("short exact", evidence["transcript"])
        self.assertNotIn("within", evidence["transcript"])
        self.assertIn("before", evidence["before_context"])
        self.assertIn("after", evidence["after_context"])
        self.assertEqual(evidence["content_duration_sec"], 2.0)
        self.assertEqual(evidence["evidence_source"], "short_subtitle_with_long_context")

    def test_subtitle_evidence_can_force_longform_interval(self) -> None:
        short = [{"start_sec": 0.0, "end_sec": 2.0, "text": "short exact"}]
        long = [
            {"start_sec": 8.0, "end_sec": 10.0, "text": "before"},
            {"start_sec": 10.0, "end_sec": 20.0, "text": "within"},
            {"start_sec": 20.0, "end_sec": 22.0, "text": "after"},
        ]
        evidence = subtitle_evidence(
            short,
            long,
            10.0,
            20.0,
            5.0,
            transcript_mode="long_only",
            short_video_id="S1",
            long_video_id="L1",
        )
        self.assertNotIn("short exact", evidence["transcript"])
        self.assertIn("within", evidence["transcript"])
        self.assertEqual(evidence["content_duration_sec"], 10.0)
        self.assertEqual(evidence["evidence_source"], "long_subtitle_interval")
        self.assertEqual(evidence["evidence_provider"], "yt_dlp")
        self.assertEqual(evidence["transcript_scope"], "longform_gold_interval")
        self.assertEqual(evidence["transcript_video_id"], "L1")

    def test_load_subtitle_cues_reads_json3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S1.json3"
            path.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "tStartMs": 1000,
                                "dDurationMs": 2000,
                                "segs": [{"utf8": "hello"}, {"utf8": " world"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cues = load_subtitle_cues([Path(tmp)], "S1")
        self.assertEqual(cues, [{"start_sec": 1.0, "end_sec": 3.0, "text": "hello world"}])

    def test_gold_performance_labels_are_separate_from_gold_role(self) -> None:
        self.assertEqual(performance_label("main"), "pos")
        self.assertEqual(performance_label("control"), "neg")
        self.assertEqual(performance_label("pilot"), "unlabeled")
        self.assertEqual(performance_label("pilot", 75.0), "pos")
        self.assertEqual(performance_label("pilot", 25.0), "neg")
        self.assertEqual(performance_label("pilot", 50.0), "unlabeled")

        aggregates = [
            {"judge_run_id": "gpt", "candidate_id": "P", "overall_score_mean": 75},
            {"judge_run_id": "gpt", "candidate_id": "N", "overall_score_mean": 25},
        ]
        sources = [
            {"source_system": "gold", "candidate_id": "P", "performance_label": "pos"},
            {"source_system": "gold", "candidate_id": "N", "performance_label": "neg"},
        ]
        result = performance_group_alignment(aggregates, sources)[0]
        self.assertEqual(result["pos_count"], 1)
        self.assertEqual(result["neg_count"], 1)
        self.assertEqual(result["pos_over_neg_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
