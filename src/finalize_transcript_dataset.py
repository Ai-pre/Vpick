#!/usr/bin/env python3
"""Finalize the 60-candidate transcript set after Gemini cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


TIMESTAMP_PREFIX = re.compile(r"^\[[^\]]+\]\s*")
INITIAL_GEMINI_REPAIRED_IDS = {
    "C_519202486aef76",
    "C_4479c41b6aa80d",
}
KEEP_EXISTING_RECHECK_IDS = {"C_4479c41b6aa80d"}
USE_TIMESTAMPED_GEMINI_LONG_IDS = {"C_f9645ed47107f3"}
RECONCILED_TRANSCRIPTS = {
    "C_87c2f9c2d22522": "\n".join(
        [
            "[원본 구간 17:23-18:15; Gemini 쇼츠ㆍ롱폼 교차확인]",
            "S1: 종서야. 대답 안 하나? 사람이 말을 하는데 대답은 해야지.",
            "S2: 응?",
            "S1: 헤어지자.",
            "S2: 왜?",
            "S1: 사실 처음부터 널 별로 안 좋아했어. 네가 갖고 장난친 거야.",
            "S2: 7년을?",
            "S1: 종서야, 제발 부탁이니까 꺼지라.",
            "S2: 싫어.",
            "S3: 저 미소는 뭐였지?",
            "S4: 왜 웃고 있어?",
            "S1: 아니, 진짜 그만하자. 종서야, 진짜 미안하다. 그렇게 쳐다보지 말고 들어가라.",
            "S2: 오빠. 가지 마.",
        ]
    ),
    "C_c3f86bdf30fa21": "\n".join(
        [
            "[원본 구간 17:49-18:06; Gemini 쇼츠ㆍ롱폼 교차확인]",
            "S1: 어떤 스탠스를 취해야 해? 고부 갈등 상황에서?",
            "S2: Put 'em in a ring.",
            "S3: UFC?",
            "S2: Yeah! UFC! And then whoever wins, it's over, right? They don't fight no more.",
            "S4: 근데 네가 잘 중재를...",
            "S2: Listen to what I'm saying.",
            "S5: No, but we never had that.",
            "S2: No, you don't know!",
        ]
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def keyed(rows: list[dict[str, str]], source: str) -> dict[str, dict[str, str]]:
    result = {row["candidate_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{source} contains duplicate candidate IDs")
    return result


def timestamp_label(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    if abs(remainder - round(remainder)) < 0.05:
        return f"{minutes}:{int(round(remainder)):02d}"
    return f"{minutes}:{remainder:04.1f}"


def interval_anchored_transcript(
    transcript: str,
    *,
    start_sec: float,
    end_sec: float,
) -> str:
    lines = []
    for line in transcript.splitlines():
        cleaned = TIMESTAMP_PREFIX.sub("", line.strip())
        if cleaned:
            lines.append(cleaned)
    if not lines:
        raise ValueError("Gemini longform transcript has no usable speech lines")
    header = (
        f"[원본 구간 {timestamp_label(start_sec)}-"
        f"{timestamp_label(end_sec)}; 세부 시각은 사용하지 않음]"
    )
    return "\n".join([header, *lines])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--short-audit", required=True)
    parser.add_argument("--long-recheck", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    candidates = keyed(read_csv(Path(args.candidates)), "candidates")
    manifest = keyed(read_csv(Path(args.manifest)), "manifest")
    audit = keyed(read_csv(Path(args.short_audit)), "short audit")
    recheck = keyed(read_csv(Path(args.long_recheck)), "long recheck")
    expected_ids = set(candidates)
    for name, rows in (("manifest", manifest), ("short audit", audit)):
        if set(rows) != expected_ids:
            raise ValueError(f"{name} candidate IDs do not match candidates")
    if len(expected_ids) != 60:
        raise ValueError(f"Expected exactly 60 candidates, found {len(expected_ids)}")

    flagged_ids = {
        candidate_id
        for candidate_id, row in audit.items()
        if row["needs_longform_recheck"].strip().lower() == "true"
    }
    rechecked_ids = set(recheck)
    if not flagged_ids <= rechecked_ids:
        raise ValueError("Automatic flags are missing from longform recheck")
    unresolved = [
        candidate_id
        for candidate_id, row in recheck.items()
        if row["decision"] != "replace_with_gemini_long"
    ]
    if unresolved:
        raise ValueError(f"Unresolved longform rechecks: {unresolved}")
    strategy_ids = (
        KEEP_EXISTING_RECHECK_IDS
        | USE_TIMESTAMPED_GEMINI_LONG_IDS
        | set(RECONCILED_TRANSCRIPTS)
    )
    if not strategy_ids <= rechecked_ids:
        raise ValueError("Finalization strategies reference missing recheck rows")

    final_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    master_rows: list[dict[str, object]] = []
    for candidate_id in sorted(expected_ids):
        candidate = candidates[candidate_id]
        metadata = manifest[candidate_id]
        gemini = audit[candidate_id]
        is_rechecked = candidate_id in recheck

        if candidate_id in KEEP_EXISTING_RECHECK_IDS:
            final_transcript = candidate["transcript"]
            comparison_result = "existing_gemini_repair_retained"
            final_source = "manifest_interval+existing_gemini_repair"
            reason = (
                "The earlier Gemini long/short repair is more tightly focused "
                "than the later longform recheck; retain the repaired canonical "
                "transcript after cross-validation."
            )
        elif candidate_id in RECONCILED_TRANSCRIPTS:
            final_transcript = RECONCILED_TRANSCRIPTS[candidate_id]
            comparison_result = "cross_source_reconciled_transcript"
            final_source = "manifest_interval+gemini_short_long_reconciliation"
            reason = (
                "Gemini short and longform evidence confirm the same scene, but "
                "neither source alone preserves both exact wording and reliable "
                "timing. Use a conservative reconciled transcript."
            )
        elif candidate_id in USE_TIMESTAMPED_GEMINI_LONG_IDS:
            final_transcript = recheck[candidate_id]["gemini_long_transcript"]
            comparison_result = "gemini_long_timestamp_verified"
            final_source = "gemini_long_transcript"
            reason = (
                "Gemini longform and short transcripts agree, and the returned "
                "longform timestamps fall inside the manifest interval."
            )
        elif is_rechecked:
            final_transcript = interval_anchored_transcript(
                gemini["gemini_short_transcript"],
                start_sec=float(metadata["start_sec"]),
                end_sec=float(metadata["end_sec"]),
            )
            comparison_result = "gemini_short_text_after_long_confirmation"
            final_source = "manifest_interval+gemini_short_transcript"
            reason = (
                "The short transcript is clearer than the source ASR, and an "
                "independent Gemini longform recheck confirms that the same "
                "speech belongs to the mapped source interval."
            )
        else:
            final_transcript = candidate["transcript"]
            if candidate_id in INITIAL_GEMINI_REPAIRED_IDS:
                comparison_result = "existing_gemini_repair_cross_validated"
                final_source = "manifest_interval+existing_gemini_repair"
                reason = (
                    "A prior Gemini long/short repair fixed a structurally "
                    "incomplete transcript, and the 60-item audit cross-validates "
                    "the repaired text."
                )
            else:
                comparison_result = "cross_validated_equivalent"
                final_source = metadata["evidence_provider"]
                reason = (
                    "Canonical and Gemini short transcripts agree; retain the "
                    "canonical source because it preserves longform timestamps "
                    "and surrounding context."
                )
        if not final_transcript.strip():
            raise ValueError(f"Blank final transcript: {candidate_id}")

        final_candidate = dict(candidate)
        final_candidate["transcript"] = final_transcript
        final_rows.append(final_candidate)

        comparison_rows.append(
            {
                "candidate_id": candidate_id,
                "pair_id": metadata["pair_id"],
                "channel_name": metadata["channel_name"],
                "performance_label_PRIVATE": metadata["performance_label"],
                "evidence_provider": metadata["evidence_provider"],
                "gemini_model": gemini["model"],
                "gemini_status": gemini["gemini_status"],
                "gemini_confidence": gemini["gemini_confidence"],
                "sequence_similarity": gemini["sequence_similarity"],
                "containment_overlap": gemini["containment_overlap"],
                "canonical_speech_chars": gemini["canonical_speech_chars"],
                "gemini_short_speech_chars": gemini[
                    "gemini_short_speech_chars"
                ],
                "longform_rechecked": is_rechecked,
                "comparison_result": comparison_result,
                "final_transcript_source": final_source,
                "decision_reason": reason,
                "canonical_transcript": candidate["transcript"],
                "gemini_short_transcript": gemini["gemini_short_transcript"],
                "final_transcript": final_transcript,
            }
        )
        master_rows.append(
            {
                "candidate_id": candidate_id,
                "pair_id": metadata["pair_id"],
                "channel_name": metadata["channel_name"],
                "performance_label_PRIVATE": metadata["performance_label"],
                "channel_performance_percentile_PRIVATE": metadata[
                    "channel_performance_percentile"
                ],
                "label_confidence": metadata["label_confidence"],
                "mapping_confidence": metadata["mapping_confidence"],
                "longform_id": metadata["longform_id"],
                "long_video_url": metadata["long_video_url"],
                "short_video_id": metadata["short_video_id"],
                "short_video_url": metadata["short_video_url"],
                "start_sec": metadata["start_sec"],
                "end_sec": metadata["end_sec"],
                "duration_sec": metadata["duration_sec"],
                "timestamp_method": metadata["timestamp_method"],
                "timestamp_confidence": metadata["timestamp_confidence"],
                "evidence_provider": metadata["evidence_provider"],
                "transcript_validation_status": comparison_result,
                "final_transcript_source": final_source,
                "gemini_model": gemini["model"],
                "gemini_confidence": gemini["gemini_confidence"],
                "description": candidate["description"],
                "transcript": final_transcript,
                "before_context": candidate["before_context"],
                "after_context": candidate["after_context"],
                "dataset_role_v2": metadata["dataset_role_v2"],
                "split_lock_version": metadata["split_lock_version"],
            }
        )

    out_dir = Path(args.out_dir)
    final_path = out_dir / "candidates_transcript_final_60.csv"
    comparison_path = out_dir / "transcript_source_comparison_60_PRIVATE.csv"
    master_path = out_dir / "goldlabel_master_transcript_final_60_PRIVATE.csv"
    write_csv(final_path, final_rows)
    write_csv(comparison_path, comparison_rows)
    write_csv(master_path, master_rows)

    dev_ids = {
        row["candidate_id"] for row in read_csv(Path(args.dev_manifest))
    }
    test_ids = {
        row["candidate_id"] for row in read_csv(Path(args.test_manifest))
    }
    if dev_ids & test_ids or dev_ids | test_ids != expected_ids:
        raise ValueError("Dev/test manifests are not a disjoint 60-ID partition")
    write_csv(
        out_dir / "dev_candidates_transcript_final_12.csv",
        [row for row in final_rows if row["candidate_id"] in dev_ids],
    )
    write_csv(
        out_dir / "locked_test_candidates_transcript_final_48.csv",
        [row for row in final_rows if row["candidate_id"] in test_ids],
    )

    summary = {
        "candidate_count": len(final_rows),
        "dev_count": len(dev_ids),
        "locked_test_count": len(test_ids),
        "gemini_status_counts": dict(
            Counter(row["gemini_status"] for row in comparison_rows)
        ),
        "gemini_model_counts": dict(
            Counter(row["gemini_model"] for row in comparison_rows)
        ),
        "evidence_provider_counts": dict(
            Counter(row["evidence_provider"] for row in comparison_rows)
        ),
        "comparison_result_counts": dict(
            Counter(row["comparison_result"] for row in comparison_rows)
        ),
        "longform_rechecked_candidate_ids": sorted(rechecked_ids),
        "outputs": {
            "final_candidates": str(final_path),
            "comparison_private": str(comparison_path),
            "master_private": str(master_path),
        },
    }
    (out_dir / "transcript_finalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
