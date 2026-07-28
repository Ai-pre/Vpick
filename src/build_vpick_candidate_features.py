from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from segments import extract_scene_list


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLIND = ROOT / "results" / "gold_reference_judge_v9_v7" / "input" / "candidates_blind_v7.csv"
DEFAULT_PRIVATE = (
    ROOT / "results" / "gold_reference_judge_v8_ko" / "input" / "candidate_sources_private.csv"
)
DEFAULT_GOLD = ROOT / "deliverables" / "2026-07-23" / "vpick_goldlabel_60_normalized.csv"
DEFAULT_RAW = ROOT / "data" / "raw" / "vpick"
DEFAULT_FALLBACK = ROOT / "data" / "raw" / "subtitle_fallback_scenes"
DEFAULT_OUTPUT_DIR = ROOT / "deliverables" / "2026-07-24" / "performance_ranker"

TIMED_LINE_RE = re.compile(
    r"^\[(?P<start>\d{1,2}:\d{2}(?::\d{2})?)-(?P<end>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<text>.*)$"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def clock_to_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return (parts[0] * 60.0) + parts[1]
    if len(parts) == 3:
        return (parts[0] * 3600.0) + (parts[1] * 60.0) + parts[2]
    return 0.0


def transcript_stats(text: str, duration_sec: float) -> dict[str, float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    timed = []
    speakers: list[str] = []
    for line in lines:
        match = TIMED_LINE_RE.match(line)
        if not match:
            continue
        timed.append(
            (
                clock_to_seconds(match.group("start")),
                clock_to_seconds(match.group("end")),
                match.group("text"),
            )
        )
        speaker_match = re.match(r"(?:>>\s*)?(S\d+|[^:]{1,20}):", match.group("text"))
        if speaker_match:
            speakers.append(speaker_match.group(1))
    turns = sum(1 for left, right in zip(speakers, speakers[1:]) if left != right)
    speech_intervals = [(start, end) for start, end, _ in timed]
    nonspace_chars = len(re.sub(r"\s+", "", text))
    return {
        "transcript_char_count": float(nonspace_chars),
        "transcript_line_count": float(len(lines)),
        "transcript_timed_line_count": float(len(timed)),
        "transcript_unique_speakers": float(len(set(speakers))),
        "transcript_speaker_turns": float(turns),
        "transcript_question_count": float(text.count("?")),
        "transcript_exclamation_count": float(text.count("!")),
        "transcript_speech_coverage_ratio": (
            min(1.0, union_length(speech_intervals) / duration_sec) if duration_sec > 0 else 0.0
        ),
    }


def hangul_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    hangul = sum("\uac00" <= char <= "\ud7a3" for char in visible)
    return hangul / len(visible)


def mojibake_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    suspicious = sum(
        char == "\ufffd"
        or char == "?"
        or "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        for char in visible
    )
    return suspicious / len(visible)


def text_is_usable(text: str) -> bool:
    if len(text.strip()) < 8:
        return False
    return hangul_ratio(text) >= 0.10 and mojibake_ratio(text) <= 0.35


def seconds_to_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_vpick_speeches(speeches: list[dict[str, Any]]) -> str:
    lines = []
    for speech in sorted(
        speeches,
        key=lambda item: (float(item["start_sec"]), float(item["end_sec"])),
    ):
        speaker_id = str(speech.get("speaker_id") or "").strip()
        speaker = f"S{speaker_id}" if speaker_id else "S?"
        lines.append(
            f"[{seconds_to_clock(float(speech['start_sec']))}-"
            f"{seconds_to_clock(float(speech['end_sec']))}] "
            f"{speaker}: {str(speech.get('text') or '').strip()}"
        )
    return "\n".join(lines)


def vpick_transcript_is_usable(text: str, speech_count: int) -> bool:
    return (
        speech_count >= 2
        and len(re.sub(r"\s+", "", text)) >= 40
        and hangul_ratio(text) >= 0.10
        and mojibake_ratio(text) <= 0.35
    )


def build_judge_transcript(
    scene_transcript: str,
    yt_dlp_transcript: str,
    scene_provider: str = "vpick_scene_api",
) -> tuple[str, str]:
    is_vpick = scene_provider == "vpick_scene_api"
    scene_label = "VPICK_ASR" if is_vpick else "SUBTITLE_FALLBACK"
    scene_source = (
        "vpick_scene_api_asr"
        if is_vpick
        else "yt_dlp_full_longform_scene_transcript"
    )
    if scene_transcript and yt_dlp_transcript:
        return (
            f"[{scene_label}]\n"
            f"{scene_transcript}\n\n"
            "[YT_DLP_CAPTIONS]\n"
            f"{yt_dlp_transcript}",
            f"{scene_source}+yt_dlp_candidate_transcript",
        )
    if scene_transcript:
        return scene_transcript, scene_source
    return yt_dlp_transcript, "yt_dlp_candidate_transcript"


def load_scene_payload(
    raw_dir: Path,
    long_video_id: str,
    fallback_dir: Path = DEFAULT_FALLBACK,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, Path | None]:
    scene_path = raw_dir / f"{long_video_id}_scenes.json"
    if not scene_path.exists():
        scene_path = fallback_dir / f"{long_video_id}_scenes.json"
        if not scene_path.exists():
            return [], {}, "", None
        provider = "yt_dlp_transcript_fallback"
    else:
        provider = "vpick_scene_api"
    payload = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    summary = payload.get("summary") if isinstance(payload, dict) else {}
    return (
        extract_scene_list(payload),
        summary if isinstance(summary, dict) else {},
        provider,
        scene_path,
    )


def load_asset_status(raw_dir: Path, long_video_id: str) -> dict[str, Any]:
    path = raw_dir / f"{long_video_id}_asset_status.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    for state_path in (raw_dir / "accounts").glob("*/missing_analysis_state.json"):
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        asset_id = str((state.get("assets", {}).get(long_video_id) or {}).get("asset_id") or "")
        account_status = state_path.parent / f"{asset_id}_asset_status.json"
        if asset_id and account_status.exists():
            payload = json.loads(account_status.read_text(encoding="utf-8-sig"))
            return payload if isinstance(payload, dict) else {}
    return {}


def load_account_sources(raw_dir: Path) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = {}
    accounts_dir = raw_dir / "accounts"
    if not accounts_dir.exists():
        return sources
    for inventory_path in accounts_dir.glob("*/inventory.csv"):
        account_label = inventory_path.parent.name
        for row in read_csv(inventory_path):
            long_video_id = row.get("long_video_id", "").strip()
            if long_video_id and row.get("status") == "READY":
                sources.setdefault(long_video_id, set()).add(account_label)
    return sources


def clip_interval(start: float, end: float, window_start: float, window_end: float) -> tuple[float, float]:
    return max(start, window_start), min(end, window_end)


def vpick_features(
    scenes: list[dict[str, Any]],
    asset_status: dict[str, Any],
    start_sec: float,
    end_sec: float,
    *,
    preserve_raw_transcript: bool = False,
) -> tuple[dict[str, Any], str, str]:
    duration = max(0.001, end_sec - start_sec)
    overlapping = [
        scene
        for scene in scenes
        if interval_overlap(start_sec, end_sec, scene["start_sec"], scene["end_sec"]) > 0
    ]
    scene_intervals = [
        clip_interval(scene["start_sec"], scene["end_sec"], start_sec, end_sec)
        for scene in overlapping
    ]
    scene_coverage = union_length(scene_intervals)
    speeches: dict[tuple[str, float, float, str], dict[str, Any]] = {}
    speech_intervals: list[tuple[float, float]] = []
    persons: set[str] = set()
    fallback_count = 0
    descriptions: list[str] = []
    names: list[str] = []

    for scene in overlapping:
        raw = scene.get("raw") or {}
        if raw.get("is_fallback"):
            fallback_count += 1
        for person in raw.get("persons") or []:
            person_id = person.get("person_id") if isinstance(person, dict) else None
            if person_id:
                persons.add(str(person_id))
        name = str(scene.get("name") or "").strip()
        description = str(scene.get("description") or "").strip()
        if name:
            names.append(name)
        if description:
            descriptions.append(description)
        for speech in scene.get("speeches") or []:
            if interval_overlap(
                start_sec,
                end_sec,
                float(speech["start_sec"]),
                float(speech["end_sec"]),
            ) <= 0:
                continue
            speech_key = (
                str(speech.get("speech_id") or ""),
                float(speech["start_sec"]),
                float(speech["end_sec"]),
                str(speech.get("text") or ""),
            )
            speeches[speech_key] = speech
            speech_intervals.append(
                clip_interval(
                    float(speech["start_sec"]),
                    float(speech["end_sec"]),
                    start_sec,
                    end_sec,
                )
            )

    speakers = {str(speech.get("speaker_id") or "") for speech in speeches.values()}
    speakers.discard("")
    raw_text = " ".join(names + descriptions).strip()
    usable_text = raw_text if text_is_usable(raw_text) else ""
    vpick_transcript = format_vpick_speeches(list(speeches.values()))
    usable_transcript = (
        vpick_transcript
        if vpick_transcript_is_usable(vpick_transcript, len(speeches))
        else ""
    )
    scene_boundaries = [
        boundary
        for scene in scenes
        for boundary in (float(scene["start_sec"]), float(scene["end_sec"]))
    ]
    start_distance = min((abs(start_sec - boundary) for boundary in scene_boundaries), default=math.nan)
    end_distance = min((abs(end_sec - boundary) for boundary in scene_boundaries), default=math.nan)

    features = {
        "vpick_available": int(bool(scenes)),
        "vpick_text_usable": int(bool(usable_text)),
        "vpick_scene_count": len(overlapping),
        "vpick_scene_change_rate_per_min": max(0, len(overlapping) - 1) * 60.0 / duration,
        "vpick_scene_coverage_ratio": min(1.0, scene_coverage / duration),
        "vpick_fallback_scene_ratio": fallback_count / len(overlapping) if overlapping else 0.0,
        "vpick_person_count": len(persons),
        "vpick_speech_count": len(speeches),
        "vpick_speech_rate_per_min": len(speeches) * 60.0 / duration,
        "vpick_unique_speakers": len(speakers),
        "vpick_speech_coverage_ratio": min(1.0, union_length(speech_intervals) / duration),
        "vpick_transcript_usable": int(bool(usable_transcript)),
        "vpick_transcript_char_count": len(re.sub(r"\s+", "", vpick_transcript)),
        "vpick_transcript_hangul_ratio": hangul_ratio(vpick_transcript),
        "vpick_transcript_mojibake_ratio": mojibake_ratio(vpick_transcript),
        "vpick_description_char_count": len(re.sub(r"\s+", "", raw_text)),
        "vpick_description_hangul_ratio": hangul_ratio(raw_text),
        "vpick_description_mojibake_ratio": mojibake_ratio(raw_text),
        "vpick_start_boundary_distance_sec": start_distance,
        "vpick_end_boundary_distance_sec": end_distance,
        "vpick_start_aligned_2s": int(not math.isnan(start_distance) and start_distance <= 2.0),
        "vpick_end_aligned_2s": int(not math.isnan(end_distance) and end_distance <= 2.0),
        "vpick_asset_duration_sec": to_float(asset_status.get("duration_ms")) / 1000.0,
        "vpick_asset_resolution": to_float(asset_status.get("resolution")),
        "vpick_asset_person_count": len(asset_status.get("persons") or []),
        "vpick_asset_fallback_count": to_float(asset_status.get("fallback_count")),
    }
    transcript_output = vpick_transcript if preserve_raw_transcript else usable_transcript
    return features, usable_text[:1800], transcript_output[:8000]


def title_features(short_title: str, long_title: str) -> dict[str, float]:
    return {
        "short_title_char_count": float(len(re.sub(r"\s+", "", short_title))),
        "short_title_question_count": float(short_title.count("?")),
        "short_title_exclamation_count": float(short_title.count("!")),
        "short_title_has_number": float(bool(re.search(r"\d", short_title))),
        "long_title_char_count": float(len(re.sub(r"\s+", "", long_title))),
    }


def build_rows(
    blind_path: Path,
    private_path: Path,
    gold_path: Path,
    raw_dir: Path,
    fallback_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    blind_rows = read_csv(blind_path)
    private_rows = read_csv(private_path)
    gold_rows = read_csv(gold_path)
    blind_by_id = {row["candidate_id"]: row for row in blind_rows}
    private_by_id = {row["candidate_id"]: row for row in private_rows}
    gold_by_pair = {row.get("pair_id", ""): row for row in gold_rows if row.get("pair_id")}
    gold_by_short = {
        row.get("short_video_id", ""): row for row in gold_rows if row.get("short_video_id")
    }
    account_sources = load_account_sources(raw_dir)
    if set(blind_by_id) != set(private_by_id):
        raise ValueError("Blind candidate IDs and private manifest candidate IDs do not match.")

    scene_cache: dict[
        str,
        tuple[list[dict[str, Any]], dict[str, Any], str, Path | None],
    ] = {}
    asset_cache: dict[str, dict[str, Any]] = {}
    feature_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []

    for candidate_id in blind_by_id:
        blind = blind_by_id[candidate_id]
        private = private_by_id[candidate_id]
        long_video_id = private["long_video_id"]
        gold = gold_by_pair.get(private.get("pair_id", "")) or gold_by_short.get(
            private.get("short_video_id", "")
        ) or {}
        if long_video_id not in scene_cache:
            scene_cache[long_video_id] = load_scene_payload(
                raw_dir,
                long_video_id,
                fallback_dir,
            )
            asset_cache[long_video_id] = load_asset_status(raw_dir, long_video_id)
        scenes, _summary, scene_provider, scene_path = scene_cache[long_video_id]
        start_sec = to_float(private.get("start_sec"))
        end_sec = to_float(private.get("end_sec"))
        duration_sec = max(0.0, end_sec - start_sec) or to_float(blind.get("duration_sec"))
        scene_metrics, visual_description, scene_transcript = vpick_features(
            scenes,
            asset_cache[long_video_id] if scene_provider == "vpick_scene_api" else {},
            start_sec,
            end_sec,
        )
        is_vpick = scene_provider == "vpick_scene_api"
        vpick = {
            key: (value if is_vpick else 0)
            for key, value in scene_metrics.items()
        }
        yt_dlp_transcript = blind.get("transcript", "")
        transcript, transcript_source = build_judge_transcript(
            scene_transcript,
            yt_dlp_transcript,
            scene_provider or "vpick_scene_api",
        )
        short_title = gold.get("short_title_yt", "")
        long_title = gold.get("long_title_yt", "")
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "pair_id": private.get("pair_id", ""),
            "long_video_id": long_video_id,
            "short_video_id": private.get("short_video_id", ""),
            "channel_name": private.get("channel_name", ""),
            "performance_label": private.get("performance_label", ""),
            "channel_performance_percentile": to_float(
                private.get("channel_performance_percentile")
            ),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "short_title": short_title,
            "long_title": long_title,
            "upload_date": gold.get("upload_date", ""),
            "transcript_source": transcript_source,
            "description_source": scene_provider if visual_description else "",
            "scene_evidence_available": int(bool(scenes)),
            "scene_evidence_provider": scene_provider,
            "scene_evidence_visual_available": int(is_vpick),
            "scene_evidence_file": str(scene_path) if scene_path else "",
            "subtitle_fallback_available": int(
                scene_provider == "yt_dlp_transcript_fallback"
            ),
            "subtitle_fallback_scene_count": (
                scene_metrics.get("vpick_scene_count", 0) if not is_vpick else 0
            ),
            "subtitle_fallback_transcript_usable": (
                scene_metrics.get("vpick_transcript_usable", 0) if not is_vpick else 0
            ),
            "youtube_metadata_source": gold.get("youtube_metadata_source", ""),
            "vpick_scene_file": (
                f"{long_video_id}_scenes.json" if is_vpick else ""
            ),
            "vpick_account_sources": "|".join(sorted(account_sources.get(long_video_id, set()))),
            "vpick_source_status": (
                "current_account_api"
                if account_sources.get(long_video_id)
                else "cached_scene_json"
                if is_vpick
                else "unavailable"
            ),
            "description": visual_description,
            "transcript": transcript,
            "yt_dlp_transcript": yt_dlp_transcript,
            "vpick_transcript": scene_transcript if is_vpick else "",
            "subtitle_fallback_transcript": scene_transcript if not is_vpick else "",
            "judge_transcript_char_count": len(re.sub(r"\s+", "", transcript)),
            "before_context": blind.get("before_context", ""),
            "after_context": blind.get("after_context", ""),
        }
        row.update(transcript_stats(yt_dlp_transcript, duration_sec))
        row.update(title_features(short_title, long_title))
        row.update(vpick)
        feature_rows.append(row)
        judge_rows.append(
            {
                "candidate_id": candidate_id,
                "duration_sec": f"{duration_sec:.3f}",
                "description": visual_description,
                "transcript": transcript,
                "before_context": blind.get("before_context", ""),
                "after_context": blind.get("after_context", ""),
            }
        )

    covered = [row for row in feature_rows if row["vpick_available"]]
    text_usable = [row for row in feature_rows if row["vpick_text_usable"]]
    summary = {
        "candidate_count": len(feature_rows),
        "unique_longform_count": len({row["long_video_id"] for row in feature_rows}),
        "vpick_covered_candidate_count": len(covered),
        "vpick_covered_longform_count": len({row["long_video_id"] for row in covered}),
        "vpick_text_usable_candidate_count": len(text_usable),
        "vpick_transcript_used_candidate_count": sum(
            "vpick_scene_api_asr" in row["transcript_source"] for row in feature_rows
        ),
        "description_filled_candidate_count": sum(bool(row["description"]) for row in feature_rows),
        "scene_evidence_covered_candidate_count": sum(
            row["scene_evidence_available"] for row in feature_rows
        ),
        "scene_evidence_covered_longform_count": len(
            {
                row["long_video_id"]
                for row in feature_rows
                if row["scene_evidence_available"]
            }
        ),
        "subtitle_fallback_candidate_count": sum(
            row["subtitle_fallback_available"] for row in feature_rows
        ),
        "subtitle_fallback_longform_count": len(
            {
                row["long_video_id"]
                for row in feature_rows
                if row["subtitle_fallback_available"]
            }
        ),
        "coverage_by_channel": {},
        "coverage_by_performance_label": {},
        "current_account_longform_count": len(
            {
                row["long_video_id"]
                for row in feature_rows
                if row["vpick_account_sources"]
            }
        ),
        "cached_only_longform_ids": sorted(
            {
                row["long_video_id"]
                for row in feature_rows
                if row["vpick_source_status"] == "cached_scene_json"
            }
        ),
        "missing_longform_ids": sorted(
            {
                row["long_video_id"]
                for row in feature_rows
                if row["vpick_source_status"] == "unavailable"
            }
        ),
        "missing_scene_evidence_longform_ids": sorted(
            {
                row["long_video_id"]
                for row in feature_rows
                if not row["scene_evidence_available"]
            }
        ),
    }
    for channel in sorted({row["channel_name"] for row in feature_rows}):
        channel_rows = [row for row in feature_rows if row["channel_name"] == channel]
        summary["coverage_by_channel"][channel] = {
            "covered": sum(row["vpick_available"] for row in channel_rows),
            "text_usable": sum(row["vpick_text_usable"] for row in channel_rows),
            "total": len(channel_rows),
        }
    for label in sorted({row["performance_label"] for row in feature_rows}):
        label_rows = [row for row in feature_rows if row["performance_label"] == label]
        summary["coverage_by_performance_label"][label] = {
            "covered": sum(row["vpick_available"] for row in label_rows),
            "total": len(label_rows),
        }
    return feature_rows, judge_rows, summary


def longform_coverage_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        grouped.setdefault(str(row["long_video_id"]), []).append(row)
    output = []
    for long_video_id, rows in sorted(grouped.items()):
        first = rows[0]
        output.append(
            {
                "long_video_id": long_video_id,
                "channel_name": first["channel_name"],
                "candidate_count": len(rows),
                "performance_labels": "|".join(
                    sorted({str(row["performance_label"]) for row in rows})
                ),
                "short_video_ids": "|".join(
                    sorted({str(row["short_video_id"]) for row in rows})
                ),
                "vpick_available": first["vpick_available"],
                "vpick_account_sources": first["vpick_account_sources"],
                "vpick_source_status": first["vpick_source_status"],
                "vpick_scene_file": first["vpick_scene_file"],
                "scene_evidence_available": first["scene_evidence_available"],
                "scene_evidence_provider": first["scene_evidence_provider"],
                "scene_evidence_file": first["scene_evidence_file"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--private-manifest", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--fallback-dir", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    feature_rows, judge_rows, summary = build_rows(
        args.blind,
        args.private_manifest,
        args.gold,
        args.raw_dir,
        args.fallback_dir,
    )
    write_csv(args.output_dir / "candidate_features_60_PRIVATE.csv", feature_rows)
    write_csv(args.output_dir / "candidates_blind_v8_vpick_enriched.csv", judge_rows)
    coverage_rows = longform_coverage_rows(feature_rows)
    write_csv(args.output_dir / "vpick_longform_coverage_54_PRIVATE.csv", coverage_rows)
    missing_rows = [row for row in coverage_rows if not row["vpick_available"]]
    write_csv(
        args.output_dir / "vpick_missing_longforms_PRIVATE.csv",
        missing_rows,
    )
    (args.output_dir / "vpick_enrichment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
