from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from language_utils import detect_content_genre, detect_prompt_language
from segments import extract_scene_list, format_speeches, seconds_to_clock


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("source name cannot be empty")
    return name, Path(raw_path)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 999) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def note_value(notes: str, key: str) -> str:
    marker = f"{key}="
    for part in str(notes or "").split(";"):
        part = part.strip()
        if part.startswith(marker):
            return part[len(marker) :].strip()
    return ""


def load_scenes(scenes_dir: Path, long_video_id: str) -> list[dict[str, Any]]:
    path = scenes_dir / f"{long_video_id}_scenes.json"
    if not path.exists():
        return []
    return extract_scene_list(json.loads(path.read_text(encoding="utf-8")))


def compact_longform_overview(
    scenes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": str(scene.get("scene_id", "")),
            "start_ms": round(float(scene.get("start_sec", 0.0)) * 1000),
            "end_ms": round(float(scene.get("end_sec", 0.0)) * 1000),
            "scene_name": str(scene.get("scene_name", ""))[:160],
            "description": str(scene.get("description", ""))[:500],
        }
        for scene in scenes
    ]


def overlapping_scene_ids(
    scenes: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[str]:
    return [
        str(scene.get("scene_id", ""))
        for scene in scenes
        if interval_overlap(
            start,
            end,
            float(scene.get("start_sec", 0.0)),
            float(scene.get("end_sec", 0.0)),
        )
        > 0
        and str(scene.get("scene_id", ""))
    ]


def load_subtitle_cues(cache_dirs: list[Path], video_id: str) -> list[dict[str, Any]]:
    path = next((directory / f"{video_id}.json3" for directory in cache_dirs if (directory / f"{video_id}.json3").exists()), None)
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    cues: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        segments = event.get("segs", [])
        text = "".join(str(segment.get("utf8", "")) for segment in segments).replace("\n", " ").strip()
        if not text:
            continue
        start = float(event.get("tStartMs", 0)) / 1000.0
        duration = float(event.get("dDurationMs", 0)) / 1000.0
        cues.append({"start_sec": start, "end_sec": start + max(duration, 0.001), "text": text})
    return sorted(cues, key=lambda cue: (cue["start_sec"], cue["end_sec"]))


def format_subtitle_cues(cues: list[dict[str, Any]], max_chars: int) -> str:
    lines = [
        f"[{seconds_to_clock(float(cue['start_sec']))}-{seconds_to_clock(float(cue['end_sec']))}] {cue['text']}"
        for cue in cues
    ]
    return "\n".join(lines)[:max_chars]


def subtitle_evidence(
    short_cues: list[dict[str, Any]],
    long_cues: list[dict[str, Any]],
    start: float,
    end: float,
    context_pad_sec: float,
    *,
    transcript_mode: str = "prefer_short",
    short_video_id: str = "",
    long_video_id: str = "",
) -> dict[str, Any]:
    if transcript_mode not in {"prefer_short", "long_only"}:
        raise ValueError(f"Unsupported transcript_mode: {transcript_mode}")
    before = [cue for cue in long_cues if interval_overlap(max(0.0, start - context_pad_sec), start, cue["start_sec"], cue["end_sec"]) > 0]
    within = [cue for cue in long_cues if interval_overlap(start, end, cue["start_sec"], cue["end_sec"]) > 0]
    after = [cue for cue in long_cues if interval_overlap(end, end + context_pad_sec, cue["start_sec"], cue["end_sec"]) > 0]
    uses_short = transcript_mode == "prefer_short" and bool(short_cues)
    transcript_cues = short_cues if uses_short else within
    return {
        "context_start_sec": max(0.0, start - context_pad_sec),
        "context_end_sec": end + context_pad_sec,
        "long_duration_sec": max([float(cue["end_sec"]) for cue in long_cues] or [end]),
        "description": "",
        "transcript": format_subtitle_cues(transcript_cues, 5000),
        "before_context": format_subtitle_cues(before, 2500),
        "after_context": format_subtitle_cues(after, 2500),
        "content_duration_sec": (
            max([float(cue["end_sec"]) for cue in short_cues] or [end - start])
            if uses_short
            else end - start
        ),
        "evidence_source": "short_subtitle_with_long_context" if uses_short else "long_subtitle_interval",
        "evidence_provider": "yt_dlp",
        "transcript_scope": "published_short" if uses_short else "longform_gold_interval",
        "transcript_video_id": short_video_id if uses_short else long_video_id,
    }


def candidate_evidence(
    scenes: list[dict[str, Any]], start: float, end: float, context_pad_sec: float
) -> dict[str, Any]:
    context_start = max(0.0, start - context_pad_sec)
    long_duration = max([float(scene["end_sec"]) for scene in scenes] or [end])
    context_end = min(long_duration, end + context_pad_sec)
    descriptions: list[str] = []
    speeches: list[dict[str, Any]] = []
    before_speeches: list[dict[str, Any]] = []
    after_speeches: list[dict[str, Any]] = []

    for scene in scenes:
        scene_start = float(scene["start_sec"])
        scene_end = float(scene["end_sec"])
        in_candidate = interval_overlap(start, end, scene_start, scene_end) > 0
        in_context = interval_overlap(context_start, context_end, scene_start, scene_end) > 0
        if not in_candidate and not in_context:
            continue
        if in_candidate and scene.get("description"):
            descriptions.append(str(scene["description"]))
        for speech in scene.get("speeches", []):
            speech_start = float(speech["start_sec"])
            speech_end = float(speech["end_sec"])
            if interval_overlap(start, end, speech_start, speech_end) > 0:
                speeches.append(speech)
            if interval_overlap(context_start, start, speech_start, speech_end) > 0:
                before_speeches.append(speech)
            if interval_overlap(end, context_end, speech_start, speech_end) > 0:
                after_speeches.append(speech)

    speech_key = lambda item: (float(item["start_sec"]), float(item["end_sec"]), str(item.get("speech_id", "")))
    speeches = sorted({speech_key(item): item for item in speeches}.values(), key=speech_key)
    before_speeches = sorted({speech_key(item): item for item in before_speeches}.values(), key=speech_key)
    after_speeches = sorted({speech_key(item): item for item in after_speeches}.values(), key=speech_key)
    return {
        "context_start_sec": context_start,
        "context_end_sec": context_end,
        "long_duration_sec": long_duration,
        "description": " ".join(descriptions)[:1800],
        "transcript": format_speeches(speeches)[:5000],
        "before_context": format_speeches(before_speeches)[:2500],
        "after_context": format_speeches(after_speeches)[:2500],
    }


def blind_id(long_video_id: str, start: float, end: float, salt: str) -> str:
    raw = f"{salt}|{long_video_id}|{start:.3f}|{end:.3f}".encode("utf-8")
    return f"C_{hashlib.sha256(raw).hexdigest()[:14]}"


def source_rows(path: Path, source_name: str, top_k: int) -> list[dict[str, Any]]:
    rows = read_csv(path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        long_video_id = row.get("long_video_id", "").strip()
        if long_video_id:
            grouped.setdefault(long_video_id, []).append(row)

    output: list[dict[str, Any]] = []
    for long_video_id, group in grouped.items():
        seen: set[tuple[float, float]] = set()
        ranked = sorted(group, key=lambda row: to_int(row.get("rank")))
        for row in ranked:
            start = round(
                to_float(row.get("pred_start_sec") or row.get("start_sec")),
                3,
            )
            end = round(
                to_float(row.get("pred_end_sec") or row.get("end_sec")),
                3,
            )
            key = (start, end)
            if end <= start or key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "long_video_id": long_video_id,
                    "start_sec": start,
                    "end_sec": end,
                    "source_system": source_name,
                    "source_run_id": (
                        row.get("run_id")
                        or row.get("source_run_id")
                        or ""
                    ),
                    "source_rank": to_int(row.get("rank"), len(seen)),
                    "pair_id": "",
                    "short_video_id": "",
                    "short_views": "",
                    "short_likes": "",
                    "label_confidence": "",
                    "source_notes": row.get("notes", ""),
                    "dataset_split": "",
                    "evaluation_role": "",
                }
            )
            if len(seen) >= top_k:
                break
    return output


def gold_source_rows(dataset: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in dataset:
        start = round(to_float(row.get("gold_start_sec") or row.get("start_sec")), 3)
        end = round(to_float(row.get("gold_end_sec") or row.get("end_sec")), 3)
        if end <= start:
            continue
        output.append(
            {
                "long_video_id": row.get("long_video_id", ""),
                "start_sec": start,
                "end_sec": end,
                "source_system": "gold",
                "source_run_id": "published_short_gold",
                "source_rank": 1,
                "pair_id": row.get("pair_id", ""),
                "short_video_id": row.get("short_video_id", ""),
                "short_views": row.get("short_views", ""),
                "short_likes": row.get("short_likes", ""),
                "channel_name": row.get("channel_name", ""),
                "channel_performance_percentile": row.get("channel_performance_percentile", ""),
                "label_confidence": row.get("label_confidence", ""),
                "source_notes": row.get("label_notes") or row.get("source_notes", ""),
                "dataset_split": row.get("_dataset_split", ""),
                "evaluation_role": "gold",
                "performance_label": row.get("performance_label", ""),
                "mapping_confidence": row.get("mapping_confidence", ""),
                "performance_evidence_status": row.get("performance_evidence_status", ""),
                "alignment_status": row.get("alignment_status", ""),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a blind candidate pool for LLM-as-a-Judge evaluation.")
    parser.add_argument("--dataset", action="append", required=True, help="Gold long-short pair CSV. Repeat to combine splits.")
    parser.add_argument("--source", action="append", type=parse_source, default=[], help="Candidate source NAME=PATH.")
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--subtitle-cache-dir", action="append", default=[])
    parser.add_argument(
        "--evidence-mode",
        choices=("scenes", "subtitles", "long_subtitles", "prefer_scenes", "prefer_subtitles"),
        default="scenes",
    )
    parser.add_argument(
        "--require-evidence-source",
        default="",
        help="Fail if any generated candidate does not use this exact evidence_source.",
    )
    parser.add_argument(
        "--require-uniform-provider",
        action="store_true",
        help="Fail unless every generated candidate uses one non-empty evidence_provider.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-pad-sec", type=float, default=15.0)
    parser.add_argument("--blind-salt", default="vpick-judge-v1")
    parser.add_argument("--exclude-gold", action="store_true")
    args = parser.parse_args()

    dataset: list[dict[str, str]] = []
    dataset_counts: dict[str, int] = {}
    for raw_path in args.dataset:
        path = Path(raw_path)
        rows = read_csv(path)
        default_split = path.stem
        for row in rows:
            split = row.get("dataset_split", "").strip() or default_split
            dataset_counts[split] = dataset_counts.get(split, 0) + 1
            dataset.append({**row, "_dataset_split": split})
    dataset_by_long: dict[str, dict[str, str]] = {}
    for row in dataset:
        dataset_by_long.setdefault(row.get("long_video_id", ""), row)

    all_sources: list[dict[str, Any]] = []
    for source_name, path in args.source:
        all_sources.extend(source_rows(path, source_name, args.top_k))
    if not args.exclude_gold:
        all_sources.extend(gold_source_rows(dataset))

    genre_hints: dict[str, str] = {}
    language_hints: dict[str, str] = {}
    for row in all_sources:
        long_video_id = str(row["long_video_id"])
        notes = str(row.get("source_notes", ""))
        genre = note_value(notes, "detected_genre")
        language = note_value(notes, "detected_language")
        if genre:
            genre_hints[long_video_id] = genre
        if language:
            language_hints[long_video_id] = language

    candidate_rows: dict[str, dict[str, Any]] = {}
    source_manifest: list[dict[str, Any]] = []
    scene_cache: dict[str, list[dict[str, Any]]] = {}
    subtitle_cache: dict[str, list[dict[str, Any]]] = {}
    subtitle_cache_dirs = [Path(path) for path in args.subtitle_cache_dir]
    for source in all_sources:
        long_video_id = str(source["long_video_id"])
        start = float(source["start_sec"])
        end = float(source["end_sec"])
        candidate_id = blind_id(long_video_id, start, end, args.blind_salt)
        if long_video_id not in scene_cache:
            scene_cache[long_video_id] = load_scenes(Path(args.scenes_dir), long_video_id)
        scenes = scene_cache[long_video_id]
        scene_based = candidate_evidence(scenes, start, end, args.context_pad_sec)
        scene_based["evidence_source"] = "vpick_scenes"
        scene_based["evidence_provider"] = "vpick_api"
        scene_based["transcript_scope"] = "longform_scene_interval"
        scene_based["transcript_video_id"] = long_video_id
        short_video_id = str(source.get("short_video_id", ""))
        for video_id in (long_video_id, short_video_id):
            if video_id and video_id not in subtitle_cache:
                subtitle_cache[video_id] = load_subtitle_cues(subtitle_cache_dirs, video_id)
        subtitle_based = subtitle_evidence(
            subtitle_cache.get(short_video_id, []),
            subtitle_cache.get(long_video_id, []),
            start,
            end,
            args.context_pad_sec,
            short_video_id=short_video_id,
            long_video_id=long_video_id,
        )
        long_subtitle_based = subtitle_evidence(
            subtitle_cache.get(short_video_id, []),
            subtitle_cache.get(long_video_id, []),
            start,
            end,
            args.context_pad_sec,
            transcript_mode="long_only",
            short_video_id=short_video_id,
            long_video_id=long_video_id,
        )
        has_scene_evidence = bool(scene_based["description"] or scene_based["transcript"])
        has_subtitle_evidence = bool(subtitle_based["transcript"])
        if args.evidence_mode == "subtitles":
            evidence = subtitle_based
        elif args.evidence_mode == "long_subtitles":
            evidence = long_subtitle_based
        elif args.evidence_mode == "prefer_subtitles":
            evidence = subtitle_based if has_subtitle_evidence else scene_based
        elif args.evidence_mode == "prefer_scenes":
            evidence = scene_based if has_scene_evidence else subtitle_based
        else:
            evidence = scene_based
        if candidate_id not in candidate_rows:
            text_sample = [evidence["description"], evidence["transcript"]]
            language = language_hints.get(long_video_id) or detect_prompt_language(text_sample, default="ko")
            genre = genre_hints.get(long_video_id) or detect_content_genre(text_sample, default="general")
            dataset_row = dataset_by_long.get(long_video_id, {})
            long_url = dataset_row.get("long_video_url", "")
            candidate_rows[candidate_id] = {
                "candidate_id": candidate_id,
                "long_video_id": long_video_id,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(float(evidence.get("content_duration_sec", end - start)), 3),
                "start_time": seconds_to_clock(start),
                "end_time": seconds_to_clock(end),
                "context_start_sec": round(float(evidence["context_start_sec"]), 3),
                "context_end_sec": round(float(evidence["context_end_sec"]), 3),
                "long_duration_sec": round(float(evidence["long_duration_sec"]), 3),
                "language": language,
                "genre": genre,
                "description": evidence["description"],
                "transcript": evidence["transcript"],
                "before_context": evidence["before_context"],
                "after_context": evidence["after_context"],
                "evidence_source": evidence["evidence_source"],
                "evidence_provider": evidence["evidence_provider"],
                "transcript_scope": evidence["transcript_scope"],
                "transcript_video_id": evidence["transcript_video_id"],
                "candidate_url": f"{long_url}&t={int(start)}s" if long_url else "",
                "evidence_available": bool(evidence["description"] or evidence["transcript"]),
            }
        source_manifest.append(
            {
                "candidate_id": candidate_id,
                **source,
            }
        )

    candidates = sorted(candidate_rows.values(), key=lambda row: (row["long_video_id"], row["start_sec"], row["end_sec"]))
    source_manifest.sort(key=lambda row: (row["source_system"], row["long_video_id"], row["source_rank"], row["candidate_id"]))
    evidence_sources = {str(row.get("evidence_source", "")) for row in candidates}
    evidence_providers = {str(row.get("evidence_provider", "")) for row in candidates}
    if args.require_evidence_source and evidence_sources != {args.require_evidence_source}:
        raise RuntimeError(
            f"Evidence source gate failed: expected {args.require_evidence_source!r}, "
            f"found {sorted(evidence_sources)!r}"
        )
    if args.require_uniform_provider and (len(evidence_providers) != 1 or "" in evidence_providers):
        raise RuntimeError(f"Uniform provider gate failed: found {sorted(evidence_providers)!r}")
    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "candidates_blind.csv",
        candidates,
        [
            "candidate_id", "long_video_id", "start_sec", "end_sec", "duration_sec", "start_time", "end_time",
            "context_start_sec", "context_end_sec", "long_duration_sec", "language", "genre", "description",
            "transcript", "before_context", "after_context", "candidate_url", "evidence_available",
            "evidence_source", "evidence_provider", "transcript_scope", "transcript_video_id",
        ],
    )
    judge_rows = []
    for row in candidates:
        scenes = scene_cache.get(str(row["long_video_id"]), [])
        judge_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "longform_id": row["long_video_id"],
                "start_ms": round(float(row["start_sec"]) * 1000),
                "end_ms": round(float(row["end_sec"]) * 1000),
                "duration_sec": row["duration_sec"],
                "longform_overview": compact_longform_overview(scenes),
                "scene_ids": overlapping_scene_ids(
                    scenes,
                    float(row["start_sec"]),
                    float(row["end_sec"]),
                ),
                "description": row["description"],
                "transcript": row["transcript"],
                "before_context": row["before_context"],
                "after_context": row["after_context"],
                "visual_evidence_available": (
                    row.get("evidence_source") == "vpick_scenes"
                ),
            }
        )
    write_jsonl(out_dir / "candidates_blind.jsonl", judge_rows)
    write_csv(
        out_dir / "candidate_sources_private.csv",
        source_manifest,
        [
            "candidate_id", "long_video_id", "start_sec", "end_sec", "source_system", "source_run_id",
            "source_rank", "pair_id", "short_video_id", "short_views", "short_likes", "channel_name",
            "channel_performance_percentile", "label_confidence", "source_notes",
            "dataset_split", "evaluation_role", "performance_label", "mapping_confidence",
            "performance_evidence_status", "alignment_status",
        ],
    )
    summary = {
        "candidate_count": len(candidates),
        "source_membership_count": len(source_manifest),
        "long_video_count": len({row["long_video_id"] for row in candidates}),
        "missing_evidence_count": sum(not bool(row["evidence_available"]) for row in candidates),
        "evidence_source_counts": {
            source: sum(row.get("evidence_source") == source for row in candidates)
            for source in sorted(evidence_sources)
        },
        "evidence_provider_counts": {
            provider: sum(row.get("evidence_provider") == provider for row in candidates)
            for provider in sorted(evidence_providers)
        },
        "transcript_scope_counts": {
            scope: sum(row.get("transcript_scope") == scope for row in candidates)
            for scope in sorted({str(row.get("transcript_scope", "")) for row in candidates})
        },
        "source_counts": {
            name: len({row["candidate_id"] for row in source_manifest if row["source_system"] == name})
            for name in sorted({str(row["source_system"]) for row in source_manifest})
        },
        "dataset_counts": dataset_counts,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
