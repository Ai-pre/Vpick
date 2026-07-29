from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_client import LLMError, call_llm  # noqa: E402


DEFAULT_GOLD = ROOT / "deliverables" / "2026-07-23" / "vpick_goldlabel_60_normalized.csv"
DEFAULT_VPICK_DIR = ROOT / "data" / "raw" / "vpick"
DEFAULT_SUBTITLE_DIR = ROOT / "data" / "raw" / "youtube_subtitles" / "fallback"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "subtitle_fallback_scenes"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def longform_rows(gold_path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in read_csv(gold_path):
        video_id = row.get("long_video_id", "").strip()
        if video_id and video_id not in rows:
            rows[video_id] = row
    return rows


def subtitle_path(subtitle_dir: Path, video_id: str) -> Path | None:
    preferred = [
        subtitle_dir / f"{video_id}.json3",
        subtitle_dir / f"{video_id}.ko-orig.json3",
        subtitle_dir / f"{video_id}.ko.json3",
    ]
    return next((path for path in preferred if path.exists() and path.stat().st_size > 20), None)


def parse_json3(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    speeches: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for event in payload.get("events", []):
        text = "".join(
            str(segment.get("utf8", ""))
            for segment in event.get("segs", [])
            if isinstance(segment, dict)
        )
        text = " ".join(text.replace("\n", " ").split())
        if not text:
            continue
        start_ms = int(event.get("tStartMs") or 0)
        duration_ms = max(1, int(event.get("dDurationMs") or 1))
        end_ms = start_ms + duration_ms
        key = (start_ms, end_ms, text)
        if key in seen:
            continue
        seen.add(key)
        speeches.append(
            {
                "speech_id": f"subtitle-{len(speeches) + 1:04d}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_id": "",
                "text": text,
            }
        )
    return sorted(speeches, key=lambda row: (row["start_ms"], row["end_ms"]))


def ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "。", "！", "？", "다", "요"))


def chunk_speeches(
    speeches: list[dict[str, Any]],
    *,
    min_scene_sec: float = 18.0,
    target_scene_sec: float = 28.0,
    max_scene_sec: float = 42.0,
    gap_break_sec: float = 6.0,
) -> list[list[dict[str, Any]]]:
    if not speeches:
        return []
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for speech in speeches:
        if current:
            span_sec = (max(item["end_ms"] for item in current) - current[0]["start_ms"]) / 1000
            gap_sec = (speech["start_ms"] - max(item["end_ms"] for item in current)) / 1000
            should_break = (
                (gap_sec >= gap_break_sec and span_sec >= min_scene_sec)
                or (
                    span_sec >= target_scene_sec
                    and ends_sentence(str(current[-1]["text"]))
                )
                or span_sec >= max_scene_sec
            )
            if should_break:
                chunks.append(current)
                current = []
        current.append(speech)
    if current:
        if chunks:
            final_span = (max(item["end_ms"] for item in current) - current[0]["start_ms"]) / 1000
            if final_span < min_scene_sec / 2:
                chunks[-1].extend(current)
            else:
                chunks.append(current)
        else:
            chunks.append(current)
    return chunks


def extractive_description(speeches: list[dict[str, Any]]) -> tuple[str, str]:
    text = " ".join(str(row["text"]) for row in speeches)
    text = " ".join(text.split())
    excerpt = text[:110].rstrip()
    if len(text) > 110:
        excerpt += "..."
    return "자막 기반 대화", f"자막에서 다음 내용이 전개되는 구간입니다: {excerpt}"


def base_scenes(video_id: str, chunks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    for index, speeches in enumerate(chunks, start=1):
        scene_name, description = extractive_description(speeches)
        scenes.append(
            {
                "asset_id": f"subtitle-fallback:{video_id}",
                "scene_id": f"subtitle-fallback:{video_id}:{index:03d}",
                "start_ms": min(row["start_ms"] for row in speeches),
                "end_ms": max(row["end_ms"] for row in speeches),
                "persons": [],
                "description": description,
                "thumbnail_url": "",
                "scene_name": scene_name,
                "is_fallback": True,
                "speeches": speeches,
                "evidence_provider": "yt_dlp",
                "visual_evidence_available": False,
            }
        )
    return scenes


def compact_scene_input(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": scene["scene_id"],
        "start_ms": scene["start_ms"],
        "end_ms": scene["end_ms"],
        "transcript": " ".join(
            str(speech["text"]) for speech in scene.get("speeches", [])
        )[:1800],
    }


def add_llm_descriptions(
    scenes: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    batch_size: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_prompt = """
당신은 한국어 영상 자막을 장면 단위로 요약하는 데이터 작성자입니다.
입력에는 영상 프레임 없이 타임코드와 자막만 있습니다.
자막에 명시되지 않은 표정, 행동, 장소, 인물 관계를 추측하지 마십시오.
각 scene_id마다 scene_name과 description을 작성하십시오.
scene_name은 25자 이내, description은 120자 이내의 자연스러운 한국어 한 문장입니다.
description은 대화의 주제뿐 아니라 가능한 경우 질문, 변화, 반응, 결론을 포함합니다.
반드시 {"scenes":[{"scene_id":"...","scene_name":"...","description":"..."}]} JSON만 출력하십시오.
""".strip()
    by_id: dict[str, dict[str, Any]] = {}
    usages: list[dict[str, Any]] = []
    errors: list[str] = []
    batches = [
        scenes[index : index + batch_size]
        for index in range(0, len(scenes), batch_size)
    ]
    for batch_index, batch in enumerate(batches, start=1):
        user_prompt = json.dumps(
            {"scenes": [compact_scene_input(scene) for scene in batch]},
            ensure_ascii=False,
        )
        try:
            result = call_llm(
                provider,
                model,
                system_prompt,
                user_prompt,
                max_tokens=max(1400, len(batch) * 180),
            )
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"batch_{batch_index}: {type(exc).__name__}: {exc}")
            continue
        usages.append(result.get("usage", {}))
        generated = result.get("json", {}).get("scenes", [])
        for row in generated:
            if isinstance(row, dict) and row.get("scene_id"):
                by_id[str(row["scene_id"])] = row

    missing = [scene for scene in scenes if scene["scene_id"] not in by_id]
    for retry_index in range(0, len(missing), 4):
        retry_batch = missing[retry_index : retry_index + 4]
        user_prompt = json.dumps(
            {"scenes": [compact_scene_input(scene) for scene in retry_batch]},
            ensure_ascii=False,
        )
        try:
            result = call_llm(
                provider,
                model,
                system_prompt,
                user_prompt,
                max_tokens=max(900, len(retry_batch) * 220),
            )
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            errors.append(
                f"retry_{retry_index // 4 + 1}: {type(exc).__name__}: {exc}"
            )
            continue
        usages.append(result.get("usage", {}))
        generated = result.get("json", {}).get("scenes", [])
        for row in generated:
            if isinstance(row, dict) and row.get("scene_id"):
                by_id[str(row["scene_id"])] = row

    for scene in scenes:
        row = by_id.get(scene["scene_id"])
        if not row:
            continue
        name = str(row.get("scene_name") or "").strip()
        description = str(row.get("description") or "").strip()
        if name:
            scene["scene_name"] = name[:50]
        if description:
            scene["description"] = description[:300]
        scene["description_provider"] = f"{provider}:{model}"
    return scenes, {
        "provider": provider,
        "model": model,
        "usage_by_call": usages,
        "call_count": len(usages),
        "errors": errors,
        "generated_scene_count": sum(scene["scene_id"] in by_id for scene in scenes),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_payload(
    video_id: str,
    scenes: list[dict[str, Any]],
    subtitle: Path,
    subtitle_source: str,
    subtitle_language: str,
    llm_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "total": len(scenes),
        "summary": {
            "evidence_provider": "yt_dlp_transcript_fallback",
            "video_id": video_id,
            "subtitle_file": str(subtitle),
            "subtitle_source": subtitle_source,
            "subtitle_language": subtitle_language,
            "visual_evidence_available": False,
            "description_generation": llm_summary,
            "warning": "Not a Vpick multimodal analysis result.",
        },
        "data": scenes,
        "fallback_count": len(scenes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Vpick-compatible transcript fallback scenes without downloading video."
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--vpick-dir", type=Path, default=DEFAULT_VPICK_DIR)
    parser.add_argument("--subtitle-dir", type=Path, default=DEFAULT_SUBTITLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only-long-video-id", action="append", default=[])
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = longform_rows(args.gold)
    selected = sorted(set(args.only_long_video_id) or set(rows))
    collector: Any = None
    report: list[dict[str, Any]] = []

    for index, video_id in enumerate(selected, start=1):
        if video_id not in rows:
            report.append({"long_video_id": video_id, "status": "missing_gold_row"})
            continue
        vpick_path = args.vpick_dir / f"{video_id}_scenes.json"
        output_path = args.output_dir / f"{video_id}_scenes.json"
        if vpick_path.exists() and not args.force:
            report.append({"long_video_id": video_id, "status": "vpick_available"})
            continue
        if output_path.exists() and not args.force:
            report.append({"long_video_id": video_id, "status": "fallback_exists"})
            continue

        print(f"[{index}/{len(selected)}] {video_id}", flush=True)
        subtitle = subtitle_path(args.subtitle_dir, video_id)
        subtitle_source = "cached"
        subtitle_language = "cached"
        if subtitle is None:
            if collector is None:
                from audit_short_long_alignment import SubtitleCollector

                collector = SubtitleCollector(
                    args.subtitle_dir,
                    sleep_seconds=args.sleep_seconds,
                )
            subtitle, subtitle_source, subtitle_language = collector.fetch(video_id)
        if subtitle is None:
            report.append(
                {
                    "long_video_id": video_id,
                    "status": "subtitle_unavailable",
                    "subtitle_source": subtitle_source,
                    "subtitle_language": subtitle_language,
                }
            )
            continue

        speeches = parse_json3(subtitle)
        scenes = base_scenes(video_id, chunk_speeches(speeches))
        if not scenes:
            report.append(
                {
                    "long_video_id": video_id,
                    "status": "subtitle_empty",
                    "subtitle_file": str(subtitle),
                }
            )
            continue

        llm_summary: dict[str, Any] = {
            "provider": "extractive",
            "model": "none",
            "generated_scene_count": 0,
        }
        if not args.no_llm:
            try:
                scenes, llm_summary = add_llm_descriptions(
                    scenes,
                    provider=args.provider,
                    model=args.model,
                )
            except (LLMError, KeyError, TypeError, ValueError) as exc:
                llm_summary = {
                    "provider": args.provider,
                    "model": args.model,
                    "generated_scene_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback": "extractive",
                }
        payload = build_payload(
            video_id,
            scenes,
            subtitle,
            subtitle_source,
            subtitle_language,
            llm_summary,
        )
        write_json(output_path, payload)
        report.append(
            {
                "long_video_id": video_id,
                "status": "fallback_ready",
                "scene_count": len(scenes),
                "subtitle_file": str(subtitle),
                "description_provider": llm_summary.get("provider", ""),
                "description_model": llm_summary.get("model", ""),
                "visual_evidence_available": 0,
            }
        )

    write_json(args.output_dir / "build_summary.json", {"items": report})
    print(json.dumps({"items": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
