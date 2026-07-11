from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def parse_time_to_seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(p) for p in str(value).strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported time format: {value!r}")


def segment_iou(pred: Segment | None, gold: Segment) -> float | None:
    if pred is None:
        return None
    overlap = max(0.0, min(pred.end_sec, gold.end_sec) - max(pred.start_sec, gold.start_sec))
    union = max(pred.end_sec, gold.end_sec) - min(pred.start_sec, gold.start_sec)
    return overlap / union if union > 0 else 0.0


def gold_coverage(pred: Segment | None, gold: Segment) -> float | None:
    if pred is None:
        return None
    overlap = max(0.0, min(pred.end_sec, gold.end_sec) - max(pred.start_sec, gold.start_sec))
    return overlap / gold.duration_sec if gold.duration_sec > 0 else 0.0


def start_error(pred: Segment | None, gold: Segment) -> float | None:
    return None if pred is None else abs(pred.start_sec - gold.start_sec)


def end_error(pred: Segment | None, gold: Segment) -> float | None:
    return None if pred is None else abs(pred.end_sec - gold.end_sec)


def normalize_scene(raw: dict[str, Any]) -> dict[str, Any]:
    scene_id = raw.get("scene_id") or raw.get("id") or raw.get("sceneId")
    start_ms = raw.get("start_ms", raw.get("startTimeMs", raw.get("start_time_ms", 0)))
    end_ms = raw.get("end_ms", raw.get("endTimeMs", raw.get("end_time_ms", 0)))
    description = (
        raw.get("description")
        or raw.get("scene_description")
        or raw.get("summary")
        or raw.get("name")
        or ""
    )
    speeches = normalize_speeches(raw.get("speeches") or raw.get("transcript") or raw.get("script") or [])
    transcript = raw.get("speech") or format_speeches(speeches)
    return {
        "scene_id": str(scene_id),
        "start_sec": float(start_ms) / 1000.0,
        "end_sec": float(end_ms) / 1000.0,
        "duration_sec": max(0.0, (float(end_ms) - float(start_ms)) / 1000.0),
        "name": raw.get("name") or raw.get("scene_name") or "",
        "description": description,
        "transcript": transcript,
        "speeches": speeches,
        "raw": raw,
    }


def normalize_speeches(raw_speeches: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_speeches, list):
        return []
    speeches: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_speeches, start=1):
        if not isinstance(item, dict):
            continue
        start_ms = item.get("start_ms", item.get("startTimeMs", item.get("start_time_ms")))
        end_ms = item.get("end_ms", item.get("endTimeMs", item.get("end_time_ms")))
        text = str(item.get("text") or item.get("utterance") or item.get("speech") or "").strip()
        if start_ms is None or end_ms is None or not text:
            continue
        speeches.append(
            {
                "speech_id": str(item.get("speech_id") or item.get("id") or idx),
                "start_sec": float(start_ms) / 1000.0,
                "end_sec": float(end_ms) / 1000.0,
                "speaker_id": str(item.get("speaker_id") or item.get("speaker") or ""),
                "text": text,
            }
        )
    speeches.sort(key=lambda speech: (speech["start_sec"], speech["end_sec"]))
    return speeches


def seconds_to_clock(seconds: float) -> str:
    total = int(seconds)
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def format_speeches(speeches: list[dict[str, Any]]) -> str:
    lines = []
    for speech in speeches:
        speaker = f"S{speech['speaker_id']}" if speech.get("speaker_id") else "S?"
        lines.append(
            f"[{seconds_to_clock(speech['start_sec'])}-{seconds_to_clock(speech['end_sec'])}] "
            f"{speaker}: {speech['text']}"
        )
    return "\n".join(lines)


def extract_scene_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw_scenes = payload
    elif isinstance(payload, dict):
        raw_scenes = payload.get("data") or payload.get("items") or payload.get("scenes") or []
    else:
        raw_scenes = []
    return [normalize_scene(s) for s in raw_scenes if isinstance(s, dict)]


def build_adjacent_candidates(
    scenes: list[dict[str, Any]],
    min_duration_sec: float = 15.0,
    max_duration_sec: float = 90.0,
    max_window_scenes: int = 4,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for i in range(len(scenes)):
        for window in range(1, max_window_scenes + 1):
            chunk = scenes[i : i + window]
            if not chunk:
                continue
            start = chunk[0]["start_sec"]
            end = chunk[-1]["end_sec"]
            duration = end - start
            if min_duration_sec <= duration <= max_duration_sec:
                text = "\n".join(
                    filter(
                        None,
                        [
                            " / ".join(str(c.get("name", "")) for c in chunk if c.get("name")),
                            " ".join(str(c.get("description", "")) for c in chunk),
                            " ".join(str(c.get("transcript", "")) for c in chunk),
                        ],
                    )
                )
                candidates.append(
                    {
                        "candidate_id": f"scene_{i + 1}_w{window}",
                        "scene_ids": [c["scene_id"] for c in chunk],
                        "start_sec": start,
                        "end_sec": end,
                        "duration_sec": duration,
                        "text": text[:5000],
                    }
                )
    return candidates


def build_speech_trim_candidates(
    scene: dict[str, Any],
    min_duration_sec: float = 8.0,
    max_duration_sec: float = 30.0,
    max_gap_sec: float = 3.0,
    max_speeches_per_candidate: int = 8,
) -> list[dict[str, Any]]:
    speeches = scene.get("speeches") or []
    candidates: list[dict[str, Any]] = []
    if not speeches:
        return build_sliding_trim_candidates(scene, min_duration_sec=min_duration_sec, max_duration_sec=max_duration_sec)

    scene_index = str(scene.get("scene_id", "scene"))[:8]
    for i in range(len(speeches)):
        for j in range(i, min(len(speeches), i + max_speeches_per_candidate)):
            chunk = speeches[i : j + 1]
            gap_too_large = any(
                float(chunk[k + 1]["start_sec"]) - float(chunk[k]["end_sec"]) > max_gap_sec
                for k in range(len(chunk) - 1)
            )
            if gap_too_large:
                break
            start = float(chunk[0]["start_sec"])
            end = float(chunk[-1]["end_sec"])
            duration = end - start
            if min_duration_sec <= duration <= max_duration_sec:
                text = format_speeches(chunk)
                candidates.append(
                    {
                        "trim_candidate_id": f"trim_{scene_index}_s{i + 1}_{j + 1}",
                        "scene_id": scene.get("scene_id"),
                        "start_sec": start,
                        "end_sec": end,
                        "duration_sec": duration,
                        "speech_ids": [speech["speech_id"] for speech in chunk],
                        "text": text[:4000],
                    }
                )
    return candidates


def build_sliding_trim_candidates(
    scene: dict[str, Any],
    min_duration_sec: float = 15.0,
    max_duration_sec: float = 30.0,
    stride_sec: float = 5.0,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    scene_start = float(scene["start_sec"])
    scene_end = float(scene["end_sec"])
    scene_index = str(scene.get("scene_id", "scene"))[:8]
    durations = sorted({float(min_duration_sec), float(max_duration_sec)})
    speeches = scene.get("speeches") or []
    for duration in durations:
        start = scene_start
        idx = 1
        while start + duration <= scene_end + 0.001:
            end = start + duration
            window_speeches = [
                speech
                for speech in speeches
                if max(float(speech["start_sec"]), start) < min(float(speech["end_sec"]), end)
            ]
            text = format_speeches(window_speeches) if window_speeches else scene.get("transcript") or scene.get("description", "")
            candidates.append(
                {
                    "trim_candidate_id": f"trim_{scene_index}_win{int(duration)}_{idx}",
                    "scene_id": scene.get("scene_id"),
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": duration,
                    "speech_ids": [],
                    "text": text,
                }
            )
            idx += 1
            start += stride_sec
    return candidates


def heuristic_select(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    hook_terms = [
        "왜",
        "근데",
        "진짜",
        "문제",
        "갈등",
        "충격",
        "비밀",
        "반전",
        "웃",
        "돈",
        "성공",
        "실패",
        "처음",
        "마지막",
    ]
    scored = []
    for c in candidates:
        text = c.get("text", "")
        term_score = sum(text.count(term) for term in hook_terms)
        duration = c["duration_sec"]
        duration_score = max(0.0, 1.0 - abs(duration - 45.0) / 45.0)
        score = term_score + duration_score
        scored.append((score, c))
    scored.sort(key=lambda item: (item[0], -abs(item[1]["duration_sec"] - 45.0)), reverse=True)
    best = dict(scored[0][1])
    best["selector"] = "heuristic_placeholder"
    best["selector_score"] = round(scored[0][0], 4)
    return best


def metrics_row(label: str, pred: Segment | None, gold: Segment) -> dict[str, Any]:
    return {
        "prediction_label": label,
        "pred_start_sec": None if pred is None else pred.start_sec,
        "pred_end_sec": None if pred is None else pred.end_sec,
        "temporal_iou": segment_iou(pred, gold),
        "gold_coverage": gold_coverage(pred, gold),
        "start_error_sec": start_error(pred, gold),
        "end_error_sec": end_error(pred, gold),
    }
