from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from segments import extract_scene_list


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "vpick"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def source_group_key(row: dict[str, str]) -> str:
    return "__".join(
        [
            row.get("long_video_id", ""),
            row.get("run_id", ""),
            row.get("rank", ""),
            row.get("pred_start_sec", ""),
            row.get("pred_end_sec", ""),
            row.get("selected_scene_ids", ""),
        ]
    )


def group_stage1_rows(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(source_group_key(row), []).append(row)
    return sorted(grouped.items(), key=lambda item: (int(float(item[1][0].get("rank", "999"))), item[0]))


def group_source_rows_by_long(rows: list[dict[str, str]]) -> list[tuple[str, list[tuple[str, list[dict[str, str]]]]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("long_video_id", ""), []).append(row)
    return [
        (long_video_id, group_stage1_rows(long_rows))
        for long_video_id, long_rows in sorted(grouped.items(), key=lambda item: item[0])
    ]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scenes(long_video_id: str) -> list[dict[str, Any]]:
    path = RAW_DIR / f"{long_video_id}_scenes.json"
    if not path.exists():
        return []
    return extract_scene_list(load_json(path))


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def dedupe_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for window in windows:
        start = float(window["start_sec"])
        end = float(window["end_sec"])
        if end <= start:
            continue
        key = (round(start * 1000), round(end * 1000))
        if key not in seen:
            output.append(window)
            seen.add(key)
    return output


def make_windows(start_sec: float, end_sec: float, duration_sec: float, windows_per_source: int) -> list[dict[str, Any]]:
    source_duration = max(0.0, end_sec - start_sec)
    if source_duration <= 0:
        return []
    if source_duration <= duration_sec:
        return [
            {
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "window_kind": "source_full",
                "target_duration_sec": round(duration_sec, 3),
            }
        ]

    count = max(1, windows_per_source)
    available = source_duration - duration_sec
    starts = [start_sec + (available * idx / max(1, count - 1)) for idx in range(count)]
    windows = []
    for start in starts:
        end = min(end_sec, start + duration_sec)
        windows.append(
            {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "window_kind": "sliding",
                "target_duration_sec": round(duration_sec, 3),
            }
        )
    return dedupe_windows(windows)


def make_prepend_windows(start_sec: float, end_sec: float, durations: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "start_sec": round(start_sec, 3),
            "end_sec": round(min(end_sec, start_sec + duration), 3),
            "window_kind": "prepend",
            "target_duration_sec": round(duration, 3),
        }
        for duration in durations
        if end_sec > start_sec
    ]


def scene_sort_key(scene: dict[str, Any]) -> tuple[float, float]:
    return (float(scene["start_sec"]), float(scene["end_sec"]))


def make_scene_boundary_windows(
    scenes: list[dict[str, Any]],
    selected_scene_ids: str,
    source_start_sec: float,
    source_end_sec: float,
    target_durations_sec: list[float],
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    if not scenes:
        return []

    selected_ids = {part for part in str(selected_scene_ids).split("|") if part}
    related = [
        scene
        for scene in scenes
        if (selected_ids and str(scene["scene_id"]) in selected_ids)
        or interval_overlap(source_start_sec, source_end_sec, float(scene["start_sec"]), float(scene["end_sec"])) > 0
    ]
    related.sort(key=scene_sort_key)

    windows: list[dict[str, Any]] = []
    for index, scene in enumerate(related):
        for end_index in range(index, len(related)):
            chunk = related[index : end_index + 1]
            start = max(source_start_sec, float(chunk[0]["start_sec"]))
            end = min(source_end_sec, float(chunk[-1]["end_sec"]))
            duration = end - start
            if min_duration_sec <= duration <= max_duration_sec:
                windows.append(
                    {
                        "start_sec": round(start, 3),
                        "end_sec": round(end, 3),
                        "window_kind": "scene_exact",
                        "target_duration_sec": round(duration, 3),
                    }
                )
            if duration > max_duration_sec:
                break

    boundaries = sorted(
        {
            round(max(source_start_sec, min(source_end_sec, float(scene["start_sec"]))), 3)
            for scene in related
        }
        | {
            round(max(source_start_sec, min(source_end_sec, float(scene["end_sec"]))), 3)
            for scene in related
        }
    )
    for boundary in boundaries:
        for duration in target_durations_sec:
            forward_end = boundary + duration
            if forward_end <= source_end_sec + 0.001:
                windows.append(
                    {
                        "start_sec": round(boundary, 3),
                        "end_sec": round(forward_end, 3),
                        "window_kind": "scene_boundary_forward",
                        "target_duration_sec": round(duration, 3),
                    }
                )
            backward_start = boundary - duration
            if backward_start >= source_start_sec - 0.001:
                windows.append(
                    {
                        "start_sec": round(backward_start, 3),
                        "end_sec": round(boundary, 3),
                        "window_kind": "scene_boundary_backward",
                        "target_duration_sec": round(duration, 3),
                    }
                )
    return dedupe_windows(windows)


def related_speeches(
    scenes: list[dict[str, Any]],
    selected_scene_ids: str,
    source_start_sec: float,
    source_end_sec: float,
) -> list[dict[str, Any]]:
    selected_ids = {part for part in str(selected_scene_ids).split("|") if part}
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for scene in scenes:
        scene_overlaps = interval_overlap(source_start_sec, source_end_sec, float(scene["start_sec"]), float(scene["end_sec"])) > 0
        if selected_ids and str(scene["scene_id"]) not in selected_ids and not scene_overlaps:
            continue
        for speech in scene.get("speeches", []):
            speech_start = float(speech["start_sec"])
            speech_end = float(speech["end_sec"])
            if interval_overlap(source_start_sec, source_end_sec, speech_start, speech_end) <= 0:
                continue
            key = (round(speech_start * 1000), round(speech_end * 1000), str(speech.get("text", "")))
            if key in seen:
                continue
            item = dict(speech)
            item["start_sec"] = max(source_start_sec, speech_start)
            item["end_sec"] = min(source_end_sec, speech_end)
            output.append(item)
            seen.add(key)
    output.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    return output


def make_speech_boundary_windows(
    scenes: list[dict[str, Any]],
    selected_scene_ids: str,
    source_start_sec: float,
    source_end_sec: float,
    min_duration_sec: float,
    max_duration_sec: float,
    max_gap_sec: float,
    max_speeches_per_window: int,
    lead_pad_sec: float,
    tail_pad_sec: float,
) -> list[dict[str, Any]]:
    speeches = related_speeches(scenes, selected_scene_ids, source_start_sec, source_end_sec)
    windows: list[dict[str, Any]] = []
    if not speeches:
        return windows

    for start_index in range(len(speeches)):
        last_end = float(speeches[start_index]["end_sec"])
        for end_index in range(start_index, min(len(speeches), start_index + max_speeches_per_window)):
            if end_index > start_index:
                gap = float(speeches[end_index]["start_sec"]) - last_end
                if gap > max_gap_sec:
                    break
            raw_start = float(speeches[start_index]["start_sec"])
            raw_end = float(speeches[end_index]["end_sec"])
            last_end = raw_end
            start = max(source_start_sec, raw_start - lead_pad_sec)
            end = min(source_end_sec, raw_end + tail_pad_sec)
            duration = end - start
            if min_duration_sec <= duration <= max_duration_sec:
                windows.append(
                    {
                        "start_sec": round(start, 3),
                        "end_sec": round(end, 3),
                        "window_kind": "speech_boundary",
                        "target_duration_sec": round(duration, 3),
                    }
                )
            if duration > max_duration_sec:
                break
    return dedupe_windows(windows)


def duration_label(durations: list[float]) -> str:
    return "_".join(str(int(duration)) if duration.is_integer() else str(duration).replace(".", "p") for duration in durations)


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand stage-1 scene candidates into deterministic sliding trim windows.")
    parser.add_argument("--stage1-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-rank", type=int, default=10)
    parser.add_argument("--source-run-id", action="append")
    parser.add_argument("--duration-sec", action="append", type=float)
    parser.add_argument("--windows-per-source", type=int, default=5)
    parser.add_argument("--prepend-duration-sec", action="append", type=float)
    parser.add_argument("--include-scene-boundary-windows", action="store_true")
    parser.add_argument("--scene-boundary-min-duration-sec", type=float, default=15.0)
    parser.add_argument("--scene-boundary-max-duration-sec", type=float, default=90.0)
    parser.add_argument("--include-speech-boundary-windows", action="store_true")
    parser.add_argument("--speech-boundary-min-duration-sec", type=float, default=20.0)
    parser.add_argument("--speech-boundary-max-duration-sec", type=float, default=90.0)
    parser.add_argument("--speech-boundary-max-gap-sec", type=float, default=4.0)
    parser.add_argument("--speech-boundary-max-speeches", type=int, default=18)
    parser.add_argument("--speech-boundary-lead-pad-sec", type=float, default=1.0)
    parser.add_argument("--speech-boundary-tail-pad-sec", type=float, default=1.5)
    args = parser.parse_args()

    durations_sec = sorted(set(args.duration_sec or [60.0]))
    prepend_durations_sec = sorted(set(args.prepend_duration_sec or []))
    source_run_ids = set(args.source_run_id or [])
    stage1_rows = [
        row
        for row in read_csv(Path(args.stage1_predictions))
        if int(float(row.get("rank", "999"))) <= args.source_rank
        and (not source_run_ids or row.get("run_id") in source_run_ids)
    ]

    output_rows: list[dict[str, Any]] = []
    scenes_by_long: dict[str, list[dict[str, Any]]] = {}
    run_id_suffix = f"trim_variable_{duration_label(durations_sec)}s_x{args.windows_per_source}"
    for long_video_id, source_groups in group_source_rows_by_long(stage1_rows):
        rank_offset = 0
        if args.include_scene_boundary_windows or args.include_speech_boundary_windows:
            scenes_by_long[long_video_id] = load_scenes(long_video_id)
        for compact_source_rank, (_, group_rows) in enumerate(source_groups, start=1):
            source = group_rows[0]
            source_rank = int(float(source["rank"]))
            start_sec = float(source["pred_start_sec"])
            end_sec = float(source["pred_end_sec"])
            windows: list[dict[str, Any]] = []
            windows.extend(make_prepend_windows(start_sec, end_sec, prepend_durations_sec))
            for duration_sec in durations_sec:
                windows.extend(make_windows(start_sec, end_sec, duration_sec, args.windows_per_source))
            if args.include_scene_boundary_windows:
                windows.extend(
                    make_scene_boundary_windows(
                        scenes_by_long.get(long_video_id, []),
                        source.get("selected_scene_ids", ""),
                        start_sec,
                        end_sec,
                        durations_sec,
                        args.scene_boundary_min_duration_sec,
                        args.scene_boundary_max_duration_sec,
                    )
                )
            if args.include_speech_boundary_windows:
                windows.extend(
                    make_speech_boundary_windows(
                        scenes_by_long.get(long_video_id, []),
                        source.get("selected_scene_ids", ""),
                        start_sec,
                        end_sec,
                        args.speech_boundary_min_duration_sec,
                        args.speech_boundary_max_duration_sec,
                        args.speech_boundary_max_gap_sec,
                        args.speech_boundary_max_speeches,
                        args.speech_boundary_lead_pad_sec,
                        args.speech_boundary_tail_pad_sec,
                    )
                )
            windows = dedupe_windows(windows)
            for target_pair in group_rows:
                for window_rank, window in enumerate(windows, start=1):
                    window_start = float(window["start_sec"])
                    window_end = float(window["end_sec"])
                    window_duration = max(0.0, window_end - window_start)
                    output_rows.append(
                        {
                            "pair_id": target_pair["pair_id"],
                            "long_video_id": target_pair.get("long_video_id", ""),
                            "short_video_id": target_pair.get("short_video_id", ""),
                            "run_id": f"{source['run_id']}__{run_id_suffix}",
                            "selector_type": "deterministic_trim_window",
                            "prompt_id": "variable_duration_windows",
                            "model_name": "none",
                            "rank": rank_offset + window_rank,
                            "pred_start_sec": window_start,
                            "pred_end_sec": window_end,
                            "selected_scene_ids": source.get("selected_scene_ids", ""),
                            "confidence": "",
                            "notes": (
                                f"source_rank={source_rank}; compact_source_rank={compact_source_rank}; "
                                f"window_rank={window_rank}; window_kind={window.get('window_kind', '')}; "
                                f"target_duration={window.get('target_duration_sec', '')}; duration={window_duration:.3f}"
                            ),
                        }
                    )
            rank_offset += len(windows)

    write_csv(Path(args.output), output_rows)
    print(json.dumps({"predictions": args.output, "rows": len(output_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
