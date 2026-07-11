from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from segments import extract_scene_list, format_speeches, seconds_to_clock


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scenes(long_video_id: str) -> list[dict[str, Any]]:
    path = RAW_DIR / f"{long_video_id}_scenes.json"
    if not path.exists():
        return []
    return extract_scene_list(load_json(path))


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def candidate_text(row: dict[str, str], scenes: list[dict[str, Any]], context_pad_sec: float) -> tuple[str, str, str, float, float]:
    start = float(row["pred_start_sec"])
    end = float(row["pred_end_sec"])
    context_start = max(0.0, start - context_pad_sec)
    context_end = end + context_pad_sec
    descriptions: list[str] = []
    speeches: list[dict[str, Any]] = []
    context_speeches: list[dict[str, Any]] = []
    for scene in scenes:
        if interval_overlap(start, end, float(scene["start_sec"]), float(scene["end_sec"])) <= 0:
            if interval_overlap(context_start, context_end, float(scene["start_sec"]), float(scene["end_sec"])) <= 0:
                continue
        if scene.get("description") and interval_overlap(start, end, float(scene["start_sec"]), float(scene["end_sec"])) > 0:
            descriptions.append(str(scene["description"]))
        for speech in scene.get("speeches", []):
            if interval_overlap(start, end, float(speech["start_sec"]), float(speech["end_sec"])) > 0:
                speeches.append(speech)
            if interval_overlap(context_start, context_end, float(speech["start_sec"]), float(speech["end_sec"])) > 0:
                context_speeches.append(speech)
    speeches.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    context_speeches.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    return (
        " ".join(descriptions)[:1200],
        format_speeches(speeches)[:3000],
        format_speeches(context_speeches)[:5000],
        context_start,
        context_end,
    )


def unique_prediction_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("long_video_id", ""),
        f"{float(row['pred_start_sec']):.3f}",
        f"{float(row['pred_end_sec']):.3f}",
        row.get("selected_scene_ids", ""),
    )


def group_by_long(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("long_video_id", ""), []).append(row)
    return grouped


def candidate_center(row: dict[str, str]) -> float:
    return (float(row["pred_start_sec"]) + float(row["pred_end_sec"])) / 2.0


def candidate_score(row: dict[str, str]) -> float:
    try:
        return float(row.get("rerank_score") or row.get("confidence") or 0.0)
    except ValueError:
        return 0.0


def row_overlap_ratio(candidate: dict[str, str], selected: dict[str, str]) -> float:
    overlap = interval_overlap(
        float(candidate["pred_start_sec"]),
        float(candidate["pred_end_sec"]),
        float(selected["pred_start_sec"]),
        float(selected["pred_end_sec"]),
    )
    shortest = min(
        float(candidate["pred_end_sec"]) - float(candidate["pred_start_sec"]),
        float(selected["pred_end_sec"]) - float(selected["pred_start_sec"]),
    )
    return overlap / shortest if shortest > 0 else 0.0


def can_add_row(candidate: dict[str, str], selected: list[dict[str, str]], max_overlap: float = 0.45) -> bool:
    return all(row_overlap_ratio(candidate, row) <= max_overlap for row in selected)


def note_text(row: dict[str, str], key: str) -> str:
    marker = f"{key}="
    for part in str(row.get("notes", "")).split(";"):
        part = part.strip()
        if part.startswith(marker):
            return part[len(marker) :].strip()
    return ""


def row_duration(row: dict[str, str]) -> float:
    return max(0.0, float(row["pred_end_sec"]) - float(row["pred_start_sec"]))


def duration_fit_score(row: dict[str, str], ideal_sec: float = 45.0, tolerance_sec: float = 45.0) -> float:
    duration = row_duration(row)
    return max(0.0, 1.0 - abs(duration - ideal_sec) / tolerance_sec)


def boundary_quality_score(row: dict[str, str]) -> float:
    window_kind = note_text(row, "window_kind")
    if window_kind == "speech_boundary":
        return 0.10
    if window_kind == "sliding":
        return 0.06
    if window_kind == "scene_exact":
        return 0.04
    return 0.0


def representative_score(row: dict[str, str]) -> float:
    return candidate_score(row) + (0.18 * duration_fit_score(row)) + boundary_quality_score(row)


def scene_key(row: dict[str, str]) -> str:
    parts = [part for part in str(row.get("selected_scene_ids", "")).split("|") if part]
    return "|".join(sorted(parts))


def row_relation_strength(a: dict[str, str], b: dict[str, str]) -> float:
    overlap = row_overlap_ratio(a, b)
    a_scene_key = scene_key(a)
    b_scene_key = scene_key(b)
    same_scene = 1.0 if a_scene_key and a_scene_key == b_scene_key else 0.0
    center_distance = abs(candidate_center(a) - candidate_center(b))
    duration_scale = max(row_duration(a), row_duration(b), 1.0)
    center_affinity = max(0.0, 1.0 - center_distance / (duration_scale * 1.5))
    return max(overlap, 0.75 * same_scene, 0.55 * center_affinity)


def cluster_event_variants(rows: list[dict[str, str]], relation_threshold: float = 0.55) -> list[list[dict[str, str]]]:
    clusters: list[list[dict[str, str]]] = []
    for row in sorted(rows, key=lambda item: representative_score(item), reverse=True):
        best_index = -1
        best_strength = 0.0
        for index, cluster in enumerate(clusters):
            strength = max(row_relation_strength(row, member) for member in cluster)
            if strength > best_strength:
                best_strength = strength
                best_index = index
        if best_index >= 0 and best_strength >= relation_threshold:
            clusters[best_index].append(row)
        else:
            clusters.append([row])
    return clusters


def event_representative(cluster: list[dict[str, str]]) -> dict[str, str]:
    return max(cluster, key=lambda row: (representative_score(row), -abs(row_duration(row) - 45.0)))


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def select_adaptive_event(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    if len(rows) <= top_k:
        return sorted(rows, key=candidate_center)

    events = []
    for cluster in cluster_event_variants(rows):
        rep = event_representative(cluster)
        events.append(
            {
                "row": rep,
                "raw_score": representative_score(rep),
                "support": len(cluster),
                "center": candidate_center(rep),
            }
        )
    if len(events) <= top_k:
        return sorted([event["row"] for event in events], key=candidate_center)

    score_norms = normalize_scores([float(event["raw_score"]) for event in events])
    support_norms = normalize_scores([float(event["support"]) for event in events])
    for event, score_norm, support_norm in zip(events, score_norms, support_norms):
        event["score_norm"] = score_norm
        event["support_norm"] = support_norm

    min_center = min(float(event["center"]) for event in events)
    max_center = max(float(event["center"]) for event in events)
    span = max(1.0, max_center - min_center)
    selected: list[dict[str, Any]] = []

    while len(selected) < top_k:
        best_event: dict[str, Any] | None = None
        best_value = -1.0
        for event in events:
            row = event["row"]
            if event in selected or not can_add_row(row, [item["row"] for item in selected], max_overlap=0.62):
                continue
            if selected:
                min_distance = min(abs(float(event["center"]) - float(item["center"])) for item in selected)
                diversity = min(1.0, min_distance / (span / max(2, top_k - 1)))
            else:
                diversity = 0.0
            value = (0.72 * float(event["score_norm"])) + (0.22 * diversity) + (0.06 * float(event["support_norm"]))
            if value > best_value:
                best_value = value
                best_event = event
        if best_event is None:
            break
        selected.append(best_event)

    if len(selected) < top_k:
        for event in sorted(events, key=lambda item: float(item["raw_score"]), reverse=True):
            if event in selected:
                continue
            selected.append(event)
            if len(selected) >= top_k:
                break

    return sorted([event["row"] for event in selected[:top_k]], key=candidate_center)


def coverage_event_value(event: dict[str, Any]) -> float:
    return (
        (0.74 * float(event["score_norm"]))
        + (0.16 * float(event["support_norm"]))
        + (0.10 * duration_fit_score(event["row"]))
    )


def build_event_pool(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    events = []
    for cluster in cluster_event_variants(rows):
        rep = event_representative(cluster)
        events.append(
            {
                "row": rep,
                "raw_score": representative_score(rep),
                "support": len(cluster),
                "center": candidate_center(rep),
            }
        )
    score_norms = normalize_scores([float(event["raw_score"]) for event in events])
    support_norms = normalize_scores([float(event["support"]) for event in events])
    for event, score_norm, support_norm in zip(events, score_norms, support_norms):
        event["score_norm"] = score_norm
        event["support_norm"] = support_norm
    return events


def infer_coverage_bin_count(rows: list[dict[str, str]]) -> int:
    if not rows:
        return 5
    duration_min = max(float(row["pred_end_sec"]) for row in rows) / 60.0
    if duration_min <= 15.0:
        return 5
    extra_five_min_blocks = int((duration_min - 15.0 + 4.999) / 5.0)
    return min(13, 5 + (extra_five_min_blocks * 2))


def select_adaptive_coverage(
    rows: list[dict[str, str]],
    top_k: int,
    coverage_bin_count: int | None = None,
    coverage_per_bin: int = 1,
) -> list[dict[str, str]]:
    bin_count = max(1, coverage_bin_count or (infer_coverage_bin_count(rows) if coverage_per_bin > 1 else top_k))
    per_bin = max(1, coverage_per_bin)
    target_k = min(top_k, bin_count * per_bin)

    if len(rows) <= target_k:
        return sorted(rows, key=candidate_center)

    min_center_all = min(candidate_center(row) for row in rows)
    max_center_all = max(candidate_center(row) for row in rows)
    span_all = max(1.0, max_center_all - min_center_all)
    intro_cutoff = max(120.0, min_center_all + (span_all * 0.08))
    candidates = [row for row in rows if candidate_center(row) >= intro_cutoff]
    if len(candidates) < target_k:
        candidates = list(rows)

    events = build_event_pool(candidates)
    if len(events) <= target_k:
        return sorted([event["row"] for event in events], key=candidate_center)

    min_center = min(float(event["center"]) for event in events)
    max_center = max(float(event["center"]) for event in events)
    span = max(1.0, max_center - min_center)
    selected: list[dict[str, Any]] = []

    # One long-form can produce multiple short-form-worthy events. We first reserve
    # slots by timeline bin, then optionally keep several candidates per bin so the
    # downstream LLM can rerank across a wider slate.
    for bin_index in range(bin_count):
        lo = min_center + (span * bin_index / bin_count)
        hi = min_center + (span * (bin_index + 1) / bin_count)
        if bin_index == bin_count - 1:
            hi += 0.001
        bucket = [
            event
            for event in events
            if lo <= float(event["center"]) < hi
            and event not in selected
            and can_add_row(event["row"], [item["row"] for item in selected], max_overlap=0.58)
        ]
        if not bucket:
            continue
        for event in sorted(bucket, key=coverage_event_value, reverse=True):
            if len(selected) >= target_k:
                break
            if not can_add_row(event["row"], [item["row"] for item in selected], max_overlap=0.58):
                continue
            selected.append(event)
            if sum(1 for item in selected if lo <= float(item["center"]) < hi) >= per_bin:
                break

    while len(selected) < target_k:
        best_event: dict[str, Any] | None = None
        best_value = -1.0
        for event in events:
            row = event["row"]
            if event in selected or not can_add_row(row, [item["row"] for item in selected], max_overlap=0.58):
                continue
            if selected:
                min_distance = min(abs(float(event["center"]) - float(item["center"])) for item in selected)
                diversity = min(1.0, min_distance / (span / max(2, bin_count - 1)))
            else:
                diversity = 0.0
            value = (0.68 * coverage_event_value(event)) + (0.32 * diversity)
            if value > best_value:
                best_value = value
                best_event = event
        if best_event is None:
            break
        selected.append(best_event)

    if len(selected) < top_k:
        for event in sorted(events, key=coverage_event_value, reverse=True):
            if event in selected:
                continue
            selected.append(event)
            if len(selected) >= target_k:
                break

    return sorted([event["row"] for event in selected[:target_k]], key=candidate_center)


def select_rank_order(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    return sorted(rows, key=lambda item: (int(float(item.get("rank", "999"))), float(item["pred_start_sec"])))[:top_k]


def select_timeline_diverse(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    if len(rows) <= top_k:
        return select_rank_order(rows, top_k)
    sorted_by_rank = select_rank_order(rows, len(rows))
    min_center = min(candidate_center(row) for row in sorted_by_rank)
    max_center = max(candidate_center(row) for row in sorted_by_rank)
    span = max(1.0, max_center - min_center)
    selected: list[dict[str, str]] = []
    used: set[tuple[str, str, str]] = set()

    for bin_index in range(top_k):
        lo = min_center + (span * bin_index / top_k)
        hi = min_center + (span * (bin_index + 1) / top_k)
        if bin_index == top_k - 1:
            hi += 0.001
        bucket = [
            row
            for row in sorted_by_rank
            if lo <= candidate_center(row) < hi
            and (row.get("long_video_id", ""), row["pred_start_sec"], row["pred_end_sec"]) not in used
        ]
        if not bucket:
            continue
        bucket.sort(key=lambda item: (int(float(item.get("rank", "999"))), -float(item.get("rerank_score") or 0)))
        selected.append(bucket[0])
        used.add((bucket[0].get("long_video_id", ""), bucket[0]["pred_start_sec"], bucket[0]["pred_end_sec"]))

    for row in sorted_by_rank:
        key = (row.get("long_video_id", ""), row["pred_start_sec"], row["pred_end_sec"])
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) >= top_k:
            break

    selected.sort(key=lambda item: int(float(item.get("rank", "999"))))
    return selected[:top_k]


def select_timeline_score(
    rows: list[dict[str, str]],
    top_k: int,
    min_center_sec: float = 0.0,
    reserve_tail: bool = False,
) -> list[dict[str, str]]:
    candidates = [row for row in rows if candidate_center(row) >= min_center_sec]
    if len(candidates) < top_k:
        candidates = list(rows)
    if len(candidates) <= top_k:
        return sorted(candidates, key=candidate_center)

    min_center = min(candidate_center(row) for row in candidates)
    max_center = max(candidate_center(row) for row in candidates)
    span = max(1.0, max_center - min_center)
    selected: list[dict[str, str]] = []
    used: set[tuple[str, str, str]] = set()

    if reserve_tail and top_k > 1:
        tail_start = min_center + (span * 0.82)
        tail_rows = [row for row in candidates if candidate_center(row) >= tail_start]
        tail_rows.sort(key=lambda item: (candidate_score(item), -int(float(item.get("rank", "999")))), reverse=True)
        for row in tail_rows:
            if can_add_row(row, selected):
                selected.append(row)
                used.add((row.get("long_video_id", ""), row["pred_start_sec"], row["pred_end_sec"]))
                break

    for bin_index in range(top_k):
        lo = min_center + (span * bin_index / top_k)
        hi = min_center + (span * (bin_index + 1) / top_k)
        if bin_index == top_k - 1:
            hi += 0.001
        bucket = [
            row
            for row in candidates
            if lo <= candidate_center(row) < hi
            and (row.get("long_video_id", ""), row["pred_start_sec"], row["pred_end_sec"]) not in used
            and can_add_row(row, selected)
        ]
        bucket.sort(key=lambda item: (candidate_score(item), -int(float(item.get("rank", "999")))), reverse=True)
        if bucket:
            selected.append(bucket[0])
            used.add((bucket[0].get("long_video_id", ""), bucket[0]["pred_start_sec"], bucket[0]["pred_end_sec"]))
        if len(selected) >= top_k:
            break

    fill = sorted(candidates, key=lambda item: (candidate_score(item), -int(float(item.get("rank", "999")))), reverse=True)
    for row in fill:
        key = (row.get("long_video_id", ""), row["pred_start_sec"], row["pred_end_sec"])
        if key in used or not can_add_row(row, selected):
            continue
        selected.append(row)
        used.add(key)
        if len(selected) >= top_k:
            break

    return sorted(selected[:top_k], key=candidate_center)


def select_late_quota(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    if not rows:
        return []
    max_center = max(candidate_center(row) for row in rows)
    min_center = min(candidate_center(row) for row in rows)
    intro_cutoff = max(120.0, min_center + ((max_center - min_center) * 0.08))
    return select_timeline_score(rows, top_k=top_k, min_center_sec=intro_cutoff, reserve_tail=True)


def build_slate(
    rows: list[dict[str, str]],
    top_k: int,
    selection_strategy: str,
    context_pad_sec: float,
    coverage_bin_count: int | None = None,
    coverage_per_bin: int = 1,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for long_video_id, long_rows in sorted(group_by_long(rows).items()):
        seen: set[tuple[str, str, str, str]] = set()
        unique_rows: list[dict[str, str]] = []
        for row in sorted(long_rows, key=lambda item: (int(float(item.get("rank", "999"))), float(item["pred_start_sec"]))):
            key = unique_prediction_key(row)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)

        scenes = load_scenes(long_video_id)
        if selection_strategy == "timeline":
            selected_rows = select_timeline_diverse(unique_rows, top_k)
        elif selection_strategy == "timeline_score":
            selected_rows = select_timeline_score(unique_rows, top_k)
        elif selection_strategy == "late_quota":
            selected_rows = select_late_quota(unique_rows, top_k)
        elif selection_strategy == "adaptive_event":
            selected_rows = select_adaptive_event(unique_rows, top_k)
        elif selection_strategy == "adaptive_coverage":
            selected_rows = select_adaptive_coverage(unique_rows, top_k, coverage_bin_count, coverage_per_bin)
        else:
            selected_rows = select_rank_order(unique_rows, top_k)
        for rank, row in enumerate(selected_rows, start=1):
            start = float(row["pred_start_sec"])
            end = float(row["pred_end_sec"])
            description, transcript, context_transcript, context_start, context_end = candidate_text(row, scenes, context_pad_sec)
            output.append(
                {
                    "long_video_id": long_video_id,
                    "candidate_id": f"{long_video_id}_cand_{rank:02d}",
                    "rank": rank,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "duration_sec": round(end - start, 3),
                    "context_start_sec": round(context_start, 3),
                    "context_end_sec": round(context_end, 3),
                    "start_time": seconds_to_clock(start),
                    "end_time": seconds_to_clock(end),
                    "context_start_time": seconds_to_clock(context_start),
                    "context_end_time": seconds_to_clock(context_end),
                    "youtube_at_url": f"https://www.youtube.com/watch?v={long_video_id}&t={int(start)}s",
                    "source_run_id": row.get("run_id", ""),
                    "selector_type": row.get("selector_type", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "model_name": row.get("model_name", ""),
                    "selected_scene_ids": row.get("selected_scene_ids", ""),
                    "rerank_score": row.get("rerank_score", ""),
                    "source_rank": row.get("source_rank", ""),
                    "description": description,
                    "transcript": transcript,
                    "context_transcript": context_transcript,
                    "notes": row.get("notes", ""),
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one long-form to many short-form candidate slate.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--coverage-bin-count", type=int, default=0)
    parser.add_argument("--coverage-per-bin", type=int, default=1)
    parser.add_argument(
        "--selection-strategy",
        choices=["rank", "timeline", "timeline_score", "late_quota", "adaptive_event", "adaptive_coverage"],
        default="timeline",
    )
    parser.add_argument("--context-pad-sec", type=float, default=20.0)
    args = parser.parse_args()

    rows = build_slate(
        read_csv(Path(args.predictions)),
        args.top_k,
        args.selection_strategy,
        args.context_pad_sec,
        args.coverage_bin_count or None,
        args.coverage_per_bin,
    )
    write_csv(Path(args.output), rows)
    print(json.dumps({"slate": args.output, "rows": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
