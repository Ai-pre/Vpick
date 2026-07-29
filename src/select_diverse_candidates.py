from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from segments import build_adjacent_candidates, extract_scene_list


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "vpick"

BOOST_TERMS = [
    "전화",
    "친구",
    "반응",
    "주문",
    "메뉴",
    "사투리",
    "방언",
    "말투",
    "고향",
    "자랑",
    "대결",
    "미션",
    "상황극",
    "호칭",
    "애교",
    "안전",
    "평화",
    "서러움",
    "에피소드",
    "일화",
    "도전",
    "긴장",
    "유쾌",
    "포시라워",
    "고백",
    "서울말",
]

PAYOFF_TERMS = [
    "전화",
    "친구",
    "주문",
    "메뉴",
    "고향",
    "자랑",
    "대결",
    "안전",
    "평화",
    "포시라워",
    "고백",
]

PENALTY_TERMS = [
    "소개",
    "규칙",
    "설명",
    "준비",
    "검색",
    "위치",
    "풍경",
    "탐방",
    "트럭",
    "규모",
    "특징",
    "이동",
    "콘텐츠",
]

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


def load_scenes_for_group(
    group_rows: list[dict[str, str]],
    raw_dir: Path = RAW_DIR,
) -> list[dict[str, Any]]:
    first = group_rows[0]
    pair_id = first["pair_id"]
    long_video_id = first.get("long_video_id", "").strip()
    candidates = [
        raw_dir / f"{pair_id}_scenes.json",
        raw_dir / f"{long_video_id}_scenes.json"
        if long_video_id
        else raw_dir / "__missing__.json",
    ]
    if not long_video_id:
        candidates.append(raw_dir / "scenes.json")
    for path in candidates:
        if path.exists():
            return extract_scene_list(load_json(path))
    raise FileNotFoundError(f"No Vpick scenes JSON found for pair_id={pair_id}, long_video_id={long_video_id}")


def group_dataset(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = row.get("long_video_id") or row.get("long_video_url") or row["pair_id"]
        grouped.setdefault(key, []).append(row)
    return sorted(grouped.items(), key=lambda item: item[0])


def count_terms(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term and term in text)


def duration_score(duration_sec: float, ideal_sec: float, tolerance_sec: float) -> float:
    if tolerance_sec <= 0:
        return 0.0
    return max(0.0, 1.0 - (abs(duration_sec - ideal_sec) / tolerance_sec))


def candidate_score(candidate: dict[str, Any], ideal_duration_sec: float = 80.0) -> float:
    text = str(candidate.get("text", ""))
    duration = float(candidate["duration_sec"])
    scene_count = len(candidate.get("scene_ids", []))
    score = 0.0
    score += 0.85 * count_terms(text, BOOST_TERMS)
    score += 0.65 * count_terms(text, PAYOFF_TERMS)
    score -= 0.9 * count_terms(text, PENALTY_TERMS)
    score += 2.5 * duration_score(duration, ideal_duration_sec, 95.0)
    score -= 0.25 * max(0, scene_count - 1)
    if 45.0 <= duration <= 115.0:
        score += 0.7
    if scene_count == 2 and duration <= 160.0:
        score += 0.25
    return score


def overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = max(0.0, min(float(a["end_sec"]), float(b["end_sec"])) - max(float(a["start_sec"]), float(b["start_sec"])))
    shortest = min(float(a["duration_sec"]), float(b["duration_sec"]))
    return overlap / shortest if shortest > 0 else 0.0


def center_sec(candidate: dict[str, Any]) -> float:
    return (float(candidate["start_sec"]) + float(candidate["end_sec"])) / 2.0


def is_allowed(candidate: dict[str, Any], selected: list[dict[str, Any]], max_overlap_ratio: float) -> bool:
    return all(overlap_ratio(candidate, item) <= max_overlap_ratio for item in selected)


def timeline_bins_select(
    candidates: list[dict[str, Any]],
    video_end_sec: float,
    top_k: int,
    bins: int,
    max_overlap_ratio: float,
    ideal_duration_sec: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    bin_count = max(1, bins)

    for bin_index in range(bin_count):
        lo = video_end_sec * bin_index / bin_count
        hi = video_end_sec * (bin_index + 1) / bin_count
        in_bin = [
            item
            for item in candidates
            if item["candidate_id"] not in used_ids and lo <= center_sec(item) < hi and is_allowed(item, selected, max_overlap_ratio)
        ]
        in_bin.sort(key=lambda item: candidate_score(item, ideal_duration_sec), reverse=True)
        if in_bin:
            selected.append(in_bin[0])
            used_ids.add(in_bin[0]["candidate_id"])
        if len(selected) >= top_k:
            return selected

    fill = [item for item in candidates if item["candidate_id"] not in used_ids]
    fill.sort(key=lambda item: candidate_score(item, ideal_duration_sec), reverse=True)
    for item in fill:
        if is_allowed(item, selected, max_overlap_ratio):
            selected.append(item)
            used_ids.add(item["candidate_id"])
        if len(selected) >= top_k:
            break
    return selected


def left_bridge_candidates(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    bridge_reserve: int,
    ideal_duration_sec: float,
) -> list[tuple[str, dict[str, Any]]]:
    used = {str(item["candidate_id"]) for item in selected}
    bridges: list[tuple[int, float, str, dict[str, Any]]] = []
    for source in selected:
        source_scene_ids = [str(scene_id) for scene_id in source.get("scene_ids", [])]
        if not source_scene_ids:
            continue
        source_start = float(source["start_sec"])
        source_end = float(source["end_sec"])
        source_id = str(source["candidate_id"])
        source_order = selected.index(source)

        source_bridges: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            if str(candidate["candidate_id"]) in used:
                continue
            candidate_scene_ids = [str(scene_id) for scene_id in candidate.get("scene_ids", [])]
            if len(candidate_scene_ids) != len(source_scene_ids) + 1:
                continue
            same_end = abs(float(candidate["end_sec"]) - source_end) <= 0.01
            starts_earlier = float(candidate["start_sec"]) < source_start
            keeps_source_suffix = candidate_scene_ids[1:] == source_scene_ids
            if same_end and starts_earlier and keeps_source_suffix:
                score = candidate_score(candidate, ideal_duration_sec)
                source_bridges.append((score, candidate))

        if source_bridges:
            source_bridges.sort(key=lambda item: item[0], reverse=True)
            bridges.append((source_order, source_bridges[0][0], source_id, source_bridges[0][1]))
    bridges.sort(key=lambda item: (item[0], -item[1]))

    output: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    max_bridges = len(bridges) if bridge_reserve <= 0 else bridge_reserve
    for _, _, source_id, candidate in bridges:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen:
            continue
        output.append((source_id, candidate))
        seen.add(candidate_id)
        if len(output) >= max_bridges:
            break
    return output


def insert_bridges(
    selected: list[dict[str, Any]],
    bridges: list[tuple[str, dict[str, Any]]],
    top_k: int,
) -> list[dict[str, Any]]:
    if not bridges:
        return selected
    by_source: dict[str, list[dict[str, Any]]] = {}
    for source_id, bridge in bridges:
        by_source.setdefault(source_id, []).append(bridge)

    output: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in selected:
        item_id = str(item["candidate_id"])
        if item_id not in used:
            output.append(item)
            used.add(item_id)
        for bridge in by_source.get(item_id, []):
            bridge_id = str(bridge["candidate_id"])
            if bridge_id not in used:
                output.append(bridge)
                used.add(bridge_id)
    return output


def mmr_select(
    candidates: list[dict[str, Any]],
    video_end_sec: float,
    top_k: int,
    max_overlap_ratio: float,
    ideal_duration_sec: float,
    diversity_weight: float,
    overlap_penalty: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        best = None
        best_score = -math.inf
        for item in remaining:
            base = candidate_score(item, ideal_duration_sec)
            if selected:
                nearest = min(abs(center_sec(item) - center_sec(chosen)) for chosen in selected) / max(video_end_sec, 1.0)
                diversity = min(nearest / 0.13, 1.0)
                overlap = max(overlap_ratio(item, chosen) for chosen in selected)
            else:
                diversity = 1.0
                overlap = 0.0
            score = base + (diversity_weight * diversity) - (overlap_penalty * overlap)
            if score > best_score:
                best = item
                best_score = score
        if best is None:
            break
        selected.append(best)
        remaining = [
            item
            for item in remaining
            if item["candidate_id"] != best["candidate_id"] and overlap_ratio(item, best) <= max_overlap_ratio
        ]
    return selected


def select_candidates(
    candidates: list[dict[str, Any]],
    video_end_sec: float,
    strategy: str,
    top_k: int,
    bins: int,
    max_overlap_ratio: float,
    ideal_duration_sec: float,
    diversity_weight: float,
    overlap_penalty: float,
    bridge_reserve: int,
    bridge_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if strategy in {"timeline_bins", "timeline_bins_bridge"}:
        selected = timeline_bins_select(candidates, video_end_sec, top_k, bins, max_overlap_ratio, ideal_duration_sec)
        if strategy == "timeline_bins_bridge":
            bridges = left_bridge_candidates(selected, bridge_candidates or candidates, bridge_reserve, ideal_duration_sec)
            return insert_bridges(selected, bridges, top_k)
        return selected
    if strategy == "mmr":
        return mmr_select(
            candidates,
            video_end_sec,
            top_k,
            max_overlap_ratio,
            ideal_duration_sec,
            diversity_weight,
            overlap_penalty,
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def rows_for_group(
    group_rows: list[dict[str, str]],
    selected: list[dict[str, Any]],
    run_id: str,
    strategy: str,
    prompt_id: str,
    ideal_duration_sec: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in group_rows:
        for rank, candidate in enumerate(selected, start=1):
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "long_video_id": pair.get("long_video_id", ""),
                    "short_video_id": pair.get("short_video_id", ""),
                    "run_id": run_id,
                    "selector_type": "deterministic_diverse",
                    "prompt_id": prompt_id,
                    "model_name": "none",
                    "rank": rank,
                    "pred_start_sec": candidate["start_sec"],
                    "pred_end_sec": candidate["end_sec"],
                    "selected_scene_ids": "|".join(str(x) for x in candidate.get("scene_ids", [])),
                    "confidence": round(candidate_score(candidate, ideal_duration_sec), 4),
                    "notes": f"{strategy}:{candidate['candidate_id']}",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a diverse no-API slate from Vpick scene candidates.")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "processed" / "pilot_dataset_pairs.csv"))
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing <long_video_id>_scenes.json files.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", choices=["timeline_bins", "timeline_bins_bridge", "mmr"], default="timeline_bins")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-duration-sec", type=float, default=15.0)
    parser.add_argument("--max-duration-sec", type=float, default=210.0)
    parser.add_argument("--max-window-scenes", type=int, default=4)
    parser.add_argument("--bridge-max-duration-sec", type=float, default=210.0)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.72)
    parser.add_argument("--ideal-duration-sec", type=float, default=80.0)
    parser.add_argument("--diversity-weight", type=float, default=2.0)
    parser.add_argument("--overlap-penalty", type=float, default=5.0)
    parser.add_argument("--bridge-reserve", type=int, default=0, help="Max bridge candidates to add. 0 means add one valid left-bridge per selected source.")
    parser.add_argument(
        "--skip-missing-scenes",
        action="store_true",
        help="Skip longforms without a scene JSON instead of stopping the run.",
    )
    parser.add_argument(
        "--missing-scenes-report",
        type=Path,
        default=None,
        help="Optional CSV audit of skipped longforms.",
    )
    args = parser.parse_args()

    dataset = read_csv(Path(args.dataset))
    prediction_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    run_id = args.run_id or (
        f"{args.strategy}_k{args.top_k}_b{args.bins}_max{int(args.max_duration_sec)}_"
        f"ov{int(args.max_overlap_ratio * 100)}"
    )

    for group_id, group_rows in group_dataset(dataset):
        try:
            scenes = load_scenes_for_group(group_rows, args.raw_dir)
        except FileNotFoundError:
            if not args.skip_missing_scenes:
                raise
            missing_rows.append(
                {
                    "long_video_id": group_rows[0].get("long_video_id", ""),
                    "pair_ids": "|".join(row["pair_id"] for row in group_rows),
                    "pair_count": len(group_rows),
                    "reason": "missing_scene_json",
                }
            )
            continue
        candidates = build_adjacent_candidates(
            scenes,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
            max_window_scenes=args.max_window_scenes,
        )[: args.max_candidates]
        bridge_candidates = candidates
        if args.strategy == "timeline_bins_bridge" and args.bridge_max_duration_sec > args.max_duration_sec:
            bridge_candidates = build_adjacent_candidates(
                scenes,
                min_duration_sec=args.min_duration_sec,
                max_duration_sec=args.bridge_max_duration_sec,
                max_window_scenes=args.max_window_scenes,
            )[: args.max_candidates]
        video_end = max(float(scene["end_sec"]) for scene in scenes)
        selected = select_candidates(
            candidates,
            video_end,
            args.strategy,
            args.top_k,
            args.bins,
            args.max_overlap_ratio,
            args.ideal_duration_sec,
            args.diversity_weight,
            args.overlap_penalty,
            args.bridge_reserve,
            bridge_candidates,
        )
        prediction_rows.extend(
            rows_for_group(
                group_rows,
                selected,
                run_id=run_id,
                strategy=args.strategy,
                prompt_id=f"{args.strategy}_b{args.bins}_max{int(args.max_duration_sec)}",
                ideal_duration_sec=args.ideal_duration_sec,
            )
        )

    write_csv(Path(args.output), prediction_rows)
    if args.missing_scenes_report is not None:
        write_csv(args.missing_scenes_report, missing_rows)
    print(
        json.dumps(
            {
                "predictions": args.output,
                "rows": len(prediction_rows),
                "run_id": run_id,
                "skipped_longforms": len(missing_rows),
                "skipped_pairs": sum(int(row["pair_count"]) for row in missing_rows),
                "missing_scenes_report": (
                    str(args.missing_scenes_report)
                    if args.missing_scenes_report is not None
                    else ""
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
