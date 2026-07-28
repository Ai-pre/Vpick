from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from language_utils import detect_content_genre, detect_prompt_language
from segments import extract_scene_list, format_speeches


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "vpick"


VARIETY_TERMS = {
    "미션": 2.5,
    "도전": 2.0,
    "주문": 2.0,
    "전화": 2.0,
    "친구": 1.5,
    "반응": 2.5,
    "웃": 2.0,
    "당황": 2.0,
    "놀라": 2.0,
    "대박": 1.5,
    "진짜": 1.0,
    "왜": 1.0,
    "뭐야": 1.5,
    "아니": 0.8,
    "사투리": 2.0,
    "말투": 1.5,
    "호칭": 1.5,
    "고향": 1.5,
    "경쟁": 1.5,
    "reaction": 2.0,
    "challenge": 2.0,
    "friend": 1.5,
}

LECTURE_TERMS = {
    "핵심": 2.5,
    "결론": 2.5,
    "정리": 2.0,
    "중요": 1.8,
    "방법": 1.8,
    "이유": 1.4,
    "문제": 1.2,
    "해결": 1.8,
    "오해": 2.0,
    "예를 들어": 2.0,
    "사례": 1.8,
    "비유": 1.8,
    "프레임워크": 2.5,
    "체크": 1.5,
    "질문": 1.5,
    "답변": 1.5,
    "concept": 1.5,
    "framework": 2.0,
    "example": 1.5,
}

FILLER_TERMS = {
    "소개": 1.2,
    "목차": 2.0,
    "준비": 1.2,
    "이동": 1.2,
    "풍경": 1.5,
    "검색": 1.0,
    "안내": 1.2,
    "구독": 1.5,
    "좋아요": 1.5,
}


VARIETY_TERMS = {
    "미션": 2.5,
    "도전": 2.0,
    "주문": 1.8,
    "전화": 2.0,
    "친구": 1.4,
    "반응": 2.2,
    "당황": 2.0,
    "놀라": 1.6,
    "웃": 1.0,
    "진짜": 0.8,
    "뭐야": 1.4,
    "고백": 1.8,
    "처음": 1.0,
    "사투리": 2.0,
    "말투": 1.5,
    "호칭": 1.5,
    "고향": 1.4,
    "지역": 1.2,
    "문화": 1.0,
    "게임": 1.2,
    "벌칙": 1.4,
    "성공": 1.2,
    "실패": 1.2,
    "맛있": 1.0,
    "먹어": 0.8,
    "건배": 1.0,
    "춤": 1.2,
    "물고기": 1.2,
    "reaction": 2.0,
    "challenge": 2.0,
    "friend": 1.5,
}

LECTURE_TERMS = {
    "핵심": 2.5,
    "결론": 2.5,
    "정리": 2.0,
    "중요": 1.8,
    "방법": 1.8,
    "이유": 1.4,
    "문제": 1.2,
    "해결": 1.8,
    "오해": 2.0,
    "예를 들어": 2.0,
    "사례": 1.8,
    "비유": 1.8,
    "프레임워크": 2.5,
    "체크": 1.5,
    "질문": 1.5,
    "답변": 1.5,
    "concept": 1.5,
    "framework": 2.0,
    "example": 1.5,
}

FILLER_TERMS = {
    "소개": 1.2,
    "목차": 2.0,
    "준비": 1.2,
    "이동": 1.2,
    "풍경": 1.5,
    "설명": 1.0,
    "안내": 1.2,
    "구독": 1.5,
    "좋아요": 1.5,
}


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


def load_scenes(
    long_video_id: str,
    raw_dir: Path = RAW_DIR,
) -> list[dict[str, Any]]:
    path = raw_dir / f"{long_video_id}_scenes.json"
    if not path.exists():
        raise FileNotFoundError(f"No scene JSON for long_video_id={long_video_id}: {path}")
    return extract_scene_list(load_json(path))


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    return dict(grouped)


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlap_ratio(candidate: dict[str, Any], selected: dict[str, Any]) -> float:
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


def count_weighted_terms(text: str, terms: dict[str, float]) -> float:
    lowered = text.lower()
    return sum(lowered.count(term.lower()) * weight for term, weight in terms.items())


def duration_score(duration_sec: float, genre: str) -> float:
    ideal = 45.0 if genre != "lecture" else 55.0
    tolerance = 45.0 if genre != "lecture" else 55.0
    return max(0.0, 1.0 - abs(duration_sec - ideal) / tolerance)


def candidate_text(row: dict[str, str], scenes: list[dict[str, Any]]) -> str:
    start = float(row["pred_start_sec"])
    end = float(row["pred_end_sec"])
    related_scenes = [
        scene
        for scene in scenes
        if interval_overlap(start, end, float(scene["start_sec"]), float(scene["end_sec"])) > 0
    ]
    if not related_scenes:
        selected_ids = {part for part in str(row.get("selected_scene_ids", "")).split("|") if part}
        related_scenes = [scene for scene in scenes if selected_ids and str(scene["scene_id"]) in selected_ids]
    speeches: list[dict[str, Any]] = []
    descriptions: list[str] = []
    for scene in related_scenes:
        if scene.get("description"):
            descriptions.append(str(scene["description"]))
        for speech in scene.get("speeches", []):
            if interval_overlap(start, end, float(speech["start_sec"]), float(speech["end_sec"])) > 0:
                speeches.append(speech)
    speeches.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    text_parts = []
    if descriptions:
        text_parts.append(" ".join(descriptions))
    if speeches:
        text_parts.append(format_speeches(speeches))
    return "\n".join(text_parts)[:6000]


def source_rank_for_scoring(row: dict[str, str]) -> int | None:
    match = re.search(r"source_rank=(\d+)", str(row.get("notes", "")))
    if match:
        return int(match.group(1))
    return None


def note_int(row: dict[str, str], key: str) -> int | None:
    match = re.search(rf"{re.escape(key)}=(\d+)", str(row.get("notes", "")))
    if match:
        return int(match.group(1))
    return None


def note_text(row: dict[str, str], key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^;]+)", str(row.get("notes", "")))
    return match.group(1).strip() if match else ""


def duration_bucket_bonus(duration_sec: float) -> float:
    if 25.0 <= duration_sec <= 35.0:
        return 0.5
    if 40.0 <= duration_sec <= 50.0:
        return 0.5
    if 55.0 <= duration_sec <= 65.0:
        return 0.5
    if 70.0 <= duration_sec <= 80.0:
        return 0.2
    return 0.0


def titleability_score(text: str) -> float:
    title_terms = ["?", "!", "진짜", "뭐야", "처음", "고백", "당황", "성공", "실패", "왜", "어떻게"]
    return min(1.0, sum(text.count(term) for term in title_terms) / 5.0)


def intro_position_score(start_sec: float) -> float:
    if start_sec < 90.0:
        return 0.0
    if start_sec < 180.0:
        return 0.35
    if start_sec < 300.0:
        return 0.7
    return 1.0


def content_volume_score(text: str, speech_lines: int) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    return min(1.0, max(len(stripped) / 450.0, speech_lines / 8.0))


def score_candidate(row: dict[str, str], text: str, language: str, genre: str) -> tuple[float, dict[str, float]]:
    start = float(row["pred_start_sec"])
    end = float(row["pred_end_sec"])
    duration = max(0.0, end - start)
    if genre == "lecture":
        signal = count_weighted_terms(text, LECTURE_TERMS)
    else:
        signal = count_weighted_terms(text, VARIETY_TERMS)
    filler = count_weighted_terms(text, FILLER_TERMS)
    speech_lines = max(0, text.count("\n"))
    speech_density = min(1.0, speech_lines / 10.0)
    titleability = min(1.0, (text.count("?") + text.count("!") + text.count("왜") + text.count("진짜")) / 5.0)
    titleability = titleability_score(text)
    duration_component = duration_score(duration, genre)
    source_rank = source_rank_for_scoring(row)
    if source_rank is not None:
        rank_prior = max(0.0, 1.0 - ((source_rank - 1) / 20.0))
    else:
        rank_prior = max(0.0, 1.0 - ((int(float(row.get("rank", "999"))) - 1) / 120.0))
    window_rank = note_int(row, "window_rank")
    window_prior = max(0.0, 1.0 - (((window_rank or 999) - 1) / 35.0))
    window_kind = note_text(row, "window_kind")
    sliding_window = 1.0 if window_kind == "sliding" else 0.0
    speech_boundary = 1.0 if window_kind == "speech_boundary" else 0.0
    components = {
        "signal": min(8.0, signal) / 8.0,
        "duration": duration_component,
        "speech_density": speech_density,
        "titleability": titleability,
        "rank_prior": rank_prior,
        "window_prior": window_prior,
        "sliding_window": sliding_window,
        "speech_boundary": speech_boundary,
        "duration_bucket": duration_bucket_bonus(duration),
        "position": intro_position_score(start),
        "content_volume": content_volume_score(text, speech_lines),
        "filler_penalty": min(5.0, filler) / 5.0,
    }
    score = (
        0.5 * components["signal"]
        + 1.7 * components["duration"]
        + 1.8 * components["speech_density"]
        + 0.4 * components["titleability"]
        + 0.5 * components["rank_prior"]
        + 0.5 * components["sliding_window"]
        + 0.5 * components["speech_boundary"]
        - 2.5 * components["filler_penalty"]
    )
    if language not in {"ko", "en"}:
        score -= 0.2
    return score, components


def parse_source_rank(row: dict[str, str]) -> int | None:
    match = re.search(r"source_rank=(\d+)", str(row.get("notes", "")))
    if match:
        return int(match.group(1))
    rank = row.get("rank")
    if rank:
        return max(1, math.ceil(int(float(rank)) / 6.0))
    return None


def unique_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        key = (
            row.get("long_video_id", ""),
            f"{float(row['pred_start_sec']):.3f}",
            f"{float(row['pred_end_sec']):.3f}",
            row.get("selected_scene_ids", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def source_rank_band(source_rank: int, max_source_rank: int, band_count: int) -> int:
    if band_count <= 0 or max_source_rank <= 0:
        return 0
    return min(band_count - 1, int((source_rank - 1) * band_count / max_source_rank))


def can_add(candidate: dict[str, Any], selected: list[dict[str, Any]], max_overlap: float) -> bool:
    return all(overlap_ratio(candidate, item) <= max_overlap for item in selected)


def select_with_diversity(
    scored: list[dict[str, Any]],
    top_k: int,
    max_overlap: float,
    source_band_count: int = 0,
) -> list[dict[str, Any]]:
    ranked = sorted(scored, key=lambda item: float(item["rerank_score"]), reverse=True)
    selected: list[dict[str, Any]] = []

    if source_band_count > 0:
        source_ranks = [int(row["source_rank"]) for row in scored if row.get("source_rank")]
        max_source_rank = max(source_ranks) if source_ranks else 0
        for band in range(source_band_count):
            band_rows = [
                row
                for row in ranked
                if row.get("source_rank")
                and source_rank_band(int(row["source_rank"]), max_source_rank, source_band_count) == band
            ]
            for row in band_rows:
                if can_add(row, selected, max_overlap):
                    selected.append(row)
                    break
            if len(selected) >= top_k:
                break

    for row in ranked:
        if row not in selected and can_add(row, selected, max_overlap):
            selected.append(row)
        if len(selected) >= top_k:
            return selected
    for row in ranked:
        if row not in selected:
            selected.append(row)
        if len(selected) >= top_k:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Final deterministic reranker for trim candidate pools.")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "processed" / "pilot_dataset_pairs.csv"))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing <long_video_id>_scenes.json files.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-overlap", type=float, default=0.45)
    parser.add_argument("--source-band-count", type=int, default=12)
    args = parser.parse_args()

    dataset_rows = read_csv(Path(args.dataset))
    predictions = read_csv(Path(args.predictions))
    dataset_by_long = group_rows(dataset_rows, "long_video_id")
    predictions_by_long = group_rows(predictions, "long_video_id")
    output_rows: list[dict[str, Any]] = []

    for long_video_id, pair_rows in sorted(dataset_by_long.items()):
        if long_video_id not in predictions_by_long:
            continue
        scenes = load_scenes(long_video_id, args.raw_dir)
        scene_texts = [
            "\n".join([str(scene.get("description", "")), str(scene.get("transcript", ""))])
            for scene in scenes
        ]
        language = detect_prompt_language(scene_texts)
        genre = detect_content_genre(scene_texts, default="variety_vlog")

        scored: list[dict[str, Any]] = []
        for candidate in unique_candidates(predictions_by_long[long_video_id]):
            text = candidate_text(candidate, scenes)
            score, components = score_candidate(candidate, text, language, genre)
            source_rank = parse_source_rank(candidate)
            enriched = dict(candidate)
            enriched["rerank_score"] = round(score, 6)
            enriched["source_rank"] = "" if source_rank is None else source_rank
            enriched["detected_language"] = language
            enriched["detected_genre"] = genre
            enriched["rerank_components"] = json.dumps(components, ensure_ascii=False, sort_keys=True)
            enriched["notes"] = f"{candidate.get('notes', '')}; final_rerank_text_chars={len(text)}"
            scored.append(enriched)

        selected = select_with_diversity(
            scored,
            top_k=args.top_k,
            max_overlap=args.max_overlap,
            source_band_count=args.source_band_count,
        )
        for rank, selected_row in enumerate(selected, start=1):
            for pair in pair_rows:
                output = dict(selected_row)
                output["pair_id"] = pair["pair_id"]
                output["short_video_id"] = pair.get("short_video_id", "")
                output["run_id"] = f"{selected_row.get('run_id', 'trim')}__final_rerank"
                output["selector_type"] = "deterministic_final_reranker"
                output["prompt_id"] = f"genre_signal_reranker_{genre}"
                output["model_name"] = "none"
                output["rank"] = rank
                output_rows.append(output)

    write_csv(Path(args.output), output_rows)
    print(json.dumps({"predictions": args.output, "rows": len(output_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
