from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rapidfuzz import fuzz, process
from yt_dlp import YoutubeDL

from youtube_metadata import extract_youtube_id


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Match:
    short_index: int
    source_start: float
    source_end: float
    score: float


def normalize_text(value: str) -> str:
    return re.sub(r"[^\w]", "", value, flags=re.UNICODE).replace("_", "").lower()


def parse_json3(path: Path) -> list[Cue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cues: list[Cue] = []
    for event in payload.get("events", []):
        text = "".join(str(segment.get("utf8", "")) for segment in event.get("segs", []))
        text = " ".join(text.split())
        normalized = normalize_text(text)
        if not normalized:
            continue
        start = float(event.get("tStartMs", 0)) / 1000.0
        duration = float(event.get("dDurationMs", 0) or 0) / 1000.0
        cues.append(Cue(start=start, end=start + min(max(duration, 0.1), 10.0), text=text))
    return cues


def chunk_cues(cues: list[Cue], target_seconds: float = 6.0, min_chars: int = 10) -> list[Cue]:
    chunks: list[Cue] = []
    index = 0
    while index < len(cues):
        start = cues[index].start
        end = cues[index].end
        texts: list[str] = []
        normalized_chars = 0
        cursor = index
        while cursor < len(cues):
            cue = cues[cursor]
            if texts and cue.start - start >= target_seconds and normalized_chars >= min_chars:
                break
            texts.append(cue.text)
            normalized_chars += len(normalize_text(cue.text))
            end = max(end, cue.end)
            cursor += 1
            if cue.start - start >= target_seconds * 1.6:
                break
        chunks.append(Cue(start=start, end=end, text=" ".join(texts)))
        index = cursor
    return [chunk for chunk in chunks if normalize_text(chunk.text)]


def build_source_windows(cues: list[Cue], max_cues: int = 6) -> tuple[list[str], list[Cue]]:
    choices: list[str] = []
    windows: list[Cue] = []
    seen: set[tuple[int, int, str]] = set()
    for start_index in range(len(cues)):
        texts: list[str] = []
        for end_index in range(start_index, min(len(cues), start_index + max_cues)):
            if end_index > start_index:
                cue_gap = cues[end_index].start - cues[end_index - 1].start
                window_span = cues[end_index].start - cues[start_index].start
                if cue_gap > 12.0 or window_span > 30.0:
                    break
            texts.append(cues[end_index].text)
            normalized = normalize_text(" ".join(texts))
            if len(normalized) < 4:
                continue
            key = (round(cues[start_index].start), round(cues[end_index].end), normalized)
            if key in seen:
                continue
            seen.add(key)
            choices.append(normalized)
            windows.append(
                Cue(
                    start=cues[start_index].start,
                    end=cues[end_index].end,
                    text=" ".join(texts),
                )
            )
    return choices, windows


def align_transcripts(short_cues: list[Cue], long_cues: list[Cue], top_k: int = 12) -> dict[str, Any]:
    short_chunks = chunk_cues(short_cues)
    source_choices, source_windows = build_source_windows(long_cues)
    if not short_chunks or not source_choices:
        return {"status": "insufficient_transcript", "matches": []}

    candidate_layers: list[list[Match]] = []
    for short_index, chunk in enumerate(short_chunks):
        query = normalize_text(chunk.text)
        extracted = process.extract(
            query,
            source_choices,
            scorer=fuzz.WRatio,
            limit=top_k,
            score_cutoff=30,
        )
        layer = [
            Match(
                short_index=short_index,
                source_start=source_windows[choice_index].start,
                source_end=source_windows[choice_index].end,
                score=float(score),
            )
            for _choice, score, choice_index in extracted
        ]
        if not layer:
            layer = [Match(short_index, 0.0, 0.0, 0.0)]
        candidate_layers.append(layer)

    scores: list[list[float]] = []
    parents: list[list[int]] = []
    for layer_index, layer in enumerate(candidate_layers):
        if layer_index == 0:
            scores.append([candidate.score for candidate in layer])
            parents.append([-1] * len(layer))
            continue
        previous_layer = candidate_layers[layer_index - 1]
        previous_scores = scores[layer_index - 1]
        current_scores: list[float] = []
        current_parents: list[int] = []
        short_delta = short_chunks[layer_index].start - short_chunks[layer_index - 1].start
        for candidate in layer:
            best_score = float("-inf")
            best_parent = 0
            for previous_index, previous in enumerate(previous_layer):
                source_delta = candidate.source_start - previous.source_start
                backward_penalty = 80.0 if source_delta < -2.0 else 0.0
                pace_penalty = min(abs(source_delta - short_delta), 60.0) * 0.08
                value = previous_scores[previous_index] + candidate.score - backward_penalty - pace_penalty
                if value > best_score:
                    best_score = value
                    best_parent = previous_index
            current_scores.append(best_score)
            current_parents.append(best_parent)
        scores.append(current_scores)
        parents.append(current_parents)

    cursor = max(range(len(scores[-1])), key=scores[-1].__getitem__)
    path: list[Match] = []
    for layer_index in range(len(candidate_layers) - 1, -1, -1):
        path.append(candidate_layers[layer_index][cursor])
        cursor = parents[layer_index][cursor]
    path.reverse()

    trusted = [match for match in path if match.score >= 55.0]
    total_chars = sum(len(normalize_text(chunk.text)) for chunk in short_chunks)
    matched_chars = sum(
        len(normalize_text(short_chunks[match.short_index].text)) for match in trusted
    )
    coverage = matched_chars / total_chars if total_chars else 0.0
    if len(trusted) < 2 or coverage < 0.55:
        return {
            "status": "insufficient_alignment",
            "coverage": round(coverage, 4),
            "matches": trusted,
            "short_chunks": short_chunks,
        }

    segment_count = 1
    backward_jumps = 0
    excess_gap_seconds = 0.0
    for previous, current in zip(trusted, trusted[1:]):
        previous_short = short_chunks[previous.short_index]
        current_short = short_chunks[current.short_index]
        source_gap = current.source_start - previous.source_end
        short_gap = max(0.0, current_short.start - previous_short.end)
        excess_gap = max(0.0, source_gap - short_gap - 3.0)
        excess_gap_seconds += excess_gap
        if current.source_start < previous.source_start - 2.0:
            backward_jumps += 1
            segment_count += 1
        elif excess_gap > 8.0:
            segment_count += 1

    predicted_start = min(match.source_start for match in trusted)
    predicted_end = max(match.source_end for match in trusted)
    short_start = short_chunks[trusted[0].short_index].start
    short_end = short_chunks[trusted[-1].short_index].end
    short_span = max(0.1, short_end - short_start)
    source_span = max(0.1, predicted_end - predicted_start)
    removed_or_unmatched_span = max(0.0, source_span - short_span)

    if (
        backward_jumps > 0
        or segment_count >= 3
        or removed_or_unmatched_span > 30.0
    ):
        status = "heavy_edit"
    elif segment_count == 2 or removed_or_unmatched_span > 15.0:
        status = "light_edit"
    else:
        status = "continuous"

    return {
        "status": status,
        "coverage": round(coverage, 4),
        "mean_match_score": round(sum(match.score for match in trusted) / len(trusted), 2),
        "predicted_start": round(predicted_start, 3),
        "predicted_end": round(predicted_end, 3),
        "source_span": round(source_span, 3),
        "short_span": round(short_span, 3),
        "segment_count": segment_count,
        "backward_jumps": backward_jumps,
        "excess_gap_seconds": round(excess_gap_seconds, 3),
        "matches": trusted,
        "short_chunks": short_chunks,
    }


def interval_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return intersection / union if union > 0 else 0.0


def row_video_id(row: dict[str, str], kind: str) -> str:
    direct = row.get(f"{kind}_video_id", "").strip()
    if direct:
        return direct
    url = row.get(f"{kind}_video_url", "").strip()
    return extract_youtube_id(url) if url else ""


def display_timestamp(seconds: float | str) -> str:
    if seconds == "":
        return ""
    value = float(seconds)
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    remaining = value % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:06.3f}"
    return f"{minutes:02d}:{remaining:06.3f}"


class SubtitleCollector:
    def __init__(self, cache_dir: Path, sleep_seconds: float = 0.4) -> None:
        self.cache_dir = cache_dir
        self.sleep_seconds = sleep_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ydl = YoutubeDL(
            {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
            }
        )

    @staticmethod
    def _preferred_track(info: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
        automatic = info.get("automatic_captions", {})
        manual = info.get("subtitles", {})
        original_languages = [
            language
            for language in ("ko-orig", "en-orig")
            if language in automatic
        ]
        original_languages.extend(
            language
            for language in automatic
            if language.endswith("-orig") and language not in original_languages
        )
        source_preferences = (
            ("automatic", automatic, tuple(original_languages)),
            ("manual", manual, ("ko-orig", "ko", "ko-KR")),
            ("manual", manual, ("en-orig", "en")),
            ("automatic", automatic, ("ko", "ko-KR")),
            ("automatic", automatic, ("en",)),
        )
        for source_name, tracks_by_language, preferred_languages in source_preferences:
            ordered = [language for language in preferred_languages if language in tracks_by_language]
            for language in ordered:
                formats = tracks_by_language.get(language, [])
                track = next((item for item in formats if item.get("ext") == "json3"), None)
                if track and track.get("url"):
                    return source_name, language, track
        return None

    def fetch(self, video_id: str) -> tuple[Path | None, str, str]:
        path = self.cache_dir / f"{video_id}.json3"
        metadata_path = self.cache_dir / f"{video_id}.meta.json"
        if path.exists() and path.stat().st_size > 20:
            language = "cached"
            source = "cached"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                language = str(metadata.get("language", language))
                source = str(metadata.get("source", source))
            return path, source, language

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            info = self.ydl.extract_info(url, download=False)
            selected = self._preferred_track(info)
            if not selected:
                return None, "missing", ""
            source, language, track = selected
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    with self.ydl.urlopen(track["url"]) as response:
                        payload = response.read()
                    json.loads(payload.decode("utf-8"))
                    path.write_bytes(payload)
                    metadata_path.write_text(
                        json.dumps(
                            {
                                "video_id": video_id,
                                "title": info.get("title", ""),
                                "source": source,
                                "language": language,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    time.sleep(self.sleep_seconds)
                    return path, source, language
                except Exception as exc:  # Network retries are intentionally broad.
                    last_error = exc
                    time.sleep((attempt + 1) * 1.5)
            return None, "error", type(last_error).__name__ if last_error else "download_error"
        except Exception as exc:
            return None, "error", type(exc).__name__


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_rows(
    rows: Iterable[dict[str, str]],
    collector: SubtitleCollector,
    gold_margin_sec: float = 0.0,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rows = list(rows)
    for index, row in enumerate(rows, start=1):
        short_id = row_video_id(row, "short")
        long_id = row_video_id(row, "long")
        print(f"[{index}/{len(rows)}] {short_id} -> {long_id}", flush=True)
        short_path, short_source, short_language = collector.fetch(short_id)
        long_path, long_source, long_language = collector.fetch(long_id)
        base: dict[str, Any] = {
            "candidate_id": row.get("candidate_id", ""),
            "pair_id": row.get("pair_id", ""),
            "channel_name": row.get("channel_name", ""),
            "performance_label": row.get("performance_label", ""),
            "short_video_id": short_id,
            "long_video_id": long_id,
            "short_video_url": row.get("short_video_url", f"https://www.youtube.com/shorts/{short_id}"),
            "long_video_url": row.get("long_video_url", f"https://www.youtube.com/watch?v={long_id}"),
            "short_subtitle_source": short_source,
            "short_subtitle_language": short_language,
            "long_subtitle_source": long_source,
            "long_subtitle_language": long_language,
            "gold_start": row.get("start_sec", ""),
            "gold_end": row.get("end_sec", ""),
        }
        if not short_path or not long_path:
            base.update(
                {
                    "alignment_status": "missing_subtitle",
                    "coverage": "",
                    "mean_match_score": "",
                    "predicted_start": "",
                    "predicted_end": "",
                    "predicted_start_time": "",
                    "predicted_end_time": "",
                    "source_span": "",
                    "short_span": "",
                    "segment_count": "",
                    "backward_jumps": "",
                    "excess_gap_seconds": "",
                    "start_error_seconds": "",
                    "end_error_seconds": "",
                    "gold_interval_iou": "",
                    "auto_accept": 0,
                }
            )
            output.append(base)
            continue

        short_cues = parse_json3(short_path)
        long_cues = parse_json3(long_path)
        gold_start = float(row["start_sec"]) if row.get("start_sec") else None
        gold_end = float(row["end_sec"]) if row.get("end_sec") else None
        if gold_margin_sec > 0 and gold_start is not None and gold_end is not None:
            search_start = max(0.0, gold_start - gold_margin_sec)
            search_end = gold_end + gold_margin_sec
            long_cues = [
                cue
                for cue in long_cues
                if cue.end >= search_start and cue.start <= search_end
            ]
            base["alignment_search_scope"] = "gold_local"
            base["gold_margin_sec"] = gold_margin_sec
        else:
            base["alignment_search_scope"] = "full_longform"
            base["gold_margin_sec"] = ""

        result = align_transcripts(short_cues, long_cues)
        predicted_start = result.get("predicted_start", "")
        predicted_end = result.get("predicted_end", "")
        comparable = predicted_start != "" and predicted_end != "" and gold_start is not None and gold_end is not None
        base.update(
            {
                "alignment_status": result.get("status", ""),
                "coverage": result.get("coverage", ""),
                "mean_match_score": result.get("mean_match_score", ""),
                "predicted_start": predicted_start,
                "predicted_end": predicted_end,
                "predicted_start_time": display_timestamp(predicted_start),
                "predicted_end_time": display_timestamp(predicted_end),
                "source_span": result.get("source_span", ""),
                "short_span": result.get("short_span", ""),
                "segment_count": result.get("segment_count", ""),
                "backward_jumps": result.get("backward_jumps", ""),
                "excess_gap_seconds": result.get("excess_gap_seconds", ""),
                "start_error_seconds": round(abs(float(predicted_start) - gold_start), 3) if comparable else "",
                "end_error_seconds": round(abs(float(predicted_end) - gold_end), 3) if comparable else "",
                "gold_interval_iou": round(
                    interval_iou(float(predicted_start), float(predicted_end), gold_start, gold_end), 4
                )
                if comparable
                else "",
                "auto_accept": int(result.get("status") in {"continuous", "light_edit"}),
            }
        )
        output.append(base)
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    subtitle_pairs = 0
    comparable: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("alignment_status", ""))
        statuses[status] = statuses.get(status, 0) + 1
        if status != "missing_subtitle":
            subtitle_pairs += 1
        if row.get("gold_interval_iou") != "":
            comparable.append(row)
    return {
        "pairs": len(rows),
        "subtitle_pair_success": subtitle_pairs,
        "subtitle_pair_success_rate": round(subtitle_pairs / len(rows), 4) if rows else 0.0,
        "alignment_status_counts": statuses,
        "auto_accept_pairs": sum(str(row.get("auto_accept", "0")) == "1" or row.get("auto_accept") == 1 for row in rows),
        "comparable_pairs": len(comparable),
        "mean_gold_interval_iou": round(
            sum(float(row["gold_interval_iou"]) for row in comparable) / len(comparable), 4
        )
        if comparable
        else None,
        "within_5s_start_rate": round(
            sum(float(row["start_error_seconds"]) <= 5.0 for row in comparable) / len(comparable), 4
        )
        if comparable
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Shorts-to-long-form transcript continuity with subtitle-only downloads.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/gold_reference_judge_v6/input/candidate_sources_private.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/subtitle_alignment_audit"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.4)
    parser.add_argument(
        "--gold-margin-sec",
        type=float,
        default=0.0,
        help="Restrict alignment to the labelled gold interval plus this margin. Use only for re-verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    cache_dir = args.output_dir / "subtitles"
    collector = SubtitleCollector(cache_dir=cache_dir, sleep_seconds=args.sleep_seconds)
    audited = audit_rows(rows, collector, gold_margin_sec=args.gold_margin_sec)
    write_rows(args.output_dir / "alignment_audit.csv", audited)
    write_rows(
        args.output_dir / "auto_accepted.csv",
        [row for row in audited if row.get("auto_accept") == 1],
    )
    write_rows(
        args.output_dir / "review_queue.csv",
        [row for row in audited if row.get("auto_accept") != 1],
    )
    summary = summarize(audited)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
