from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_frame(capture: cv2.VideoCapture, second: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
    ok, frame = capture.read()
    return frame if ok else None


def duration_seconds(capture: cv2.VideoCapture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    return frames / fps if fps > 0 else 0.0


def resize_for_features(frame: np.ndarray, max_side: int = 480) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def visual_variants(frame: np.ndarray, is_short: bool) -> list[np.ndarray]:
    height, width = frame.shape[:2]
    variants = [frame]
    if is_short:
        variants.append(frame[round(height * 0.08) : round(height * 0.92), :])
        variants.append(frame[round(height * 0.10) : round(height * 0.80), :])
        variants.append(frame[round(height * 0.16) : round(height * 0.76), :])
    elif width > height:
        crop_width = max(1, round(height * 9 / 16))
        for center_ratio in (0.28, 0.5, 0.72):
            center = round(width * center_ratio)
            left = max(0, min(width - crop_width, center - crop_width // 2))
            variants.append(frame[:, left : left + crop_width])
    return variants


def phash(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    transformed = cv2.dct(np.float32(resized))
    low_frequency = transformed[:8, :8]
    threshold = np.median(low_frequency[1:, :])
    return (low_frequency > threshold).reshape(-1)


def sample_hashes(
    video_path: Path,
    interval: float,
    is_short: bool,
    end_margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    duration = duration_seconds(capture)
    start = min(1.0, max(0.0, duration / 10))
    end = max(start, duration - end_margin)
    times = np.arange(start, end + 0.001, interval, dtype=np.float32)
    hashes: list[np.ndarray] = []
    valid_times: list[float] = []
    for second in times:
        frame = read_frame(capture, float(second))
        if frame is None:
            continue
        hashes.append(np.stack([phash(item) for item in visual_variants(frame, is_short)]))
        valid_times.append(float(second))
    capture.release()
    return np.asarray(valid_times, dtype=np.float32), np.asarray(hashes, dtype=bool)


def top_hash_candidates(
    short_hashes: np.ndarray,
    long_hashes: np.ndarray,
    top_k: int,
) -> list[list[tuple[int, int]]]:
    output: list[list[tuple[int, int]]] = []
    for short_variants in short_hashes:
        distances = np.count_nonzero(
            short_variants[:, None, None, :] != long_hashes[None, :, :, :],
            axis=-1,
        )
        best_by_time = distances.min(axis=(0, 2))
        indices = np.argsort(best_by_time)[:top_k]
        output.append([(int(index), int(best_by_time[index])) for index in indices])
    return output


def sift_descriptors(
    capture: cv2.VideoCapture,
    second: float,
    sift: cv2.SIFT,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    frame = read_frame(capture, second)
    if frame is None:
        return [], None
    return sift.detectAndCompute(resize_for_features(frame), None)


def sift_match_score(
    short_descriptor: np.ndarray | None,
    long_descriptor: np.ndarray | None,
    matcher: cv2.BFMatcher,
) -> tuple[int, float]:
    if short_descriptor is None or long_descriptor is None:
        return 0, 0.0
    matches = matcher.knnMatch(short_descriptor, long_descriptor, k=2)
    ratios = [first.distance / max(second.distance, 1e-6) for first, second in matches]
    good = [ratio for ratio in ratios if ratio < 0.72]
    return len(good), round(float(np.mean(good)), 4) if good else 0.0


def align_visual(
    short_path: Path,
    long_path: Path,
    interval: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    short_times, short_hashes = sample_hashes(
        short_path,
        interval,
        is_short=True,
        end_margin=1.0,
    )
    long_times, long_hashes = sample_hashes(
        long_path,
        interval,
        is_short=False,
        end_margin=15.0,
    )
    hash_candidates = top_hash_candidates(short_hashes, long_hashes, top_k)

    short_capture = cv2.VideoCapture(str(short_path))
    long_capture = cv2.VideoCapture(str(long_path))
    sift = cv2.SIFT_create(nfeatures=450)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    long_descriptor_cache: dict[int, np.ndarray | None] = {}
    frame_rows: list[dict[str, Any]] = []

    for short_index, candidates in enumerate(hash_candidates):
        short_second = float(short_times[short_index])
        _keypoints, short_descriptor = sift_descriptors(short_capture, short_second, sift)
        scored: list[tuple[int, float, int, int]] = []
        for long_index, hash_distance in candidates:
            if long_index not in long_descriptor_cache:
                _keypoints, descriptor = sift_descriptors(
                    long_capture,
                    float(long_times[long_index]),
                    sift,
                )
                long_descriptor_cache[long_index] = descriptor
            good_matches, mean_ratio = sift_match_score(
                short_descriptor,
                long_descriptor_cache[long_index],
                matcher,
            )
            scored.append((good_matches, -mean_ratio, long_index, hash_distance))
        best = max(scored, default=(0, 0.0, 0, 64))
        good_matches, negative_mean_ratio, long_index, hash_distance = best
        long_second = float(long_times[long_index])
        frame_rows.append(
            {
                "short_time": round(short_second, 3),
                "long_time": round(long_second, 3),
                "offset": round(long_second - short_second, 3),
                "good_sift_matches": good_matches,
                "mean_sift_ratio": round(-negative_mean_ratio, 4),
                "phash_distance": hash_distance,
            }
        )

    short_capture.release()
    long_capture.release()

    trusted = [
        row
        for row in frame_rows
        if row["good_sift_matches"] >= 10 and row["phash_distance"] <= 20
    ]
    offset_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trusted:
        offset_buckets[round(float(row["offset"]) / 2.0)].append(row)
    clusters = sorted(
        offset_buckets.values(),
        key=lambda group: (
            len(group),
            sum(int(row["good_sift_matches"]) for row in group),
        ),
        reverse=True,
    )
    dominant = clusters[0] if clusters else []
    summary = {
        "short_video": str(short_path),
        "long_video": str(long_path),
        "sample_interval_sec": interval,
        "short_sample_count": len(frame_rows),
        "trusted_frame_count": len(trusted),
        "dominant_offset_frame_count": len(dominant),
        "dominant_offset": round(float(np.median([row["offset"] for row in dominant])), 3)
        if dominant
        else None,
        "predicted_start": round(
            min(float(row["long_time"]) for row in dominant), 3
        )
        if dominant
        else None,
        "predicted_end": round(
            max(float(row["long_time"]) for row in dominant) + interval, 3
        )
        if dominant
        else None,
        "max_sift_matches": max(
            (int(row["good_sift_matches"]) for row in frame_rows),
            default=0,
        ),
        "median_sift_matches_dominant": round(
            float(np.median([row["good_sift_matches"] for row in dominant])),
            2,
        )
        if dominant
        else None,
        "offset_cluster_count": len(clusters),
    }
    return frame_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fallback visual timestamp alignment for Shorts with unusable subtitles."
    )
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--long", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = align_visual(args.short, args.long, args.interval, args.top_k)
    write_csv(args.output_dir / "frame_matches.csv", rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
