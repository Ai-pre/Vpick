from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_BLIND_KEYS = {
    "channel",
    "channel_id",
    "channel_name",
    "channel_name_raw",
    "channel_performance_percentile",
    "channel_view_percentile",
    "is_target",
    "label",
    "label_confidence",
    "like_count",
    "likes",
    "performance_label",
    "percentile",
    "relative_log_view_score",
    "short_likes",
    "short_views",
    "source_system",
    "view_count",
    "views",
}


def load_config(path: Path) -> dict[str, Any]:
    """Load the JSON-compatible YAML config without adding a YAML dependency."""
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(path.resolve())
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or (list(rows[0]) if rows else ["status"]))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def as_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return None if number is None else int(number)


def parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def upload_age_days(upload_date: Any, snapshot_date: Any) -> int | None:
    uploaded = parse_date(upload_date)
    observed = parse_date(snapshot_date)
    if not uploaded or not observed or observed < uploaded:
        return None
    return (observed - uploaded).days


def rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = average_rank
        cursor = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson(rankdata(left), rankdata(right))


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0 or len(labels) != len(scores):
        return None
    ranks = rankdata(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0 or len(labels) != len(scores):
        return None
    ordered = sorted(zip(scores, labels), reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float | None:
    if len(labels) != len(predictions) or not labels:
        return None
    scores: list[float] = []
    for target_class in (0, 1):
        tp = sum(a == target_class and b == target_class for a, b in zip(labels, predictions))
        fp = sum(a != target_class and b == target_class for a, b in zip(labels, predictions))
        fn = sum(a == target_class and b != target_class for a, b in zip(labels, predictions))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return statistics.mean(scores)


def percentile(values: Sequence[float], value: float) -> float | None:
    if not values:
        return None
    lower = sum(candidate < value for candidate in values)
    equal = sum(candidate == value for candidate in values)
    return 100.0 * (lower + 0.5 * equal) / len(values)


def relative_log_view_score(views: float, channel_median: float) -> float:
    return math.log2((views + 1.0) / (channel_median + 1.0))


def stable_id(prefix: str, *parts: Any, length: int = 14) -> str:
    material = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def deterministic_group_split(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str = "longform_id",
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: str = "vpick-evaluation-v1",
) -> dict[str, str]:
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Invalid split fractions")
    groups = sorted({str(row[group_key]) for row in rows})
    assignment: dict[str, str] = {}
    for group in groups:
        digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        if value < train_fraction:
            split = "train"
        elif value < train_fraction + validation_fraction:
            split = "validation"
        else:
            split = "test"
        assignment[group] = split
    return assignment


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def assert_blind_payload(payload: dict[str, Any]) -> None:
    leaked = sorted(nested_keys(payload) & FORBIDDEN_BLIND_KEYS)
    if leaked:
        raise ValueError(f"Blind payload leaks private fields: {', '.join(leaked)}")


def aggregate_by_candidate(
    rows: Sequence[dict[str, Any]],
    candidate_key: str,
    score_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[candidate_key])].append(row)
    output: list[dict[str, Any]] = []
    for candidate_id, group in sorted(groups.items()):
        item: dict[str, Any] = {
            "candidate_id": candidate_id,
            "repeat_count": len(group),
        }
        for field in score_fields:
            values = [value for row in group if (value := as_float(row.get(field))) is not None]
            item[f"{field}_mean"] = statistics.mean(values) if values else None
            item[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(item)
    return output


def bootstrap_group_metric(
    rows: Sequence[dict[str, Any]],
    metric,
    *,
    group_key: str = "longform_id",
    iterations: int = 1000,
    seed: int = 20260724,
) -> dict[str, float | int | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    groups = sorted(grouped)
    observed = metric(list(rows))
    if observed is None or len(groups) < 2:
        return {"estimate": observed, "ci_lower": None, "ci_upper": None, "iterations": 0}
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for group in rng.choices(groups, k=len(groups)):
            sampled.extend(grouped[group])
        value = metric(sampled)
        if value is not None and math.isfinite(value):
            samples.append(float(value))
    if not samples:
        return {"estimate": observed, "ci_lower": None, "ci_upper": None, "iterations": 0}
    samples.sort()
    lower = samples[max(0, int(0.025 * (len(samples) - 1)))]
    upper = samples[min(len(samples) - 1, int(0.975 * (len(samples) - 1)))]
    return {
        "estimate": float(observed),
        "ci_lower": lower,
        "ci_upper": upper,
        "iterations": len(samples),
    }


def rounded(value: Any, digits: int = 4) -> Any:
    return None if value is None else round(float(value), digits)
