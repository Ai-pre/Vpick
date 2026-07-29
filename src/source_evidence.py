from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BLIND_FIELDS = (
    "candidate_id",
    "duration_sec",
    "description",
    "transcript",
    "before_context",
    "after_context",
)


def read_longforms(path: Path) -> dict[str, dict[str, Any]]:
    longforms: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            longforms[row["longform_id"]] = row
    return longforms


def format_timestamp(milliseconds: Any) -> str:
    seconds = max(0, int(float(milliseconds or 0) / 1000))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _scene_line(
    scene: dict[str, Any],
    *,
    local: bool,
    description_chars: int = 120,
) -> str:
    start = format_timestamp(scene.get("start_ms"))
    end = format_timestamp(scene.get("end_ms"))
    name = str(scene.get("scene_name") or "").strip()
    description = str(scene.get("description") or "").strip()
    if len(description) > description_chars:
        description = description[: description_chars - 1].rstrip() + "…"
    prefix = "[후보 주변] " if local else ""
    line = f"{prefix}[{start}-{end}] {name}"
    if description:
        line += f": {description}"
    return line


def _evenly_spaced_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return list(indices)
    if count == 1:
        return [indices[len(indices) // 2]]
    positions = {
        round(position * (len(indices) - 1) / (count - 1))
        for position in range(count)
    }
    return [indices[position] for position in sorted(positions)]


def source_outline(
    longform: dict[str, Any],
    max_chars: int,
    *,
    candidate_start_sec: float | None = None,
    candidate_end_sec: float | None = None,
    local_margin_sec: float = 90.0,
) -> str:
    """Build a bounded source outline without dropping late candidate context.

    All scene descriptions are first compacted. If the complete compact outline
    still exceeds the limit, every scene near the candidate is preserved and
    the rest of the timeline is sampled evenly.
    """

    if max_chars < 500:
        raise ValueError("max_chars must be at least 500")
    scenes = list(longform.get("scenes", []))
    title = str(longform.get("title") or "").strip()
    duration_ms = float(longform.get("duration_ms") or 0)
    header_parts = []
    if title:
        header_parts.append(f"제목: {title}")
    if duration_ms > 0:
        header_parts.append(f"원본 길이: {format_timestamp(duration_ms)}")
    header = "\n".join(header_parts)

    candidate_start_ms = (
        float(candidate_start_sec) * 1000
        if candidate_start_sec is not None
        else None
    )
    candidate_end_ms = (
        float(candidate_end_sec) * 1000
        if candidate_end_sec is not None
        else candidate_start_ms
    )
    local_indices: set[int] = set()
    if candidate_start_ms is not None and candidate_end_ms is not None:
        local_start = candidate_start_ms - local_margin_sec * 1000
        local_end = candidate_end_ms + local_margin_sec * 1000
        for index, scene in enumerate(scenes):
            scene_start = float(scene.get("start_ms") or 0)
            scene_end = float(scene.get("end_ms") or scene_start)
            if scene_end >= local_start and scene_start <= local_end:
                local_indices.add(index)

    compact_lines = [
        _scene_line(scene, local=index in local_indices)
        for index, scene in enumerate(scenes)
    ]
    complete = "\n".join(part for part in [header, *compact_lines] if part)
    if len(complete) <= max_chars:
        return complete

    header_cost = len(header) + (1 if header else 0)
    local_cost = sum(len(compact_lines[index]) + 1 for index in local_indices)
    if local_cost + header_cost > max_chars:
        center_ms = (
            (candidate_start_ms + candidate_end_ms) / 2
            if candidate_start_ms is not None and candidate_end_ms is not None
            else 0
        )
        local_by_distance = sorted(
            local_indices,
            key=lambda index: abs(
                (
                    float(scenes[index].get("start_ms") or 0)
                    + float(scenes[index].get("end_ms") or 0)
                )
                / 2
                - center_ms
            ),
        )
        kept_local: set[int] = set()
        used = header_cost
        for index in local_by_distance:
            cost = len(compact_lines[index]) + 1
            if used + cost > max_chars:
                continue
            kept_local.add(index)
            used += cost
        local_indices = kept_local
        local_cost = used - header_cost

    global_indices = [
        index for index in range(len(scenes)) if index not in local_indices
    ]
    remaining = max_chars - header_cost - local_cost
    average_global_cost = (
        sum(len(compact_lines[index]) + 1 for index in global_indices)
        / max(1, len(global_indices))
    )
    global_count = max(0, int(remaining / max(1, average_global_cost)))
    selected = set(local_indices)
    selected.update(_evenly_spaced_indices(global_indices, global_count))

    # If rounding exceeded the budget, discard sampled scenes farthest from
    # evenly representing the timeline while always preserving local scenes.
    rendered = "\n".join(
        part
        for part in [
            header,
            *(compact_lines[index] for index in sorted(selected)),
        ]
        if part
    )
    while len(rendered) > max_chars:
        removable = [index for index in selected if index not in local_indices]
        if not removable:
            rendered = rendered[: max_chars - 1].rstrip() + "…"
            break
        selected.remove(removable[len(removable) // 2])
        rendered = "\n".join(
            part
            for part in [
                header,
                *(compact_lines[index] for index in sorted(selected)),
            ]
            if part
        )
    return rendered
