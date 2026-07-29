from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vpick_client import VpickClient, save_json  # noqa: E402


DEFAULT_MANIFEST = (
    ROOT / "results" / "gold_reference_judge_v8_ko" / "input" / "candidate_sources_private.csv"
)
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "vpick"
VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def response_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "projects", "assets"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def read_target_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {
            row.get("long_video_id", "").strip()
            for row in csv.DictReader(f)
            if row.get("long_video_id", "").strip()
        }


def infer_long_video_id(asset_name: str, target_ids: set[str]) -> str:
    target_matches = [video_id for video_id in target_ids if video_id in asset_name]
    if len(target_matches) == 1:
        return target_matches[0]
    normalized = asset_name.removesuffix("-longform")
    suffix = normalized.rsplit("_", 1)[-1]
    if len(suffix) == 11 and VIDEO_ID_RE.fullmatch(suffix):
        return suffix
    tokens = [token.strip("_") for token in VIDEO_ID_RE.findall(normalized)]
    tokens = [token for token in tokens if len(token) == 11]
    if not tokens:
        return ""
    likely = [
        token
        for token in tokens
        if any(char.isdigit() for char in token)
        and any(char.isalpha() for char in token)
    ]
    return likely[-1] if likely else tokens[-1]


def payload_scene_count(payload: Any) -> int:
    return len(response_rows(payload))


def paged_projects(client: VpickClient) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = client.list_projects(offset=offset, limit=100)
        rows = response_rows(payload)
        output.extend(rows)
        total = int(payload.get("total", len(output))) if isinstance(payload, dict) else len(output)
        if not rows or len(output) >= total:
            return output
        offset += len(rows)


def paged_assets(client: VpickClient, project_id: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = client.list_assets(project_id, offset=offset, limit=100)
        rows = response_rows(payload)
        output.extend(rows)
        total = int(payload.get("total", len(output))) if isinstance(payload, dict) else len(output)
        if not rows or len(output) >= total:
            return output
        offset += len(rows)


def merge_canonical_scene(
    raw_dir: Path,
    long_video_id: str,
    payload: Any,
) -> str:
    if not long_video_id:
        return "unmapped"
    canonical = raw_dir / f"{long_video_id}_scenes.json"
    if not canonical.exists():
        save_json(canonical, payload)
        return "saved_missing_canonical"
    try:
        existing = json.loads(canonical.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        save_json(canonical, payload)
        return "replaced_unreadable_canonical"
    if payload_scene_count(payload) > payload_scene_count(existing):
        save_json(canonical, payload)
        return "replaced_with_more_scenes"
    return "kept_existing_canonical"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-label", required=True)
    parser.add_argument("--target-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--skip-scenes", action="store_true")
    args = parser.parse_args()

    if not os.getenv("VPICK_EMAIL") or not (
        os.getenv("VPICK_PASSWORD") or os.getenv("VPICK_ACCESS_TOKEN")
    ):
        raise RuntimeError("Set VPICK_EMAIL and VPICK_PASSWORD, or VPICK_ACCESS_TOKEN.")

    target_ids = read_target_ids(args.target_manifest)
    account_dir = args.raw_dir / "accounts" / args.account_label
    account_dir.mkdir(parents=True, exist_ok=True)
    client = VpickClient()
    client.login_from_env()
    projects = paged_projects(client)
    inventory: list[dict[str, Any]] = []

    for project in projects:
        project_id = str(project.get("project_id") or "")
        project_name = str(project.get("project_name") or "")
        for asset in paged_assets(client, project_id):
            asset_id = str(asset.get("asset_id") or "")
            asset_name = str(asset.get("asset_name") or "")
            status = str(asset.get("status") or "")
            long_video_id = infer_long_video_id(asset_name, target_ids)
            scene_count = 0
            canonical_action = "not_ready"
            error = ""
            if status == "READY" and not args.skip_scenes:
                try:
                    scenes = client.get_scenes(project_id, asset_id)
                    scene_count = payload_scene_count(scenes)
                    save_json(account_dir / f"{asset_id}_scenes.json", scenes)
                    save_json(account_dir / f"{asset_id}_asset_status.json", asset)
                    if long_video_id in target_ids:
                        canonical_action = merge_canonical_scene(
                            args.raw_dir,
                            long_video_id,
                            scenes,
                        )
                    else:
                        canonical_action = "non_target_account_copy_only"
                except Exception as exc:
                    canonical_action = "scene_fetch_failed"
                    error = str(exc)[:300]
            inventory.append(
                {
                    "account_label": args.account_label,
                    "project_id": project_id,
                    "project_name": project_name,
                    "asset_id": asset_id,
                    "asset_name": asset_name,
                    "status": status,
                    "long_video_id": long_video_id,
                    "is_target_longform": int(long_video_id in target_ids),
                    "scene_count": scene_count,
                    "canonical_action": canonical_action,
                    "error": error,
                }
            )

    write_csv(account_dir / "inventory.csv", inventory)
    summary = {
        "account_label": args.account_label,
        "project_count": len(projects),
        "asset_count": len(inventory),
        "ready_asset_count": sum(row["status"] == "READY" for row in inventory),
        "mapped_asset_count": sum(bool(row["long_video_id"]) for row in inventory),
        "target_asset_count": sum(row["is_target_longform"] for row in inventory),
        "target_longform_ids": sorted(
            {row["long_video_id"] for row in inventory if row["is_target_longform"]}
        ),
        "canonical_added_count": sum(
            row["canonical_action"] == "saved_missing_canonical" for row in inventory
        ),
        "scene_fetch_failure_count": sum(
            row["canonical_action"] == "scene_fetch_failed" for row in inventory
        ),
    }
    save_json(account_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
