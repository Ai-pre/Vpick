from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fetch_vpick_account_inventory import (  # noqa: E402
    merge_canonical_scene,
    paged_assets,
    paged_projects,
)
from src.vpick_client import VpickClient, save_json  # noqa: E402


DEFAULT_FEATURES = (
    ROOT
    / "deliverables"
    / "2026-07-24"
    / "performance_ranker"
    / "candidate_features_60_PRIVATE.csv"
)
DEFAULT_JUDGE = (
    ROOT
    / "results"
    / "gold_reference_judge_v9_v7"
    / "direct_codex"
    / "reference_judge_v7_scores.csv"
)
DEFAULT_GOLD = ROOT / "deliverables" / "2026-07-23" / "vpick_goldlabel_60_normalized.csv"
DEFAULT_RAW = ROOT / "data" / "raw" / "vpick"
TERMINAL = {
    "READY",
    "FAILED",
    "PRETRANSCODING_FAILED",
    "MI_ANALYSIS_FAILED",
    "INSUFFICIENT_CREDITS",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def credit_remaining(client: VpickClient) -> int:
    payload = client.request("GET", "/plan-page")
    return int(
        payload.get("current_plan", {})
        .get("credit_usage", {})
        .get("remaining", 0)
    )


def project_id_for_name(client: VpickClient, project_name: str) -> str:
    for project in paged_projects(client):
        if str(project.get("project_name") or "") == project_name:
            return str(project["project_id"])
    return client.create_project(project_name)


def existing_assets_by_name(client: VpickClient, project_id: str) -> dict[str, dict[str, Any]]:
    return {
        str(asset.get("asset_name") or ""): asset
        for asset in paged_assets(client, project_id)
    }


def priority_longforms(
    feature_rows: list[dict[str, str]],
    judge_rows: list[dict[str, str]],
    mode: str,
) -> list[dict[str, Any]]:
    feature_by_id = {row["candidate_id"]: row for row in feature_rows}
    selected: dict[str, dict[str, Any]] = {}
    for judge in judge_rows:
        feature = feature_by_id.get(judge["candidate_id"])
        if not feature or feature.get("vpick_available") == "1":
            continue
        flags = set(filter(None, (judge.get("failure_flags") or "").split("|")))
        intelligibility = to_float(judge.get("evidence_transcript_intelligibility"), 5.0)
        boundary = to_float(judge.get("evidence_boundary_observability"), 5.0)
        verdict = judge.get("verdict", "")
        is_priority = (
            verdict == "abstain"
            or intelligibility <= 3.0
            or boundary <= 3.0
            or "asr_degraded" in flags
            or "insufficient_evidence" in flags
        )
        if mode == "priority" and not is_priority:
            continue
        long_video_id = feature["long_video_id"]
        severity = (
            int(verdict == "abstain") * 100
            + int("insufficient_evidence" in flags) * 50
            + int("asr_degraded" in flags) * 20
            + int(max(0.0, 4.0 - intelligibility) * 10)
            + int(max(0.0, 4.0 - boundary) * 5)
        )
        current = selected.get(long_video_id)
        if current is None or severity > current["severity"]:
            selected[long_video_id] = {
                "long_video_id": long_video_id,
                "channel_name": feature.get("channel_name", ""),
                "candidate_id": feature["candidate_id"],
                "severity": severity,
                "judge_verdict": verdict,
                "transcript_intelligibility": intelligibility,
                "boundary_observability": boundary,
                "failure_flags": "|".join(sorted(flags)),
            }
    return sorted(
        selected.values(),
        key=lambda row: (-row["severity"], row["long_video_id"]),
    )


def longform_urls(gold_rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        row["long_video_id"]: row["long_video_url"]
        for row in gold_rows
        if row.get("long_video_id") and row.get("long_video_url")
    }


def poll_until_terminal(
    client: VpickClient,
    project_id: str,
    asset_id: str,
    max_wait_min: int,
    interval_sec: int,
) -> dict[str, Any]:
    deadline = time.time() + max_wait_min * 60
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = client.get_asset(project_id, asset_id)
        if str(last.get("status") or "") in TERMINAL:
            return last
        time.sleep(interval_sec)
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-label", required=True)
    parser.add_argument("--project-name", default="judge_missing_20260724")
    parser.add_argument("--mode", choices=("priority", "all"), default="priority")
    parser.add_argument("--only-long-video-id", action="append", default=[])
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--min-credit-reserve", type=int, default=500)
    parser.add_argument("--max-wait-min", type=int, default=90)
    parser.add_argument("--poll-interval", type=int, default=20)
    parser.add_argument("--submit-delay", type=int, default=10)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()

    if not os.getenv("VPICK_EMAIL") or not (
        os.getenv("VPICK_PASSWORD") or os.getenv("VPICK_ACCESS_TOKEN")
    ):
        raise RuntimeError("Set Vpick credentials in the environment.")

    client = VpickClient()
    client.login_from_env()
    start_credits = credit_remaining(client)
    if start_credits <= args.min_credit_reserve:
        raise RuntimeError(
            f"Remaining credits {start_credits} do not exceed reserve {args.min_credit_reserve}."
        )

    account_dir = args.raw_dir / "accounts" / args.account_label
    account_dir.mkdir(parents=True, exist_ok=True)
    state_path = account_dir / "missing_analysis_state.json"
    state = load_json(state_path, {"project_id": "", "assets": {}})
    project_id = project_id_for_name(client, args.project_name)
    state["project_id"] = project_id
    save_json(state_path, state)

    targets = priority_longforms(
        read_csv(args.features),
        read_csv(args.judge),
        args.mode,
    )
    if args.only_long_video_id:
        allowed = set(args.only_long_video_id)
        targets = [row for row in targets if row["long_video_id"] in allowed]
    urls = longform_urls(read_csv(args.gold))
    assets_by_name = existing_assets_by_name(client, project_id)
    submitted = 0
    completed = 0
    pending: dict[str, dict[str, Any]] = {}

    # Submit first so Vpick can analyze independent longforms concurrently.
    for target in targets:
        long_video_id = target["long_video_id"]
        canonical_scene = args.raw_dir / f"{long_video_id}_scenes.json"
        if canonical_scene.exists():
            target["action"] = "already_has_scenes"
            continue
        asset_name = f"judge_{long_video_id}"
        state_row = state["assets"].get(long_video_id, {})
        asset = assets_by_name.get(asset_name)
        attempt = int(state_row.get("attempt") or 0)
        failed_before = str(state_row.get("status") or "") in {
            "FAILED",
            "PRETRANSCODING_FAILED",
            "MI_ANALYSIS_FAILED",
            "SUBMIT_FAILED",
            "POLL_TIMEOUT",
        }
        if args.retry_failed and failed_before:
            attempt += 1
            asset_name = f"judge_{long_video_id}_retry{attempt}"
            asset = assets_by_name.get(asset_name)
            state_row = {
                "attempt_history": [
                    *(state_row.get("attempt_history") or []),
                    {
                        "attempt": int(state_row.get("attempt") or 0),
                        "asset_id": state_row.get("asset_id", ""),
                        "asset_name": state_row.get("asset_name", ""),
                        "status": state_row.get("status", ""),
                        "error": state_row.get("error", ""),
                    },
                ]
            }
        asset_id = str(
            state_row.get("asset_id")
            or (asset or {}).get("asset_id")
            or ""
        )
        if not asset_id:
            if args.max_new and submitted >= args.max_new:
                target["action"] = "deferred_by_max_new"
                continue
            remaining = credit_remaining(client)
            if remaining <= args.min_credit_reserve:
                target["action"] = "deferred_credit_reserve"
                continue
            url = urls.get(long_video_id)
            if not url:
                target["action"] = "missing_longform_url"
                continue
            try:
                asset_id = client.create_asset_from_youtube(
                    project_id,
                    url,
                    asset_name,
                )
                submitted += 1
                target["action"] = "submitted"
                state["assets"][long_video_id] = {
                    **state_row,
                    **target,
                    "project_id": project_id,
                    "asset_id": asset_id,
                    "asset_name": asset_name,
                    "attempt": attempt,
                    "status": "SUBMITTED",
                }
                save_json(state_path, state)
                time.sleep(args.submit_delay)
            except Exception as exc:
                target["action"] = "submit_failed"
                target["error"] = str(exc)[:400]
                state["assets"][long_video_id] = {
                    **state_row,
                    **target,
                    "asset_name": asset_name,
                    "attempt": attempt,
                    "status": "SUBMIT_FAILED",
                }
                save_json(state_path, state)
                continue

        pending[long_video_id] = {
            **target,
            "project_id": project_id,
            "asset_id": asset_id,
            "asset_name": asset_name,
        }

    deadline = time.time() + args.max_wait_min * 60
    while pending and time.time() < deadline:
        for long_video_id in list(pending):
            target = pending[long_video_id]
            try:
                result = client.get_asset(project_id, target["asset_id"])
            except Exception as exc:
                state["assets"].setdefault(long_video_id, {}).update(
                    {"poll_error": str(exc)[:300]}
                )
                continue
            status = str(result.get("status") or "")
            state["assets"].setdefault(long_video_id, {}).update(
                {**target, "status": status}
            )
            if status == "READY":
                scenes = client.get_scenes(project_id, target["asset_id"])
                save_json(account_dir / f"{target['asset_id']}_scenes.json", scenes)
                save_json(
                    account_dir / f"{target['asset_id']}_asset_status.json",
                    result,
                )
                save_json(
                    args.raw_dir / f"{long_video_id}_asset_status.json",
                    result,
                )
                action = merge_canonical_scene(args.raw_dir, long_video_id, scenes)
                state["assets"][long_video_id]["scene_count"] = len(
                    scenes.get("data", []) if isinstance(scenes, dict) else scenes
                )
                state["assets"][long_video_id]["canonical_action"] = action
                completed += 1
                del pending[long_video_id]
            elif status in TERMINAL:
                del pending[long_video_id]
            save_json(state_path, state)
        if pending:
            time.sleep(args.poll_interval)

    for long_video_id, target in pending.items():
        state["assets"].setdefault(long_video_id, {}).update(
            {**target, "status": "POLL_TIMEOUT"}
        )
    save_json(state_path, state)

    end_credits = credit_remaining(client)
    summary = {
        "account_label": args.account_label,
        "project_id": project_id,
        "project_name": args.project_name,
        "mode": args.mode,
        "target_count": len(targets),
        "submitted_now": submitted,
        "ready_now": completed,
        "start_credits": start_credits,
        "end_credits": end_credits,
        "credits_consumed": start_credits - end_credits,
        "status_counts": {},
    }
    for row in state["assets"].values():
        status = str(row.get("status") or "UNKNOWN")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
    save_json(account_dir / "missing_analysis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
