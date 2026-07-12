# -*- coding: utf-8 -*-
"""
gold 데이터셋 롱폼 일괄 Vpick 업로드 + ID 자동 기입

Vpick 레포 루트에 놓고 실행:
    scripts/batch_upload_assets.py 로 저장

    export VPICK_EMAIL="..." VPICK_PASSWORD="..."   # 또는 VPICK_ACCESS_TOKEN
    python scripts/batch_upload_assets.py \
        --csv data/processed/gold_dataset_pairs_main.csv \
        --csv data/processed/gold_dataset_pairs_control.csv

동작:
  1. CSV에서 vpick_asset_id가 비어 있는 고유 롱폼 목록 추출
  2. 채널별 프로젝트 생성/재사용 (이름: gold_{채널명})
  3. 롱폼 전부 업로드 제출 (분석은 서버에서 병렬 진행)
  4. 완료 폴링 → READY면 scene 개수 확인 후 CSV에 project/asset ID 기입
  5. 진행 상태는 data/raw/vpick/upload_state.json에 저장 (중단 후 재실행 시 이어서)

실패 처리:
  - FAILED / INSUFFICIENT_CREDITS 등은 state에 기록하고 계속 진행
  - 401 등 인증 만료 시 중단됨 → 토큰 갱신 후 재실행하면 이어서 진행
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vpick_client import VpickClient  # noqa: E402

STATE_PATH = Path("data/raw/vpick/upload_state.json")
TERMINAL_FAIL = {"FAILED", "PRETRANSCODING_FAILED", "MI_ANALYSIS_FAILED",
                 "INSUFFICIENT_CREDITS"}


def count_scenes(resp) -> int:
    """scenes API 응답 구조가 무엇이든 리스트를 찾아 개수 반환. 못 찾으면 -1."""
    if isinstance(resp, list):
        return len(resp)
    if isinstance(resp, dict):
        for k in ("scenes", "data", "items", "results", "scene_list", "sceneList"):
            v = resp.get(k)
            if isinstance(v, list):
                return len(v)
            if isinstance(v, dict):  # 한 겹 더 감싼 경우
                for vv in v.values():
                    if isinstance(vv, list):
                        return len(vv)
        for v in resp.values():
            if isinstance(v, list):
                return len(v)
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, list):
                        return len(vv)
    return -1


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"projects": {}, "assets": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def read_rows(paths: list[str]) -> dict[str, list[dict]]:
    """CSV들을 읽어 {csv경로: rows} 반환."""
    all_rows = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as f:
            all_rows[p] = list(csv.DictReader(f))
    return all_rows


def unique_longforms(all_rows: dict) -> dict[str, dict]:
    """asset_id 미기입 고유 롱폼: {long_video_id: {url, channel, title힌트}}"""
    out = {}
    for rows in all_rows.values():
        for r in rows:
            lid = (r.get("long_video_id") or "").strip()
            if not lid or (r.get("vpick_asset_id") or "").strip():
                continue
            if lid not in out:
                out[lid] = {"url": r["long_video_url"].strip(),
                            "channel": (r.get("channel_name") or "misc").strip()}
    return out


def write_back(all_rows: dict, paths_written: set, assets: dict) -> None:
    """READY 된 롱폼의 ID를 모든 CSV 행에 기입 후 저장."""
    for path, rows in all_rows.items():
        changed = False
        for r in rows:
            lid = (r.get("long_video_id") or "").strip()
            a = assets.get(lid)
            if a and a.get("status") == "READY" and not (r.get("vpick_asset_id") or "").strip():
                r["vpick_project_id"] = a["project_id"]
                r["vpick_asset_id"] = a["asset_id"]
                changed = True
        if changed:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            paths_written.add(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="gold pair CSV (여러 번 지정 가능)")
    ap.add_argument("--poll-interval", type=int, default=30,
                    help="폴링 간격(초), 기본 30")
    ap.add_argument("--max-wait-min", type=int, default=120,
                    help="전체 폴링 최대 대기(분), 기본 120")
    ap.add_argument("--single-project", default=None,
                    help="채널별 프로젝트 대신 지정한 project_id 하나만 사용")
    ap.add_argument("--submit-delay", type=int, default=20,
                    help="제출 사이 대기(초), 기본 20 (다운로더 과부하 방지)")
    ap.add_argument("--max-new", type=int, default=8,
                    help="이번 실행에서 새로 제출할 최대 개수, 기본 8 (0=무제한)")
    args = ap.parse_args()

    client = VpickClient()
    client.login_from_env()
    print("로그인 OK")

    state = load_state()
    all_rows = read_rows(args.csv)
    todo = unique_longforms(all_rows)
    print(f"업로드 대상 고유 롱폼: {len(todo)}개")

    # 1) 업로드 제출
    submitted_now = 0
    consecutive_fail = 0
    for lid, info in todo.items():
        st = state["assets"].get(lid, {})
        if st.get("asset_id"):
            continue  # 이미 제출됨 (재실행 이어서)
        if args.max_new and submitted_now >= args.max_new:
            print(f"[제출 상한] 이번 실행 제출 {args.max_new}개 도달, 나머지는 다음 실행에서")
            break
        if consecutive_fail >= 3:
            print("[제출 중단] 연속 실패 3회 - 서버 다운로더 불안정. "
                  "제출은 멈추고 폴링으로 넘어감. 잠시 후 재실행할 것.", file=sys.stderr)
            break
        ch = info["channel"]
        if args.single_project:
            pid = args.single_project
        else:
            pid = state["projects"].get(ch)
            if not pid:
                pid = client.create_project(f"gold_{ch}")
                state["projects"][ch] = pid
                save_state(state)
                print(f"[프로젝트 생성] gold_{ch} -> {pid}")
        try:
            aid = client.create_asset_from_youtube(pid, info["url"], f"{ch}_{lid}")
            state["assets"][lid] = {"project_id": pid, "asset_id": aid,
                                    "channel": ch, "status": "SUBMITTED"}
            save_state(state)
            submitted_now += 1
            consecutive_fail = 0
            print(f"[제출] {ch} {lid} -> asset {aid}")
        except Exception as ex:
            consecutive_fail += 1
            state["assets"][lid] = {"project_id": pid, "channel": ch,
                                    "status": "SUBMIT_FAILED", "error": str(ex)[:300]}
            save_state(state)
            print(f"[제출 실패] {ch} {lid}: {ex}", file=sys.stderr)
        time.sleep(args.submit_delay)

    # 2) 완료 폴링
    deadline = time.time() + args.max_wait_min * 60
    paths_written: set = set()
    while time.time() < deadline:
        pending = [lid for lid, a in state["assets"].items()
                   if a.get("asset_id") and a.get("status") not in ({"READY"} | TERMINAL_FAIL)]
        if not pending:
            break
        for lid in pending:
            a = state["assets"][lid]
            try:
                res = client.get_asset(a["project_id"], a["asset_id"])
                status = str(res.get("status", ""))
                if status and status != a.get("status"):
                    a["status"] = status
                    save_state(state)
                    print(f"[상태] {a['channel']} {lid}: {status}")
                if status == "READY":
                    try:
                        scenes = client.get_scenes(a["project_id"], a["asset_id"])
                        n = count_scenes(scenes)
                        a["scene_count"] = n
                        save_state(state)
                        if n >= 0:
                            print(f"    scenes: {n}개")
                            if n < 5:
                                print(f"    [주의] scene {n}개 - 분석 품질 확인 필요",
                                      file=sys.stderr)
                        else:
                            keys = list(scenes.keys()) if isinstance(scenes, dict) else type(scenes).__name__
                            print(f"    scenes 응답 구조 미확인 (keys={keys}) - UI에서 확인")
                    except Exception as ex:
                        print(f"    scenes 확인 실패: {ex}", file=sys.stderr)
            except Exception as ex:
                print(f"[폴링 오류] {lid}: {ex}", file=sys.stderr)
        write_back(all_rows, paths_written, state["assets"])
        remaining = [lid for lid, a in state["assets"].items()
                     if a.get("status") not in ({"READY"} | TERMINAL_FAIL) and a.get("asset_id")]
        if not remaining:
            break
        print(f"... 대기 중 {len(remaining)}개, {args.poll_interval}초 후 재확인")
        time.sleep(args.poll_interval)

    # 3) 최종 기입 및 요약
    write_back(all_rows, paths_written, state["assets"])
    ready = sum(1 for a in state["assets"].values() if a.get("status") == "READY")
    failed = {lid: a for lid, a in state["assets"].items()
              if a.get("status") in TERMINAL_FAIL or a.get("status") == "SUBMIT_FAILED"}
    print(f"\n=== 요약 ===\nREADY: {ready} / {len(state['assets'])}")
    if failed:
        print("실패 목록:")
        for lid, a in failed.items():
            print(f"  {a.get('channel')} {lid}: {a.get('status')} {a.get('error','')[:120]}")
    if paths_written:
        print("ID 기입된 CSV:", ", ".join(sorted(paths_written)))
    print(f"상태 파일: {STATE_PATH}")


if __name__ == "__main__":
    main()
