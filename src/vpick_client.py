from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class VpickClient:
    def __init__(self, base_url: str = "https://api.yettey.ai/api/v1/vpick") -> None:
        self.base_url = base_url.rstrip("/")
        self.access_token: str | None = os.getenv("VPICK_ACCESS_TOKEN")

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                text = res.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: HTTP {exc.code} {detail}") from exc

    def login_from_env(self) -> None:
        if self.access_token:
            return
        email = os.getenv("VPICK_EMAIL")
        password = os.getenv("VPICK_PASSWORD")
        if not email or not password:
            raise RuntimeError("Set VPICK_EMAIL/VPICK_PASSWORD or VPICK_ACCESS_TOKEN.")
        data = self.request(
            "POST",
            "/auth/login",
            {
                "email": email,
                "password": password,
                "fingerprint": os.getenv("VPICK_FINGERPRINT", "vpick-codex-pipeline"),
            },
        )
        self.access_token = data["access_token"]

    def get_asset(self, project_id: str, asset_id: str) -> Any:
        return self.request("GET", f"/projects/{project_id}/assets/{asset_id}")

    def list_projects(self, offset: int = 0, limit: int = 100) -> Any:
        return self.request("GET", f"/projects?offset={offset}&limit={limit}")

    def get_project(self, project_id: str) -> Any:
        return self.request("GET", f"/projects/{project_id}")

    def list_assets(self, project_id: str, offset: int = 0, limit: int = 100) -> Any:
        return self.request(
            "GET",
            f"/projects/{project_id}/assets?offset={offset}&limit={limit}",
        )

    def get_scenes(self, project_id: str, asset_id: str) -> Any:
        return self.request("GET", f"/projects/{project_id}/assets/{asset_id}/scenes")

    def list_shortforms(self, project_id: str, limit: int = 50) -> Any:
        return self.request("GET", f"/projects/{project_id}/shortforms?offset=0&limit={limit}")

    def get_shortform(self, project_id: str, shortform_id: str) -> Any:
        return self.request("GET", f"/projects/{project_id}/shortforms/{shortform_id}")

    def shortform_status(self, project_id: str) -> Any:
        return self.request("GET", f"/projects/{project_id}/shortforms/status")

    def create_shortforms_for_asset(self, project_id: str, asset_id: str, payload: dict[str, Any]) -> Any:
        return self.request("POST", f"/projects/{project_id}/assets/{asset_id}/shortforms", payload)

    def create_project(self, project_name: str) -> str:
        data = self.request("POST", "/projects", {"project_name": project_name})
        return data["project_id"]

    def youtube_metadata(self, project_id: str, youtube_url: str) -> Any:
        return self.request("POST", f"/projects/{project_id}/youtube/metadata", {"youtube_url": youtube_url})

    def create_asset_from_youtube(self, project_id: str, youtube_url: str, asset_name: str) -> str:
        meta = self.youtube_metadata(project_id, youtube_url)
        payload = {
            "youtube_url": youtube_url,
            "asset_name": asset_name,
            "duration_ms": meta.get("duration_ms"),
            "resolution": meta.get("resolution"),
            "filesize_approx": meta.get("filesize_approx"),
        }
        data = self.request("POST", f"/projects/{project_id}/assets/from-youtube", payload)
        return data["asset_id"]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def poll_asset(client: VpickClient, project_id: str, asset_id: str, seconds: int = 0) -> dict[str, Any]:
    deadline = time.time() + seconds
    last: dict[str, Any] = {}
    while True:
        last = client.get_asset(project_id, asset_id)
        status = str(last.get("status", ""))
        if status in {"READY", "FAILED", "PRETRANSCODING_FAILED", "MI_ANALYSIS_FAILED", "INSUFFICIENT_CREDITS"}:
            return last
        if seconds <= 0 or time.time() >= deadline:
            return last
        time.sleep(10)
