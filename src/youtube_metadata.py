from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Iterable


def extract_youtube_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    if "/shorts/" in parsed.path:
        return parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
    return urllib.parse.parse_qs(parsed.query).get("v", [""])[0]


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_json(url: str) -> dict[str, Any]:
    return json.loads(fetch_text(url))


def official_youtube_stats(video_ids: Iterable[str], api_key: str | None = None) -> dict[str, dict[str, Any]]:
    key = api_key or os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY is not set")

    unique_ids = list(dict.fromkeys(video_id for video_id in video_ids if video_id))
    output: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(unique_ids), 50):
        params = urllib.parse.urlencode(
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(unique_ids[offset : offset + 50]),
                "key": key,
            }
        )
        response = fetch_json(f"https://www.googleapis.com/youtube/v3/videos?{params}")
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            video_id = str(item.get("id", ""))
            output[video_id] = {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "channel_name": snippet.get("channelTitle", ""),
                "channel_id": snippet.get("channelId", ""),
                "published_at": snippet.get("publishedAt", ""),
                "view_count": int(statistics["viewCount"]) if statistics.get("viewCount") else None,
                "like_count": int(statistics["likeCount"]) if statistics.get("likeCount") else None,
                "source": "youtube_data_api_v3",
            }
    return output


def oembed_metadata(url: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({"url": url, "format": "json"})
    response = fetch_json(f"https://www.youtube.com/oembed?{params}")
    return {
        "title": response.get("title", ""),
        "channel_name": response.get("author_name", ""),
        "channel_url": response.get("author_url", ""),
        "source": "youtube_oembed",
    }


def page_html_stats(video_id: str) -> dict[str, Any]:
    url = f"https://www.youtube.com/shorts/{video_id}"
    html = fetch_text(url)
    metadata: dict[str, Any] = {"video_id": video_id, "source": "youtube_public_page"}
    patterns = {
        "view_count": r'"viewCount"\s*:\s*"?(\d+)"?',
        "like_count": r'"likeCount"\s*:\s*"?(\d+)"?',
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, html)
        if match:
            metadata[field] = int(match.group(1))
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.DOTALL)
    if title_match:
        metadata["title"] = re.sub(r"\s+-\s+YouTube\s*$", "", title_match.group(1)).strip()
    try:
        metadata.update({key: value for key, value in oembed_metadata(url).items() if value})
        metadata["source"] = "youtube_public_page"
    except Exception:
        pass
    return metadata


def _extract_json_after_marker(html: str, marker: str) -> dict[str, Any]:
    marker_index = html.find(marker)
    if marker_index < 0:
        return {}
    object_index = html.find("{", marker_index + len(marker))
    if object_index < 0:
        return {}
    try:
        value, _end = json.JSONDecoder().raw_decode(html[object_index:])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def channel_shorts_ids(channel_shorts_url: str) -> list[str]:
    html = fetch_text(channel_shorts_url)
    initial_data = _extract_json_after_marker(html, "ytInitialData")
    if not initial_data:
        raise RuntimeError(f"Could not parse ytInitialData from {channel_shorts_url}")

    output: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            video_id = value.get("videoId")
            if isinstance(video_id, str) and len(video_id) == 11 and video_id not in output:
                output.append(video_id)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(initial_data)
    return output


def collect_stats(video_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    ids = list(dict.fromkeys(video_id for video_id in video_ids if video_id))
    if os.getenv("YOUTUBE_API_KEY"):
        return official_youtube_stats(ids)
    return {video_id: page_html_stats(video_id) for video_id in ids}
