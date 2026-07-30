from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_SCENES = Path(__file__).resolve().parent / "samples" / "BETA_DEMO01_scenes.json"
RAW_VPICK_DIR = ROOT / "data" / "raw" / "vpick"
SRC_DIR = ROOT / "src"


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file(ROOT / ".env")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_longform_slate import (  # noqa: E402
    select_adaptive_coverage,
)
from segments import (  # noqa: E402
    build_adjacent_candidates,
    extract_scene_list,
    seconds_to_clock,
)
from vpick_client import VpickClient  # noqa: E402
from youtube_metadata import extract_youtube_id  # noqa: E402


JUDGE_WEIGHTS = {
    "change_or_surprise": 0.40,
    "title_packaging": 0.15,
    "thumbnail_packaging": 0.45,
}
SELECTION_WEIGHTS = {
    "opening_clarity_pull_0_4": 0.15,
    "event_reaction_change_0_4": 0.25,
    "progression_payoff_0_4": 0.20,
    "self_contained_0_4": 0.15,
    "boundary_integrity_0_4": 0.15,
    "titleability_0_4": 0.10,
}
CHANGE_TERMS = (
    "갑자기",
    "결국",
    "그런데",
    "근데",
    "반전",
    "사실",
    "성공",
    "실패",
    "왜",
    "처음",
    "최초",
    "문제",
    "비밀",
    "놀라",
    "충격",
    "웃",
    "진짜",
    "알고 보니",
    "결과",
    "정답",
    "바뀌",
)
PAYOFF_TERMS = (
    "결국",
    "그래서",
    "결론",
    "정답",
    "성공",
    "실패",
    "알고 보니",
    "밝혀",
    "해결",
    "결과",
)
FILLER_PREFIXES = (
    "안녕하세요",
    "자 그러면",
    "오늘은",
    "네 여러분",
    "준비",
    "이동",
)


class AppError(RuntimeError):
    def __init__(self, message: str, status: int = 400, code: str = "bad_request") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha1(raw).hexdigest()[:12]}"


def youtube_thumbnail(video_url: str) -> str:
    video_id = extract_youtube_id(video_url)
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else ""


def parse_json3_transcript(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    previous_text = ""
    for event in payload.get("events", []):
        segments = event.get("segs") or []
        text = "".join(str(segment.get("utf8", "")) for segment in segments)
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text == previous_text:
            continue
        start = number(event.get("tStartMs")) / 1000.0
        duration = number(event.get("dDurationMs")) / 1000.0
        end = start + max(duration, 0.1)
        lines.append(f"[{seconds_to_clock(start)}-{seconds_to_clock(end)}] {text}")
        previous_text = text
    return "\n".join(lines)


def preferred_caption(
    tracks: dict[str, list[dict[str, Any]]],
) -> tuple[str, dict[str, Any]] | None:
    if not tracks:
        return None
    language_order = ("ko-orig", "ko", "ko-KR", "en-orig", "en")
    languages = list(tracks)
    ordered_languages = [
        *[language for language in language_order if language in tracks],
        *[language for language in languages if language.startswith("ko") and language not in language_order],
        *[language for language in languages if language.startswith("en") and language not in language_order],
    ]
    for language in ordered_languages:
        entries = tracks.get(language) or []
        for extension in ("json3", "vtt", "srt"):
            entry = next(
                (
                    item
                    for item in entries
                    if item.get("ext") == extension and item.get("url")
                ),
                None,
            )
            if entry:
                return language, entry
    return None


def fetch_caption(entry: dict[str, Any]) -> str:
    request = urllib.request.Request(
        str(entry["url"]),
        headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "ko-KR,ko;q=0.9"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    extension = str(entry.get("ext", ""))
    if extension == "json3":
        return parse_json3_transcript(json.loads(raw.decode("utf-8", errors="replace")))
    text = raw.decode("utf-8", errors="replace")
    output: list[str] = []
    for line in text.splitlines():
        stripped = re.sub(r"<[^>]+>", "", line).strip()
        if (
            not stripped
            or stripped.startswith(("WEBVTT", "Kind:", "Language:", "NOTE"))
            or "-->" in stripped
            or stripped.isdigit()
        ):
            continue
        if not output or stripped != output[-1]:
            output.append(stripped)
    return "\n".join(output)


def multipart_body(
    fields: dict[str, str],
    *,
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    boundary = f"vpick-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), boundary


def transcribe_youtube_audio_openai(video_url: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        import yt_dlp
    except ImportError:
        return ""
    with tempfile.TemporaryDirectory(prefix="vpick-short-") as temp_dir:
        output_template = str(Path(temp_dir) / "%(id)s.%(ext)s")
        options = {
            "format": "bestaudio[filesize<24M]/bestaudio",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 30,
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.extract_info(video_url, download=True)
        except Exception:
            return ""
        files = [
            path
            for path in Path(temp_dir).iterdir()
            if path.is_file() and path.suffix not in {".part", ".ytdl"}
        ]
        if not files:
            return ""
        audio_path = max(files, key=lambda path: path.stat().st_size)
        if audio_path.stat().st_size > 25 * 1024 * 1024:
            return ""
        body, boundary = multipart_body(
            {
                "model": os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
                "response_format": "json",
                "language": "ko",
            },
            file_field="file",
            file_path=audio_path,
        )
        request = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return ""
    segments = result.get("segments") or []
    if segments:
        return "\n".join(
            (
                f"[{seconds_to_clock(number(segment.get('start')))}-"
                f"{seconds_to_clock(number(segment.get('end')))}] "
                f"{str(segment.get('text', '')).strip()}"
            )
            for segment in segments
            if str(segment.get("text", "")).strip()
        )
    return str(result.get("text", "")).strip()


def gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def gemini_response_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: gemini_response_schema(item)
            for key, item in value.items()
            if key != "additionalProperties"
        }
    if isinstance(value, list):
        return [gemini_response_schema(item) for item in value]
    return value


def extract_gemini_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            text = str(part.get("text", "")).strip()
            if text:
                chunks.append(text)
    if not chunks:
        raise ValueError("Gemini 응답에서 텍스트를 찾지 못했습니다.")
    return "\n".join(chunks)


def gemini_generate_content(
    payload: dict[str, Any],
    *,
    model: str,
    timeout: int = 180,
) -> dict[str, Any]:
    api_key = gemini_api_key()
    if not api_key:
        raise AppError(
            "GEMINI_API_KEY가 설정되지 않았습니다.",
            status=503,
            code="missing_gemini_key",
        )
    encoded_model = urllib.parse.quote(model, safe="-_.")
    request = urllib.request.Request(
        (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{encoded_model}:generateContent"
        ),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AppError(
            f"Gemini 호출 실패: HTTP {exc.code} {detail[:400]}",
            status=502,
            code="gemini_error",
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AppError(
            f"Gemini 연결 실패: {exc}",
            status=502,
            code="gemini_error",
        ) from exc


def transcribe_youtube_audio_gemini(video_url: str) -> str:
    model = os.getenv(
        "GEMINI_TRANSCRIBE_MODEL",
        os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.6-flash"),
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["segments"],
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start_sec", "end_sec", "text"],
                    "properties": {
                        "start_sec": {"type": "number", "minimum": 0},
                        "end_sec": {"type": "number", "minimum": 0},
                        "text": {"type": "string"},
                    },
                },
            }
        },
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "fileUri": video_url,
                            "mimeType": "video/mp4",
                        }
                    },
                    {
                        "text": (
                            "이 공개 YouTube Shorts의 음성을 한국어로 정확히 전사하세요. "
                            "발화가 바뀌거나 의미 단위가 끝날 때마다 구간을 나누고, "
                            "각 구간의 시작·종료 초와 발화 원문만 반환하세요. "
                            "화면 자막을 임의로 요약하거나 내용을 만들어내지 마세요."
                        )
                    },
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": gemini_response_schema(schema),
            "temperature": 0,
        },
    }
    try:
        response = gemini_generate_content(payload, model=model, timeout=240)
        result = json.loads(extract_gemini_text(response))
    except (AppError, ValueError, json.JSONDecodeError):
        return ""
    return "\n".join(
        (
            f"[{seconds_to_clock(number(segment.get('start_sec')))}-"
            f"{seconds_to_clock(number(segment.get('end_sec')))}] "
            f"{str(segment.get('text', '')).strip()}"
        )
        for segment in result.get("segments", [])
        if str(segment.get("text", "")).strip()
    )


def transcribe_youtube_audio(video_url: str) -> tuple[str, str]:
    if os.getenv("OPENAI_API_KEY"):
        transcript = transcribe_youtube_audio_openai(video_url)
        if transcript:
            return transcript, "openai_audio_asr"
    if gemini_api_key():
        transcript = transcribe_youtube_audio_gemini(video_url)
        if transcript:
            return transcript, "gemini_youtube_asr"
    return "", ""


def clean_short_description(value: str) -> str:
    text = re.sub(r"https?://\S+", "", value)
    text = re.sub(r"(?<!\w)#[^\s#]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:700]


def collect_youtube_short(video_url: str) -> dict[str, Any]:
    video_id = extract_youtube_id(video_url)
    if not video_id:
        raise AppError("올바른 YouTube 또는 Shorts 주소를 입력해 주세요.")
    try:
        import yt_dlp
    except ImportError as exc:
        raise AppError(
            "자동 수집에 필요한 yt-dlp가 설치되지 않았습니다.",
            status=503,
            code="yt_dlp_missing",
        ) from exc
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(video_url, download=False)
    except Exception as exc:
        raise AppError(
            f"YouTube 정보를 가져오지 못했습니다: {str(exc)[:300]}",
            status=502,
            code="youtube_metadata_error",
        ) from exc

    transcript = ""
    transcript_source = ""
    caption_language = ""
    for source_name, tracks in (
        ("youtube_subtitles", info.get("subtitles") or {}),
        ("youtube_auto_captions", info.get("automatic_captions") or {}),
    ):
        selected = preferred_caption(tracks)
        if not selected:
            continue
        language, entry = selected
        try:
            transcript = fetch_caption(entry)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            transcript = ""
        if transcript:
            transcript_source = source_name
            caption_language = language
            break

    if not transcript:
        transcript, transcript_source = transcribe_youtube_audio(video_url)
        if transcript:
            caption_language = "auto"
    if not transcript:
        raise AppError(
            "이 Shorts에서 자막을 찾지 못했고 자동 ASR도 사용할 수 없습니다. 고급 수정에서 자막을 직접 넣어 주세요.",
            status=422,
            code="transcript_unavailable",
        )

    duration = number(info.get("duration"))
    return {
        "video_id": str(info.get("id") or video_id),
        "title": str(info.get("title") or "").strip(),
        "description": clean_short_description(str(info.get("description") or "")),
        "transcript": transcript[:12000],
        "thumbnail_url": str(info.get("thumbnail") or youtube_thumbnail(video_url)),
        "duration_sec": duration,
        "start_time": "0:00",
        "end_time": seconds_to_clock(duration) if duration else "",
        "transcript_source": transcript_source,
        "caption_language": caption_language,
        "metadata_source": "yt_dlp",
    }


def bool_value(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def autofill_short_candidate(
    payload: dict[str, Any],
    collector: Any = collect_youtube_short,
) -> tuple[dict[str, Any], dict[str, Any]]:
    enriched = dict(payload)
    video_url = str(enriched.get("video_url", "")).strip()
    if not video_url:
        raise AppError("Shorts 주소를 입력해 주세요.")
    required_missing = not str(enriched.get("title", "")).strip() or not str(
        enriched.get("transcript", "")
    ).strip()
    collected: dict[str, Any] = {}
    if bool_value(enriched.get("auto_enrich"), True) and required_missing:
        collected = collector(video_url)
        for field in (
            "title",
            "description",
            "transcript",
            "thumbnail_url",
            "start_time",
            "end_time",
        ):
            if not str(enriched.get(field, "")).strip() and collected.get(field):
                enriched[field] = collected[field]
    if not str(enriched.get("transcript", "")).strip():
        raise AppError(
            "평가 근거가 될 자막을 확보하지 못했습니다.",
            status=422,
            code="transcript_required",
        )
    if not str(enriched.get("title", "")).strip():
        enriched["title"] = "제목 없음"
    summary = {
        "metadata_source": collected.get("metadata_source") or "manual_override",
        "transcript_source": collected.get("transcript_source") or "manual_input",
        "caption_language": collected.get("caption_language") or "",
        "duration_sec": collected.get("duration_sec")
        or max(
            0.0,
            number(enriched.get("end_time")) - number(enriched.get("start_time")),
        ),
        "auto_filled": bool(collected),
        "title": enriched["title"],
    }
    return enriched, summary


def score_package_formula(scores: dict[str, int]) -> float:
    weighted = sum(
        JUDGE_WEIGHTS[name] * number(scores.get(f"{name}_0_4"))
        for name in JUDGE_WEIGHTS
    )
    return round(clamp(weighted / 4.0) * 100.0, 1)


def extract_response_text(response: dict[str, Any]) -> str:
    for output in response.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                return str(content.get("text", ""))
    raise ValueError("OpenAI 응답에서 JSON 텍스트를 찾지 못했습니다.")


def call_openai_json(
    *,
    instructions: str,
    text_input: str,
    schema_name: str,
    schema: dict[str, Any],
    image_url: str = "",
) -> tuple[dict[str, Any], str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AppError(
            "OPENAI_API_KEY가 없어 오프라인 프리뷰로 실행합니다.",
            status=503,
            code="missing_openai_key",
        )
    model = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-sol")
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text_input}]
    if image_url:
        content.append({"type": "input_image", "image_url": image_url})
    payload = {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AppError(
            f"OpenAI 호출 실패: HTTP {exc.code} {detail[:400]}",
            status=502,
            code="openai_error",
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AppError(
            f"OpenAI 연결 실패: {exc}",
            status=502,
            code="openai_error",
        ) from exc
    return json.loads(extract_response_text(body)), model


def gemini_inline_image(image_url: str) -> dict[str, Any] | None:
    if not image_url:
        return None
    parsed = urllib.parse.urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        return None
    request = urllib.request.Request(
        image_url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "image/*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            media_type = str(response.headers.get_content_type() or "")
            data = response.read(6 * 1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    if len(data) > 6 * 1024 * 1024 or not media_type.startswith("image/"):
        return None
    return {
        "inlineData": {
            "mimeType": media_type,
            "data": base64.b64encode(data).decode("ascii"),
        }
    }


def call_gemini_json(
    *,
    instructions: str,
    text_input: str,
    schema_name: str,
    schema: dict[str, Any],
    image_url: str = "",
) -> tuple[dict[str, Any], str]:
    del schema_name
    model = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.6-flash")
    parts: list[dict[str, Any]] = [{"text": text_input}]
    image_part = gemini_inline_image(image_url)
    if image_part:
        parts.append(image_part)
    payload = {
        "systemInstruction": {"parts": [{"text": instructions}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": gemini_response_schema(schema),
            "temperature": 0,
        },
    }
    response = gemini_generate_content(payload, model=model)
    try:
        result = json.loads(extract_gemini_text(response))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AppError(
            f"Gemini JSON 응답을 읽지 못했습니다: {exc}",
            status=502,
            code="gemini_invalid_json",
        ) from exc
    return result, f"Gemini · {model}"


def call_judge_json(
    *,
    instructions: str,
    text_input: str,
    schema_name: str,
    schema: dict[str, Any],
    image_url: str = "",
) -> tuple[dict[str, Any], str]:
    errors: list[str] = []
    kwargs = {
        "instructions": instructions,
        "text_input": text_input,
        "schema_name": schema_name,
        "schema": schema,
        "image_url": image_url,
    }
    if os.getenv("OPENAI_API_KEY"):
        try:
            result, model = call_openai_json(**kwargs)
            return result, f"OpenAI · {model}"
        except AppError as exc:
            errors.append(str(exc))
    if gemini_api_key():
        try:
            return call_gemini_json(**kwargs)
        except AppError as exc:
            errors.append(str(exc))
    if errors:
        raise AppError(
            " / ".join(errors),
            status=502,
            code="llm_provider_error",
        )
    raise AppError(
        "OPENAI_API_KEY 또는 GEMINI_API_KEY가 없어 오프라인 프리뷰로 실행합니다.",
        status=503,
        code="missing_llm_key",
    )


def package_judge_schema() -> dict[str, Any]:
    score = {"type": "integer", "minimum": 0, "maximum": 4}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "evidence_first",
            "change_or_surprise_0_4",
            "title_packaging_0_4",
            "thumbnail_packaging_0_4",
            "confidence_1_5",
            "strengths",
            "risks",
        ],
        "properties": {
            "evidence_first": {"type": "string"},
            "change_or_surprise_0_4": score,
            "title_packaging_0_4": score,
            "thumbnail_packaging_0_4": score,
            "confidence_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            },
        },
    }


def package_judge_instructions() -> str:
    return """
당신은 공개 전 숏폼 후보의 성공 잠재력을 블라인드 평가하는 Pointwise Judge입니다.
채널명, 조회수, 좋아요 수, 성과 등급은 추정하거나 사용하지 마십시오.
후보의 설명·자막·제목과 제공된 썸네일만 근거로 아래 세 축을 각각 0~4 정수로 평가하십시오.

1. change_or_surprise: 후보 안에서 사건, 반응, 변화, 반전 또는 기억에 남는 결론이 실제로 발생하는가.
2. title_packaging: 제목이 구체적이고 즉시 이해되며, 과장 없이 후보의 핵심 변화와 궁금증을 전달하는가.
3. thumbnail_packaging: 썸네일만 보아도 인물·행동·대상이 명확하고, 작은 화면에서도 초점과 감정이 읽히는가.

후보를 다른 후보와 비교하지 말고 독립 채점하십시오. 불확실성은 confidence로만 표현하십시오.
먼저 관찰한 근거를 쓰고 그 근거에 묶인 점수를 출력하십시오.
""".strip()


def offline_package_judge(payload: dict[str, Any]) -> dict[str, Any]:
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    transcript = str(payload.get("transcript", "")).strip()
    thumbnail_url = str(payload.get("thumbnail_url", "")).strip()
    text = f"{description}\n{transcript}".strip()
    term_hits = sum(1 for term in CHANGE_TERMS if term in text)
    payoff_hits = sum(1 for term in PAYOFF_TERMS if term in text)
    dialogue_bonus = min(1.0, (text.count("?") + text.count("!") + text.count(":")) / 5.0)
    change_raw = clamp(0.18 + term_hits * 0.12 + payoff_hits * 0.08 + dialogue_bonus * 0.18)

    title_len = len(title.replace(" ", ""))
    title_specific = any(term in title for term in CHANGE_TERMS) or bool(re.search(r"\d", title))
    title_raw = clamp(
        (0.25 if 8 <= title_len <= 32 else 0.12)
        + (0.35 if title_specific else 0.12)
        + (0.20 if title and title[-1] not in ".," else 0.08)
    )
    scores = {
        "change_or_surprise_0_4": int(round(change_raw * 4)),
        "title_packaging_0_4": int(round(title_raw * 4)),
        "thumbnail_packaging_0_4": 2 if thumbnail_url else 1,
    }
    return {
        "evidence_first": (
            f"텍스트에서 변화·반응 신호 {term_hits}개와 결말 신호 {payoff_hits}개를 확인했습니다. "
            "이미지는 오프라인 프리뷰에서 직접 판독하지 않습니다."
        ),
        **scores,
        "confidence_1_5": 2,
        "strengths": [
            item
            for item in (
                "후보 안에 변화 또는 반응 단서가 있음" if term_hits else "",
                "제목이 후보의 구체적 상황을 드러냄" if title_specific else "",
            )
            if item
        ],
        "risks": [
            item
            for item in (
                "썸네일은 중립값으로 처리됨",
                "결말 회수 단서가 약함" if payoff_hits == 0 else "",
                "자막 근거가 짧음" if len(text) < 120 else "",
            )
            if item
        ],
    }


def run_package_judge(payload: dict[str, Any]) -> dict[str, Any]:
    payload, input_summary = autofill_short_candidate(payload)
    video_url = str(payload.get("video_url", "")).strip()
    thumbnail_url = str(payload.get("thumbnail_url", "")).strip() or youtube_thumbnail(video_url)
    mode = str(payload.get("mode", "auto")).strip()
    model = "offline-preview-v1"
    warning = ""
    text_input = json.dumps(
        {
            "candidate_id": payload.get("candidate_id") or stable_id("WEB", video_url),
            "source_interval": {
                "start": payload.get("start_time", ""),
                "end": payload.get("end_time", ""),
            },
            "title": str(payload.get("title", "")).strip(),
            "candidate_description": str(payload.get("description", "")).strip(),
            "candidate_transcript": str(payload.get("transcript", "")).strip(),
            "thumbnail_available": bool(thumbnail_url),
        },
        ensure_ascii=False,
    )
    if mode != "preview":
        try:
            result, model = call_judge_json(
                instructions=package_judge_instructions(),
                text_input=text_input,
                image_url=thumbnail_url,
                schema_name="vpick_package_judge",
                schema=package_judge_schema(),
            )
        except AppError as exc:
            if mode == "live":
                raise
            result = offline_package_judge({**payload, "thumbnail_url": thumbnail_url})
            warning = str(exc)
    else:
        result = offline_package_judge({**payload, "thumbnail_url": thumbnail_url})
    score = score_package_formula(result)
    return {
        "candidate_id": payload.get("candidate_id") or stable_id("WEB", video_url, payload.get("start_time")),
        "mode": "live_llm" if model != "offline-preview-v1" else "offline_preview",
        "model": model,
        "editorial_success_score": score,
        "formula": "40% 변화·반전 + 15% 제목 패키징 + 45% 썸네일 패키징",
        "thumbnail_url": thumbnail_url,
        "input_summary": input_summary,
        "warning": warning,
        **result,
    }


def scene_text(scenes: list[dict[str, Any]], start: float, end: float) -> tuple[str, str, list[str]]:
    descriptions: list[str] = []
    transcripts: list[str] = []
    scene_ids: list[str] = []
    for scene in scenes:
        if max(start, number(scene["start_sec"])) >= min(end, number(scene["end_sec"])):
            continue
        scene_ids.append(str(scene["scene_id"]))
        description = str(scene.get("description", "")).strip()
        transcript = str(scene.get("transcript", "")).strip()
        if description and description not in descriptions:
            descriptions.append(description)
        if transcript:
            transcripts.append(transcript)
    return " ".join(descriptions)[:1400], "\n".join(transcripts)[:4200], scene_ids


def structural_features(
    *,
    description: str,
    transcript: str,
    duration: float,
    source_kind: str,
) -> dict[str, float]:
    text = f"{description} {transcript}".strip()
    term_hits = sum(1 for term in CHANGE_TERMS if term in text)
    payoff_hits = sum(1 for term in PAYOFF_TERMS if term in text)
    text_per_sec = len(text.replace(" ", "")) / max(duration, 1.0)
    speech_density = clamp(text_per_sec / 5.0)
    content_volume = clamp(len(text) / 700.0)
    signal = clamp(0.08 + term_hits * 0.105 + payoff_hits * 0.08)
    titleability = clamp(
        0.18
        + signal * 0.52
        + min(0.22, (text.count("?") + text.count("!")) * 0.05)
    )
    duration_fit = clamp(1.0 - abs(duration - 45.0) / 45.0)
    speech_boundary = 1.0 if source_kind in {"scene", "scene_bridge"} else 0.78
    filler_penalty = 0.35 if any(text.startswith(prefix) for prefix in FILLER_PREFIXES) else 0.0
    structural_score = clamp(
        0.28 * signal
        + 0.18 * titleability
        + 0.18 * speech_density
        + 0.16 * content_volume
        + 0.12 * duration_fit
        + 0.08 * speech_boundary
        - 0.10 * filler_penalty
    )
    return {
        "signal": signal,
        "titleability": titleability,
        "speech_density": speech_density,
        "content_volume": content_volume,
        "duration": duration_fit,
        "speech_boundary": speech_boundary,
        "filler_penalty": filler_penalty,
        "structural_score": structural_score,
    }


def add_candidate(
    output: dict[tuple[float, float], dict[str, Any]],
    scenes: list[dict[str, Any]],
    *,
    start: float,
    end: float,
    source_kind: str,
) -> None:
    start = round(max(0.0, start), 3)
    end = round(max(start, end), 3)
    duration = end - start
    if duration < 15.0 or duration > 90.0:
        return
    description, transcript, scene_ids = scene_text(scenes, start, end)
    if not description and not transcript:
        return
    features = structural_features(
        description=description,
        transcript=transcript,
        duration=duration,
        source_kind=source_kind,
    )
    candidate = {
        "candidate_id": stable_id("C", start, end),
        "pred_start_sec": start,
        "pred_end_sec": end,
        "duration_sec": round(duration, 3),
        "selected_scene_ids": "|".join(scene_ids),
        "description": description,
        "transcript": transcript,
        "source_kind": source_kind,
        "rerank_score": round(features["structural_score"], 6),
        "rerank_components": json.dumps(features, ensure_ascii=False),
        "notes": f"window_kind={source_kind}",
        **features,
    }
    key = (start, end)
    previous = output.get(key)
    if previous is None or number(candidate["rerank_score"]) > number(previous["rerank_score"]):
        output[key] = candidate


def interval_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    overlap = max(
        0.0,
        min(number(left["pred_end_sec"]), number(right["pred_end_sec"]))
        - max(number(left["pred_start_sec"]), number(right["pred_start_sec"])),
    )
    shortest = min(number(left["duration_sec"]), number(right["duration_sec"]))
    return overlap / shortest if shortest > 0 else 0.0


def compress_candidates(candidates: list[dict[str, Any]], threshold: float = 0.85) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: number(item["rerank_score"]), reverse=True):
        if any(interval_overlap_ratio(candidate, previous) >= threshold for previous in kept):
            continue
        kept.append(candidate)
    return kept


def generate_candidate_pool(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not scenes:
        return []
    scenes = sorted(scenes, key=lambda item: number(item["start_sec"]))
    video_end = max(number(scene["end_sec"]) for scene in scenes)
    output: dict[tuple[float, float], dict[str, Any]] = {}
    for row in build_adjacent_candidates(
        scenes,
        min_duration_sec=15.0,
        max_duration_sec=90.0,
        max_window_scenes=4,
    ):
        source_kind = "scene" if len(row["scene_ids"]) == 1 else "scene_bridge"
        add_candidate(
            output,
            scenes,
            start=number(row["start_sec"]),
            end=number(row["end_sec"]),
            source_kind=source_kind,
        )
    for scene in scenes:
        center = (number(scene["start_sec"]) + number(scene["end_sec"])) / 2.0
        for duration in (30.0, 45.0, 60.0, 75.0):
            if video_end < 15.0:
                continue
            actual_duration = min(duration, video_end)
            start = clamp(center - actual_duration / 2.0, 0.0, max(0.0, video_end - actual_duration))
            add_candidate(
                output,
                scenes,
                start=start,
                end=min(video_end, start + actual_duration),
                source_kind=f"window_{int(duration)}",
            )
    return compress_candidates(list(output.values()))


def selection_schema() -> dict[str, Any]:
    score = {"type": "integer", "minimum": 0, "maximum": 4}
    properties = {
        "candidate_id": {"type": "string"},
        "evidence_first": {"type": "string"},
        "generated_title": {"type": "string"},
        "confidence_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
    }
    properties.update({field: score for field in SELECTION_WEIGHTS})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["content_mode", "candidate_scores"],
        "properties": {
            "content_mode": {
                "type": "string",
                "enum": [
                    "entertainment_vlog",
                    "interview_conversation",
                    "lecture_information",
                    "mixed_other",
                ],
            },
            "candidate_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(properties),
                    "properties": properties,
                },
            },
        },
    }


def selection_instructions() -> str:
    return """
당신은 롱폼에서 만든 숏폼 후보들을 텍스트 근거로 채점하는 Listwise Judge입니다.
채널명, 조회수, 실제 Shorts 구간은 제공되지 않으며 추정해서도 안 됩니다.
모든 후보를 읽어 중복 사건을 파악하되, 각 점수는 그 후보 안의 근거로만 0~4 정수 채점하십시오.

- opening_clarity_pull: 초반부터 상황·질문·주장이 이해되고 다음 내용을 궁금하게 하는가.
- event_reaction_change: 사건, 반응, 정보 이득 또는 변화가 실제로 발생하는가.
- progression_payoff: 질문-답, 행동-반응, 주장-결론이 후보 안에서 전개되고 회수되는가.
- self_contained: 원본을 보지 않은 시청자도 인물·주제·사건을 이해할 수 있는가.
- boundary_integrity: 문장 중간에 시작하거나 핵심 반응·답·결론 전에 끝나지 않는가.
- titleability: 과장 없이 한 문장 제목으로 요약할 구체적 상황이 있는가.

보이지 않는 표정, 화면 전환, 음성 톤, 자막 연출은 추정하지 마십시오.
각 후보에 근거 문장과 과장 없는 짧은 제목을 함께 출력하십시오.
""".strip()


def offline_selection_judge(candidate: dict[str, Any]) -> dict[str, Any]:
    description = str(candidate.get("description", ""))
    transcript = str(candidate.get("transcript", ""))
    text = f"{description} {transcript}".strip()
    first = text[:180]
    signal = number(candidate.get("signal"))
    titleability = number(candidate.get("titleability"))
    boundary = number(candidate.get("speech_boundary"))
    payoff_hits = sum(1 for term in PAYOFF_TERMS if term in text)
    event = int(round(clamp(signal + 0.10) * 4))
    progression = int(round(clamp(0.24 + payoff_hits * 0.18 + signal * 0.35) * 4))
    self_contained = int(round(clamp(0.25 + len(description) / 500.0 + len(text) / 1800.0) * 4))
    opening = int(round(clamp(0.22 + signal * 0.45 + (0.15 if "?" in first else 0.0)) * 4))
    boundary_score = int(round(boundary * 4))
    title_score = int(round(titleability * 4))
    cleaned = re.sub(r"\[[^\]]+\]\s*S[^:]*:\s*", "", description or transcript)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return {
        "candidate_id": candidate["candidate_id"],
        "evidence_first": (
            f"변화 신호 {sum(1 for term in CHANGE_TERMS if term in text)}개, "
            f"회수 신호 {payoff_hits}개가 텍스트에서 확인됩니다."
        ),
        "opening_clarity_pull_0_4": opening,
        "event_reaction_change_0_4": event,
        "progression_payoff_0_4": progression,
        "self_contained_0_4": self_contained,
        "boundary_integrity_0_4": boundary_score,
        "titleability_0_4": title_score,
        "generated_title": cleaned[:32].rstrip("., ") or "핵심 장면",
        "confidence_1_5": 2,
    }


def completion_gate(scores: dict[str, Any]) -> float:
    values = [
        int(scores["progression_payoff_0_4"]),
        int(scores["self_contained_0_4"]),
        int(scores["boundary_integrity_0_4"]),
    ]
    if any(value == 0 for value in values):
        return 0.50
    if sum(value <= 1 for value in values) >= 2:
        return 0.65
    if sum(value <= 1 for value in values) == 1:
        return 0.80
    return 1.00


def selection_score(scores: dict[str, Any]) -> float:
    raw = sum(
        weight * number(scores[field])
        for field, weight in SELECTION_WEIGHTS.items()
    ) / 4.0
    return clamp(raw) * completion_gate(scores)


def judge_candidate_pool(
    candidates: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[dict[str, dict[str, Any]], str, str]:
    if not candidates:
        return {}, "offline-preview-v1", ""
    prioritized = sorted(
        candidates,
        key=lambda item: number(item["rerank_score"]),
        reverse=True,
    )[:30]
    model = "offline-preview-v1"
    warning = ""
    if mode != "preview":
        batch = {
            "longform_id": "web_beta",
            "candidates": [
                {
                    "candidate_id": item["candidate_id"],
                    "start_time": seconds_to_clock(number(item["pred_start_sec"])),
                    "end_time": seconds_to_clock(number(item["pred_end_sec"])),
                    "description": item["description"],
                    "transcript": item["transcript"],
                }
                for item in prioritized
            ],
        }
        try:
            result, model = call_judge_json(
                instructions=selection_instructions(),
                text_input=json.dumps(batch, ensure_ascii=False),
                schema_name="vpick_selection_judge",
                schema=selection_schema(),
            )
            scored = list(result["candidate_scores"])
        except AppError as exc:
            if mode == "live":
                raise
            scored = [offline_selection_judge(item) for item in prioritized]
            warning = str(exc)
    else:
        scored = [offline_selection_judge(item) for item in prioritized]
    return {str(item["candidate_id"]): item for item in scored}, model, warning


def load_scene_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_vpick_asset_url(value: str) -> tuple[str, str] | None:
    match = re.search(r"/project/([^/]+)/([^/?#]+)", value)
    if not match:
        return None
    return match.group(1), match.group(2)


def resolve_scenes(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    supplied = payload.get("scene_payload")
    if supplied:
        scenes = extract_scene_list(supplied)
        if not scenes:
            raise AppError("업로드한 JSON에서 Vpick 장면 목록을 찾지 못했습니다.")
        return scenes, "uploaded_scene_json"
    vpick_url = str(payload.get("vpick_asset_url", "")).strip()
    if vpick_url:
        ids = parse_vpick_asset_url(vpick_url)
        if not ids:
            raise AppError("Vpick 주소에서 project ID와 asset ID를 읽지 못했습니다.")
        client = VpickClient()
        client.login_from_env()
        scenes = extract_scene_list(client.get_scenes(*ids))
        if not scenes:
            raise AppError("Vpick API 응답에 장면이 없습니다.", status=422, code="empty_scenes")
        return scenes, "vpick_api"
    video_url = str(payload.get("video_url", "")).strip()
    video_id = extract_youtube_id(video_url)
    candidates = [
        RAW_VPICK_DIR / f"{video_id}_scenes.json",
        Path(__file__).resolve().parent / "samples" / f"{video_id}_scenes.json",
    ]
    for path in candidates:
        if video_id and path.exists():
            scenes = extract_scene_list(load_scene_payload(path))
            if scenes:
                return scenes, "repository_vpick_scenes" if path.parent == RAW_VPICK_DIR else "sample_scene_json"
    raise AppError(
        "이 URL에 연결된 장면 분석 자료가 없습니다. 새 영상은 Vpick asset 주소를 입력하거나 장면 JSON을 업로드해 주세요.",
        status=422,
        code="scene_data_required",
    )


def public_candidate(candidate: dict[str, Any], video_url: str, rank: int) -> dict[str, Any]:
    start = number(candidate["pred_start_sec"])
    end = number(candidate["pred_end_sec"])
    video_id = extract_youtube_id(video_url)
    watch_url = (
        f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"
        if video_id
        else video_url
    )
    judge = candidate.get("judge", {})
    return {
        "rank": rank,
        "candidate_id": candidate["candidate_id"],
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "start_time": seconds_to_clock(start),
        "end_time": seconds_to_clock(end),
        "duration_sec": round(end - start, 1),
        "watch_url": watch_url,
        "source_kind": candidate["source_kind"],
        "generated_title": judge.get("generated_title") or "핵심 장면",
        "description": candidate["description"],
        "transcript_excerpt": candidate["transcript"][:420],
        "structural_score": round(number(candidate["rerank_score"]) * 100.0, 1),
        "judge_score": round(number(candidate.get("judge_score")) * 100.0, 1),
        "completion_gate": number(candidate.get("completion_gate"), 1.0),
        "final_score": round(number(candidate.get("final_score")) * 100.0, 1),
        "selection_role": candidate.get("selection_role", ""),
        "evidence": judge.get("evidence_first", ""),
    }


def run_highlight_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    video_url = str(payload.get("video_url", "")).strip()
    if not video_url and not payload.get("scene_payload"):
        raise AppError("YouTube URL 또는 장면 JSON이 필요합니다.")
    mode = str(payload.get("mode", "auto")).strip()
    scenes, data_source = resolve_scenes(payload)
    pool = generate_candidate_pool(scenes)
    if len(pool) < 5:
        raise AppError(
            f"유효 후보가 {len(pool)}개뿐이라 Top5를 만들 수 없습니다.",
            status=422,
            code="insufficient_candidates",
        )
    rows = [
        {
            **candidate,
            "pred_start_sec": str(candidate["pred_start_sec"]),
            "pred_end_sec": str(candidate["pred_end_sec"]),
            "duration_sec": str(candidate["duration_sec"]),
            "rerank_score": str(candidate["rerank_score"]),
        }
        for candidate in pool
    ]
    adaptive = select_adaptive_coverage(
        rows,
        top_k=5,
        coverage_bin_count=5,
        coverage_per_bin=1,
    )
    adaptive_ids = [str(item["candidate_id"]) for item in adaptive]
    score_map, model, warning = judge_candidate_pool(pool, mode=mode)
    for candidate in pool:
        scores = score_map.get(candidate["candidate_id"]) or offline_selection_judge(candidate)
        judge_value = selection_score(scores)
        candidate["judge"] = scores
        candidate["judge_score"] = judge_value
        candidate["completion_gate"] = completion_gate(scores)
        candidate["final_score"] = clamp(
            0.75 * number(candidate["rerank_score"]) + 0.25 * judge_value
        )
    by_id = {candidate["candidate_id"]: candidate for candidate in pool}
    anchors = [by_id[candidate_id] for candidate_id in adaptive_ids[:4] if candidate_id in by_id]
    for candidate in anchors:
        candidate["selection_role"] = "adaptive_coverage_anchor"
    supplement = None
    for candidate in sorted(pool, key=lambda item: number(item["final_score"]), reverse=True):
        if candidate in anchors:
            continue
        if all(interval_overlap_ratio(candidate, anchor) <= 0.58 for anchor in anchors):
            supplement = candidate
            break
    if supplement is None:
        supplement = next(
            (by_id[item] for item in adaptive_ids[4:] if item in by_id),
            sorted(pool, key=lambda item: number(item["final_score"]), reverse=True)[0],
        )
    supplement["selection_role"] = "judge_supplement"
    final = (anchors + [supplement])[:5]
    duration_sec = max(number(scene["end_sec"]) for scene in scenes)
    return {
        "video_id": extract_youtube_id(video_url),
        "video_url": video_url,
        "data_source": data_source,
        "mode": "live_llm" if model != "offline-preview-v1" else "offline_preview",
        "model": model,
        "warning": warning,
        "scene_count": len(scenes),
        "video_duration_sec": round(duration_sec, 1),
        "compressed_candidate_count": len(pool),
        "judge_candidate_count": min(30, len(pool)),
        "adaptive_coverage_ids": adaptive_ids,
        "final_candidates": [
            public_candidate(candidate, video_url, rank)
            for rank, candidate in enumerate(final, start=1)
        ],
        "top_ranked_candidates": [
            public_candidate(candidate, video_url, rank)
            for rank, candidate in enumerate(
                sorted(pool, key=lambda item: number(item["final_score"]), reverse=True)[:10],
                start=1,
            )
        ],
        "pipeline": {
            "candidate_windows": [30, 45, 60, 75],
            "adaptive_coverage_top_k": 5,
            "preserved_anchors": 4,
            "structural_weight": 0.75,
            "judge_weight": 0.25,
            "supplement_overlap_limit": 0.58,
        },
    }


def available_library() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for path in sorted(RAW_VPICK_DIR.glob("*_scenes.json")):
        video_id = path.name.removesuffix("_scenes.json")
        output.append(
            {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "source": "repository",
            }
        )
    output.insert(
        0,
        {
            "video_id": "BETA_DEMO01",
            "video_url": "https://www.youtube.com/watch?v=BETA_DEMO01",
            "source": "sample",
        },
    )
    return output


class VpickBetaHandler(BaseHTTPRequestHandler):
    server_version = "VpickBeta/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {format_string % args}")

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, relative: str) -> None:
        requested = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in requested.parents and requested != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not requested.exists() or not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(requested.suffix, "application/octet-stream")
        body = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 20 * 1024 * 1024:
            raise AppError("요청 본문 크기가 올바르지 않습니다.")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AppError("JSON 요청을 읽지 못했습니다.") from exc
        if not isinstance(payload, dict):
            raise AppError("JSON 객체를 보내야 합니다.")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            openai_ready = bool(os.getenv("OPENAI_API_KEY"))
            gemini_ready = bool(gemini_api_key())
            provider = "OpenAI" if openai_ready else "Gemini" if gemini_ready else ""
            model = (
                os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-sol")
                if openai_ready
                else os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.6-flash")
            )
            self.send_json(
                {
                    "status": "ok",
                    "openai_ready": openai_ready,
                    "gemini_ready": gemini_ready,
                    "llm_ready": openai_ready or gemini_ready,
                    "provider": provider,
                    "vpick_ready": bool(
                        os.getenv("VPICK_ACCESS_TOKEN")
                        or (os.getenv("VPICK_EMAIL") and os.getenv("VPICK_PASSWORD"))
                    ),
                    "model": model,
                    "library_count": len(available_library()),
                }
            )
            return
        if path == "/api/library":
            self.send_json({"items": available_library()})
            return
        if path == "/api/sample-scenes":
            self.send_json(load_scene_payload(SAMPLE_SCENES))
            return
        if path in {"/", "/index.html"}:
            self.serve_static("index.html")
            return
        self.serve_static(path.removeprefix("/"))

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self.read_json()
            if self.path == "/api/judge":
                result = run_package_judge(payload)
            elif self.path == "/api/highlights":
                result = run_highlight_pipeline(payload)
            else:
                raise AppError("지원하지 않는 API 경로입니다.", status=404, code="not_found")
            self.send_json(result)
        except AppError as exc:
            self.send_json(
                {"error": str(exc), "code": exc.code},
                status=exc.status,
            )
        except Exception as exc:  # pragma: no cover - final API boundary
            self.send_json(
                {"error": f"서버 오류: {exc}", "code": "internal_error"},
                status=500,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Vpick beta web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), VpickBetaHandler)
    print(f"Vpick beta: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
