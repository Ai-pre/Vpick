from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


def pointwise_judgment_json_schema() -> dict[str, Any]:
    def score_dimensions(names: tuple[str, ...]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {name: {"type": "integer", "minimum": 1, "maximum": 5} for name in names},
            "required": list(names),
            "additionalProperties": False,
        }

    evidence = score_dimensions(("description_support", "transcript_intelligibility", "boundary_observability"))
    editorial = score_dimensions(
        ("context_clarity", "event_progression", "completeness", "boundary_naturalness", "content_density", "standalone")
    )
    performance = score_dimensions(
        (
            "emotional_intensity", "change_or_surprise", "specificity_novelty",
            "relatability_shareability", "payoff_strength", "hook_title_potential",
        )
    )
    return {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["score", "abstain"]},
            "evidence": evidence,
            "editorial": {**editorial, "type": ["object", "null"]},
            "performance": {**performance, "type": ["object", "null"]},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            "failure_flags": {"type": "array", "items": {"type": "string"}, "maxItems": 13},
            "reason": {"type": "string"},
        },
        "required": [
            "candidate_id", "verdict", "evidence", "editorial", "performance",
            "confidence", "failure_flags", "reason",
        ],
        "additionalProperties": False,
    }


def pairwise_judge_json_schema() -> dict[str, Any]:
    score_dimensions = lambda names: {
        "type": "object",
        "properties": {
            name: {"type": "integer", "minimum": 1, "maximum": 5}
            for name in names
        },
        "required": list(names),
        "additionalProperties": False,
    }
    evidence = score_dimensions(
        ("description_support", "transcript_intelligibility", "boundary_observability", "visual_dependency")
    )
    editorial = score_dimensions(
        ("context_clarity", "event_progression", "completeness", "boundary_naturalness", "content_density", "standalone")
    )
    performance = score_dimensions(
        (
            "emotional_intensity", "change_or_surprise", "specificity_novelty",
            "relatability_shareability", "payoff_strength", "hook_title_potential",
        )
    )
    side = {
        "type": "object",
        "properties": {
            "evidence": evidence,
            "editorial": {**editorial, "type": ["object", "null"]},
            "performance": {**performance, "type": ["object", "null"]},
        },
        "required": ["evidence", "editorial", "performance"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "comparison_id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["score", "abstain"]},
            "left": side,
            "right": side,
            "editorial_preference": {"type": "string", "enum": ["left", "right", "tie"]},
            "performance_preference": {"type": "string", "enum": ["left", "right", "tie"]},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            "failure_flags": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "reason": {"type": "string"},
        },
        "required": [
            "comparison_id", "verdict", "left", "right", "editorial_preference",
            "performance_preference", "confidence", "failure_flags", "reason",
        ],
        "additionalProperties": False,
    }


def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> dict[str, Any]:
    provider = provider.lower().strip()
    if provider == "openai":
        return call_openai(model, system_prompt, user_prompt, max_tokens=max_tokens)
    if provider == "openrouter":
        return call_openrouter(model, system_prompt, user_prompt, max_tokens=max_tokens)
    if provider in {"anthropic", "claude"}:
        return call_anthropic(model, system_prompt, user_prompt, max_tokens=max_tokens)
    if provider in {"gemini", "google"}:
        return call_gemini(model, system_prompt, user_prompt, max_tokens=max_tokens)
    raise ValueError(f"Unsupported provider: {provider}")


def call_openrouter(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set.")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "provider": {
            "require_parameters": True,
            "allow_fallbacks": True,
        },
    }
    data = post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/",
            "X-OpenRouter-Title": "Vpick LLM-as-a-Judge",
        },
        timeout_sec=360,
    )
    text = data["choices"][0]["message"]["content"]
    return {
        "provider": "openrouter",
        "model": model,
        "text": text,
        "json": parse_json_text(text),
        "usage": data.get("usage", {}),
        "raw": data,
    }


def call_openai(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set.")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0
    data = post_json(
        "https://api.openai.com/v1/chat/completions",
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    text = data["choices"][0]["message"]["content"]
    return {
        "provider": "openai",
        "model": model,
        "text": text,
        "json": parse_json_text(text),
        "usage": data.get("usage", {}),
        "raw": data,
    }


def call_anthropic(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY or CLAUDE_API_KEY is not set.")
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    adaptive_models = ("claude-opus-4-7", "claude-opus-4-8")
    no_sampling_models = (
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
        *adaptive_models,
    )
    if model.startswith(adaptive_models):
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": "high"}
    if not model.startswith(no_sampling_models):
        payload["temperature"] = 0
    data = post_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    text_parts = [
        item.get("text", "")
        for item in data.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    text = "\n".join(text_parts).strip()
    return {
        "provider": "anthropic",
        "model": model,
        "text": text,
        "json": parse_json_text(text),
        "usage": data.get("usage", {}),
        "raw": data,
    }


def call_gemini(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    generation_config: dict[str, Any] = {
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
    }
    if model.startswith("gemini-3.6"):
        generation_config["thinkingConfig"] = {
            "thinkingLevel": os.getenv("GEMINI_THINKING_LEVEL", "high"),
        }
    else:
        generation_config["temperature"] = 0
    data = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        },
        {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )
    return parse_gemini_response(data, model)


def call_gemini_video_pair(
    model: str,
    system_prompt: str,
    user_prompt: str,
    left_video_url: str,
    left_start_sec: float,
    left_end_sec: float,
    right_video_url: str,
    right_start_sec: float,
    right_end_sec: float,
    max_tokens: int,
    fps: float = 2.0,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    def video_part(url: str, start_sec: float, end_sec: float) -> dict[str, Any]:
        if not url.startswith(("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")):
            raise LLMError(f"Gemini video input must be a YouTube URL: {url}")
        if start_sec < 0 or end_sec <= start_sec:
            raise LLMError(f"Invalid video interval: {start_sec}-{end_sec}")
        return {
            "fileData": {"fileUri": url, "mimeType": "video/*"},
            "videoMetadata": {
                "startOffset": f"{start_sec:g}s",
                "endOffset": f"{end_sec:g}s",
                "fps": fps,
            },
        }

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "다음 영상 구간은 LEFT 후보입니다."},
                    video_part(left_video_url, left_start_sec, left_end_sec),
                    {"text": "다음 영상 구간은 RIGHT 후보입니다."},
                    video_part(right_video_url, right_start_sec, right_end_sec),
                    {"text": user_prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": pairwise_judge_json_schema(),
        },
    }
    data = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload,
        {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        timeout_sec=360,
    )
    return parse_gemini_response(data, model)


def call_gemini_video_batch(
    model: str,
    system_prompt: str,
    user_prompt: str,
    comparisons: list[dict[str, Any]],
    max_tokens: int,
    fps: float = 2.0,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    if not 1 <= len(comparisons) <= 5:
        raise LLMError(f"Gemini video batch must contain 1-5 comparisons: {len(comparisons)}")

    def video_part(url: str, start_sec: float, end_sec: float) -> dict[str, Any]:
        if not url.startswith(("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")):
            raise LLMError(f"Gemini video input must be a YouTube URL: {url}")
        if start_sec < 0 or end_sec <= start_sec:
            raise LLMError(f"Invalid video interval: {start_sec}-{end_sec}")
        return {
            "fileData": {"fileUri": url, "mimeType": "video/*"},
            "videoMetadata": {
                "startOffset": f"{start_sec:g}s",
                "endOffset": f"{end_sec:g}s",
                "fps": fps,
            },
        }

    parts: list[dict[str, Any]] = []
    for index, comparison in enumerate(comparisons, start=1):
        comparison_id = str(comparison["comparison_id"])
        parts.extend(
            [
                {"text": f"비교 {index} ({comparison_id})의 LEFT 후보 영상입니다."},
                video_part(
                    str(comparison["left"]["url"]),
                    float(comparison["left"]["start_sec"]),
                    float(comparison["left"]["end_sec"]),
                ),
                {"text": f"비교 {index} ({comparison_id})의 RIGHT 후보 영상입니다."},
                video_part(
                    str(comparison["right"]["url"]),
                    float(comparison["right"]["start_sec"]),
                    float(comparison["right"]["end_sec"]),
                ),
            ]
        )
    parts.append({"text": user_prompt})
    schema = {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "items": pairwise_judge_json_schema(),
                "minItems": len(comparisons),
                "maxItems": len(comparisons),
            }
        },
        "required": ["judgments"],
        "additionalProperties": False,
    }
    data = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        },
        {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        timeout_sec=900,
    )
    return parse_gemini_response(data, model)


def call_gemini_text_batch(
    model: str,
    system_prompt: str,
    user_prompt: str,
    comparison_count: int,
    max_tokens: int,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    if not 1 <= comparison_count <= 5:
        raise LLMError(f"Gemini text batch must contain 1-5 comparisons: {comparison_count}")

    schema = {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "items": pairwise_judge_json_schema(),
                "minItems": comparison_count,
                "maxItems": comparison_count,
            }
        },
        "required": ["judgments"],
        "additionalProperties": False,
    }
    data = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        },
        {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        timeout_sec=360,
    )
    return parse_gemini_response(data, model)


def call_gemini_video_pointwise_batch(
    model: str,
    system_prompt: str,
    user_prompt: str,
    candidates: list[dict[str, Any]],
    max_tokens: int,
    fps: float = 2.0,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    if not 1 <= len(candidates) <= 5:
        raise LLMError(f"Gemini pointwise video batch must contain 1-5 candidates: {len(candidates)}")

    def video_part(url: str, start_sec: float, end_sec: float) -> dict[str, Any]:
        if not url.startswith(("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")):
            raise LLMError(f"Gemini video input must be a YouTube URL: {url}")
        if start_sec < 0 or end_sec <= start_sec:
            raise LLMError(f"Invalid video interval: {start_sec}-{end_sec}")
        return {
            "fileData": {"fileUri": url, "mimeType": "video/*"},
            "videoMetadata": {
                "startOffset": f"{start_sec:g}s",
                "endOffset": f"{end_sec:g}s",
                "fps": fps,
            },
        }

    parts: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate["candidate_id"])
        parts.extend(
            [
                {"text": f"독립 평가 후보 {index} ({candidate_id})의 영상 구간입니다."},
                video_part(
                    str(candidate["url"]),
                    float(candidate["start_sec"]),
                    float(candidate["end_sec"]),
                ),
            ]
        )
    parts.append({"text": user_prompt})
    schema = {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "items": pointwise_judgment_json_schema(),
                "minItems": len(candidates),
                "maxItems": len(candidates),
            }
        },
        "required": ["judgments"],
        "additionalProperties": False,
    }
    data = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        },
        {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        timeout_sec=900,
    )
    return parse_gemini_response(data, model)


def parse_gemini_response(data: dict[str, Any], model: str) -> dict[str, Any]:
    candidates = data.get("candidates", [])
    if not candidates:
        raise LLMError(f"Gemini returned no candidates: {data.get('promptFeedback', {})}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise LLMError(f"Gemini returned no text: finishReason={candidates[0].get('finishReason')}")
    return {
        "provider": "gemini",
        "model": model,
        "text": text,
        "json": parse_json_text(text),
        "usage": data.get("usageMetadata", {}),
        "raw": data,
    }


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: int = 120) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as res:
            text = res.read().decode("utf-8")
            return json.loads(text)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"POST {url} failed: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"POST {url} failed: {type(exc).__name__}: {exc}") from exc


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM did not return valid JSON: {text[:500]}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("LLM JSON root must be an object.")
    return parsed
