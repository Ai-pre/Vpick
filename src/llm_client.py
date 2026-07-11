from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> dict[str, Any]:
    provider = provider.lower().strip()
    if provider == "openai":
        return call_openai(model, system_prompt, user_prompt, max_tokens=max_tokens)
    if provider in {"anthropic", "claude"}:
        return call_anthropic(model, system_prompt, user_prompt, max_tokens=max_tokens)
    raise ValueError(f"Unsupported provider: {provider}")


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


def call_anthropic(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY or CLAUDE_API_KEY is not set.")
    data = post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
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


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            text = res.read().decode("utf-8")
            return json.loads(text)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"POST {url} failed: HTTP {exc.code} {detail}") from exc


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
