"""Small OpenAI-compatible streaming client for the OpenRouter fast path."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from greenloop_rag_crew.llm import GenerationSettings, LLMSettings
from greenloop_rag_crew.ollama_client import GenerationResponse


class OpenRouterGenerationError(RuntimeError):
    """Raised when one direct OpenRouter generation request cannot complete."""


def generate_chat_stream(
    *,
    settings: LLMSettings,
    generation: GenerationSettings,
    messages: list[dict[str, str]],
    on_token: Callable[[str], None],
) -> GenerationResponse:
    """Make exactly one streaming OpenAI-compatible request to OpenRouter."""

    if settings.provider != "openrouter" or not settings.api_key:
        raise OpenRouterGenerationError(
            "Direct OpenRouter generation requires LLM_PROVIDER=openrouter."
        )

    payload = {
        "model": settings.model,
        "messages": messages,
        "stream": True,
        "temperature": generation.temperature,
        "max_tokens": generation.max_tokens,
        "reasoning": {"enabled": False},
    }
    request = Request(
        f"{settings.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = perf_counter()
    try:
        with urlopen(request, timeout=600) as response:
            content, usage, first_token_at, thinking_present = _read_sse_stream(
                response, on_token=on_token, started=started
            )
    except HTTPError as exc:
        raise OpenRouterGenerationError(f"OpenRouter returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise OpenRouterGenerationError("Could not connect to OpenRouter.") from exc
    except TimeoutError as exc:
        raise OpenRouterGenerationError("OpenRouter generation timed out.") from exc

    total_seconds = perf_counter() - started
    if not content:
        raise OpenRouterGenerationError("OpenRouter returned an empty answer.")
    return GenerationResponse(
        content=content,
        request_count=1,
        input_tokens=_optional_int(usage.get("prompt_tokens")),
        output_tokens=_optional_int(usage.get("completion_tokens")),
        load_seconds=None,
        prompt_evaluation_seconds=None,
        generation_seconds=total_seconds,
        total_seconds=total_seconds,
        thinking_present=thinking_present,
        time_to_first_token_seconds=(
            first_token_at - started if first_token_at is not None else None
        ),
        model_already_loaded=None,
    )


def _read_sse_stream(
    response: Any,
    *,
    on_token: Callable[[str], None],
    started: float,
) -> tuple[str, dict[str, Any], float | None, bool]:
    """Consume SSE events without retaining provider diagnostics or hidden reasoning."""

    parts: list[str] = []
    usage: dict[str, Any] = {}
    first_token_at: float | None = None
    thinking_present = False
    try:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("reasoning") or delta.get("reasoning_content"):
                thinking_present = True
            token = str(delta.get("content") or "")
            if token:
                if first_token_at is None:
                    first_token_at = perf_counter()
                parts.append(token)
                on_token(token)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenRouterGenerationError("OpenRouter returned an invalid streaming response.") from exc
    return "".join(parts).strip(), usage, first_token_at, thinking_present


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
