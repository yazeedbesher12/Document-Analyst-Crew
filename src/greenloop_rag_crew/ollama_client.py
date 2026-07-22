"""Small supported Ollama HTTP client for the single-generation fast path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from greenloop_rag_crew.llm import GenerationSettings, LLMSettings, ollama_model_name


class OllamaGenerationError(RuntimeError):
    """Raised when one direct Ollama generation request cannot complete."""


@dataclass(frozen=True)
class GenerationResponse:
    """One answer and safe request-level metrics returned by a provider."""

    content: str
    request_count: int
    input_tokens: int | None
    output_tokens: int | None
    load_seconds: float | None
    prompt_evaluation_seconds: float | None
    generation_seconds: float | None
    total_seconds: float
    thinking_present: bool
    think_requested: bool = False
    keep_alive_requested: str | None = None
    time_to_first_token_seconds: float | None = None
    model_already_loaded: bool | None = None


def generate_chat(
    *,
    settings: LLMSettings,
    generation: GenerationSettings,
    messages: list[dict[str, str]],
) -> GenerationResponse:
    """Make exactly one non-streaming request to Ollama's documented chat API."""

    return _generate_chat(
        settings=settings,
        generation=generation,
        messages=messages,
        stream=False,
        on_token=None,
    )


def generate_chat_stream(
    *,
    settings: LLMSettings,
    generation: GenerationSettings,
    messages: list[dict[str, str]],
    on_token: Callable[[str], None],
) -> GenerationResponse:
    """Make one streaming Ollama request and yield safe final-answer tokens promptly."""

    return _generate_chat(
        settings=settings,
        generation=generation,
        messages=messages,
        stream=True,
        on_token=on_token,
    )


def _generate_chat(
    *,
    settings: LLMSettings,
    generation: GenerationSettings,
    messages: list[dict[str, str]],
    stream: bool,
    on_token: Callable[[str], None] | None,
) -> GenerationResponse:
    """Make one Ollama chat request, consuming newline-delimited stream events when requested."""

    if settings.provider != "ollama":
        raise OllamaGenerationError("Direct Ollama generation requires LLM_PROVIDER=ollama.")

    payload = {
        "model": ollama_model_name(settings.model),
        "messages": messages,
        "stream": stream,
        "think": generation.think,
        "keep_alive": generation.keep_alive,
        "options": {
            "temperature": generation.temperature,
            "top_p": 0.95,
            "num_predict": generation.max_tokens,
            "num_ctx": generation.num_ctx,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{settings.base_url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = perf_counter()
    first_token_at: float | None = None
    try:
        with urlopen(request, timeout=600) as response:
            if stream:
                response_payload, content, thinking_present, first_token_at = _read_stream(
                    response,
                    on_token=on_token,
                    started=started,
                )
            else:
                response_payload, content, thinking_present = _read_response(response.read())
    except HTTPError as exc:
        raise OllamaGenerationError(f"Ollama returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise OllamaGenerationError("Could not connect to Ollama.") from exc
    except TimeoutError as exc:
        raise OllamaGenerationError("Ollama generation timed out.") from exc

    total_seconds = perf_counter() - started
    if not content:
        raise OllamaGenerationError("Ollama returned an empty answer.")

    load_seconds = _nanoseconds_to_seconds(response_payload.get("load_duration"))
    return GenerationResponse(
        content=content,
        request_count=1,
        input_tokens=_optional_int(response_payload.get("prompt_eval_count")),
        output_tokens=_optional_int(response_payload.get("eval_count")),
        load_seconds=load_seconds,
        prompt_evaluation_seconds=_nanoseconds_to_seconds(
            response_payload.get("prompt_eval_duration")
        ),
        generation_seconds=_nanoseconds_to_seconds(response_payload.get("eval_duration")),
        total_seconds=total_seconds,
        thinking_present=thinking_present,
        think_requested=generation.think,
        keep_alive_requested=generation.keep_alive,
        time_to_first_token_seconds=(
            first_token_at - started if first_token_at is not None else None
        ),
        model_already_loaded=(load_seconds < 0.01 if load_seconds is not None else None),
    )


def _read_response(response_bytes: bytes) -> tuple[dict[str, Any], str, bool]:
    try:
        response_payload: dict[str, Any] = json.loads(response_bytes)
        message = response_payload["message"]
        content = str(message.get("content", "")).strip()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OllamaGenerationError("Ollama returned an invalid response.") from exc
    thinking = message.get("thinking") or response_payload.get("thinking")
    return response_payload, content, bool(thinking)


def _read_stream(
    response: Any,
    *,
    on_token: Callable[[str], None] | None,
    started: float,
) -> tuple[dict[str, Any], str, bool, float | None]:
    """Consume Ollama's documented newline-delimited JSON stream without logging content."""

    parts: list[str] = []
    final_payload: dict[str, Any] | None = None
    thinking_present = False
    first_token_at: float | None = None
    try:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            event = json.loads(line)
            message = event.get("message", {})
            token = str(message.get("content", ""))
            if message.get("thinking") or event.get("thinking"):
                thinking_present = True
            if token:
                if first_token_at is None:
                    first_token_at = perf_counter()
                parts.append(token)
                if on_token is not None:
                    on_token(token)
            if event.get("done"):
                final_payload = event
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OllamaGenerationError("Ollama returned an invalid streaming response.") from exc
    if final_payload is None:
        raise OllamaGenerationError("Ollama streaming response ended before completion.")
    return final_payload, "".join(parts).strip(), thinking_present, first_token_at


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _nanoseconds_to_seconds(value: Any) -> float | None:
    return value / 1_000_000_000 if isinstance(value, int | float) else None
