"""Run one safe, short direct-Ollama fast-path generation diagnostic."""

from __future__ import annotations

import json

from greenloop_rag_crew.llm import get_generation_settings, get_provider_settings
from greenloop_rag_crew.ollama_client import generate_chat_stream


def run_smoke() -> dict[str, object]:
    """Make one request and return metrics only, never prompt or response content."""

    settings = get_provider_settings()
    generation = get_generation_settings()
    if settings.provider != "ollama":
        raise RuntimeError("The fast-path smoke diagnostic requires LLM_PROVIDER=ollama.")
    response = generate_chat_stream(
        settings=settings,
        generation=generation,
        messages=[{"role": "user", "content": "Reply with exactly: healthy"}],
        on_token=lambda _token: None,
    )
    return {
        "request_count": response.request_count,
        "stream_requested": True,
        "think_requested": response.think_requested,
        "thinking_present": response.thinking_present,
        "keep_alive_requested": response.keep_alive_requested,
        "output_present": bool(response.content),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "load_seconds": response.load_seconds,
        "prompt_evaluation_seconds": response.prompt_evaluation_seconds,
        "generation_seconds": response.generation_seconds,
        "total_seconds": response.total_seconds,
        "time_to_first_token_seconds": response.time_to_first_token_seconds,
        "model_already_loaded": response.model_already_loaded,
    }


def main() -> None:
    print(json.dumps(run_smoke(), sort_keys=True))


if __name__ == "__main__":
    main()
