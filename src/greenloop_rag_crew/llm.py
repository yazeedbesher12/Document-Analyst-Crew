"""Shared CrewAI LLM configuration for local Ollama."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from crewai import LLM
from dotenv import load_dotenv

DEFAULT_MODEL = "ollama/qwen3:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
# qwen3:8b generates at about 2 tokens/second on the target local machine.
# Allow a complete response instead of triggering LiteLLM's automatic retries.
DEFAULT_TIMEOUT_SECONDS = 600


def create_llm() -> LLM:
    """Create a CrewAI LLM configured for the local Ollama qwen3:8b model."""

    model, base_url = get_llm_settings()
    return LLM(
        model=model,
        base_url=base_url,
        temperature=0.6,
        top_p=0.95,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_tokens=1000,
    )


def get_llm_settings() -> tuple[str, str]:
    """Return validated `(model, base_url)` settings from the environment."""

    load_dotenv()
    raw_model = os.getenv("MODEL")
    raw_base_url = os.getenv("OLLAMA_BASE_URL")
    model = DEFAULT_MODEL if raw_model is None else raw_model.strip()
    base_url = DEFAULT_OLLAMA_BASE_URL if raw_base_url is None else raw_base_url.strip()

    if not model:
        raise ValueError("MODEL must not be empty.")
    if not model.startswith("ollama/"):
        raise ValueError("MODEL must use the ollama/ provider prefix.")

    base_url = normalize_base_url(base_url)
    return model, base_url


def normalize_base_url(base_url: str) -> str:
    """Validate and normalize an HTTP(S) base URL."""

    stripped = base_url.strip().rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OLLAMA_BASE_URL must be a valid HTTP or HTTPS URL.")
    return stripped


def ollama_model_name(model: str | None = None) -> str:
    """Convert a CrewAI Ollama model identifier to the installed Ollama model name."""

    configured_model = model or get_llm_settings()[0]
    if not configured_model.startswith("ollama/"):
        raise ValueError("MODEL must use the ollama/ provider prefix.")
    return configured_model.removeprefix("ollama/")
