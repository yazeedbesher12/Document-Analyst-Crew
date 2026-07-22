"""Validated, environment-selected CrewAI LLM configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from crewai import LLM
from dotenv import load_dotenv

DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_MODEL = f"ollama/{DEFAULT_OLLAMA_MODEL}"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3-8b"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_MAX_TOKENS = 400
DEFAULT_LLM_NUM_CTX = 3072
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"
# qwen3:8b generates at about 2 tokens/second on the target local machine.
# Allow a complete response instead of triggering automatic retries.
DEFAULT_TIMEOUT_SECONDS = 600


class LLMConfigurationError(ValueError):
    """Raised when the selected LLM provider is not configured safely."""


class OllamaConfigurationError(LLMConfigurationError):
    """Raised when local Ollama settings are invalid."""


class OpenRouterConfigurationError(LLMConfigurationError):
    """Raised when the OpenRouter runtime configuration is incomplete or invalid."""


@dataclass(frozen=True)
class LLMSettings:
    """Validated settings required to construct one CrewAI LLM instance."""

    provider: str
    model: str
    base_url: str
    api_version: str | None = None
    api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class GenerationSettings:
    """Validated runtime controls shared by direct and CrewAI LLM calls."""

    temperature: float
    max_tokens: int
    num_ctx: int
    think: bool
    keep_alive: str


def create_llm() -> LLM:
    """Create a CrewAI LLM for the environment-selected provider."""

    settings = get_provider_settings()
    generation = get_generation_settings()
    shared_options = {
        "model": settings.model,
        "temperature": generation.temperature,
        "top_p": 0.95,
        "timeout": DEFAULT_TIMEOUT_SECONDS,
        "max_tokens": generation.max_tokens,
    }

    if settings.provider == "ollama":
        additional_params = {
            "extra_body": {
                "think": generation.think,
                "keep_alive": generation.keep_alive,
                "options": {"num_ctx": generation.num_ctx},
            }
        }
        return LLM(
            base_url=settings.base_url,
            additional_params=additional_params,
            **shared_options,
        )

    # Legacy CrewAI mode uses LiteLLM's OpenRouter provider identifier.
    return LLM(
        model=f"openrouter/{settings.model}",
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=generation.temperature,
        top_p=0.95,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_tokens=generation.max_tokens,
    )


def get_provider_settings() -> LLMSettings:
    """Return validated settings for the selected provider without contacting it."""

    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    if provider == "ollama":
        return _ollama_settings()
    if provider == "openrouter":
        return _openrouter_settings()
    raise LLMConfigurationError(
        "LLM_PROVIDER must be either 'ollama' or 'openrouter'."
    )


def get_generation_settings() -> GenerationSettings:
    """Return validated generation controls without making an LLM request."""

    return GenerationSettings(
        temperature=_configured_temperature(),
        max_tokens=_configured_max_tokens(),
        num_ctx=_configured_num_ctx(),
        think=ollama_thinking_enabled(),
        keep_alive=_configured_keep_alive(),
    )


def get_llm_settings() -> tuple[str, str]:
    """Return validated Ollama ``(model, base_url)`` settings for legacy callers."""

    settings = get_provider_settings()
    if settings.provider != "ollama":
        raise OllamaConfigurationError(
            "Ollama settings were requested while LLM_PROVIDER is 'openrouter'."
        )
    return settings.model, settings.base_url


def normalize_base_url(base_url: str, variable_name: str = "OLLAMA_BASE_URL") -> str:
    """Validate and normalize an HTTP(S) service endpoint."""

    stripped = base_url.strip().rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigurationError(
            f"{variable_name} must be a valid HTTP or HTTPS URL."
        )
    return stripped


def ollama_model_name(model: str | None = None) -> str:
    """Convert a CrewAI Ollama identifier to the installed Ollama model name."""

    configured_model = model or get_llm_settings()[0]
    if not configured_model.startswith("ollama/"):
        raise OllamaConfigurationError("OLLAMA_MODEL must not include another provider prefix.")
    return configured_model.removeprefix("ollama/")


def _ollama_settings() -> LLMSettings:
    raw_model = os.getenv("OLLAMA_MODEL")
    if raw_model is None:
        # MODEL was used before provider selection existed; retain it for local users.
        raw_model = os.getenv("MODEL", DEFAULT_MODEL)
    raw_model = raw_model.strip()
    if not raw_model:
        raise OllamaConfigurationError("OLLAMA_MODEL must not be empty.")

    model = raw_model if raw_model.startswith("ollama/") else f"ollama/{raw_model}"
    if model.count("/") != 1 or not model.removeprefix("ollama/").strip():
        raise OllamaConfigurationError("OLLAMA_MODEL must name an Ollama model, for example qwen3:8b.")

    raw_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    try:
        base_url = normalize_base_url(raw_base_url, "OLLAMA_BASE_URL")
    except LLMConfigurationError as exc:
        raise OllamaConfigurationError(str(exc)) from exc
    return LLMSettings(provider="ollama", model=model, base_url=base_url)


def _openrouter_settings() -> LLMSettings:
    """Read OpenRouter settings only when explicitly selected by LLM_PROVIDER."""

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
    if not model:
        raise OpenRouterConfigurationError("OPENROUTER_MODEL must not be empty.")
    _reject_placeholder(model, "OPENROUTER_MODEL", OpenRouterConfigurationError)

    raw_base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
    try:
        base_url = normalize_base_url(raw_base_url, "OPENROUTER_BASE_URL")
    except LLMConfigurationError as exc:
        raise OpenRouterConfigurationError(str(exc)) from exc
    if "localhost" in base_url.lower() or "127.0.0.1" in base_url:
        raise OpenRouterConfigurationError(
            "OPENROUTER_BASE_URL must point to the OpenRouter API, not localhost."
        )
    api_key = _required_env("OPENROUTER_API_KEY", OpenRouterConfigurationError, "openrouter")
    _reject_placeholder(api_key, "OPENROUTER_API_KEY", OpenRouterConfigurationError)
    return LLMSettings(
        provider="openrouter",
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def _required_env(
    name: str,
    error_type: type[LLMConfigurationError],
    provider: str,
) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise error_type(f"{name} is required when LLM_PROVIDER={provider}.")
    return value


def _reject_placeholder(
    value: str,
    variable_name: str,
    error_type: type[LLMConfigurationError],
) -> None:
    """Reject example placeholders before a request reaches a cloud provider."""

    if "<" in value or ">" in value:
        raise error_type(f"{variable_name} must be replaced with a real value.")


def ollama_thinking_enabled() -> bool:
    """Return whether Ollama reasoning mode is explicitly enabled at runtime."""

    return _env_bool("OLLAMA_THINK", default=False)


def _configured_temperature() -> float:
    value = _env_float("LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE)
    if not 0 <= value <= 2:
        raise LLMConfigurationError("LLM_TEMPERATURE must be between 0 and 2.")
    return value


def _configured_max_tokens() -> int:
    raw_value = os.getenv("LLM_MAX_TOKENS", str(DEFAULT_LLM_MAX_TOKENS)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise LLMConfigurationError("LLM_MAX_TOKENS must be a positive integer.") from exc
    if not 64 <= value <= 2000:
        raise LLMConfigurationError("LLM_MAX_TOKENS must be between 64 and 2000.")
    return value


def _configured_num_ctx() -> int:
    raw_value = os.getenv("LLM_NUM_CTX", str(DEFAULT_LLM_NUM_CTX)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise LLMConfigurationError("LLM_NUM_CTX must be a positive integer.") from exc
    if not 512 <= value <= 32768:
        raise LLMConfigurationError("LLM_NUM_CTX must be between 512 and 32768.")
    return value


def _configured_keep_alive() -> str:
    value = os.getenv("OLLAMA_KEEP_ALIVE", DEFAULT_OLLAMA_KEEP_ALIVE).strip()
    if not value:
        raise LLMConfigurationError("OLLAMA_KEEP_ALIVE must not be empty.")
    return value


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be a number.") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise LLMConfigurationError(f"{name} must be true or false.")
