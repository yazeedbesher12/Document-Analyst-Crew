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
# qwen3:8b generates at about 2 tokens/second on the target local machine.
# Allow a complete response instead of triggering automatic retries.
DEFAULT_TIMEOUT_SECONDS = 600


class LLMConfigurationError(ValueError):
    """Raised when the selected LLM provider is not configured safely."""


class OllamaConfigurationError(LLMConfigurationError):
    """Raised when local Ollama settings are invalid."""


class AzureConfigurationError(LLMConfigurationError):
    """Raised when Azure runtime configuration is incomplete or invalid."""


@dataclass(frozen=True)
class LLMSettings:
    """Validated settings required to construct one CrewAI LLM instance."""

    provider: str
    model: str
    base_url: str
    api_version: str | None = None
    api_key: str | None = field(default=None, repr=False)


def create_llm() -> LLM:
    """Create a CrewAI LLM for the environment-selected provider."""

    settings = get_provider_settings()
    shared_options = {
        "model": settings.model,
        "temperature": 0.6,
        "top_p": 0.95,
        "timeout": DEFAULT_TIMEOUT_SECONDS,
        "max_tokens": 1000,
    }

    if settings.provider == "ollama":
        return LLM(base_url=settings.base_url, **shared_options)

    # CrewAI's native Azure provider accepts endpoint/api_version directly.
    return LLM(
        endpoint=settings.base_url,
        api_key=settings.api_key,
        api_version=settings.api_version,
        **shared_options,
    )


def get_provider_settings() -> LLMSettings:
    """Return validated settings for the selected provider without contacting it."""

    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    if provider == "ollama":
        return _ollama_settings()
    if provider == "azure":
        return _azure_settings()
    raise LLMConfigurationError(
        "LLM_PROVIDER must be either 'ollama' or 'azure'."
    )


def get_llm_settings() -> tuple[str, str]:
    """Return validated Ollama ``(model, base_url)`` settings for legacy callers."""

    settings = get_provider_settings()
    if settings.provider != "ollama":
        raise OllamaConfigurationError(
            "Ollama settings were requested while LLM_PROVIDER is 'azure'."
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


def _azure_settings() -> LLMSettings:
    model = _required_env("AZURE_LLM_MODEL", AzureConfigurationError)
    if not model.startswith("azure/") or model == "azure/":
        raise AzureConfigurationError(
            "AZURE_LLM_MODEL must use the azure/ prefix, for example azure/my-deployment."
        )

    endpoint = _required_env("AZURE_ENDPOINT", AzureConfigurationError)
    try:
        endpoint = normalize_base_url(endpoint, "AZURE_ENDPOINT")
    except LLMConfigurationError as exc:
        raise AzureConfigurationError(str(exc)) from exc
    api_key = _required_env("AZURE_API_KEY", AzureConfigurationError)
    api_version = os.getenv("AZURE_API_VERSION", "").strip() or None
    return LLMSettings(
        provider="azure",
        model=model,
        base_url=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def _required_env(name: str, error_type: type[LLMConfigurationError]) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise error_type(f"{name} is required when LLM_PROVIDER=azure.")
    return value
