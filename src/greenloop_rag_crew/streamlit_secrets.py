"""Safe root-level Streamlit secrets ingestion for runtime provider settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from streamlit.runtime.secrets import StreamlitSecretNotFoundError


STREAMLIT_RUNTIME_VARIABLES = (
    "LLM_PROVIDER",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_THINK",
    "OLLAMA_KEEP_ALIVE",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_BASE_URL",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
)


def apply_streamlit_secrets(secrets: Mapping[str, Any]) -> None:
    """Expose root-level Streamlit secrets to code without logging their values.

    Existing environment values win, so local shell configuration remains explicit.
    Missing secrets files are treated as empty configuration instead of crashing.
    """

    for name in STREAMLIT_RUNTIME_VARIABLES:
        try:
            value = secrets.get(name)
        except StreamlitSecretNotFoundError:
            continue
        if value is not None and not os.getenv(name):
            os.environ[name] = str(value)
