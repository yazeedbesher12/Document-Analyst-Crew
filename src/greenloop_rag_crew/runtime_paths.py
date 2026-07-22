"""Runtime filesystem locations shared by local and container execution."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_KNOWLEDGE_DIRECTORY = Path("knowledge")
DEFAULT_STORAGE_DIRECTORY = Path("storage")
DEFAULT_CHROMA_DIRECTORY = DEFAULT_STORAGE_DIRECTORY / "chroma"
DEFAULT_OUTPUT_DIRECTORY = Path("output")
DEFAULT_CHUNKS_FILE = DEFAULT_STORAGE_DIRECTORY / "chunks.jsonl"
DEFAULT_MANIFEST_FILE = DEFAULT_STORAGE_DIRECTORY / "index_manifest.json"
DEFAULT_ANSWER_CACHE_DIRECTORY = DEFAULT_STORAGE_DIRECTORY / "answer_cache"


def knowledge_dir() -> Path:
    """Return the packaged knowledge directory without creating it."""

    return _environment_path("KNOWLEDGE_DIR", DEFAULT_KNOWLEDGE_DIRECTORY)


def chroma_persist_dir(*, create: bool = True) -> Path:
    """Return the writable Chroma directory, honoring old local configuration."""

    path = _environment_path(
        "CHROMA_PERSIST_DIR",
        _environment_path("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_DIRECTORY),
    )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(*, create: bool = True) -> Path:
    """Return the writable report directory."""

    path = _environment_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIRECTORY)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def answer_cache_dir(*, create: bool = True) -> Path:
    """Return the persistent exact-answer cache directory."""

    path = _environment_path("RAG_ANSWER_CACHE_DIR", DEFAULT_ANSWER_CACHE_DIRECTORY)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def chunks_file() -> Path:
    """Return the generated chunks file location."""

    return DEFAULT_CHUNKS_FILE


def manifest_file() -> Path:
    """Return the generated index manifest location."""

    return DEFAULT_MANIFEST_FILE


def prepare_model_cache_dirs() -> tuple[Path, ...]:
    """Create explicitly configured Hugging Face cache directories when needed."""

    paths = tuple(
        path
        for variable in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME")
        if (path := _configured_path(variable)) is not None
    )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_configured_output_path(output_file: str | Path) -> Path:
    """Map configured ``output/...`` paths into an optional OUTPUT_DIR safely."""

    path = Path(output_file)
    configured_root = os.getenv("OUTPUT_DIR", "").strip()
    if not configured_root or path.is_absolute() or not path.parts or path.parts[0] != "output":
        return path
    return output_dir() / Path(*path.parts[1:])


def _environment_path(variable: str, default: Path) -> Path:
    configured = _configured_path(variable)
    return configured if configured is not None else default


def _configured_path(variable: str) -> Path | None:
    value = os.getenv(variable, "").strip()
    return Path(value).expanduser() if value else None
