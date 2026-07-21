"""Lazy local embedding service for dense retrieval."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from functools import lru_cache
import logging
from time import perf_counter

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from greenloop_rag_crew.runtime_paths import prepare_model_cache_dirs

LOGGER = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE = 16
EXPECTED_EMBEDDING_DIMENSION = 768


class EmbeddingValidationError(ValueError):
    """Raised when an embedding vector is malformed."""


class GreenLoopEmbedder:
    """Small wrapper around SentenceTransformer with lazy model loading."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self.device = device or os.getenv("EMBEDDING_DEVICE") or DEFAULT_EMBEDDING_DEVICE
        self.batch_size = batch_size or _env_int(
            "EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE
        )
        self._model: SentenceTransformer | None = None
        self.model_load_seconds: float | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            started = perf_counter()
            prepare_model_cache_dirs()
            model_options = {
                "device": self.device,
                "trust_remote_code": False,
            }
            cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME") or None
            if cache_folder is not None:
                model_options["cache_folder"] = cache_folder
            self._model = SentenceTransformer(
                self.model_name,
                **model_options,
            )
            self.model_load_seconds = perf_counter() - started
            LOGGER.info(
                "timing event=embedding_model_loaded elapsed_seconds=%.3f model=%s",
                self.model_load_seconds,
                self.model_name,
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts with normalized float32 vectors."""

        if not texts:
            return []
        embeddings = self._encode(texts)
        return self._validate_embeddings(embeddings, expected_count=len(texts))

    def embed_query(self, query: str) -> list[float]:
        """Embed a raw user query with the same model and normalization."""

        if not query.strip():
            raise ValueError("Query text must not be empty.")
        embeddings = self._encode([query])
        return self._validate_embeddings(embeddings, expected_count=1)[0]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Embedding texts must be non-empty strings.")

        encoded = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            precision="float32",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(encoded, dtype=np.float32)

    def _validate_embeddings(
        self, embeddings: np.ndarray, expected_count: int
    ) -> list[list[float]]:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[0] != expected_count:
            raise EmbeddingValidationError(
                f"Expected {expected_count} embeddings, received {embeddings.shape[0]}."
            )
        if embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
            raise EmbeddingValidationError(
                f"Expected embedding dimension {EXPECTED_EMBEDDING_DIMENSION}, "
                f"received {embeddings.shape[1]}."
            )
        if not np.isfinite(embeddings).all():
            raise EmbeddingValidationError("Embedding vectors must contain only finite values.")

        vectors = embeddings.astype(np.float32, copy=False).tolist()
        for vector in vectors:
            if not vector or not all(math.isfinite(value) for value in vector):
                raise EmbeddingValidationError("Embedding vectors must be finite and non-empty.")
        return vectors


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


@lru_cache(maxsize=4)
def _cached_embedder(model_name: str, device: str, batch_size: int) -> GreenLoopEmbedder:
    """Return one reusable lazy embedder for a resolved runtime configuration."""

    return GreenLoopEmbedder(model_name=model_name, device=device, batch_size=batch_size)


def get_cached_embedder() -> GreenLoopEmbedder:
    """Return the process-level embedding service without loading weights yet."""

    load_dotenv()
    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    device = os.getenv("EMBEDDING_DEVICE") or DEFAULT_EMBEDDING_DEVICE
    batch_size = _env_int("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)
    return _cached_embedder(model_name, device, batch_size)


def clear_cached_embedders() -> None:
    """Clear cached embedder wrappers for tests or explicit runtime reconfiguration."""

    _cached_embedder.cache_clear()
