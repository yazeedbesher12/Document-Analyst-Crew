"""Process-level lifecycle management for GreenLoop hybrid retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter

from greenloop_rag_crew.rag.build_chunks import build_chunks
from greenloop_rag_crew.rag.build_index import (
    INDEX_SCHEMA_VERSION,
    build_index,
    chunking_configuration,
    read_manifest,
    sha256_file,
)
from greenloop_rag_crew.rag.chroma_store import DEFAULT_CHROMA_COLLECTION
from greenloop_rag_crew.rag.document_registry import DOCUMENT_REGISTRY
from greenloop_rag_crew.rag.embedder import DEFAULT_EMBEDDING_MODEL
from greenloop_rag_crew.rag.hybrid_retriever import HybridRetriever
from greenloop_rag_crew.rag.schemas import HybridSearchResult
from greenloop_rag_crew.runtime_paths import chroma_persist_dir, chunks_file, knowledge_dir

LOGGER = logging.getLogger(__name__)
_SERVICE_LOCK = Lock()


@dataclass(frozen=True)
class IndexPreparation:
    """Safe lifecycle details for index initialization and timing reports."""

    action: str
    reason: str
    elapsed_seconds: float


class RetrievalService:
    """One reusable, thread-safe hybrid retriever for a process configuration."""

    def __init__(self, retriever: HybridRetriever, preparation: IndexPreparation) -> None:
        self.retriever = retriever
        self.preparation = preparation
        self._search_lock = Lock()
        self.retrieval_calls = 0
        self.total_retrieval_seconds = 0.0

    def search(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[HybridSearchResult]:
        """Run one serialized local Chroma/BM25 search and log only timing metadata."""

        started = perf_counter()
        with self._search_lock:
            results = self.retriever.search(
                query=query,
                top_k=top_k,
                document_id=document_id,
            )
            elapsed = perf_counter() - started
            self.retrieval_calls += 1
            self.total_retrieval_seconds += elapsed
        LOGGER.info(
            "timing event=retrieval_call elapsed_seconds=%.3f result_count=%s document_filter=%s",
            elapsed,
            len(results),
            document_id or "all",
        )
        return _deduplicate_results(results)

    def metrics_snapshot(self) -> tuple[int, float]:
        """Return aggregate request-safe retrieval metrics without any query content."""

        with self._search_lock:
            return self.retrieval_calls, self.total_retrieval_seconds


def get_retrieval_service() -> RetrievalService:
    """Return the reusable service, rebuilding only when its fingerprint changes."""

    signature = runtime_index_signature()
    with _SERVICE_LOCK:
        return _cached_retrieval_service(signature)


def clear_retrieval_service_cache() -> None:
    """Clear process-level retrieval state for tests or explicit reconfiguration."""

    with _SERVICE_LOCK:
        _cached_retrieval_service.cache_clear()


def runtime_index_signature() -> str:
    """Return a content/config fingerprint without extracting PDF text or loading models."""

    knowledge_path = knowledge_dir()
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "pdfs": _pdf_fingerprints(knowledge_path),
        "embedding_model": _embedding_model_name(),
        "chunking": chunking_configuration(),
        "collection_name": DEFAULT_CHROMA_COLLECTION,
        "chunks_file": str(chunks_file()),
        "chroma_persist_dir": str(chroma_persist_dir(create=False)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=8)
def _cached_retrieval_service(signature: str) -> RetrievalService:
    del signature  # The cache key controls lifecycle; settings are resolved below.
    started = perf_counter()
    preparation = ensure_index_ready()
    retriever = HybridRetriever(
        chunks_file=chunks_file(),
        persist_dir=chroma_persist_dir(),
        collection_name=DEFAULT_CHROMA_COLLECTION,
    )
    elapsed = perf_counter() - started
    LOGGER.info(
        "timing event=retrieval_service_initialized elapsed_seconds=%.3f index_action=%s reason=%s",
        elapsed,
        preparation.action,
        preparation.reason,
    )
    return RetrievalService(retriever=retriever, preparation=preparation)


def ensure_index_ready() -> IndexPreparation:
    """Load a current index or rebuild deterministic artifacts when fingerprints changed."""

    started = perf_counter()
    manifest = read_manifest()
    reason = _chunk_rebuild_reason(manifest)
    if reason is not None:
        build_chunks(knowledge_dir=knowledge_dir(), output=chunks_file())
        action = build_index(
            chunks_file=chunks_file(),
            persist_dir=chroma_persist_dir(),
            collection_name=DEFAULT_CHROMA_COLLECTION,
            knowledge_dir=knowledge_dir(),
            rebuild=False,
        )
        action = "rebuilt" if action in {"rebuilt", "skipped"} else action
        result = IndexPreparation(action=action, reason=reason, elapsed_seconds=perf_counter() - started)
    else:
        action = build_index(
            chunks_file=chunks_file(),
            persist_dir=chroma_persist_dir(),
            collection_name=DEFAULT_CHROMA_COLLECTION,
            knowledge_dir=knowledge_dir(),
            rebuild=False,
        )
        result = IndexPreparation(
            action="loaded" if action == "skipped" else action,
            reason="manifest_current" if action == "skipped" else "chroma_missing_or_stale",
            elapsed_seconds=perf_counter() - started,
        )
    LOGGER.info(
        "index_lifecycle action=%s reason=%s elapsed_seconds=%.3f",
        result.action,
        result.reason,
        result.elapsed_seconds,
    )
    return result


def _chunk_rebuild_reason(manifest: dict | None) -> str | None:
    if manifest is None:
        return "manifest_missing"
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        return "index_schema_changed"
    if manifest.get("embedding_model") != _embedding_model_name():
        return "embedding_model_changed"
    if manifest.get("chunking") != chunking_configuration():
        return "chunking_configuration_changed"
    if manifest.get("pdfs") != _pdf_fingerprints(knowledge_dir()):
        return "source_pdf_fingerprint_changed"
    if not chunks_file().exists():
        return "chunks_file_missing"
    return None


def _pdf_fingerprints(directory: Path) -> list[dict[str, str]]:
    fingerprints: list[dict[str, str]] = []
    for metadata in DOCUMENT_REGISTRY:
        path = directory / metadata.filename
        if not path.exists():
            return [
                {
                    "filename": metadata.filename,
                    "document_id": metadata.document_id,
                    "sha256": "missing",
                }
            ]
        fingerprints.append(
            {
                "filename": metadata.filename,
                "document_id": metadata.document_id,
                "sha256": sha256_file(path),
            }
        )
    return fingerprints


def _embedding_model_name() -> str:
    from greenloop_rag_crew.rag.build_index import _configured_embedding_model

    return _configured_embedding_model() or DEFAULT_EMBEDDING_MODEL


def _deduplicate_results(results: list[HybridSearchResult]) -> list[HybridSearchResult]:
    """Keep the highest-ranked occurrence of each chunk without changing metadata."""

    seen: set[str] = set()
    unique: list[HybridSearchResult] = []
    for result in results:
        if result.chunk_id not in seen:
            unique.append(result)
            seen.add(result.chunk_id)
    return unique
