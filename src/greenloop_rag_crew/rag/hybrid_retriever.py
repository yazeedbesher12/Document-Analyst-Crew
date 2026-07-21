"""Hybrid dense plus BM25 retrieval using weighted reciprocal rank fusion."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from greenloop_rag_crew.rag.bm25_retriever import BM25Retriever
from greenloop_rag_crew.rag.build_index import (
    DEFAULT_CHUNKS_FILE,
    build_manifest,
    load_chunks,
    read_manifest,
)
from greenloop_rag_crew.rag.chroma_store import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_PERSIST_DIRECTORY,
    ChromaStore,
)
from greenloop_rag_crew.rag.dense_search import DenseRetriever
from greenloop_rag_crew.rag.embedder import GreenLoopEmbedder, get_cached_embedder
from greenloop_rag_crew.rag.schemas import (
    BM25SearchResult,
    DenseSearchResult,
    HybridSearchResult,
)
from greenloop_rag_crew.runtime_paths import chroma_persist_dir, chunks_file as configured_chunks_file

DEFAULT_DENSE_WEIGHT = 0.50
DEFAULT_BM25_WEIGHT = 0.50
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_K = 20


class HybridRetriever:
    """Compose dense and BM25 retrieval with weighted RRF."""

    def __init__(
        self,
        chunks_file: str | Path | None = None,
        persist_dir: str | Path | None = None,
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        embedder: GreenLoopEmbedder | None = None,
    ) -> None:
        self.chunks_file = Path(chunks_file) if chunks_file is not None else configured_chunks_file()
        self.persist_dir = Path(persist_dir) if persist_dir is not None else chroma_persist_dir()
        self.collection_name = collection_name
        self.embedder = embedder or get_cached_embedder()
        self.dense_retriever = dense_retriever or DenseRetriever(
            persist_dir=self.persist_dir,
            collection_name=collection_name,
            chunks_file=self.chunks_file,
            embedder=self.embedder,
        )
        self.bm25_retriever = bm25_retriever or BM25Retriever(chunks_file=self.chunks_file)
        self._index_verified = False
        self._verification_lock = Lock()

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[HybridSearchResult]:
        weights = _validate_search_args(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
        )
        self._ensure_index_verified()

        dense_results = self.dense_retriever.search(
            query, top_k=candidate_k, document_id=document_id
        )
        bm25_results = self.bm25_retriever.search(
            query, top_k=candidate_k, document_id=document_id
        )
        fused = fuse_results(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=top_k,
            dense_weight=weights[0],
            bm25_weight=weights[1],
            rrf_k=rrf_k,
        )
        return fused

    def _ensure_index_verified(self) -> None:
        if self._index_verified:
            return
        with self._verification_lock:
            if not self._index_verified:
                self._verify_compatible_indexes()
                self._index_verified = True

    def _verify_compatible_indexes(self) -> None:
        manifest = read_manifest()
        if manifest is None:
            raise FileNotFoundError("Index manifest not found. Run build_index first.")

        chunks = load_chunks(self.chunks_file)
        expected = build_manifest(
            chunks,
            self.chunks_file,
            self.collection_name,
            embedding_model=self.embedder.model_name,
        )
        comparable_manifest = dict(manifest)
        comparable_manifest["created_at"] = None
        if comparable_manifest != expected:
            raise ValueError(
                "Lexical chunks and dense index do not match. Run "
                "`uv run python -m greenloop_rag_crew.rag.build_index --rebuild`."
            )

        store = ChromaStore(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
        )
        if store.count() != manifest["chunk_count"]:
            raise ValueError(
                "Chroma collection count does not match the index manifest. Rebuild the index."
            )


def fuse_results(
    dense_results: list[DenseSearchResult],
    bm25_results: list[BM25SearchResult],
    top_k: int,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridSearchResult]:
    """Merge dense and BM25 candidates using weighted reciprocal rank fusion."""

    dense_by_id = {result.chunk_id: result for result in dense_results}
    bm25_by_id = {result.chunk_id: result for result in bm25_results}
    chunk_ids = set(dense_by_id) | set(bm25_by_id)
    scored = []

    for chunk_id in chunk_ids:
        dense = dense_by_id.get(chunk_id)
        bm25 = bm25_by_id.get(chunk_id)
        base = dense or bm25
        assert base is not None
        fusion_score = 0.0
        if dense is not None:
            fusion_score += dense_weight / (rrf_k + dense.rank)
        if bm25 is not None:
            fusion_score += bm25_weight / (rrf_k + bm25.rank)
        best_rank = min(
            rank for rank in [dense.rank if dense else None, bm25.rank if bm25 else None]
            if rank is not None
        )
        scored.append((fusion_score, best_rank, chunk_id, base, dense, bm25))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    results: list[HybridSearchResult] = []
    for rank, (fusion_score, _best_rank, chunk_id, base, dense, bm25) in enumerate(
        scored[:top_k], start=1
    ):
        matched_by = []
        if dense is not None:
            matched_by.append("dense")
        if bm25 is not None:
            matched_by.append("bm25")
        results.append(
            HybridSearchResult(
                rank=rank,
                chunk_id=chunk_id,
                source=base.source,
                document_id=base.document_id,
                title=base.title,
                page=base.page,
                section=base.section,
                text=base.text,
                fusion_score=fusion_score,
                dense_rank=dense.rank if dense else None,
                dense_score=dense.score if dense else None,
                bm25_rank=bm25.rank if bm25 else None,
                bm25_score=bm25.bm25_score if bm25 else None,
                matched_by=matched_by,
            )
        )
    return results


def _validate_search_args(
    query: str,
    top_k: int,
    candidate_k: int,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> tuple[float, float]:
    if not query.strip():
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k.")
    if dense_weight < 0 or bm25_weight < 0:
        raise ValueError("retrieval weights must be non-negative.")
    total_weight = dense_weight + bm25_weight
    if total_weight <= 0:
        raise ValueError("at least one retrieval weight must be greater than zero.")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    return dense_weight / total_weight, bm25_weight / total_weight
