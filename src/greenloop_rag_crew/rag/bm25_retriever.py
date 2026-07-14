"""In-memory BM25 lexical retrieval over Step 2 chunks."""

from __future__ import annotations

from pathlib import Path

from rank_bm25 import BM25Okapi

from greenloop_rag_crew.rag.build_index import DEFAULT_CHUNKS_FILE, load_chunks
from greenloop_rag_crew.rag.schemas import BM25SearchResult, DocumentChunk
from greenloop_rag_crew.rag.tokenizer import tokenize


class BM25Retriever:
    """Small deterministic BM25 retriever backed by storage/chunks.jsonl."""

    def __init__(self, chunks_file: str | Path = DEFAULT_CHUNKS_FILE) -> None:
        self.chunks_file = Path(chunks_file)
        self.chunks = load_chunks(self.chunks_file)
        self._indexed_text = [_searchable_text(chunk) for chunk in self.chunks]
        self._tokenized_corpus = [tokenize(text) for text in self._indexed_text]
        if not self._tokenized_corpus:
            raise ValueError("BM25 corpus is empty.")
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    @property
    def corpus_size(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: str | None = None,
    ) -> list[BM25SearchResult]:
        """Search chunks lexically, with optional document filtering before ranking."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        candidate_indexes = [
            index
            for index, chunk in enumerate(self.chunks)
            if document_id is None or chunk.document_id == document_id
        ]
        if not candidate_indexes:
            return []

        if document_id is None:
            scores = self._bm25.get_scores(query_tokens)
            scored = [
                (float(scores[index]), index, self.chunks[index])
                for index in candidate_indexes
            ]
        else:
            filtered_corpus = [self._tokenized_corpus[index] for index in candidate_indexes]
            filtered_bm25 = BM25Okapi(filtered_corpus)
            filtered_scores = filtered_bm25.get_scores(query_tokens)
            scored = [
                (float(score), original_index, self.chunks[original_index])
                for score, original_index in zip(filtered_scores, candidate_indexes)
            ]

        scored.sort(key=lambda item: (-item[0], item[1], item[2].chunk_id))
        return [
            _to_result(rank, chunk, score)
            for rank, (score, _index, chunk) in enumerate(scored[:top_k], start=1)
        ]


def _searchable_text(chunk: DocumentChunk) -> str:
    return f"{chunk.title}\n{chunk.section}\n{chunk.text}"


def _to_result(rank: int, chunk: DocumentChunk, score: float) -> BM25SearchResult:
    return BM25SearchResult(
        rank=rank,
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        document_id=chunk.document_id,
        title=chunk.title,
        page=chunk.page,
        section=chunk.section,
        bm25_score=score,
        text=chunk.text,
    )
