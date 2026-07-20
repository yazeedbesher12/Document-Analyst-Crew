"""Persistent Chroma storage wrapper for GreenLoop dense chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv

from greenloop_rag_crew.rag.schemas import DocumentChunk
from greenloop_rag_crew.runtime_paths import chroma_persist_dir

DEFAULT_CHROMA_PERSIST_DIRECTORY = "storage/chroma"
DEFAULT_CHROMA_COLLECTION = "greenloop_documents"


class ChromaStore:
    """Narrow wrapper around persistent Chroma operations used by this project."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        load_dotenv()
        self.persist_dir = (
            Path(persist_dir) if persist_dir is not None else chroma_persist_dir()
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = (
            collection_name or os.getenv("CHROMA_COLLECTION") or DEFAULT_CHROMA_COLLECTION
        )
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

    def get_or_create_collection(self) -> Collection:
        """Create or return a cosine HNSW collection without Chroma embeddings."""

        return self.client.get_or_create_collection(
            name=self.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def recreate_collection(self) -> Collection:
        """Delete and recreate the target collection."""

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        return self.get_or_create_collection()

    def count(self) -> int:
        return self.get_or_create_collection().count()

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 16,
    ) -> None:
        """Add chunks and precomputed embeddings in deterministic batches."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
            )
        collection = self.get_or_create_collection()
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            collection.add(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                embeddings=batch_embeddings,
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[_metadata_for_chunk(chunk) for chunk in batch_chunks],
            )

    def get_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        collection = self.get_or_create_collection()
        result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "document": result["documents"][0],
            "metadata": result["metadatas"][0],
        }

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        collection = self.get_or_create_collection()
        where = {"document_id": document_id} if document_id else None
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )


def _metadata_for_chunk(chunk: DocumentChunk) -> dict[str, str | int]:
    return {
        "source": chunk.source,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "page": chunk.page,
        "section": chunk.section,
        "chunk_id": chunk.chunk_id,
        "token_count": chunk.token_count,
    }
