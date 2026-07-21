"""Standalone dense retrieval over the local Chroma index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from greenloop_rag_crew.rag.build_index import DEFAULT_CHUNKS_FILE, build_manifest, load_chunks, read_manifest
from greenloop_rag_crew.rag.chroma_store import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_PERSIST_DIRECTORY,
    ChromaStore,
)
from greenloop_rag_crew.rag.embedder import GreenLoopEmbedder, get_cached_embedder
from greenloop_rag_crew.rag.schemas import DenseSearchResult
from greenloop_rag_crew.runtime_paths import chroma_persist_dir, chunks_file as configured_chunks_file


class DenseRetriever:
    """Reusable dense retriever for the standalone RAG preparation index."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        chunks_file: str | Path | None = None,
        embedder: GreenLoopEmbedder | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir is not None else chroma_persist_dir()
        self.collection_name = collection_name
        self.chunks_file = Path(chunks_file) if chunks_file is not None else configured_chunks_file()
        self.embedder = embedder or get_cached_embedder()
        self.store = ChromaStore(persist_dir=self.persist_dir, collection_name=collection_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[DenseSearchResult]:
        self._verify_index()
        query_embedding = self.embedder.embed_query(query)
        raw = self.store.query(query_embedding, top_k=top_k, document_id=document_id)
        return _format_results(raw)

    def _verify_index(self) -> None:
        manifest = read_manifest()
        if manifest is None:
            raise FileNotFoundError("Index manifest not found. Run build_index first.")
        if manifest.get("collection_name") != self.collection_name:
            raise ValueError(
                f"Manifest collection {manifest.get('collection_name')!r} does not match "
                f"configured collection {self.collection_name!r}."
            )
        if manifest.get("embedding_model") != self.embedder.model_name:
            raise ValueError(
                f"Manifest embedding model {manifest.get('embedding_model')!r} does not "
                f"match configured model {self.embedder.model_name!r}."
            )

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
            raise ValueError("Index manifest is stale. Run build_index again.")
        if self.store.count() != manifest["chunk_count"]:
            raise ValueError("Chroma collection count does not match the manifest.")


def _format_results(raw: dict) -> list[DenseSearchResult]:
    results: list[DenseSearchResult] = []
    ids = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for index, chunk_id in enumerate(ids, start=1):
        metadata = metadatas[index - 1]
        distance = float(distances[index - 1])
        results.append(
            DenseSearchResult(
                rank=index,
                chunk_id=chunk_id,
                source=str(metadata["source"]),
                document_id=str(metadata["document_id"]),
                title=str(metadata["title"]),
                page=int(metadata["page"]),
                section=str(metadata["section"]),
                distance=distance,
                score=1.0 - distance,
                text=documents[index - 1],
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--collection", default=DEFAULT_CHROMA_COLLECTION)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    retriever = DenseRetriever(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedder=GreenLoopEmbedder(),
    )
    results = retriever.search(args.query, top_k=args.top_k, document_id=args.document_id)

    if args.json:
        print(json.dumps([result.model_dump() for result in results], ensure_ascii=False))
        return

    for result in results:
        preview = " ".join(result.text.split())[:220]
        print(
            f"{result.rank}. score={result.score:.4f} "
            f"source={result.source} page={result.page} section={result.section}"
        )
        print(f"   chunk_id={result.chunk_id}")
        print(f"   {preview}")


if __name__ == "__main__":
    main()
