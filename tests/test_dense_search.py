import json
from pathlib import Path

import pytest

from greenloop_rag_crew.rag import build_index as build_index_module
from greenloop_rag_crew.rag.build_index import (
    ChunkValidationError,
    build_index,
    build_manifest,
    is_index_current,
    load_chunks,
)
from greenloop_rag_crew.rag.chroma_store import ChromaStore
from greenloop_rag_crew.rag.dense_search import DenseRetriever, _format_results
from greenloop_rag_crew.rag.embedder import EXPECTED_EMBEDDING_DIMENSION


def _write_chunks(path: Path, count: int = 3) -> list:
    chunks = load_chunks("storage/chunks.jsonl")[:count]
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
    return chunks


class FakeEmbedder:
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model_name") or "sentence-transformers/all-mpnet-base-v2"
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        vectors = []
        for index, _text in enumerate(texts):
            vector = [0.0] * EXPECTED_EMBEDDING_DIMENSION
            vector[index % EXPECTED_EMBEDDING_DIMENSION] = 1.0
            vectors.append(vector)
        return vectors


def test_load_chunks_rejects_malformed_and_duplicate_records(tmp_path):
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ChunkValidationError):
        load_chunks(malformed)

    duplicate = tmp_path / "duplicate.jsonl"
    chunk = load_chunks("storage/chunks.jsonl")[0]
    line = json.dumps(chunk.model_dump())
    duplicate.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ChunkValidationError):
        load_chunks(duplicate)


def test_manifest_matching_and_mismatch_detection(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(build_index_module, "MANIFEST_PATH", manifest_path)
    chunks = load_chunks("storage/chunks.jsonl")[:2]
    expected = build_manifest(chunks, "storage/chunks.jsonl", "manifest_collection")
    manifest_path.write_text(
        json.dumps({**expected, "created_at": "now"}, sort_keys=True),
        encoding="utf-8",
    )
    store = ChromaStore(tmp_path / "chroma", "manifest_collection")
    store.recreate_collection()
    store.add_chunks(
        chunks,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        batch_size=2,
    )

    assert is_index_current(expected, store)

    mismatched = {**expected, "collection_name": "other_collection"}
    assert not is_index_current(mismatched, store)


def test_idempotent_indexing_forced_rebuild_and_no_duplicates(tmp_path, monkeypatch):
    manifest_path = tmp_path / "index_manifest.json"
    chunks_file = tmp_path / "chunks.jsonl"
    chunks = _write_chunks(chunks_file, count=3)
    monkeypatch.setattr(build_index_module, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(build_index_module, "GreenLoopEmbedder", FakeEmbedder)

    first = build_index(
        chunks_file=chunks_file,
        persist_dir=tmp_path / "chroma",
        collection_name="idempotent_collection",
        batch_size=2,
        rebuild=True,
    )
    second = build_index(
        chunks_file=chunks_file,
        persist_dir=tmp_path / "chroma",
        collection_name="idempotent_collection",
        batch_size=2,
        rebuild=False,
    )
    third = build_index(
        chunks_file=chunks_file,
        persist_dir=tmp_path / "chroma",
        collection_name="idempotent_collection",
        batch_size=2,
        rebuild=True,
    )

    store = ChromaStore(tmp_path / "chroma", "idempotent_collection")
    assert first == "rebuilt"
    assert second == "skipped"
    assert third == "rebuilt"
    assert store.count() == len(chunks)


def test_score_calculation_is_one_minus_cosine_distance():
    raw = {
        "ids": [["chunk-1"]],
        "documents": [["document text"]],
        "metadatas": [
            [
                {
                    "source": "source.pdf",
                    "document_id": "DOC",
                    "title": "Title",
                    "page": 3,
                    "section": "Section",
                }
            ]
        ],
        "distances": [[0.25]],
    }

    result = _format_results(raw)[0]

    assert result.score == 0.75
    assert result.distance == 0.25


@pytest.fixture(scope="session")
def real_retriever():
    build_index(rebuild=False)
    return DenseRetriever()


@pytest.mark.parametrize(
    ("query", "expected_document", "expected_page", "expected_text", "top_k"),
    [
        (
            "What is the remote work policy for software and AI employees?",
            "HR-HBK-2025-v1.4",
            6,
            "up to three days per week",
            5,
        ),
        (
            "What macro accuracy did the Sorter X1 achieve in laboratory validation?",
            "PRD-GLX1-2025-v2.1",
            12,
            "93.2%",
            5,
        ),
        (
            "How did Q3 field macro accuracy compare with Q2?",
            "FIN-Q3-2025-v1.0",
            12,
            "89.1%",
            5,
        ),
        (
            "What is the dashboard availability SLA target?",
            "PRD-GLX1-2025-v2.1",
            27,
            "99.5%",
            5,
        ),
        (
            "What dashboard uptime was achieved during Q3?",
            "FIN-Q3-2025-v1.0",
            14,
            "99.72%",
            5,
        ),
        (
            "How much revenue was lost because of multi-item errors?",
            "FIN-Q3-2025-v1.0",
            13,
            "",
            8,
        ),
    ],
)
def test_real_dense_retrieval_finds_expected_pages(
    real_retriever, query, expected_document, expected_page, expected_text, top_k
):
    results = real_retriever.search(query, top_k=top_k)

    matching = [
        result
        for result in results
        if result.document_id == expected_document and result.page == expected_page
    ]

    assert matching, [(result.document_id, result.page, result.chunk_id) for result in results]
    if expected_text:
        assert expected_text in " ".join(result.text for result in matching)


def test_real_dense_retrieval_document_id_filter(real_retriever):
    results = real_retriever.search(
        "What is the dashboard availability SLA target?",
        top_k=5,
        document_id="PRD-GLX1-2025-v2.1",
    )

    assert results
    assert all(result.document_id == "PRD-GLX1-2025-v2.1" for result in results)
    assert any(result.page == 27 and "99.5%" in result.text for result in results)
