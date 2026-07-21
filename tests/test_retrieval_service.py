import threading
import time
from concurrent.futures import ThreadPoolExecutor

from greenloop_rag_crew.rag import embedder as embedder_module
from greenloop_rag_crew.rag import retrieval_service as service_module
from greenloop_rag_crew.rag.embedder import (
    GreenLoopEmbedder,
    clear_cached_embedders,
    get_cached_embedder,
)
from greenloop_rag_crew.rag.dense_search import DenseRetriever
from greenloop_rag_crew.rag.retrieval_service import IndexPreparation, RetrievalService
from greenloop_rag_crew.rag.schemas import HybridSearchResult


class FakeHybridRetriever:
    instances = 0

    def __init__(self, **_kwargs):
        type(self).instances += 1


def test_retrieval_service_is_reused_for_an_unchanged_runtime(monkeypatch):
    service_module.clear_retrieval_service_cache()
    FakeHybridRetriever.instances = 0
    preparations = []
    monkeypatch.setattr(service_module, "runtime_index_signature", lambda: "unchanged")
    monkeypatch.setattr(
        service_module,
        "ensure_index_ready",
        lambda: preparations.append(True)
        or IndexPreparation("loaded", "manifest_current", 0.01),
    )
    monkeypatch.setattr(service_module, "HybridRetriever", FakeHybridRetriever)

    first = service_module.get_retrieval_service()
    second = service_module.get_retrieval_service()

    assert first is second
    assert FakeHybridRetriever.instances == 1
    assert preparations == [True]


def test_changed_runtime_fingerprint_creates_a_fresh_service(monkeypatch):
    service_module.clear_retrieval_service_cache()
    FakeHybridRetriever.instances = 0
    signatures = iter(["before-pdf-change", "after-pdf-change"])
    preparations = []
    monkeypatch.setattr(service_module, "runtime_index_signature", lambda: next(signatures))
    monkeypatch.setattr(
        service_module,
        "ensure_index_ready",
        lambda: preparations.append(True)
        or IndexPreparation("rebuilt", "source_pdf_fingerprint_changed", 0.01),
    )
    monkeypatch.setattr(service_module, "HybridRetriever", FakeHybridRetriever)

    first = service_module.get_retrieval_service()
    second = service_module.get_retrieval_service()

    assert first is not second
    assert FakeHybridRetriever.instances == 2
    assert preparations == [True, True]


def test_manifest_detects_changed_pdf_fingerprint(monkeypatch, tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(service_module, "chunks_file", lambda: chunks)
    monkeypatch.setattr(service_module, "_embedding_model_name", lambda: "embedding-model")
    monkeypatch.setattr(service_module, "chunking_configuration", lambda: {"version": "current"})
    monkeypatch.setattr(
        service_module,
        "_pdf_fingerprints",
        lambda _directory: [{"filename": "source.pdf", "document_id": "DOC", "sha256": "new"}],
    )
    manifest = {
        "schema_version": service_module.INDEX_SCHEMA_VERSION,
        "embedding_model": "embedding-model",
        "chunking": {"version": "current"},
        "pdfs": [{"filename": "source.pdf", "document_id": "DOC", "sha256": "old"}],
    }

    assert service_module._chunk_rebuild_reason(manifest) == "source_pdf_fingerprint_changed"


def test_changed_pdf_fingerprint_rebuilds_chunks_and_index(monkeypatch, tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    old_fingerprints = [{"filename": "source.pdf", "document_id": "DOC", "sha256": "old"}]
    new_fingerprints = [{"filename": "source.pdf", "document_id": "DOC", "sha256": "new"}]
    manifest = {
        "schema_version": service_module.INDEX_SCHEMA_VERSION,
        "embedding_model": "embedding-model",
        "chunking": {"version": "current"},
        "pdfs": old_fingerprints,
    }
    calls = []

    monkeypatch.setattr(service_module, "chunks_file", lambda: chunks)
    monkeypatch.setattr(service_module, "knowledge_dir", lambda: tmp_path)
    monkeypatch.setattr(service_module, "chroma_persist_dir", lambda: tmp_path / "chroma")
    monkeypatch.setattr(service_module, "_embedding_model_name", lambda: "embedding-model")
    monkeypatch.setattr(service_module, "chunking_configuration", lambda: {"version": "current"})
    monkeypatch.setattr(service_module, "_pdf_fingerprints", lambda _directory: new_fingerprints)
    monkeypatch.setattr(service_module, "read_manifest", lambda: manifest)
    monkeypatch.setattr(
        service_module,
        "build_chunks",
        lambda **kwargs: calls.append(("build_chunks", kwargs)),
    )
    monkeypatch.setattr(
        service_module,
        "build_index",
        lambda **kwargs: calls.append(("build_index", kwargs)) or "rebuilt",
    )

    preparation = service_module.ensure_index_ready()

    assert preparation.action == "rebuilt"
    assert preparation.reason == "source_pdf_fingerprint_changed"
    assert calls == [
        ("build_chunks", {"knowledge_dir": tmp_path, "output": chunks}),
        (
            "build_index",
            {
                "chunks_file": chunks,
                "persist_dir": tmp_path / "chroma",
                "collection_name": service_module.DEFAULT_CHROMA_COLLECTION,
                "knowledge_dir": tmp_path,
                "rebuild": False,
            },
        ),
    ]


def test_manifest_with_unchanged_inputs_does_not_rebuild_chunks(monkeypatch, tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    fingerprints = [{"filename": "source.pdf", "document_id": "DOC", "sha256": "same"}]
    monkeypatch.setattr(service_module, "chunks_file", lambda: chunks)
    monkeypatch.setattr(service_module, "_embedding_model_name", lambda: "embedding-model")
    monkeypatch.setattr(service_module, "chunking_configuration", lambda: {"version": "current"})
    monkeypatch.setattr(service_module, "_pdf_fingerprints", lambda _directory: fingerprints)
    manifest = {
        "schema_version": service_module.INDEX_SCHEMA_VERSION,
        "embedding_model": "embedding-model",
        "chunking": {"version": "current"},
        "pdfs": fingerprints,
    }

    assert service_module._chunk_rebuild_reason(manifest) is None


def test_current_manifest_skips_chunking_and_chroma_rebuild(monkeypatch, tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    fingerprints = [{"filename": "source.pdf", "document_id": "DOC", "sha256": "same"}]
    manifest = {
        "schema_version": service_module.INDEX_SCHEMA_VERSION,
        "embedding_model": "embedding-model",
        "chunking": {"version": "current"},
        "pdfs": fingerprints,
    }
    calls = []

    monkeypatch.setattr(service_module, "chunks_file", lambda: chunks)
    monkeypatch.setattr(service_module, "knowledge_dir", lambda: tmp_path)
    monkeypatch.setattr(service_module, "chroma_persist_dir", lambda: tmp_path / "chroma")
    monkeypatch.setattr(service_module, "_embedding_model_name", lambda: "embedding-model")
    monkeypatch.setattr(service_module, "chunking_configuration", lambda: {"version": "current"})
    monkeypatch.setattr(service_module, "_pdf_fingerprints", lambda _directory: fingerprints)
    monkeypatch.setattr(service_module, "read_manifest", lambda: manifest)
    monkeypatch.setattr(
        service_module,
        "build_chunks",
        lambda **_kwargs: calls.append("build_chunks"),
    )

    def fake_build_index(**kwargs):
        calls.append(kwargs)
        return "skipped"

    monkeypatch.setattr(service_module, "build_index", fake_build_index)

    preparation = service_module.ensure_index_ready()

    assert preparation.action == "loaded"
    assert preparation.reason == "manifest_current"
    assert "build_chunks" not in calls
    assert calls == [
        {
            "chunks_file": chunks,
            "persist_dir": tmp_path / "chroma",
            "collection_name": service_module.DEFAULT_CHROMA_COLLECTION,
            "knowledge_dir": tmp_path,
            "rebuild": False,
        }
    ]


def test_cached_embedder_wrapper_is_reused_without_loading_weights(monkeypatch):
    clear_cached_embedders()
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-model")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")

    first = get_cached_embedder()
    second = get_cached_embedder()

    assert first is second
    assert first.model_loaded is False


def test_embedding_model_initializes_once_when_first_access_is_concurrent(monkeypatch):
    class FakeSentenceTransformer:
        instances = 0
        lock = threading.Lock()

        def __init__(self, *_args, **_kwargs):
            with self.lock:
                type(self).instances += 1
            time.sleep(0.02)

    embedder = GreenLoopEmbedder(model_name="fake-model", device="cpu", batch_size=2)
    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(embedder_module, "prepare_model_cache_dirs", lambda: ())

    with ThreadPoolExecutor(max_workers=2) as executor:
        models = list(executor.map(lambda _index: embedder.model, range(2)))

    assert models[0] is models[1]
    assert FakeSentenceTransformer.instances == 1


def test_dense_retriever_verifies_its_index_once_across_concurrent_searches(monkeypatch):
    retriever = object.__new__(DenseRetriever)
    retriever._index_verified = False
    retriever._verification_lock = threading.Lock()
    calls = []
    start = threading.Barrier(2)

    def verify_index(_self):
        calls.append(True)
        time.sleep(0.02)

    monkeypatch.setattr(DenseRetriever, "_verify_index", verify_index)

    def verify_once(_index):
        start.wait()
        retriever._ensure_index_verified()

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(verify_once, range(2)))

    assert calls == [True]


def test_retrieval_service_deduplicates_chunks_without_losing_citation_metadata():
    first = HybridSearchResult(
        rank=1,
        chunk_id="HR-HBK-2025-v1.4_p06_c01",
        source="GreenLoop_Employee_Handbook_2025.pdf",
        document_id="HR-HBK-2025-v1.4",
        title="GreenLoop Employee Handbook 2025",
        page=6,
        section="Work modes and role eligibility",
        text="Eligible software and AI employees may work remotely up to three days.",
        fusion_score=0.031,
        dense_rank=1,
        dense_score=0.92,
        bm25_rank=1,
        bm25_score=8.5,
        matched_by=["dense", "bm25"],
    )
    duplicate = first.model_copy(update={"rank": 2, "fusion_score": 0.02})
    second = first.model_copy(
        update={
            "rank": 3,
            "chunk_id": "FIN-Q3-2025-v1.0_p10_c01",
            "source": "GreenLoop_Q3_FY2025_Report.pdf",
            "document_id": "FIN-Q3-2025-v1.0",
            "page": 10,
            "section": "Financial performance",
            "fusion_score": 0.015,
        }
    )

    class DuplicateResultRetriever:
        def search(self, **_kwargs):
            return [first, duplicate, second]

    service = RetrievalService(
        retriever=DuplicateResultRetriever(),
        preparation=IndexPreparation("loaded", "manifest_current", 0.0),
    )

    results = service.search("remote work and revenue", top_k=6)

    assert [result.chunk_id for result in results] == [first.chunk_id, second.chunk_id]
    assert results[0].source == first.source
    assert results[0].page == first.page
    assert results[0].section == first.section
    assert results[0].fusion_score == first.fusion_score
