from pathlib import Path

from greenloop_rag_crew.rag import retrieval_service as service_module
from greenloop_rag_crew.rag.embedder import clear_cached_embedders, get_cached_embedder
from greenloop_rag_crew.rag.retrieval_service import IndexPreparation


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


def test_cached_embedder_wrapper_is_reused_without_loading_weights(monkeypatch):
    clear_cached_embedders()
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-model")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")

    first = get_cached_embedder()
    second = get_cached_embedder()

    assert first is second
    assert first.model_loaded is False
