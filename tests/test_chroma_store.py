from greenloop_rag_crew.rag.chroma_store import ChromaStore
from greenloop_rag_crew.rag.schemas import DocumentChunk


def _chunk(chunk_id: str, document_id: str = "DOC-1", page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        source="source.pdf",
        document_id=document_id,
        title="Source",
        page=page,
        section="Section",
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        embedding_text=f"embedding text for {chunk_id}",
        token_count=12,
    )


def test_chroma_persistence_count_metadata_and_get_by_id(tmp_path):
    store = ChromaStore(persist_dir=tmp_path, collection_name="test_collection")
    store.recreate_collection()
    chunks = [_chunk("DOC-1_p01_c01"), _chunk("DOC-1_p02_c01", page=2)]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], batch_size=1)

    assert store.count() == 2

    stored = store.get_by_id("DOC-1_p01_c01")
    assert stored is not None
    assert stored["document"] == "text for DOC-1_p01_c01"
    assert stored["metadata"]["chunk_id"] == "DOC-1_p01_c01"
    assert stored["metadata"]["page"] == 1

    reopened = ChromaStore(persist_dir=tmp_path, collection_name="test_collection")
    assert reopened.count() == 2


def test_chroma_query_document_id_filter(tmp_path):
    store = ChromaStore(persist_dir=tmp_path, collection_name="filter_collection")
    store.recreate_collection()
    chunks = [
        _chunk("DOC-1_p01_c01", document_id="DOC-1"),
        _chunk("DOC-2_p01_c01", document_id="DOC-2"),
    ]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], batch_size=2)

    result = store.query([1.0, 0.0, 0.0], top_k=2, document_id="DOC-2")

    assert result["ids"][0] == ["DOC-2_p01_c01"]
    assert result["metadatas"][0][0]["document_id"] == "DOC-2"
