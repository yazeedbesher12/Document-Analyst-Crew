import pytest

from greenloop_rag_crew.rag import hybrid_retriever as hybrid_module
from greenloop_rag_crew.rag.hybrid_retriever import HybridRetriever, fuse_results
from greenloop_rag_crew.rag.schemas import BM25SearchResult, DenseSearchResult


def _dense(chunk_id: str, rank: int, score: float = 0.9) -> DenseSearchResult:
    return DenseSearchResult(
        rank=rank,
        chunk_id=chunk_id,
        source="source.pdf",
        document_id="DOC",
        title="Title",
        page=rank,
        section="Section",
        distance=1.0 - score,
        score=score,
        text=f"text {chunk_id}",
    )


def _bm25(chunk_id: str, rank: int, score: float = 4.0) -> BM25SearchResult:
    return BM25SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        source="source.pdf",
        document_id="DOC",
        title="Title",
        page=rank,
        section="Section",
        bm25_score=score,
        text=f"text {chunk_id}",
    )


class FakeDenseRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k=5, document_id=None):
        self.calls.append((query, top_k, document_id))
        return [_dense("both", 1), _dense("dense-only", 2)]


class FakeBM25Retriever:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k=5, document_id=None):
        self.calls.append((query, top_k, document_id))
        return [_bm25("both", 1), _bm25("bm25-only", 2)]


def test_weighted_rrf_formula_missing_candidates_and_one_based_ranks():
    results = fuse_results(
        dense_results=[_dense("both", 1, 0.8), _dense("dense-only", 2, 0.7)],
        bm25_results=[_bm25("both", 3, 5.0), _bm25("bm25-only", 1, 6.0)],
        top_k=3,
        dense_weight=0.25,
        bm25_weight=0.75,
        rrf_k=10,
    )

    by_id = {result.chunk_id: result for result in results}
    expected_both = 0.25 / 11 + 0.75 / 13
    assert by_id["both"].rank == 1
    assert by_id["both"].dense_rank == 1
    assert by_id["both"].bm25_rank == 3
    assert by_id["both"].matched_by == ["dense", "bm25"]
    assert by_id["both"].fusion_score == pytest.approx(expected_both)
    assert by_id["dense-only"].bm25_rank is None
    assert by_id["bm25-only"].dense_rank is None


def test_fusion_deterministic_tie_breaking():
    results = fuse_results(
        dense_results=[_dense("b", 1), _dense("a", 1)],
        bm25_results=[],
        top_k=2,
        dense_weight=1.0,
        bm25_weight=0.0,
        rrf_k=60,
    )

    assert [result.chunk_id for result in results] == ["a", "b"]


def test_hybrid_search_normalizes_weights_and_passes_filters(monkeypatch):
    monkeypatch.setattr(HybridRetriever, "_verify_compatible_indexes", lambda self: None)
    dense = FakeDenseRetriever()
    bm25 = FakeBM25Retriever()
    retriever = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = retriever.search(
        "policy",
        top_k=2,
        candidate_k=4,
        document_id="DOC",
        dense_weight=2.0,
        bm25_weight=2.0,
    )

    assert dense.calls == [("policy", 4, "DOC")]
    assert bm25.calls == [("policy", 4, "DOC")]
    assert results[0].chunk_id == "both"
    assert results[0].matched_by == ["dense", "bm25"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "", "top_k": 1, "candidate_k": 1},
        {"query": "x", "top_k": 0, "candidate_k": 1},
        {"query": "x", "top_k": 2, "candidate_k": 1},
        {"query": "x", "top_k": 1, "candidate_k": 1, "dense_weight": -1},
        {"query": "x", "top_k": 1, "candidate_k": 1, "dense_weight": 0, "bm25_weight": 0},
        {"query": "x", "top_k": 1, "candidate_k": 1, "rrf_k": 0},
    ],
)
def test_hybrid_invalid_arguments(monkeypatch, kwargs):
    monkeypatch.setattr(HybridRetriever, "_verify_compatible_indexes", lambda self: None)
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        bm25_retriever=FakeBM25Retriever(),
    )
    with pytest.raises(ValueError):
        retriever.search(**kwargs)


def test_hybrid_mismatched_manifest_detection(monkeypatch):
    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def count(self):
            return 999

    monkeypatch.setattr(hybrid_module, "read_manifest", lambda: {"created_at": "now"})
    monkeypatch.setattr(hybrid_module, "ChromaStore", FakeStore)
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        bm25_retriever=FakeBM25Retriever(),
    )

    with pytest.raises(ValueError, match="Lexical chunks and dense index do not match"):
        retriever._verify_compatible_indexes()
