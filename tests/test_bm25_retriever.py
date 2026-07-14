import json

import pytest

from greenloop_rag_crew.rag.bm25_retriever import BM25Retriever
from greenloop_rag_crew.rag.build_index import ChunkValidationError, load_chunks
from greenloop_rag_crew.rag.tokenizer import tokenize


def test_tokenizer_preserves_business_numeric_forms():
    assert tokenize("Q2 and Q3") == ["q2", "and", "q3"]

    revenue_tokens = tokenize("Q3 revenue was $3.06M")
    assert "q3" in revenue_tokens
    assert "revenue" in revenue_tokens
    assert "$3.06m" in revenue_tokens
    assert "3.06m" in revenue_tokens
    assert "3.06" in revenue_tokens

    uptime_tokens = tokenize("99.72% dashboard uptime")
    assert "99.72%" in uptime_tokens
    assert "99.72" in uptime_tokens
    assert "dashboard" in uptime_tokens
    assert "uptime" in uptime_tokens

    hyphen_tokens = tokenize("remote-work policy")
    assert "remote-work" in hyphen_tokens
    assert "remote" in hyphen_tokens
    assert "work" in hyphen_tokens
    assert "policy" in hyphen_tokens


def test_tokenizer_is_deterministic_and_handles_empty_text():
    text = "X1 remote-work: 89.1% in Q3"
    assert tokenize(text) == tokenize(text)
    assert tokenize("") == []


def test_bm25_corpus_size_and_exact_numeric_matching():
    retriever = BM25Retriever()

    assert retriever.corpus_size == 78

    results = retriever.search("99.72% dashboard uptime", top_k=5)
    assert any(
        result.document_id == "FIN-Q3-2025-v1.0"
        and result.page == 14
        and "99.72%" in result.text
        for result in results
    )


def test_bm25_phrase_matching_document_filter_and_deterministic_ranking():
    retriever = BM25Retriever()

    first = retriever.search(
        "dashboard availability SLA target",
        top_k=5,
        document_id="PRD-GLX1-2025-v2.1",
    )
    second = retriever.search(
        "dashboard availability SLA target",
        top_k=5,
        document_id="PRD-GLX1-2025-v2.1",
    )

    assert [result.chunk_id for result in first] == [result.chunk_id for result in second]
    assert all(result.document_id == "PRD-GLX1-2025-v2.1" for result in first)
    assert first[0].page == 27
    assert "99.5%" in first[0].text


def test_bm25_rejects_duplicate_and_malformed_chunks(tmp_path):
    chunk = load_chunks("storage/chunks.jsonl")[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        json.dumps(chunk.model_dump()) + "\n" + json.dumps(chunk.model_dump()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ChunkValidationError):
        BM25Retriever(chunks_file=duplicate)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ChunkValidationError):
        BM25Retriever(chunks_file=malformed)


def test_bm25_empty_query_and_top_k_validation():
    retriever = BM25Retriever()

    assert retriever.search("!!!", top_k=5) == []
    with pytest.raises(ValueError):
        retriever.search("policy", top_k=0)
