import json
import subprocess
import sys

import pytest

from greenloop_rag_crew.rag.bm25_retriever import BM25Retriever
from greenloop_rag_crew.rag.dense_search import DenseRetriever
from greenloop_rag_crew.rag.hybrid_retriever import HybridRetriever


@pytest.fixture(scope="session")
def hybrid_retriever():
    return HybridRetriever()


@pytest.mark.parametrize(
    ("query", "document_id", "page", "expected_text", "top_k"),
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
            "87.9%",
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
def test_real_hybrid_retrieval_finds_expected_pages(
    hybrid_retriever, query, document_id, page, expected_text, top_k
):
    results = hybrid_retriever.search(query, top_k=top_k, candidate_k=20)
    matching = [
        result
        for result in results
        if result.document_id == document_id and result.page == page
    ]

    assert matching, [(result.document_id, result.page, result.chunk_id) for result in results]
    if expected_text:
        joined = " ".join(result.text for result in matching)
        assert expected_text in joined


def test_real_hybrid_multi_document_accuracy_comparison(hybrid_retriever):
    results = hybrid_retriever.search(
        "Compare the Sorter X1 laboratory accuracy with Q3 field accuracy.",
        top_k=10,
        candidate_k=20,
    )
    found = {(result.document_id, result.page) for result in results}

    assert ("PRD-GLX1-2025-v2.1", 12) in found
    assert ("FIN-Q3-2025-v1.0", 12) in found


def test_real_hybrid_multi_document_sla_comparison(hybrid_retriever):
    results = hybrid_retriever.search(
        "Compare the dashboard SLA target with actual Q3 dashboard uptime.",
        top_k=10,
        candidate_k=20,
    )
    found = {(result.document_id, result.page) for result in results}

    assert ("PRD-GLX1-2025-v2.1", 27) in found
    assert ("FIN-Q3-2025-v1.0", 14) in found


def test_hybrid_preserves_candidates_from_dense_and_bm25_candidate_pools():
    query = "What is the dashboard availability SLA target?"
    dense = DenseRetriever().search(query, top_k=20)
    bm25 = BM25Retriever().search(query, top_k=20)
    hybrid = HybridRetriever().search(query, top_k=10, candidate_k=20)
    hybrid_ids = {result.chunk_id for result in hybrid}

    assert dense[0].chunk_id in hybrid_ids
    assert bm25[0].chunk_id in hybrid_ids


def test_hybrid_json_cli_outputs_valid_json():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "greenloop_rag_crew.rag.hybrid_search",
            "How much revenue was lost because of multi-item errors?",
            "--top-k",
            "8",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    payload = json.loads(completed.stdout)
    assert payload
    assert payload[0]["document_id"] == "FIN-Q3-2025-v1.0"
    assert payload[0]["page"] == 13
