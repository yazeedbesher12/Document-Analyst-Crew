import json

import pytest
from crewai.tools import BaseTool
from pydantic import ValidationError

from greenloop_rag_crew.rag.schemas import HybridSearchResult
from greenloop_rag_crew.tools import DocumentSearchTool
from greenloop_rag_crew.tools.document_search import DocumentSearchInput


def _result(rank: int = 1, chunk_id: str = "HR-HBK-2025-v1.4_p06_c01"):
    return HybridSearchResult(
        rank=rank,
        chunk_id=chunk_id,
        source="GreenLoop_Employee_Handbook_2025.pdf",
        document_id="HR-HBK-2025-v1.4",
        title="GreenLoop Employee Handbook 2025",
        page=6,
        section="Work modes and role eligibility",
        text="Full chunk text about software and AI employees working remotely up to three days.",
        fusion_score=0.0162,
        dense_rank=1,
        dense_score=0.7,
        bm25_rank=1,
        bm25_score=8.4,
        matched_by=["dense", "bm25"],
    )


class FakeHybridRetriever:
    instances = 0

    def __init__(self):
        FakeHybridRetriever.instances += 1
        self.calls = []

    def search(self, query, top_k=5, document_id=None):
        self.calls.append((query, top_k, document_id))
        return [_result()]


class EmptyHybridRetriever(FakeHybridRetriever):
    def search(self, query, top_k=5, document_id=None):
        self.calls.append((query, top_k, document_id))
        return []


class MissingIndexRetriever(FakeHybridRetriever):
    def search(self, query, top_k=5, document_id=None):
        raise FileNotFoundError("Index manifest not found. Run build_index first.")


class StaleIndexRetriever(FakeHybridRetriever):
    def search(self, query, top_k=5, document_id=None):
        raise ValueError("Lexical chunks and dense index do not match.")


class ExplodingRetriever(FakeHybridRetriever):
    def search(self, query, top_k=5, document_id=None):
        raise RuntimeError("secret stack details should stay out of tool output")


def test_tool_identity_and_schema():
    tool = DocumentSearchTool()

    assert tool.name == "document_search"
    assert isinstance(tool, BaseTool)
    assert tool.args_schema is DocumentSearchInput


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "ab"},
        {"query": "valid", "top_k": 0},
        {"query": "valid", "top_k": 11},
        {"query": "valid", "document_id": "UNKNOWN"},
    ],
)
def test_input_validation_rejects_invalid_values(payload):
    with pytest.raises(ValidationError):
        DocumentSearchInput.model_validate(payload)


@pytest.mark.parametrize(
    "document_id",
    ["HR-HBK-2025-v1.4", "PRD-GLX1-2025-v2.1", "FIN-Q3-2025-v1.0"],
)
def test_known_document_ids_are_accepted(document_id):
    model = DocumentSearchInput.model_validate(
        {"query": "remote work policy", "document_id": document_id}
    )

    assert model.document_id == document_id


def test_retriever_initializes_lazily_and_is_reused(monkeypatch):
    FakeHybridRetriever.instances = 0
    monkeypatch.setattr(
        "greenloop_rag_crew.tools.document_search.HybridRetriever",
        FakeHybridRetriever,
    )
    tool = DocumentSearchTool()

    assert tool._retriever is None
    first = json.loads(tool.run(query="remote work policy", top_k=5))
    second = json.loads(
        tool.run(
            query="dashboard SLA",
            top_k=3,
            document_id="HR-HBK-2025-v1.4",
        )
    )

    assert FakeHybridRetriever.instances == 1
    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert tool._retriever.calls == [
        ("remote work policy", 5, None),
        ("dashboard SLA", 3, "HR-HBK-2025-v1.4"),
    ]


def test_successful_output_is_valid_evidence_json(monkeypatch):
    monkeypatch.setattr(
        "greenloop_rag_crew.tools.document_search.HybridRetriever",
        FakeHybridRetriever,
    )
    payload = json.loads(DocumentSearchTool().run(query="remote work policy", top_k=5))

    assert payload["status"] == "ok"
    assert payload["retrieval_method"] == "hybrid_dense_bm25_rrf"
    assert payload["result_count"] == 1
    assert "not probabilities or factual-confidence scores" in payload["notice"]
    result = payload["results"][0]
    assert result["chunk_id"] == "HR-HBK-2025-v1.4_p06_c01"
    assert result["source"] == "GreenLoop_Employee_Handbook_2025.pdf"
    assert result["page"] == 6
    assert result["text"].startswith("Full chunk text")
    assert "embedding" not in result
    assert "embedding_text" not in result
    assert list(payload["results"][0].keys()).index("rank") == 0


def test_no_results_output(monkeypatch):
    monkeypatch.setattr(
        "greenloop_rag_crew.tools.document_search.HybridRetriever",
        EmptyHybridRetriever,
    )
    payload = json.loads(DocumentSearchTool().run(query="zzzz unmatched query", top_k=5))

    assert payload["status"] == "no_results"
    assert payload["result_count"] == 0
    assert payload["results"] == []


@pytest.mark.parametrize(
    ("retriever", "expected_type"),
    [
        (MissingIndexRetriever, "index_not_ready"),
        (StaleIndexRetriever, "index_not_ready"),
    ],
)
def test_index_errors_are_actionable(monkeypatch, retriever, expected_type):
    monkeypatch.setattr(
        "greenloop_rag_crew.tools.document_search.HybridRetriever",
        retriever,
    )
    payload = json.loads(DocumentSearchTool().run(query="remote work policy", top_k=5))

    assert payload["status"] == "error"
    assert payload["error_type"] == expected_type
    assert "build_chunks" in payload["message"]
    assert "build_index" in payload["message"]
    assert "Traceback" not in payload["message"]


def test_internal_exception_does_not_expose_stack_trace(monkeypatch):
    monkeypatch.setattr(
        "greenloop_rag_crew.tools.document_search.HybridRetriever",
        ExplodingRetriever,
    )
    payload = json.loads(DocumentSearchTool().run(query="remote work policy", top_k=5))

    assert payload["status"] == "error"
    assert payload["error_type"] == "retrieval_error"
    assert "Traceback" not in json.dumps(payload)
    assert "secret stack details" not in json.dumps(payload)


def test_tool_does_not_call_unrelated_services_when_retriever_is_mocked(monkeypatch):
    monkeypatch.setattr(
        "greenloop_rag_crew.tools.document_search.HybridRetriever",
        FakeHybridRetriever,
    )
    monkeypatch.setattr(
        "greenloop_rag_crew.rag.build_index.build_index",
        lambda *args, **kwargs: pytest.fail("tool must not rebuild index"),
    )
    tool = DocumentSearchTool()
    payload = json.loads(tool.run(query="remote work policy", top_k=5))

    assert payload["status"] == "ok"
