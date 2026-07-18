import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from greenloop_rag_crew.tools import DocumentSearchTool


def _search(query: str, document_id: str, top_k: int = 5) -> dict:
    output = DocumentSearchTool().run(
        query=query,
        document_id=document_id,
        top_k=top_k,
    )
    return json.loads(output)


def _page_result(payload: dict, page: int) -> dict:
    for result in payload["results"]:
        if result["page"] == page:
            return result
    raise AssertionError([(result["document_id"], result["page"]) for result in payload["results"]])


def test_remote_work_search_returns_page_6_evidence():
    payload = _search(
        "remote work policy eligibility and number of remote days",
        "HR-HBK-2025-v1.4",
    )

    assert payload["status"] == "ok"
    result = _page_result(payload, 6)
    assert "software and AI employees may work remotely up to three days per week" in result["text"]


def test_laboratory_accuracy_search_returns_page_12_evidence():
    payload = _search(
        "Sorter X1 laboratory macro accuracy",
        "PRD-GLX1-2025-v2.1",
    )

    result = _page_result(payload, 12)
    assert "93.2%" in result["text"]


def test_concurrent_document_searches_both_succeed():
    tool = DocumentSearchTool()
    searches = [
        ("remote work policy", "HR-HBK-2025-v1.4"),
        ("laboratory accuracy", "PRD-GLX1-2025-v2.1"),
    ]

    def run_search(search):
        query, document_id = search
        return json.loads(tool.run(query=query, document_id=document_id, top_k=5))

    with ThreadPoolExecutor(max_workers=2) as executor:
        remote_payload, accuracy_payload = executor.map(run_search, searches)

    assert remote_payload["status"] == "ok"
    assert _page_result(remote_payload, 6)["chunk_id"] == "HR-HBK-2025-v1.4_p06_c01"
    assert accuracy_payload["status"] == "ok"
    accuracy_result = _page_result(accuracy_payload, 12)
    assert accuracy_result["chunk_id"] == "PRD-GLX1-2025-v2.1_p12_c01"
    assert "93.2%" in accuracy_result["text"]


def test_q3_vs_q2_field_accuracy_search_returns_page_12_evidence():
    payload = _search(
        "Q3 field macro accuracy compared with Q2",
        "FIN-Q3-2025-v1.0",
    )

    result = _page_result(payload, 12)
    assert "89.1%" in result["text"]
    assert "87.9%" in result["text"]


def test_dashboard_sla_search_returns_page_27_evidence():
    payload = _search(
        "dashboard availability SLA target",
        "PRD-GLX1-2025-v2.1",
    )

    result = _page_result(payload, 27)
    assert "99.5%" in result["text"]


def test_actual_dashboard_uptime_search_returns_page_14_evidence():
    payload = _search(
        "actual dashboard uptime achieved during Q3",
        "FIN-Q3-2025-v1.0",
    )

    result = _page_result(payload, 14)
    assert "99.72%" in result["text"]


def test_missing_revenue_loss_amount_returns_page_13_without_invention():
    payload = _search(
        "exact revenue lost because of multi-item errors",
        "FIN-Q3-2025-v1.0",
        top_k=8,
    )

    result = _page_result(payload, 13)
    assert "No exact revenue-loss amount can be supported from this report." in result["text"]
    assert "15 Q3 multi-item tickets" in result["text"]
    assert "answer" not in payload


def test_document_search_does_not_modify_pdfs():
    pdfs = sorted(Path("knowledge").glob("*.pdf"))
    before = {path.name: _sha256(path) for path in pdfs}

    payload = _search("dashboard availability SLA target", "PRD-GLX1-2025-v2.1")

    after = {path.name: _sha256(path) for path in pdfs}
    assert payload["status"] == "ok"
    assert before == after


def test_tool_output_excludes_embeddings_and_paths():
    payload = _search("Sorter X1 laboratory macro accuracy", "PRD-GLX1-2025-v2.1")
    serialized = json.dumps(payload)

    assert "embedding_text" not in serialized
    assert "embeddings" not in serialized
    assert "D:\\\\" not in serialized
    assert "/Desktop/" not in serialized


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
