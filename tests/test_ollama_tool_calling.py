import json

import pytest

from greenloop_rag_crew.diagnostics import ollama_tool_check as check_module
from greenloop_rag_crew.diagnostics.ollama_tool_check import RecordingDocumentSearchTool
from greenloop_rag_crew.tools.document_search import DocumentSearchTool


def test_tool_call_recorder_captures_calls(monkeypatch):
    def fake_run(self, query, top_k=5, document_id=None):
        return json.dumps(
            {
                "status": "ok",
                "results": [
                    {
                        "page": 6,
                        "chunk_id": "HR-HBK-2025-v1.4_p06_c01",
                    }
                ],
            }
        )

    monkeypatch.setattr(DocumentSearchTool, "_run", fake_run)
    tool = RecordingDocumentSearchTool()

    output = tool._run("remote work", top_k=3, document_id="HR-HBK-2025-v1.4")

    assert json.loads(output)["status"] == "ok"
    assert len(tool.call_records) == 1
    record = tool.call_records[0]
    assert record.query == "remote work"
    assert record.top_k == 3
    assert record.document_id == "HR-HBK-2025-v1.4"
    assert record.status == "ok"
    assert record.pages == [6]


def test_production_tool_remains_unchanged():
    tool = DocumentSearchTool()

    assert not hasattr(tool, "call_records")


@pytest.mark.ollama
def test_real_remote_policy_tool_call():
    health = check_module.check_ollama_health(timeout=2)
    if not health.reachable or not health.model_installed:
        pytest.skip(health.message or "Ollama qwen3:8b is not available.")

    result = check_module.run_remote_policy_smoke(verbose=False)

    assert result.passed, result.errors
    assert result.tool_calls
    assert any(6 in call.pages for call in result.tool_calls)
