from pathlib import Path

import pytest
from crewai import LLM, Process

from greenloop_rag_crew.crew import create_crew
from greenloop_rag_crew.tasks import create_tasks
from greenloop_rag_crew.agents import create_agents
from greenloop_rag_crew.tools import DocumentSearchTool


def _safe_llm() -> LLM:
    return LLM(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.6,
        top_p=0.95,
        timeout=1,
        max_tokens=1000,
    )


def test_create_tasks_constructs_three_tasks_with_correct_agents(tmp_path):
    agents = create_agents(llm=_safe_llm())
    tasks = create_tasks(agents=agents, output_path=tmp_path / "report.md")

    assert tasks.research_task.agent is agents.document_researcher
    assert tasks.fact_check_task.agent is agents.fact_checker
    assert tasks.report_task.agent is agents.report_writer


def test_task_context_is_connected_correctly(tmp_path):
    bundle = create_crew(llm=_safe_llm(), output_path=tmp_path / "report.md")

    assert bundle.tasks.research_task.context == []
    assert bundle.tasks.fact_check_task.context == [bundle.tasks.research_task]
    assert bundle.tasks.report_task.context == [
        bundle.tasks.research_task,
        bundle.tasks.fact_check_task,
    ]


def test_report_task_receives_output_file(tmp_path):
    output_path = tmp_path / "nested" / "report.md"
    bundle = create_crew(llm=_safe_llm(), output_path=output_path)

    assert Path(bundle.tasks.report_task.output_file) == output_path
    assert output_path.parent.exists()
    assert bundle.tasks.research_task.output_file is None
    assert bundle.tasks.fact_check_task.output_file is None


def test_crew_process_order_and_tools(tmp_path):
    bundle = create_crew(llm=_safe_llm(), output_path=tmp_path / "report.md")

    assert len(bundle.crew.agents) == 3
    assert len(bundle.crew.tasks) == 3
    assert bundle.crew.agents == [
        bundle.agents.document_researcher,
        bundle.agents.fact_checker,
        bundle.agents.report_writer,
    ]
    assert bundle.crew.tasks == [
        bundle.tasks.research_task,
        bundle.tasks.fact_check_task,
        bundle.tasks.report_task,
    ]
    assert bundle.crew.process == Process.sequential
    assert [tool.name for tool in bundle.agents.document_researcher.tools] == [
        "document_search"
    ]
    assert [tool.name for tool in bundle.agents.fact_checker.tools] == [
        "document_search"
    ]
    assert bundle.agents.report_writer.tools == []


def test_no_web_tools_exist(tmp_path):
    bundle = create_crew(llm=_safe_llm(), output_path=tmp_path / "report.md")
    tool_names = [
        tool.name
        for agent in bundle.crew.agents
        for tool in agent.tools
    ]

    assert tool_names == ["document_search", "document_search"]


def test_construction_does_not_invoke_ollama_or_document_search(monkeypatch, tmp_path):
    llm = _safe_llm()
    monkeypatch.setattr(llm, "call", lambda *args, **kwargs: pytest.fail("LLM was called"))
    monkeypatch.setattr(
        DocumentSearchTool,
        "_get_retriever",
        lambda self: pytest.fail("document_search retrieval was initialized"),
    )

    bundle = create_crew(llm=llm, output_path=tmp_path / "report.md")

    assert bundle.agents.document_search_tool._retriever is None


@pytest.mark.parametrize("bad_path", ["", "   ", "../report.md"])
def test_unsafe_output_paths_are_rejected(bad_path):
    agents = create_agents(llm=_safe_llm())

    with pytest.raises(ValueError, match="output_path"):
        create_tasks(agents=agents, output_path=bad_path)
