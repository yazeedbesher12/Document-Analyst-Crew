from pathlib import Path

import pytest
from crewai import Agent, LLM

from greenloop_rag_crew.agents import create_agents
from greenloop_rag_crew.config_loader import ConfigError
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


def test_create_agents_constructs_three_agents_with_expected_roles():
    llm = _safe_llm()
    bundle = create_agents(llm=llm)

    agents = [
        bundle.document_researcher,
        bundle.fact_checker,
        bundle.report_writer,
    ]
    assert len(agents) == 3
    assert all(isinstance(agent, Agent) for agent in agents)
    assert bundle.document_researcher.role == "Document Researcher"
    assert bundle.fact_checker.role == "Fact Checker"
    assert bundle.report_writer.role == "Report Writer"


def test_all_agents_share_supplied_llm():
    llm = _safe_llm()
    bundle = create_agents(llm=llm)

    assert bundle.document_researcher.llm is llm
    assert bundle.fact_checker.llm is llm
    assert bundle.report_writer.llm is llm


def test_tool_assignment_uses_one_shared_document_search_tool():
    bundle = create_agents(llm=_safe_llm())

    assert len(bundle.document_researcher.tools) == 1
    assert len(bundle.fact_checker.tools) == 1
    assert len(bundle.report_writer.tools) == 0
    assert bundle.document_researcher.tools[0] is bundle.document_search_tool
    assert bundle.fact_checker.tools[0] is bundle.document_search_tool
    assert isinstance(bundle.document_search_tool, DocumentSearchTool)


def test_no_web_tool_exists():
    bundle = create_agents(llm=_safe_llm())
    tool_names = [
        tool.name
        for agent in [bundle.document_researcher, bundle.fact_checker, bundle.report_writer]
        for tool in agent.tools
    ]

    assert tool_names == ["document_search", "document_search"]


def test_construction_does_not_invoke_llm_or_retrieval(monkeypatch):
    llm = _safe_llm()
    monkeypatch.setattr(llm, "call", lambda *args, **kwargs: pytest.fail("LLM was called"))
    monkeypatch.setattr(
        DocumentSearchTool,
        "_get_retriever",
        lambda self: pytest.fail("retrieval was initialized"),
    )

    bundle = create_agents(llm=llm)

    assert bundle.document_search_tool._retriever is None


def test_unknown_custom_tool_fails_clearly(tmp_path):
    path = tmp_path / "agents.jsonc"
    path.write_text(
        """
        {
          document_researcher: {
            role: "Document Researcher",
            goal: "Search local evidence.",
            backstory: "Uses only local documents.",
            tools: ["custom:document_search", "custom:unknown"],
            allow_delegation: false,
            verbose: true,
            max_iter: 8,
          },
          fact_checker: {
            role: "Fact Checker",
            goal: "Verify claims.",
            backstory: "Checks local evidence.",
            tools: ["custom:document_search"],
            allow_delegation: false,
            verbose: true,
            max_iter: 10,
          },
          report_writer: {
            role: "Report Writer",
            goal: "Write from verified context.",
            backstory: "Never retrieves.",
            tools: [],
            allow_delegation: false,
            verbose: true,
            max_iter: 6,
          },
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Unknown tool"):
        create_agents(llm=_safe_llm(), config_path=path)
