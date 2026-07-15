from pathlib import Path

import pytest

from greenloop_rag_crew.config_loader import (
    ConfigError,
    load_agents_config,
)


VALID_CONFIG = """
{
  // JSONC comments are allowed.
  document_researcher: {
    role: "Document Researcher",
    goal: "Search local evidence.",
    backstory: "Uses only local documents.",
    tools: ["custom:document_search"],
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
"""


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "agents.jsonc"
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_jsonc_with_comments_and_trailing_commas_loads(tmp_path):
    config = load_agents_config(_write_config(tmp_path, VALID_CONFIG))

    assert config.document_researcher.role == "Document Researcher"
    assert config.fact_checker.max_iter == 10
    assert config.report_writer.tools == []


def test_malformed_jsonc_fails_clearly(tmp_path):
    path = _write_config(tmp_path, "{ document_researcher: ")

    with pytest.raises(ConfigError, match="Failed to parse agents config"):
        load_agents_config(path)


def test_missing_agent_fails(tmp_path):
    content = VALID_CONFIG.replace("fact_checker:", "missing_fact_checker:")

    with pytest.raises(ConfigError, match="missing agent key"):
        load_agents_config(_write_config(tmp_path, content))


def test_extra_agent_fails(tmp_path):
    content = VALID_CONFIG.replace(
        "report_writer:",
        'extra_agent: {role:"x", goal:"x", backstory:"x", tools:[], allow_delegation:false, verbose:true, max_iter:1}, report_writer:',
    )

    with pytest.raises(ConfigError, match="unexpected agent key"):
        load_agents_config(_write_config(tmp_path, content))


def test_invalid_max_iter_fails(tmp_path):
    content = VALID_CONFIG.replace("max_iter: 8", "max_iter: 0", 1)

    with pytest.raises(ConfigError, match="max_iter"):
        load_agents_config(_write_config(tmp_path, content))


def test_web_search_tools_are_rejected(tmp_path):
    content = VALID_CONFIG.replace(
        'tools: ["custom:document_search"]',
        'tools: ["custom:document_search", "serper_dev_tool"]',
        1,
    )

    with pytest.raises(ConfigError, match="web-search tools are not allowed"):
        load_agents_config(_write_config(tmp_path, content))


def test_report_writer_cannot_have_tools(tmp_path):
    content = VALID_CONFIG.replace(
        "tools: [],",
        'tools: ["custom:document_search"],',
        1,
    )

    with pytest.raises(ConfigError, match="Report Writer must not have tools"):
        load_agents_config(_write_config(tmp_path, content))


def test_default_config_loads_outside_project_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = load_agents_config()

    assert config.document_researcher.role == "Document Researcher"
    assert config.fact_checker.role == "Fact Checker"
    assert config.report_writer.role == "Report Writer"
