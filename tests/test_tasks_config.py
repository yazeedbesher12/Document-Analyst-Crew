from pathlib import Path

import pytest

from greenloop_rag_crew.config_loader import ConfigError, load_tasks_config


VALID_TASKS_CONFIG = """
{
  // Comments and trailing commas are valid JSONC.
  research_task: {
    description: "Research this question: {question}",
    expected_output: "Research notes.",
    agent: "document_researcher",
    context: [],
  },
  fact_check_task: {
    description: "Check researcher claims.",
    expected_output: "Claim table.",
    agent: "fact_checker",
    context: ["research_task"],
  },
  report_task: {
    description: "Write final answer for: {question}",
    expected_output: "Markdown report.",
    agent: "report_writer",
    context: ["research_task", "fact_check_task"],
  },
}
"""


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "tasks.jsonc"
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_jsonc_loads():
    config = load_tasks_config()

    assert config.research_task.agent == "document_researcher"
    assert config.fact_check_task.context == ["research_task"]
    assert config.report_task.context == ["research_task", "fact_check_task"]


def test_default_tasks_keep_retrieval_errors_distinct_and_prioritize_corrections():
    config = load_tasks_config()

    assert "RETRIEVAL_ERROR must never become NOT_DISCLOSED" in config.fact_check_task.description
    assert "retrieval succeeded but relevant information is absent" in config.fact_check_task.description
    assert "corrected wording" in config.fact_check_task.expected_output
    assert "source filename, page, section, chunk_id" in config.fact_check_task.expected_output
    assert "Fact Check Task claim audit as the authority" in config.report_task.description
    assert "Answer only the original question" in config.report_task.description
    assert "under 250 words" in config.report_task.description
    assert "omit every adjacent" in config.report_task.description
    assert "Include all required citations before ending" in config.report_task.description


def test_valid_jsonc_with_comments_and_trailing_commas_loads(tmp_path):
    config = load_tasks_config(_write_config(tmp_path, VALID_TASKS_CONFIG))

    assert config.research_task.description == "Research this question: {question}"


def test_missing_task_fails(tmp_path):
    content = VALID_TASKS_CONFIG.replace("fact_check_task:", "missing_fact_check_task:")

    with pytest.raises(ConfigError, match="missing task key"):
        load_tasks_config(_write_config(tmp_path, content))


def test_extra_task_fails(tmp_path):
    content = VALID_TASKS_CONFIG.replace(
        "report_task:",
        'extra_task: {description:"x", expected_output:"x", agent:"report_writer", context:[]}, report_task:',
    )

    with pytest.raises(ConfigError, match="unexpected task key"):
        load_tasks_config(_write_config(tmp_path, content))


def test_invalid_agent_name_fails(tmp_path):
    content = VALID_TASKS_CONFIG.replace(
        'agent: "document_researcher"',
        'agent: "unknown_agent"',
        1,
    )

    with pytest.raises(ConfigError, match="unknown agent"):
        load_tasks_config(_write_config(tmp_path, content))


def test_invalid_context_reference_fails(tmp_path):
    content = VALID_TASKS_CONFIG.replace(
        'context: ["research_task"]',
        'context: ["unknown_task"]',
    )

    with pytest.raises(ConfigError, match="unknown context task"):
        load_tasks_config(_write_config(tmp_path, content))


def test_incorrect_task_dependency_fails(tmp_path):
    content = VALID_TASKS_CONFIG.replace(
        'context: ["research_task", "fact_check_task"]',
        'context: ["fact_check_task", "research_task"]',
    )

    with pytest.raises(ConfigError, match="report_task context"):
        load_tasks_config(_write_config(tmp_path, content))


def test_missing_question_placeholder_fails(tmp_path):
    content = VALID_TASKS_CONFIG.replace("{question}", "question", 1)

    with pytest.raises(ConfigError, match=r"\{question\}"):
        load_tasks_config(_write_config(tmp_path, content))
