"""JSONC configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json5
from pydantic import BaseModel, ConfigDict, Field, field_validator

EXPECTED_AGENT_KEYS = frozenset(
    {"document_researcher", "fact_checker", "report_writer"}
)
EXPECTED_TASK_KEYS = frozenset({"research_task", "fact_check_task", "report_task"})
VALID_AGENT_KEYS = EXPECTED_AGENT_KEYS
DOCUMENT_SEARCH_TOOL = "custom:document_search"


class ConfigError(ValueError):
    """Raised when JSONC agent configuration is invalid."""


class AgentConfig(BaseModel):
    """Validated config for one CrewAI Agent."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)
    backstory: str = Field(..., min_length=1)
    tools: list[str]
    allow_delegation: bool
    verbose: bool
    max_iter: int = Field(..., gt=0)

    @field_validator("role", "goal", "backstory")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("tools")
    @classmethod
    def reject_web_tools(cls, value: list[str]) -> list[str]:
        for tool_name in value:
            lowered = tool_name.strip().lower()
            if not lowered:
                raise ValueError("tool names must not be blank")
            if _is_web_tool(lowered):
                raise ValueError(f"web-search tools are not allowed: {tool_name}")
        return [tool_name.strip() for tool_name in value]


class AgentsConfig(BaseModel):
    """Validated final three-agent configuration."""

    model_config = ConfigDict(extra="forbid")

    document_researcher: AgentConfig
    fact_checker: AgentConfig
    report_writer: AgentConfig


class TaskConfig(BaseModel):
    """Validated config for one CrewAI Task."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(..., min_length=1)
    expected_output: str = Field(..., min_length=1)
    agent: str = Field(..., min_length=1)
    context: list[str]

    @field_validator("description", "expected_output", "agent")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class TasksConfig(BaseModel):
    """Validated final three-task configuration."""

    model_config = ConfigDict(extra="forbid")

    research_task: TaskConfig
    fact_check_task: TaskConfig
    report_task: TaskConfig


def load_agents_config(config_path: str | Path | None = None) -> AgentsConfig:
    """Load and strictly validate the agents JSONC configuration."""

    path = Path(config_path) if config_path is not None else _default_agents_config_path()
    try:
        with path.open("r", encoding="utf-8") as file:
            raw_config: Any = json5.load(file)
    except Exception as exc:
        raise ConfigError(f"Failed to parse agents config {path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError("Agents config must be a JSON object.")

    keys = set(raw_config)
    missing = sorted(EXPECTED_AGENT_KEYS - keys)
    unexpected = sorted(keys - EXPECTED_AGENT_KEYS)
    if missing or unexpected:
        pieces = []
        if missing:
            pieces.append("missing agent key(s): " + ", ".join(missing))
        if unexpected:
            pieces.append("unexpected agent key(s): " + ", ".join(unexpected))
        raise ConfigError("Invalid agents config keys: " + "; ".join(pieces))

    try:
        config = AgentsConfig.model_validate(raw_config)
    except Exception as exc:
        raise ConfigError(f"Invalid agents config {path}: {exc}") from exc

    _validate_agent_tool_rules(config)
    return config


def load_tasks_config(config_path: str | Path | None = None) -> TasksConfig:
    """Load and strictly validate the tasks JSONC configuration."""

    path = Path(config_path) if config_path is not None else _default_tasks_config_path()
    try:
        with path.open("r", encoding="utf-8") as file:
            raw_config: Any = json5.load(file)
    except Exception as exc:
        raise ConfigError(f"Failed to parse tasks config {path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError("Tasks config must be a JSON object.")

    keys = set(raw_config)
    missing = sorted(EXPECTED_TASK_KEYS - keys)
    unexpected = sorted(keys - EXPECTED_TASK_KEYS)
    if missing or unexpected:
        pieces = []
        if missing:
            pieces.append("missing task key(s): " + ", ".join(missing))
        if unexpected:
            pieces.append("unexpected task key(s): " + ", ".join(unexpected))
        raise ConfigError("Invalid tasks config keys: " + "; ".join(pieces))

    try:
        config = TasksConfig.model_validate(raw_config)
    except Exception as exc:
        raise ConfigError(f"Invalid tasks config {path}: {exc}") from exc

    _validate_task_rules(config)
    return config


def _default_agents_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "agents.jsonc"


def _default_tasks_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "tasks.jsonc"


def _validate_agent_tool_rules(config: AgentsConfig) -> None:
    researcher_tools = config.document_researcher.tools
    checker_tools = config.fact_checker.tools
    writer_tools = config.report_writer.tools

    if DOCUMENT_SEARCH_TOOL not in researcher_tools:
        raise ConfigError("Document Researcher must include custom:document_search.")
    if DOCUMENT_SEARCH_TOOL not in checker_tools:
        raise ConfigError("Fact Checker must include custom:document_search.")
    if writer_tools:
        raise ConfigError("Report Writer must not have tools.")


def _is_web_tool(tool_name: str) -> bool:
    if tool_name == DOCUMENT_SEARCH_TOOL:
        return False
    blocked_terms = ("serper", "web-search", "web_search", "web search", "browser")
    return any(term in tool_name for term in blocked_terms)


def _validate_task_rules(config: TasksConfig) -> None:
    task_by_key = {
        "research_task": config.research_task,
        "fact_check_task": config.fact_check_task,
        "report_task": config.report_task,
    }

    for task_key, task_config in task_by_key.items():
        if task_config.agent not in VALID_AGENT_KEYS:
            raise ConfigError(
                f"{task_key} references unknown agent {task_config.agent!r}."
            )
        for context_key in task_config.context:
            if context_key not in EXPECTED_TASK_KEYS:
                raise ConfigError(
                    f"{task_key} references unknown context task {context_key!r}."
                )
            if context_key == task_key:
                raise ConfigError(f"{task_key} cannot reference itself as context.")

    if config.research_task.agent != "document_researcher":
        raise ConfigError("research_task must use document_researcher.")
    if config.fact_check_task.agent != "fact_checker":
        raise ConfigError("fact_check_task must use fact_checker.")
    if config.report_task.agent != "report_writer":
        raise ConfigError("report_task must use report_writer.")

    if config.research_task.context != []:
        raise ConfigError("research_task must have no context.")
    if config.fact_check_task.context != ["research_task"]:
        raise ConfigError('fact_check_task context must be exactly ["research_task"].')
    if config.report_task.context != ["research_task", "fact_check_task"]:
        raise ConfigError(
            'report_task context must be exactly ["research_task", "fact_check_task"].'
        )

    if "{question}" not in config.research_task.description:
        raise ConfigError("research_task description must include {question}.")
    if "{question}" not in config.report_task.description:
        raise ConfigError("report_task description must include {question}.")
