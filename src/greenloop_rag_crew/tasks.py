"""CrewAI Task construction for the GreenLoop sequential workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crewai import Task

from greenloop_rag_crew.agents import AgentBundle
from greenloop_rag_crew.config_loader import TaskConfig, load_tasks_config


@dataclass(frozen=True)
class TaskBundle:
    """Final task objects in sequential execution order."""

    research_task: Task
    fact_check_task: Task
    report_task: Task


def create_tasks(
    agents: AgentBundle,
    output_path: str | Path,
    config_path: str | Path | None = None,
) -> TaskBundle:
    """Construct the three final CrewAI tasks and their context wiring."""

    config = load_tasks_config(config_path=config_path)
    output_file = _prepare_output_path(output_path)
    agent_by_key = {
        "document_researcher": agents.document_researcher,
        "fact_checker": agents.fact_checker,
        "report_writer": agents.report_writer,
    }

    research_task = _create_task(
        config.research_task,
        agent=agent_by_key[config.research_task.agent],
        context=[],
    )
    fact_check_task = _create_task(
        config.fact_check_task,
        agent=agent_by_key[config.fact_check_task.agent],
        context=[research_task],
    )
    report_task = _create_task(
        config.report_task,
        agent=agent_by_key[config.report_task.agent],
        context=[research_task, fact_check_task],
        output_file=str(output_file),
    )

    return TaskBundle(
        research_task=research_task,
        fact_check_task=fact_check_task,
        report_task=report_task,
    )


def _create_task(
    task_config: TaskConfig,
    agent,
    context: list[Task],
    output_file: str | None = None,
) -> Task:
    kwargs = {
        "description": task_config.description,
        "expected_output": task_config.expected_output,
        "agent": agent,
        "context": context,
    }
    if output_file is not None:
        kwargs["output_file"] = output_file
    return Task(**kwargs)


def _prepare_output_path(output_path: str | Path) -> Path:
    if output_path is None:
        raise ValueError("output_path must not be empty.")
    raw = str(output_path).strip()
    if not raw:
        raise ValueError("output_path must not be empty.")
    if "\x00" in raw:
        raise ValueError("output_path must not contain null bytes.")

    path = Path(raw)
    if any(part == ".." for part in path.parts):
        raise ValueError("output_path must not contain parent-directory traversal.")
    if path.exists() and path.is_dir():
        raise ValueError("output_path must be a file path, not a directory.")
    if path.name in {"", ".", ".."}:
        raise ValueError("output_path must include a filename.")

    parent = path.parent if str(path.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    return path
