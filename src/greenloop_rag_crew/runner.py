"""CLI runner helpers for the three final GreenLoop questions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from greenloop_rag_crew.config_loader import ConfigError, QuestionConfig, load_questions_config
from greenloop_rag_crew.crew import create_crew


@dataclass(frozen=True)
class RunResult:
    """Structured result for a question run or dry run."""

    question_id: str
    question: str
    output_path: str
    agent_order: list[str]
    task_order: list[str]
    process_type: str
    required_document_ids: list[str]
    dry_run: bool
    kickoff_result: Any | None = None


def list_questions() -> list[QuestionConfig]:
    """Return the configured final questions."""

    return load_questions_config().questions


def run_question(
    question_id: str,
    llm=None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> RunResult:
    """Create a fresh crew for one configured question and optionally run it."""

    question = _find_question(question_id)
    _ensure_output_available(question.output_file, overwrite=overwrite)
    bundle = create_crew(llm=llm, output_path=question.output_file)
    result = _to_run_result(question=question, bundle=bundle, dry_run=dry_run)
    if dry_run:
        return result

    kickoff_result = bundle.crew.kickoff(inputs={"question": question.question})
    return RunResult(**{**result.__dict__, "kickoff_result": kickoff_result})


def run_all(llm=None, overwrite: bool = False, dry_run: bool = False) -> list[RunResult]:
    """Create a fresh crew for each configured question."""

    results: list[RunResult] = []
    for question in list_questions():
        results.append(
            run_question(
                question.id,
                llm=llm,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )
    return results


def _find_question(question_id: str) -> QuestionConfig:
    for question in list_questions():
        if question.id == question_id:
            return question
    raise ConfigError(f"Unknown question_id {question_id!r}.")


def _ensure_output_available(output_file: str, overwrite: bool) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}. Use --overwrite to replace it."
        )


def _to_run_result(question: QuestionConfig, bundle, dry_run: bool) -> RunResult:
    return RunResult(
        question_id=question.id,
        question=question.question,
        output_path=question.output_file,
        agent_order=[agent.role for agent in bundle.crew.agents],
        task_order=["research_task", "fact_check_task", "report_task"],
        process_type=bundle.crew.process.value,
        required_document_ids=list(question.required_document_ids),
        dry_run=dry_run,
    )
