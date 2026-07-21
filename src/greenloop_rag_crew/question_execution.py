"""Shared execution helpers for interactive GreenLoop document questions."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from greenloop_rag_crew.crew import create_crew
from greenloop_rag_crew.diagnostics.ollama_tool_check import check_ollama_health
from greenloop_rag_crew.llm import LLMConfigurationError, LLMSettings, get_provider_settings
from greenloop_rag_crew.runtime_paths import output_dir as configured_output_dir
from greenloop_rag_crew.rag.retrieval_service import RetrievalService, get_retrieval_service
from greenloop_rag_crew.timing import RequestTiming

LOGGER = logging.getLogger(__name__)

OLLAMA_PREFLIGHT_TIMEOUT_SECONDS = 5.0


class QuestionValidationError(ValueError):
    """Raised when an interactive question is invalid."""


class QuestionExecutionError(RuntimeError):
    """Raised with a safe message when a crew run cannot complete."""


class OllamaConnectionError(QuestionExecutionError):
    """Raised when the local Ollama API cannot be reached."""


class OllamaModelMissingError(QuestionExecutionError):
    """Raised when the configured local Ollama model is unavailable."""


class RetrievalError(QuestionExecutionError):
    """Raised when the local retrieval or Chroma layer fails."""


class CrewExecutionError(QuestionExecutionError):
    """Raised when CrewAI cannot complete an otherwise available run."""


class UnexpectedExecutionError(QuestionExecutionError):
    """Raised for unexpected failures without exposing implementation details."""


class ProviderConfigurationError(QuestionExecutionError):
    """Raised when the selected LLM provider lacks required runtime settings."""


@dataclass(frozen=True)
class QuestionExecutionResult:
    """The safe, user-facing output from one fresh CrewAI run."""

    question: str
    output_path: Path
    report_markdown: str
    timings: dict[str, float]
    llm_calls: int | None
    retrieval_calls: int


def validate_question(question: str) -> str:
    """Return a non-empty question without exposing implementation details."""

    normalized = question.strip() if isinstance(question, str) else ""
    if not normalized:
        raise QuestionValidationError("Please enter a question before analyzing documents.")
    return normalized


def create_unique_report_path(
    question: str,
    output_dir: Path | None = None,
) -> Path:
    """Create a readable, collision-resistant Markdown output path."""

    normalized = validate_question(question)
    slug = re.sub(r"[^\w]+", "-", normalized.lower(), flags=re.UNICODE).strip("-")
    slug = slug[:60].rstrip("-") or "question"
    destination = output_dir if output_dir is not None else configured_output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    return destination / f"greenloop_report_{slug}_{uuid4().hex[:12]}.md"


def execute_question(
    question: str,
    output_dir: Path | None = None,
    crew_factory: Callable[..., Any] = create_crew,
    retrieval_service: RetrievalService | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> QuestionExecutionResult:
    """Run one fresh crew and return its Markdown without internal errors."""

    normalized = validate_question(question)
    output_path = create_unique_report_path(normalized, output_dir=output_dir)
    timing = RequestTiming()
    _notify_progress(progress_callback, "Preparing document index", 0.0)
    timing.begin("application_initialization")
    try:
        retrieval_service = retrieval_service or get_retrieval_service()
        check_llm_preflight()
    except Exception:
        timing.finish("application_initialization")
        timing.finish_request()
        raise
    initialization_elapsed = timing.finish("application_initialization")
    _notify_progress(progress_callback, "Researching documents", initialization_elapsed)
    retrieval_calls_before, retrieval_seconds_before = retrieval_service.metrics_snapshot()

    try:
        bundle = crew_factory(output_path=output_path)
        _attach_task_timing(bundle, timing, progress_callback)
        timing.begin("researcher_execution")
        timing.begin("total_crew_execution")
        kickoff_result = bundle.crew.kickoff(inputs={"question": normalized})
        timing.finish("researcher_execution")
        timing.finish("fact_checker_execution")
        timing.finish("report_writer_execution")
        timing.finish("total_crew_execution")
        report_markdown = _read_or_write_report(output_path, kickoff_result)
    except Exception as exc:
        timing.finish("researcher_execution")
        timing.finish("fact_checker_execution")
        timing.finish("report_writer_execution")
        timing.finish("total_crew_execution")
        LOGGER.exception("CrewAI document analysis failed.")
        raise _classify_execution_error(exc) from exc
    finally:
        timing.finish_request()

    _notify_progress(
        progress_callback,
        "Completed",
        timing.durations.get("report_writer_execution", 0.0),
    )
    retrieval_calls_after, retrieval_seconds_after = retrieval_service.metrics_snapshot()
    timing.durations["total_retrieval_execution"] = (
        retrieval_seconds_after - retrieval_seconds_before
    )

    return QuestionExecutionResult(
        question=normalized,
        output_path=output_path,
        report_markdown=report_markdown,
        timings=timing.snapshot(),
        llm_calls=_observed_llm_calls(bundle.crew),
        retrieval_calls=retrieval_calls_after - retrieval_calls_before,
    )


def check_ollama_preflight(timeout: float = OLLAMA_PREFLIGHT_TIMEOUT_SECONDS) -> None:
    """Verify the configured local Ollama server and qwen3:8b before kickoff."""

    try:
        health = check_ollama_health(timeout=timeout)
    except Exception as exc:
        LOGGER.exception("Ollama preflight request failed.")
        raise OllamaConnectionError(
            "Cannot reach the local Ollama service. Start Ollama and try again."
        ) from exc

    if not health.reachable:
        LOGGER.error("Ollama preflight failed: %s", health.error_type or "unreachable")
        raise OllamaConnectionError(
            "Cannot reach the local Ollama service. Start Ollama and try again."
        )
    if not health.model_installed:
        LOGGER.error("Ollama preflight failed: model %s is missing.", health.model)
        raise OllamaModelMissingError(
            "The local Ollama model qwen3:8b is not installed. Run `ollama pull qwen3:8b` "
            "and try again."
        )


def check_llm_preflight() -> LLMSettings:
    """Validate the selected provider and health-check only local Ollama."""

    try:
        settings = get_provider_settings()
    except LLMConfigurationError as exc:
        LOGGER.error("LLM provider configuration failed: %s", exc)
        raise ProviderConfigurationError(str(exc)) from exc

    if settings.provider == "ollama":
        check_ollama_preflight()
    return settings


def _classify_execution_error(exc: Exception) -> QuestionExecutionError:
    message = _exception_text(exc).lower()

    if _contains_any(
        message,
        (
            "model not found",
            "qwen3:8b not found",
            "qwen3:8b does not exist",
        ),
    ):
        return OllamaModelMissingError(
            "The local Ollama model qwen3:8b is not installed. Run `ollama pull qwen3:8b` "
            "and try again."
        )
    if _contains_any(
        message,
        (
            "chroma",
            "rustbindingsapi",
            "default_tenant",
            "retrieval_error",
            "index_not_ready",
            "document retrieval",
            "collection count",
            "embedding model",
        ),
    ):
        return RetrievalError(
            "Local document retrieval is unavailable. Check the local index and try again."
        )
    if _contains_any(
        message,
        (
            "connection refused",
            "connection error",
            "localhost:11434",
            "ollama connection",
            "failed to connect",
        ),
    ):
        return OllamaConnectionError(
            "The local Ollama service became unavailable. Check Ollama and try again."
        )
    if isinstance(exc, (RuntimeError, ValueError, TimeoutError)):
        return CrewExecutionError(
            "The document analysis crew could not complete this request. Please try again."
        )
    return UnexpectedExecutionError(
        "An unexpected error occurred during document analysis. Please try again."
    )


def _exception_text(exc: Exception) -> str:
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(messages)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _read_or_write_report(output_path: Path, kickoff_result: Any) -> str:
    if output_path.exists():
        content = output_path.read_text(encoding="utf-8").strip()
        if content:
            return content

    report_markdown = str(getattr(kickoff_result, "raw", kickoff_result)).strip()
    if not report_markdown:
        raise RuntimeError("Crew returned an empty report.")

    output_path.write_text(report_markdown + "\n", encoding="utf-8")
    return report_markdown


def _attach_task_timing(
    bundle: Any,
    timing: RequestTiming,
    progress_callback: Callable[[str, float], None] | None,
) -> None:
    """Attach content-free task completion callbacks to a newly created Crew only."""

    tasks = getattr(bundle, "tasks", None)
    if tasks is None:
        return

    def researcher_complete(_output: Any) -> None:
        elapsed = timing.finish("researcher_execution")
        _notify_progress(progress_callback, "Verifying claims", elapsed)
        timing.begin("fact_checker_execution")

    def fact_checker_complete(_output: Any) -> None:
        elapsed = timing.finish("fact_checker_execution")
        _notify_progress(progress_callback, "Writing report", elapsed)
        timing.begin("report_writer_execution")

    def report_writer_complete(_output: Any) -> None:
        timing.finish("report_writer_execution")

    tasks.research_task.callback = researcher_complete
    tasks.fact_check_task.callback = fact_checker_complete
    tasks.report_task.callback = report_writer_complete


def _notify_progress(
    callback: Callable[[str, float], None] | None,
    stage: str,
    elapsed_seconds: float,
) -> None:
    if callback is not None:
        callback(stage, elapsed_seconds)


def _observed_llm_calls(crew: Any) -> int | None:
    """Return CrewAI's provider-reported request count when available."""

    usage_metrics = getattr(crew, "usage_metrics", None)
    successful_requests = getattr(usage_metrics, "successful_requests", None)
    return successful_requests if isinstance(successful_requests, int) else None
