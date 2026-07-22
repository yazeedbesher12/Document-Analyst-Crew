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
from greenloop_rag_crew.fast_pipeline import (
    ExactAnswerCache,
    PipelineConfigurationError,
    pipeline_mode,
    run_fast_pipeline,
)
from greenloop_rag_crew.llm import (
    LLMConfigurationError,
    LLMSettings,
    get_generation_settings,
    get_provider_settings,
)
from greenloop_rag_crew.ollama_client import (
    GenerationResponse,
    generate_chat_stream,
)
from greenloop_rag_crew.openrouter_client import generate_chat_stream as generate_openrouter_chat_stream
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
    """The safe, user-facing output from one isolated request execution."""

    question: str
    output_path: Path
    report_markdown: str
    timings: dict[str, float]
    llm_calls: int | None
    retrieval_calls: int
    pipeline_mode: str
    metrics: dict[str, object]


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
    answer_generator: Callable[[str], GenerationResponse] | None = None,
    answer_cache: ExactAnswerCache | None = None,
    mode: str | None = None,
    token_callback: Callable[[str], None] | None = None,
) -> QuestionExecutionResult:
    """Run one fresh fast, strict, or legacy execution without leaking internals."""

    normalized = validate_question(question)
    output_path = create_unique_report_path(normalized, output_dir=output_dir)
    timing = RequestTiming()
    _notify_progress(progress_callback, "Preparing document index", 0.0)
    timing.begin("application_initialization")
    try:
        retrieval_service = retrieval_service or get_retrieval_service()
        llm_settings = check_llm_preflight()
        generation_settings = get_generation_settings()
        selected_mode = mode or pipeline_mode()
    except Exception:
        timing.finish("application_initialization")
        timing.finish_request()
        raise
    initialization_elapsed = timing.finish("application_initialization")
    _notify_progress(
        progress_callback,
        "Researching documents" if selected_mode == "legacy" else "Retrieving evidence",
        initialization_elapsed,
    )
    retrieval_calls_before, retrieval_seconds_before = retrieval_service.metrics_snapshot()

    try:
        if selected_mode == "legacy":
            report_markdown, llm_calls, metrics = _execute_legacy(
                normalized,
                output_path,
                crew_factory,
                timing,
                progress_callback,
            )
        else:
            timing.begin("retrieval_execution")

            def fast_progress(stage: str, elapsed_seconds: float) -> None:
                if stage == "Verifying citation metadata":
                    timing.finish("retrieval_execution")
                    timing.begin("citation_verification")
                elif stage == "Writing answer":
                    timing.finish("citation_verification")
                    timing.begin("answer_generation")
                _notify_progress(progress_callback, stage, elapsed_seconds)

            generator = answer_generator or _configured_answer_generator(
                llm_settings, generation_settings, token_callback=token_callback
            )
            fast_result = run_fast_pipeline(
                question=normalized,
                service=retrieval_service,
                llm_settings=llm_settings,
                generation_settings=generation_settings,
                generate=generator,
                mode=selected_mode,
                cache=answer_cache,
                progress_callback=fast_progress,
            )
            timing.finish("retrieval_execution")
            timing.finish("citation_verification")
            timing.finish("answer_generation")
            report_markdown = fast_result.report_markdown
            llm_calls = fast_result.llm_calls
            metrics = fast_result.metrics
            metrics["deterministic_verification_warnings"] = len(
                fast_result.verification.warnings
            )
            output_path.write_text(report_markdown, encoding="utf-8")
    except Exception as exc:
        timing.finish("retrieval_execution")
        timing.finish("citation_verification")
        timing.finish("answer_generation")
        timing.finish("researcher_execution")
        timing.finish("fact_checker_execution")
        timing.finish("report_writer_execution")
        timing.finish("total_crew_execution")
        LOGGER.exception("Document analysis execution failed.")
        raise _classify_execution_error(exc) from exc
    finally:
        timing.finish_request()

    completion_stage = "report_writer_execution" if selected_mode == "legacy" else "answer_generation"
    _notify_progress(progress_callback, "Completed", timing.durations.get(completion_stage, 0.0))
    retrieval_calls_after, retrieval_seconds_after = retrieval_service.metrics_snapshot()
    timing.durations["total_retrieval_execution"] = (
        retrieval_seconds_after - retrieval_seconds_before
    )
    LOGGER.info(
        "request_metrics pipeline_mode=%s retrieved_chunks=%s llm_request_count=%s "
        "answer_cache=%s first_token_seconds=%s generation_seconds=%s output_tokens=%s "
        "total_request_seconds=%.3f",
        selected_mode,
        metrics.get("retrieved_chunks", 0),
        llm_calls,
        metrics.get("answer_cache", "disabled"),
        metrics.get("time_to_first_token_seconds"),
        metrics.get("generation_seconds"),
        metrics.get("generated_output_tokens"),
        timing.durations.get("total_request_execution", 0.0),
    )

    return QuestionExecutionResult(
        question=normalized,
        output_path=output_path,
        report_markdown=report_markdown,
        timings=timing.snapshot(),
        llm_calls=llm_calls,
        retrieval_calls=retrieval_calls_after - retrieval_calls_before,
        pipeline_mode=selected_mode,
        metrics=metrics,
    )


def _execute_legacy(
    question: str,
    output_path: Path,
    crew_factory: Callable[..., Any],
    timing: RequestTiming,
    progress_callback: Callable[[str, float], None] | None,
) -> tuple[str, int | None, dict[str, object]]:
    """Run the retained multi-agent CrewAI workflow only when explicitly selected."""

    bundle = crew_factory(output_path=output_path)
    _attach_task_timing(bundle, timing, progress_callback)
    timing.begin("researcher_execution")
    timing.begin("total_crew_execution")
    kickoff_result = bundle.crew.kickoff(inputs={"question": question})
    timing.finish("researcher_execution")
    timing.finish("fact_checker_execution")
    timing.finish("report_writer_execution")
    timing.finish("total_crew_execution")
    return (
        _read_or_write_report(output_path, kickoff_result),
        _observed_llm_calls(bundle.crew),
        {"answer_cache": "disabled", "retrieved_chunks": 0},
    )


def _configured_answer_generator(
    settings: LLMSettings,
    generation_settings,
    *,
    token_callback: Callable[[str], None] | None = None,
) -> Callable[[str], GenerationResponse]:
    """Return one direct streaming call for the explicitly selected provider."""

    if settings.provider == "ollama":
        return lambda prompt: generate_chat_stream(
            settings=settings,
            generation=generation_settings,
            messages=[{"role": "user", "content": prompt}],
            on_token=token_callback or (lambda _token: None),
        )

    if settings.provider == "openrouter":
        return lambda prompt: generate_openrouter_chat_stream(
            settings=settings,
            generation=generation_settings,
            messages=[{"role": "user", "content": prompt}],
            on_token=token_callback or (lambda _token: None),
        )

    raise RuntimeError(f"Unsupported configured provider: {settings.provider}.")


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
