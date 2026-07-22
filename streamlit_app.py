"""Streamlit interface for local GreenLoop document analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import streamlit as st

from greenloop_rag_crew.question_execution import (
    CrewExecutionError,
    OllamaConnectionError,
    OllamaModelMissingError,
    ProviderConfigurationError,
    QuestionExecutionError,
    QuestionValidationError,
    RetrievalError,
    UnexpectedExecutionError,
    execute_question,
)
from greenloop_rag_crew.llm import LLMConfigurationError, get_provider_settings
from greenloop_rag_crew.rag.retrieval_service import RetrievalService, get_retrieval_service
from greenloop_rag_crew.streamlit_secrets import apply_streamlit_secrets

LOGGER = logging.getLogger(__name__)
INITIAL_STAGE = "Preparing document index"


@dataclass
class _StageProgress:
    """Render public workflow stages without exposing request content."""

    status: Any
    current_stage: str = INITIAL_STAGE
    writing_started_at: float | None = None

    def advance(self, next_stage: str, elapsed_seconds: float) -> None:
        """Record the completed public stage before showing the next one."""

        if next_stage == self.current_stage:
            return
        elapsed = max(0.0, elapsed_seconds)
        self.status.write(f"{self.current_stage}: {elapsed:.1f}s")
        self.current_stage = next_stage
        if next_stage != "Completed":
            self.status.update(label=next_stage, state="running")
        if next_stage == "Writing answer":
            self.writing_started_at = perf_counter()

    def complete(self, total_elapsed_seconds: float) -> None:
        """Finish the public status after all workflow stages have been shown."""

        if self.current_stage != "Completed":
            self.advance("Completed", 0.0)
        self.status.update(
            label=f"Completed ({max(0.0, total_elapsed_seconds):.1f}s)",
            state="complete",
        )

    def fail(self, label: str) -> None:
        """Mark the public status as failed without rendering internal details."""

        self.status.update(label=label, state="error")

    def update_writing_elapsed(self) -> None:
        """Keep the public writing timer moving as streamed answer tokens arrive."""

        if self.current_stage == "Writing answer" and self.writing_started_at is not None:
            elapsed = perf_counter() - self.writing_started_at
            self.status.update(label=f"Writing answer ({elapsed:.1f}s)", state="running")


@dataclass
class _StreamingAnswer:
    """Render final-answer tokens immediately without showing internal model data."""

    placeholder: Any
    progress: _StageProgress
    parts: list[str] = field(default_factory=list)

    def push(self, token: str) -> None:
        self.parts.append(token)
        self.placeholder.markdown("".join(self.parts) + "\n\n...")
        self.progress.update_writing_elapsed()

    def replace_with_complete(self, markdown: str) -> None:
        self.placeholder.markdown(markdown)


def _generation_metrics_text(metrics: dict[str, object], llm_calls: int | None) -> str:
    """Format safe generation metrics for the UI without exposing prompts or reasoning."""

    first_token = metrics.get("time_to_first_token_seconds")
    duration = metrics.get("generation_seconds")
    tokens = metrics.get("generated_output_tokens")
    tokens_per_second = metrics.get("tokens_per_second")
    loaded = metrics.get("model_already_loaded")
    loaded_label = "unknown" if loaded is None else ("yes" if loaded else "no")
    parts = [f"LLM requests: {llm_calls if llm_calls is not None else 'unknown'}"]
    if isinstance(first_token, (int, float)):
        parts.append(f"First token: {first_token:.1f}s")
    if isinstance(duration, (int, float)):
        parts.append(f"Writing: {duration:.1f}s")
    if isinstance(tokens, int):
        parts.append(f"Generated tokens: {tokens}")
    if isinstance(tokens_per_second, (int, float)):
        parts.append(f"Tokens/s: {tokens_per_second:.1f}")
    parts.append(f"Model already loaded: {loaded_label}")
    return " | ".join(parts)


@st.cache_resource(show_spinner=False)
def _streamlit_retrieval_service() -> RetrievalService:
    """Keep only reusable local retrieval state across Streamlit reruns."""

    return get_retrieval_service()


def main() -> None:
    """Render the local-document Streamlit interface."""

    st.set_page_config(page_title="GreenLoop Document Analyst", layout="centered")
    apply_streamlit_secrets(st.secrets)
    st.title("GreenLoop Document Analyst")
    st.write("Answers are generated only from the local GreenLoop documents.")
    try:
        provider = get_provider_settings()
    except LLMConfigurationError:
        st.caption("Provider configuration is incomplete.")
    else:
        provider_name = "Ollama" if provider.provider == "ollama" else "OpenRouter"
        st.caption(f"Provider: {provider_name} | Model: {provider.model.removeprefix('ollama/')}")

    question = st.text_area(
        "Question",
        placeholder=(
            "Ask a question about the GreenLoop employee handbook, product "
            "specification, or Q3 report."
        ),
        height=160,
    )

    if st.button("Analyze Documents", type="primary"):
        if not question.strip():
            st.error("Please enter a question before analyzing documents.")
        else:
            status = st.status(INITIAL_STAGE, expanded=True)
            progress = _StageProgress(status)
            streamed_answer = _StreamingAnswer(st.empty(), progress)

            def show_stage(stage: str, elapsed_seconds: float) -> None:
                progress.advance(stage, elapsed_seconds)

            try:
                retrieval_service = _streamlit_retrieval_service()
            except Exception:
                LOGGER.exception("Streamlit retrieval initialization failure.")
                progress.fail("Document index preparation failed")
                st.error("Local document retrieval is unavailable. Check the local index and try again.")
                return

            try:
                result = execute_question(
                    question,
                    retrieval_service=retrieval_service,
                    progress_callback=show_stage,
                    token_callback=streamed_answer.push,
                )
            except QuestionValidationError as exc:
                LOGGER.exception("Streamlit received an invalid question.")
                progress.fail("Document analysis failed")
                st.error(str(exc))
            except OllamaConnectionError as exc:
                LOGGER.exception("Streamlit Ollama connection failure.")
                progress.fail("Document analysis failed")
                st.error(str(exc))
            except OllamaModelMissingError as exc:
                LOGGER.exception("Streamlit Ollama model preflight failure.")
                progress.fail("Document analysis failed")
                st.error(str(exc))
            except ProviderConfigurationError as exc:
                LOGGER.exception("Streamlit LLM provider configuration failure.")
                progress.fail("Document analysis failed")
                st.error(str(exc))
            except RetrievalError as exc:
                LOGGER.exception("Streamlit retrieval failure.")
                progress.fail("Document analysis failed")
                st.error(str(exc))
            except CrewExecutionError as exc:
                LOGGER.exception("Streamlit CrewAI execution failure.")
                progress.fail("Document analysis failed")
                st.error(str(exc))
            except UnexpectedExecutionError as exc:
                LOGGER.exception("Streamlit unexpected execution failure.")
                progress.fail("Document analysis failed")
                st.error(str(exc))
            except QuestionExecutionError as exc:
                LOGGER.exception("Streamlit document analysis failure.")
                progress.fail("Document analysis failed")
                st.error(str(exc))
            except Exception:
                LOGGER.exception("Unexpected Streamlit application failure.")
                progress.fail("Document analysis failed")
                st.error("An unexpected error occurred. Please try again.")
            else:
                progress.complete(result.timings.get("total_request_execution", 0.0))
                streamed_answer.replace_with_complete(result.report_markdown)
                st.session_state["streamed_answer"] = result.report_markdown
                st.caption(_generation_metrics_text(result.metrics, result.llm_calls))
                st.download_button(
                    "Download Markdown report",
                    data=st.session_state["streamed_answer"],
                    file_name=result.output_path.name,
                    mime="text/markdown",
                )


if __name__ == "__main__":
    main()
