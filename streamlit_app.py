"""Streamlit interface for local GreenLoop document analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
from greenloop_rag_crew.rag.retrieval_service import RetrievalService, get_retrieval_service

LOGGER = logging.getLogger(__name__)
INITIAL_STAGE = "Preparing document index"


@dataclass
class _StageProgress:
    """Render public workflow stages without exposing request content."""

    status: Any
    current_stage: str = INITIAL_STAGE

    def advance(self, next_stage: str, elapsed_seconds: float) -> None:
        """Record the completed public stage before showing the next one."""

        if next_stage == self.current_stage:
            return
        elapsed = max(0.0, elapsed_seconds)
        self.status.write(f"{self.current_stage}: {elapsed:.1f}s")
        self.current_stage = next_stage
        if next_stage != "Completed":
            self.status.update(label=next_stage, state="running")

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


@st.cache_resource(show_spinner=False)
def _streamlit_retrieval_service() -> RetrievalService:
    """Keep only reusable local retrieval state across Streamlit reruns."""

    return get_retrieval_service()


def main() -> None:
    """Render the local-document Streamlit interface."""

    st.set_page_config(page_title="GreenLoop Document Analyst", layout="centered")
    st.title("GreenLoop Document Analyst")
    st.write("Answers are generated only from the local GreenLoop documents.")

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
                st.markdown(result.report_markdown)
                st.download_button(
                    "Download Markdown report",
                    data=result.report_markdown,
                    file_name=result.output_path.name,
                    mime="text/markdown",
                )


if __name__ == "__main__":
    main()
