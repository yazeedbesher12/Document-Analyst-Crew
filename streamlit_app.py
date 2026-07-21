"""Streamlit interface for local GreenLoop document analysis."""

from __future__ import annotations

import logging
from time import perf_counter

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
            status = st.status("Preparing document index", expanded=True)
            stage_started = perf_counter()

            def show_stage(stage: str, elapsed_seconds: float) -> None:
                elapsed = elapsed_seconds or (perf_counter() - stage_started)
                state = "complete" if stage == "Completed" else "running"
                status.update(label=f"{stage} ({elapsed:.1f}s)", state=state)

            try:
                retrieval_service = _streamlit_retrieval_service()
            except Exception:
                LOGGER.exception("Streamlit retrieval initialization failure.")
                status.update(label="Document index preparation failed", state="error")
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
                st.error(str(exc))
            except OllamaConnectionError as exc:
                LOGGER.exception("Streamlit Ollama connection failure.")
                st.error(str(exc))
            except OllamaModelMissingError as exc:
                LOGGER.exception("Streamlit Ollama model preflight failure.")
                st.error(str(exc))
            except ProviderConfigurationError as exc:
                LOGGER.exception("Streamlit LLM provider configuration failure.")
                st.error(str(exc))
            except RetrievalError as exc:
                LOGGER.exception("Streamlit retrieval failure.")
                st.error(str(exc))
            except CrewExecutionError as exc:
                LOGGER.exception("Streamlit CrewAI execution failure.")
                st.error(str(exc))
            except UnexpectedExecutionError as exc:
                LOGGER.exception("Streamlit unexpected execution failure.")
                st.error(str(exc))
            except QuestionExecutionError as exc:
                LOGGER.exception("Streamlit document analysis failure.")
                st.error(str(exc))
            except Exception:
                LOGGER.exception("Unexpected Streamlit application failure.")
                st.error("An unexpected error occurred. Please try again.")
            else:
                status.update(
                    label=f"Completed ({result.timings.get('total_request_execution', 0.0):.1f}s)",
                    state="complete",
                )
                st.markdown(result.report_markdown)
                st.download_button(
                    "Download Markdown report",
                    data=result.report_markdown,
                    file_name=result.output_path.name,
                    mime="text/markdown",
                )


if __name__ == "__main__":
    main()
