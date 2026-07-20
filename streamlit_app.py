"""Streamlit interface for local GreenLoop document analysis."""

from __future__ import annotations

import logging

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

LOGGER = logging.getLogger(__name__)


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
            with st.spinner("Analyzing local documents..."):
                try:
                    result = execute_question(question)
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
                    st.markdown(result.report_markdown)
                    st.download_button(
                        "Download Markdown report",
                        data=result.report_markdown,
                        file_name=result.output_path.name,
                        mime="text/markdown",
                    )


if __name__ == "__main__":
    main()
