"""CrewAI custom tool for local GreenLoop document evidence retrieval."""

from __future__ import annotations

import json
import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from greenloop_rag_crew.rag.document_registry import DOCUMENT_REGISTRY
from greenloop_rag_crew.rag.hybrid_retriever import HybridRetriever

LOGGER = logging.getLogger(__name__)

VALID_DOCUMENT_IDS = frozenset(metadata.document_id for metadata in DOCUMENT_REGISTRY)
RETRIEVAL_NOTICE = (
    "Retrieval scores rank relevance only. They are not probabilities or "
    "factual-confidence scores."
)


class DocumentSearchInput(BaseModel):
    """Input schema exposed to CrewAI agents for document_search."""

    query: str = Field(
        ...,
        description=(
            "One focused search query for local GreenLoop document evidence. "
            "Use separate tool calls for multi-part questions."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of evidence chunks to return, from 1 to 10.",
    )
    document_id: str | None = Field(
        default=None,
        description=(
            "Optional exact document filter. Valid values: HR-HBK-2025-v1.4, "
            "PRD-GLX1-2025-v2.1, FIN-Q3-2025-v1.0."
        ),
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("query must contain at least 3 non-whitespace characters.")
        return stripped

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if stripped not in VALID_DOCUMENT_IDS:
            valid = ", ".join(sorted(VALID_DOCUMENT_IDS))
            raise ValueError(f"Unknown document_id {value!r}. Valid document IDs: {valid}.")
        return stripped


class DocumentSearchTool(BaseTool):
    """Single CrewAI-facing tool for GreenLoop hybrid retrieval."""

    name: str = "document_search"
    description: str = (
        "Search the local GreenLoop company documents for employee policy, product "
        "specification, operational, and financial questions. This tool returns "
        "source-grounded evidence chunks with document IDs and PDF page numbers. "
        "It does not search the web, generate an answer, verify an answer, call an "
        "LLM, or prove that a claim is true. For multi-part questions, call this "
        "tool separately with focused subqueries. If evidence does not explicitly "
        "support a claim, do not invent the missing information."
    )
    args_schema: type[BaseModel] = DocumentSearchInput

    _retriever: HybridRetriever | None = PrivateAttr(default=None)

    def _run(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> str:
        """Return evidence chunks as a JSON string for CrewAI."""

        try:
            validated = DocumentSearchInput.model_validate(
                {"query": query, "top_k": top_k, "document_id": document_id}
            )
            results = self._get_retriever().search(
                validated.query,
                top_k=validated.top_k,
                document_id=validated.document_id,
            )
            if not results:
                return _json_dumps(
                    {
                        "status": "no_results",
                        "query": validated.query,
                        "result_count": 0,
                        "results": [],
                        "message": (
                            "No matching evidence was found in the selected local documents."
                        ),
                    }
                )

            return _json_dumps(
                {
                    "status": "ok",
                    "query": validated.query,
                    "retrieval_method": "hybrid_dense_bm25_rrf",
                    "document_filter": validated.document_id,
                    "result_count": len(results),
                    "results": [result.model_dump() for result in results],
                    "notice": RETRIEVAL_NOTICE,
                }
            )
        except Exception as exc:
            error_type, message = _classify_tool_error(exc)
            LOGGER.exception("document_search failed with %s", error_type)
            safe_query = query.strip() if isinstance(query, str) else ""
            return _json_dumps(
                {
                    "status": "error",
                    "query": safe_query,
                    "result_count": 0,
                    "results": [],
                    "error_type": error_type,
                    "message": message,
                }
            )

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever


def _classify_tool_error(exc: Exception) -> tuple[str, str]:
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, (FileNotFoundError, NotADirectoryError)) or any(
        marker in lowered
        for marker in [
            "manifest",
            "index",
            "chroma collection count",
            "lexical chunks and dense index",
            "run build_index",
            "rebuild the index",
            "chunks file",
        ]
    ):
        return (
            "index_not_ready",
            (
                "The local document index is missing or outdated. Run "
                "`uv run python -m greenloop_rag_crew.rag.build_chunks` and "
                "`uv run python -m greenloop_rag_crew.rag.build_index`, then retry."
            ),
        )
    if "validation" in lowered or "query" in lowered or "top_k" in lowered:
        return "invalid_input", text
    return (
        "retrieval_error",
        "Document retrieval failed. Check the local index and application logs, then retry.",
    )


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
