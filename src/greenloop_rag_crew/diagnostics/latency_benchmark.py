"""Run one cold and one warm local CrewAI request without printing report content."""

from __future__ import annotations

import argparse
import json

from greenloop_rag_crew.question_execution import execute_question
from greenloop_rag_crew.rag.embedder import clear_cached_embedders
from greenloop_rag_crew.rag.retrieval_service import (
    clear_retrieval_service_cache,
    get_retrieval_service,
)

DEFAULT_QUESTION = (
    "What is GreenLoop's remote-work policy, and how did Q3 revenue compare with Q2?"
)


def run_benchmark(question: str = DEFAULT_QUESTION) -> dict[str, object]:
    """Return safe latency metadata for repeated requests in one Python process."""

    clear_retrieval_service_cache()
    clear_cached_embedders()
    cold = execute_question(question)
    service = get_retrieval_service()
    cold_model_loaded = service.retriever.embedder.model_load_seconds is not None
    cold_index = service.preparation

    warm = execute_question(question)
    return {
        "index_action": cold_index.action,
        "index_reason": cold_index.reason,
        "cold_embedding_model_loaded": cold_model_loaded,
        "cold": _result_summary(cold),
        "warm": _result_summary(warm),
    }


def _result_summary(result) -> dict[str, object]:
    return {
        "timings": result.timings,
        "llm_calls": result.llm_calls,
        "retrieval_calls": result.retrieval_calls,
        "report_path": str(result.output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
