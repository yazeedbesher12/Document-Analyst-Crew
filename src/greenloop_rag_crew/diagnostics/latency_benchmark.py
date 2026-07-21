"""Run one cold and one warm local CrewAI request without printing report content."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from greenloop_rag_crew.llm import ollama_thinking_enabled
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
    cold = execute_question(question, progress_callback=_stage_reporter("cold"))
    service = get_retrieval_service()
    cold_model_loaded = service.retriever.embedder.model_load_seconds is not None
    cold_index = service.preparation
    model_load_seconds_before_warm = service.retriever.embedder.model_load_seconds

    warm = execute_question(question, progress_callback=_stage_reporter("warm"))
    warm_model_loaded_again = (
        service.retriever.embedder.model_load_seconds != model_load_seconds_before_warm
    )
    return {
        "cold_index": {
            "action": cold_index.action,
            "reason": cold_index.reason,
            "elapsed_seconds": cold_index.elapsed_seconds,
            "pdfs_reindexed": cold_index.action == "rebuilt",
        },
        "warm_index": {
            "action": "reused_in_memory",
            "reason": "same_process_same_runtime_signature",
            "pdfs_reindexed": False,
        },
        "cold_embedding_model_loaded": cold_model_loaded,
        "cold_embedding_model_load_seconds": service.retriever.embedder.model_load_seconds,
        "warm_embedding_model_loaded_again": warm_model_loaded_again,
        "ollama_think_enabled": ollama_thinking_enabled(),
        "ollama_ps": _ollama_ps(),
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


def _stage_reporter(run: str):
    """Emit only public stage names and durations for timeout diagnostics."""

    def report(stage: str, elapsed_seconds: float) -> None:
        print(
            f"benchmark_stage run={run} stage={stage} elapsed_seconds={elapsed_seconds:.3f}",
            file=sys.stderr,
            flush=True,
        )

    return report


def _ollama_ps() -> str:
    """Return the local processor snapshot without touching application state."""

    completed = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
