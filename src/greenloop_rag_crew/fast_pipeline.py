"""Deterministic evidence preparation and single-generation report rendering."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Callable

from greenloop_rag_crew.llm import GenerationSettings, LLMSettings
from greenloop_rag_crew.ollama_client import GenerationResponse
from greenloop_rag_crew.rag.retrieval_service import RetrievalService, runtime_index_signature
from greenloop_rag_crew.rag.schemas import HybridSearchResult
from greenloop_rag_crew.runtime_paths import answer_cache_dir

PIPELINE_VERSION = "fast-v2-streaming"
DEFAULT_PIPELINE_MODE = "fast"
CHUNK_ID_PATTERN = re.compile(r"[A-Za-z0-9.-]+_p\d+_c\d+")
_CACHE_LOCK = Lock()


class PipelineConfigurationError(ValueError):
    """Raised when deterministic pipeline configuration is invalid."""


@dataclass(frozen=True)
class RetrievalLimits:
    """Bounded evidence budgets for one generation prompt."""

    vector_top_k: int
    bm25_top_k: int
    final_context_chunks: int
    max_context_chars: int


@dataclass(frozen=True)
class RetrievalContext:
    """Typed, deduplicated retrieval evidence sent to the answer generator."""

    chunks: tuple[HybridSearchResult, ...]
    rendered_context: str
    warnings: tuple[str, ...]
    duration_seconds: float


@dataclass(frozen=True)
class VerificationResult:
    """Deterministic metadata-verification results without semantic inference."""

    allowed_chunk_ids: frozenset[str]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FastPipelineResult:
    """Safe output and observability data for one fast or strict request."""

    report_markdown: str
    retrieval: RetrievalContext
    verification: VerificationResult
    llm_calls: int
    metrics: dict[str, object]
    answer_cache_hit: bool


class ExactAnswerCache:
    """Small persistent cache for exact questions against one index signature."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory if directory is not None else answer_cache_dir()

    def get(self, key: str) -> str | None:
        path = self.directory / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        report = payload.get("report_markdown")
        return report if isinstance(report, str) and report.strip() else None

    def put(self, key: str, report_markdown: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        payload = {"pipeline_version": PIPELINE_VERSION, "report_markdown": report_markdown}
        with _CACHE_LOCK:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path)


def pipeline_mode() -> str:
    """Return the requested execution mode, defaulting to one-call fast mode."""

    mode = os.getenv("RAG_PIPELINE_MODE", DEFAULT_PIPELINE_MODE).strip().lower()
    if mode not in {"fast", "strict", "legacy"}:
        raise PipelineConfigurationError(
            "RAG_PIPELINE_MODE must be one of: fast, strict, legacy."
        )
    return mode


def strict_llm_verification_enabled() -> bool:
    """Return the explicit strict-verification opt-in."""

    return _env_bool("STRICT_LLM_VERIFICATION", False)


def answer_cache_enabled() -> bool:
    """Return whether exact answer caching is enabled for fast/strict runs."""

    return _env_bool("RAG_ANSWER_CACHE", True)


def retrieval_limits() -> RetrievalLimits:
    """Read bounded, independently configurable dense and BM25 retrieval limits."""

    return RetrievalLimits(
        vector_top_k=_env_int("RAG_TOP_K_VECTOR", 4, minimum=1, maximum=20),
        bm25_top_k=_env_int("RAG_TOP_K_BM25", 4, minimum=1, maximum=20),
        final_context_chunks=_env_int("RAG_FINAL_CONTEXT_CHUNKS", 4, minimum=1, maximum=10),
        max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 7000, minimum=1000, maximum=50000),
    )


def retrieve_evidence(
    question: str,
    service: RetrievalService,
    limits: RetrievalLimits | None = None,
) -> RetrievalContext:
    """Retrieve, deduplicate, and serialize only a bounded evidence subset."""

    limits = limits or retrieval_limits()
    started = perf_counter()
    results = service.search(
        question,
        top_k=limits.final_context_chunks,
        dense_candidate_k=limits.vector_top_k,
        bm25_candidate_k=limits.bm25_top_k,
    )
    distinct: list[HybridSearchResult] = []
    seen_ids: set[str] = set()
    for result in results:
        if result.chunk_id not in seen_ids:
            distinct.append(result)
            seen_ids.add(result.chunk_id)
    selected = tuple(distinct[: limits.final_context_chunks])
    warnings = _metadata_warnings(selected)
    return RetrievalContext(
        chunks=selected,
        rendered_context=_render_context(selected, limits.max_context_chars),
        warnings=tuple(warnings),
        duration_seconds=perf_counter() - started,
    )


def verify_retrieval_context(context: RetrievalContext) -> VerificationResult:
    """Verify citation metadata deterministically; never infer semantic support."""

    warnings = list(context.warnings)
    allowed_ids = frozenset(chunk.chunk_id for chunk in context.chunks)
    if not allowed_ids:
        warnings.append("No local evidence chunks were retrieved for this question.")
    return VerificationResult(allowed_chunk_ids=allowed_ids, warnings=tuple(warnings))


def run_fast_pipeline(
    *,
    question: str,
    service: RetrievalService,
    llm_settings: LLMSettings,
    generation_settings: GenerationSettings,
    generate: Callable[[str], GenerationResponse],
    mode: str | None = None,
    cache: ExactAnswerCache | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> FastPipelineResult:
    """Run deterministic evidence work and one, or strict-mode two, generations."""

    selected_mode = mode or pipeline_mode()
    if selected_mode == "legacy":
        raise PipelineConfigurationError("Legacy mode must be executed by the CrewAI pipeline.")
    if selected_mode == "strict" and not strict_llm_verification_enabled():
        raise PipelineConfigurationError(
            "STRICT_LLM_VERIFICATION=true is required when RAG_PIPELINE_MODE=strict."
        )

    cache_key = _cache_key(question, llm_settings, generation_settings, selected_mode)
    cache = cache or ExactAnswerCache()
    if answer_cache_enabled():
        cached = cache.get(cache_key)
        if cached is not None:
            _notify_progress(progress_callback, "Retrieving evidence", 0.0)
            _notify_progress(progress_callback, "Verifying citation metadata", 0.0)
            _notify_progress(progress_callback, "Writing answer", 0.0)
            empty_context = RetrievalContext((), "", (), 0.0)
            verification = VerificationResult(frozenset(), ("Exact answer cache hit.",))
            return FastPipelineResult(
                report_markdown=cached,
                retrieval=empty_context,
                verification=verification,
                llm_calls=0,
                metrics={"answer_cache": "hit", "retrieved_chunks": 0},
                answer_cache_hit=True,
            )

    retrieval = retrieve_evidence(question, service)
    _notify_progress(progress_callback, "Verifying citation metadata", retrieval.duration_seconds)
    verification_started = perf_counter()
    verification = verify_retrieval_context(retrieval)
    _notify_progress(
        progress_callback,
        "Writing answer",
        perf_counter() - verification_started,
    )
    answer = generate(_answer_prompt(question, retrieval.rendered_context))
    llm_calls = answer.request_count
    final_content = answer.content
    if selected_mode == "strict":
        strict = generate(_strict_prompt(question, retrieval.rendered_context, answer.content))
        final_content = strict.content
        llm_calls += strict.request_count
        answer = strict

    report, report_warnings = render_report(final_content, verification, retrieval)
    all_warnings = tuple(dict.fromkeys((*verification.warnings, *report_warnings)))
    final_verification = VerificationResult(verification.allowed_chunk_ids, all_warnings)
    if answer_cache_enabled():
        cache.put(cache_key, report)
    return FastPipelineResult(
        report_markdown=report,
        retrieval=retrieval,
        verification=final_verification,
        llm_calls=llm_calls,
        metrics={
            "answer_cache": "miss",
            "retrieved_chunks": len(retrieval.chunks),
            "prompt_input_tokens": answer.input_tokens,
            "generated_output_tokens": answer.output_tokens,
            "model_load_seconds": answer.load_seconds,
            "prompt_evaluation_seconds": answer.prompt_evaluation_seconds,
            "generation_seconds": answer.generation_seconds,
            "time_to_first_token_seconds": answer.time_to_first_token_seconds,
            "tokens_per_second": _tokens_per_second(answer),
            "model_already_loaded": answer.model_already_loaded,
            "llm_request_count": llm_calls,
        },
        answer_cache_hit=False,
    )


def render_report(
    generated_markdown: str,
    verification: VerificationResult,
    retrieval: RetrievalContext,
) -> tuple[str, tuple[str, ...]]:
    """Keep only supplied inline citation IDs and the concise answer structure."""

    warnings: list[str] = []
    unknown_ids = [
        chunk_id
        for chunk_id in CHUNK_ID_PATTERN.findall(generated_markdown)
        if chunk_id not in verification.allowed_chunk_ids
    ]
    body = generated_markdown.strip()
    for chunk_id in sorted(set(unknown_ids)):
        body = body.replace(chunk_id, "[unsupported citation removed]")
    if unknown_ids:
        warnings.append("Generated citations outside retrieved evidence were removed.")

    body = re.split(r"(?m)^## Citations\s*$", body, maxsplit=1)[0].rstrip()
    body = _ensure_concise_structure(body)
    body = _limit_evidence_bullets(body)
    warnings.extend(_uncited_claim_warnings(body, verification.allowed_chunk_ids))
    limitations = list(verification.warnings) + warnings
    if limitations:
        body = _append_limitations(body, limitations)
    else:
        body = _remove_empty_limitation(body)
    return _limit_words(body, maximum_words=300) + "\n", tuple(warnings)


def _render_context(chunks: tuple[HybridSearchResult, ...], max_chars: int) -> str:
    remaining = max_chars
    blocks: list[str] = []
    for chunk in chunks:
        header = (
            f"Source filename: {chunk.source}\nDocument ID: {chunk.document_id}\n"
            f"PDF page: {chunk.page}\nSection: {chunk.section}\nChunk ID: {chunk.chunk_id}\n"
            f"Retrieval score: {chunk.fusion_score:.6f} (ranking only, not probability)\nEvidence: "
        )
        evidence = chunk.text
        available = max(0, remaining - len(header))
        if available < len(evidence):
            evidence = evidence[:available].rstrip() + (" [truncated]" if available else "")
        block = header + evidence
        blocks.append(block)
        remaining -= len(block) + 2
        if remaining <= 0:
            break
    return "\n\n".join(blocks)


def _metadata_warnings(chunks: tuple[HybridSearchResult, ...]) -> list[str]:
    warnings: list[str] = []
    for chunk in chunks:
        if not CHUNK_ID_PATTERN.fullmatch(chunk.chunk_id):
            warnings.append("A retrieved chunk had malformed citation metadata.")
        if not all((chunk.source.strip(), chunk.document_id.strip(), chunk.section.strip())):
            warnings.append("A retrieved chunk had incomplete citation metadata.")
    return warnings


def _answer_prompt(question: str, context: str) -> str:
    return (
        "Answer only from the supplied local evidence. Do not use outside knowledge. "
        "Use this exact concise Markdown structure and stay under 300 words:\n"
        "## Direct Answer\nOne short paragraph answering the question.\n\n"
        "## Evidence\nAt most four concise bullets. Every factual bullet must include an inline "
        "supplied Chunk ID citation, for example [HR-HBK-2025-v1.4_p06_c01].\n\n"
        "## Limitation\nInclude this section only when a requested conclusion is not directly supported. "
        "Do not add an introduction, conclusion, executive summary, bibliography, source/page repetition, "
        "or duplicate citations. Do not invent sources, pages, sections, chunk IDs, or values.\n\n"
        f"User question:\n{question}\n\nRetrieved evidence:\n{context}"
    )


def _strict_prompt(question: str, context: str, answer: str) -> str:
    return (
        "Perform one strict citation review. Return a corrected concise Markdown answer only. "
        "Keep only inline supplied Chunk IDs, at most four Evidence bullets, and no bibliography. "
        "Remove any claim not supported by the supplied evidence.\n\n"
        f"User question:\n{question}\n\nEvidence:\n{context}\n\nDraft answer:\n{answer}"
    )


def _ensure_concise_structure(body: str) -> str:
    body = re.sub(r"(?m)^# GreenLoop Document Analysis\s*$\n*", "", body)
    body = body.replace("## Executive Summary", "## Direct Answer")
    body = body.replace("## Findings", "## Evidence")
    body = body.replace("## Limitations and Undisclosed Information", "## Limitation")
    if "## Direct Answer" not in body:
        body = f"## Direct Answer\n{body}"
    if "## Evidence" not in body:
        body += "\n\n## Evidence\n"
    return body.rstrip()


def _append_limitations(body: str, limitations: list[str]) -> str:
    heading = "## Limitation"
    if heading not in body:
        body += f"\n\n{heading}\n"
    prefix, suffix = body.split(heading, maxsplit=1)
    bullets = "\n".join(f"- {warning}" for warning in dict.fromkeys(limitations))
    return f"{prefix}{heading}{suffix.rstrip()}\n{bullets}".rstrip()


def _uncited_claim_warnings(body: str, allowed_chunk_ids: frozenset[str]) -> list[str]:
    """Warn when deterministic checks cannot link a finding to retrieved evidence."""

    findings_match = re.search(
        r"(?ms)^## Evidence\s*$\n(.*?)(?=^## |\Z)", body
    )
    if findings_match is None:
        return []
    for line in findings_match.group(1).splitlines():
        candidate = line.strip()
        if not candidate or candidate.lower() in {"none.", "none"}:
            continue
        if any(chunk_id in candidate for chunk_id in allowed_chunk_ids):
            continue
        if any(character.isalnum() for character in candidate):
            return ["A finding could not be deterministically linked to retrieved evidence."]
    return []


def _limit_evidence_bullets(body: str) -> str:
    heading = "## Evidence"
    if heading not in body:
        return body
    prefix, suffix = body.split(heading, maxsplit=1)
    match = re.match(r"(.*?)(?=\n## |\Z)", suffix, flags=re.DOTALL)
    if match is None:
        return body
    evidence, remainder = match.group(1), suffix[len(match.group(1)) :]
    bullets = [line for line in evidence.splitlines() if line.lstrip().startswith("-")]
    if not bullets and evidence.strip():
        bullets = [f"- {line.strip()}" for line in evidence.splitlines() if line.strip()]
    limited = "\n".join(bullets[:4])
    return f"{prefix}{heading}\n{limited}{remainder}".rstrip()


def _remove_empty_limitation(body: str) -> str:
    return re.sub(
        r"(?ms)\n*## Limitation\s*\n\s*(?:None\.?\s*)?(?=\Z)",
        "",
        body,
    ).rstrip()


def _limit_words(body: str, *, maximum_words: int) -> str:
    """Bound output without splitting a citation-bearing line in half."""

    kept: list[str] = []
    used = 0
    for line in body.splitlines():
        if line.startswith("## "):
            kept.append(line)
            continue
        count = len(line.split())
        if used + count > maximum_words:
            continue
        kept.append(line)
        used += count
    return "\n".join(kept).rstrip()


def _tokens_per_second(answer: GenerationResponse) -> float | None:
    if not answer.output_tokens or not answer.generation_seconds or answer.generation_seconds <= 0:
        return None
    return answer.output_tokens / answer.generation_seconds


def _cache_key(
    question: str,
    llm_settings: LLMSettings,
    generation_settings: GenerationSettings,
    mode: str,
) -> str:
    payload = {
        "question": " ".join(question.casefold().split()),
        "index_signature": runtime_index_signature(),
        "model": llm_settings.model,
        "generation": asdict(generation_settings),
        "mode": mode,
        "pipeline_version": PIPELINE_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError as exc:
        raise PipelineConfigurationError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise PipelineConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise PipelineConfigurationError(f"{name} must be true or false.")


def _notify_progress(
    callback: Callable[[str, float], None] | None,
    stage: str,
    elapsed_seconds: float,
) -> None:
    if callback is not None:
        callback(stage, elapsed_seconds)
