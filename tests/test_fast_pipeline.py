import pytest

from greenloop_rag_crew import fast_pipeline
from greenloop_rag_crew import question_execution
from greenloop_rag_crew.fast_pipeline import (
    ExactAnswerCache,
    render_report,
    retrieve_evidence,
    run_fast_pipeline,
    verify_retrieval_context,
)
from greenloop_rag_crew.llm import GenerationSettings, LLMSettings
from greenloop_rag_crew.ollama_client import GenerationResponse
from greenloop_rag_crew.rag.schemas import HybridSearchResult
from greenloop_rag_crew.question_execution import execute_question


def _chunk(chunk_id="HR-HBK-2025-v1.4_p06_c01"):
    return HybridSearchResult(
        rank=1,
        chunk_id=chunk_id,
        source="GreenLoop_Employee_Handbook_2025.pdf",
        document_id="HR-HBK-2025-v1.4",
        title="GreenLoop Employee Handbook 2025",
        page=6,
        section="Work modes and role eligibility",
        text="Eligible software and AI employees may work remotely up to three days per week.",
        fusion_score=0.03,
        dense_rank=1,
        dense_score=0.9,
        bm25_rank=1,
        bm25_score=8.0,
        matched_by=["dense", "bm25"],
    )


class FakeRetrievalService:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k, **kwargs):
        self.calls.append((query, top_k, kwargs))
        return [_chunk()]

    def metrics_snapshot(self):
        return len(self.calls), 0.0


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return GenerationResponse(
            content=(
                "## Direct Answer\nEligible employees may work remotely up to three days.\n\n"
                "## Evidence\n- Eligible employees may work remotely up to three days "
                "[HR-HBK-2025-v1.4_p06_c01]."
            ),
            request_count=1,
            input_tokens=50,
            output_tokens=25,
            load_seconds=0.1,
            prompt_evaluation_seconds=0.2,
            generation_seconds=0.3,
            total_seconds=0.4,
            thinking_present=False,
        )


@pytest.fixture(autouse=True)
def stable_signature(monkeypatch):
    monkeypatch.setattr(fast_pipeline, "runtime_index_signature", lambda: "index-a")
    monkeypatch.setenv("RAG_ANSWER_CACHE", "true")
    monkeypatch.setenv("STRICT_LLM_VERIFICATION", "false")


def _settings():
    return (
        LLMSettings(provider="ollama", model="ollama/qwen3:8b", base_url="http://localhost:11434"),
        GenerationSettings(temperature=0.1, max_tokens=400, num_ctx=3072, think=False, keep_alive="30m"),
    )


def _run(service, generator, cache, mode="fast"):
    llm_settings, generation_settings = _settings()
    return run_fast_pipeline(
        question="What is the remote work policy?",
        service=service,
        llm_settings=llm_settings,
        generation_settings=generation_settings,
        generate=generator,
        cache=cache,
        mode=mode,
    )


def test_fast_mode_makes_exactly_one_generation_request(tmp_path):
    service, generator = FakeRetrievalService(), RecordingGenerator()

    result = _run(service, generator, ExactAnswerCache(tmp_path))

    assert result.llm_calls == 1
    assert len(generator.calls) == 1
    assert service.calls[0][2] == {"dense_candidate_k": 4, "bm25_candidate_k": 4}
    assert result.metrics["retrieved_chunks"] == 1


def test_retrieval_and_metadata_verification_make_no_generation_request():
    service = FakeRetrievalService()

    context = retrieve_evidence("remote work", service)
    verification = verify_retrieval_context(context)

    assert len(service.calls) == 1
    assert verification.allowed_chunk_ids == {"HR-HBK-2025-v1.4_p06_c01"}


def test_strict_mode_makes_no_more_than_two_generation_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("STRICT_LLM_VERIFICATION", "true")
    service, generator = FakeRetrievalService(), RecordingGenerator()

    result = _run(service, generator, ExactAnswerCache(tmp_path), mode="strict")

    assert result.llm_calls == 2
    assert len(generator.calls) == 2


def test_exact_cache_hit_uses_zero_new_generation_requests(tmp_path):
    service, generator, cache = FakeRetrievalService(), RecordingGenerator(), ExactAnswerCache(tmp_path)
    _run(service, generator, cache)

    cached = _run(service, generator, cache)

    assert cached.answer_cache_hit is True
    assert cached.llm_calls == 0
    assert len(generator.calls) == 1


def test_changed_index_signature_invalidates_exact_cache(monkeypatch, tmp_path):
    service, generator, cache = FakeRetrievalService(), RecordingGenerator(), ExactAnswerCache(tmp_path)
    _run(service, generator, cache)
    monkeypatch.setattr(fast_pipeline, "runtime_index_signature", lambda: "index-b")

    result = _run(service, generator, cache)

    assert result.answer_cache_hit is False
    assert len(generator.calls) == 2


def test_final_report_keeps_only_retrieved_citation_ids():
    context = fast_pipeline.RetrievalContext((_chunk(),), "evidence", (), 0.0)
    verification = verify_retrieval_context(context)
    report, warnings = render_report(
        "# GreenLoop Document Analysis\n\n## Findings\n"
        "Claim [HR-HBK-2025-v1.4_p06_c01] and invented [FIN-Q3-2025-v1.0_p10_c01].\n\n"
        "## Limitations and Undisclosed Information\nNone.",
        verification,
        context,
    )

    assert "FIN-Q3-2025-v1.0_p10_c01" not in report
    assert "HR-HBK-2025-v1.4_p06_c01" in report
    assert "## Citations" not in report
    assert warnings


def test_concise_report_keeps_at_most_four_inline_evidence_bullets():
    context = fast_pipeline.RetrievalContext((_chunk(),), "evidence", (), 0.0)
    verification = verify_retrieval_context(context)
    report, _warnings = render_report(
        "## Direct Answer\nShort answer.\n\n## Evidence\n"
        "- One [HR-HBK-2025-v1.4_p06_c01]\n"
        "- Two [HR-HBK-2025-v1.4_p06_c01]\n"
        "- Three [HR-HBK-2025-v1.4_p06_c01]\n"
        "- Four [HR-HBK-2025-v1.4_p06_c01]\n"
        "- Five [HR-HBK-2025-v1.4_p06_c01]",
        verification,
        context,
    )

    evidence = report.split("## Evidence", maxsplit=1)[1]
    assert evidence.count("\n-") == 4
    assert "## Citations" not in report


def test_context_truncation_preserves_complete_citation_metadata():
    service = FakeRetrievalService()
    truncated = _chunk().model_copy(update={"text": "x" * 1000})
    service.search = lambda *_args, **_kwargs: [truncated]
    limits = fast_pipeline.RetrievalLimits(4, 4, 1, 80)

    context = retrieve_evidence("remote work", service, limits)

    assert "Source filename: GreenLoop_Employee_Handbook_2025.pdf" in context.rendered_context
    assert "Chunk ID: HR-HBK-2025-v1.4_p06_c01" in context.rendered_context


def test_uncited_finding_is_reported_as_a_deterministic_warning():
    context = fast_pipeline.RetrievalContext((_chunk(),), "evidence", (), 0.0)
    verification = verify_retrieval_context(context)

    report, warnings = render_report(
        "# GreenLoop Document Analysis\n\n## Findings\nAn uncited factual statement.\n\n"
        "## Limitations and Undisclosed Information\nNone.",
        verification,
        context,
    )

    assert any("could not be deterministically linked" in warning for warning in warnings)
    assert "could not be deterministically linked" in report


def test_fast_mode_never_constructs_a_crewai_tool_loop(monkeypatch, tmp_path):
    service, generator = FakeRetrievalService(), RecordingGenerator()
    llm_settings, _generation_settings = _settings()
    monkeypatch.setattr(question_execution, "check_llm_preflight", lambda: llm_settings)

    result = execute_question(
        "What is the remote work policy?",
        output_dir=tmp_path,
        retrieval_service=service,
        answer_generator=generator,
        crew_factory=lambda **_kwargs: pytest.fail("fast mode must not construct a CrewAI crew"),
        answer_cache=ExactAnswerCache(tmp_path / "cache"),
        mode="fast",
    )

    assert result.pipeline_mode == "fast"
    assert result.llm_calls == 1
    assert len(generator.calls) == 1
