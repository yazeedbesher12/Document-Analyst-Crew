from dataclasses import dataclass

import pytest

from greenloop_rag_crew.diagnostics.ollama_tool_check import OllamaHealth
from greenloop_rag_crew.question_execution import (
    CrewExecutionError,
    OllamaConnectionError,
    OllamaModelMissingError,
    QuestionExecutionError,
    QuestionValidationError,
    RetrievalError,
    check_ollama_preflight,
    create_unique_report_path,
    execute_question,
    validate_question,
)
from greenloop_rag_crew import question_execution as execution_module


@dataclass
class FakeKickoffResult:
    raw: str = "# GreenLoop Document Analysis\n\nVerified report."


class FakeCrew:
    def __init__(self) -> None:
        self.inputs = []

    def kickoff(self, inputs):
        self.inputs.append(inputs)
        return FakeKickoffResult()


class FakeRetrievalService:
    def metrics_snapshot(self):
        return 0, 0.0


@dataclass
class FakeBundle:
    crew: FakeCrew


@pytest.fixture(autouse=True)
def skip_real_ollama_preflight(monkeypatch):
    monkeypatch.setattr(execution_module, "check_llm_preflight", lambda: None)
    monkeypatch.setattr(
        execution_module,
        "get_retrieval_service",
        lambda: FakeRetrievalService(),
    )


def test_empty_question_is_rejected():
    with pytest.raises(QuestionValidationError, match="Please enter a question"):
        validate_question("   ")


def test_unique_markdown_output_paths(tmp_path):
    first = create_unique_report_path("Remote work policy", output_dir=tmp_path)
    second = create_unique_report_path("Remote work policy", output_dir=tmp_path)

    assert first != second
    assert first.parent == tmp_path
    assert first.suffix == ".md"
    assert second.suffix == ".md"


def test_each_execution_creates_a_fresh_crew_and_passes_the_question(tmp_path):
    crews = []

    def fake_factory(**_kwargs):
        crew = FakeCrew()
        crews.append(crew)
        return FakeBundle(crew=crew)

    first = execute_question(
        "What is the remote work policy?",
        output_dir=tmp_path,
        crew_factory=fake_factory,
        mode="legacy",
    )
    second = execute_question(
        "What laboratory accuracy was achieved?",
        output_dir=tmp_path,
        crew_factory=fake_factory,
        mode="legacy",
    )

    assert len(crews) == 2
    assert crews[0] is not crews[1]
    assert crews[0].inputs == [{"question": "What is the remote work policy?"}]
    assert crews[1].inputs == [{"question": "What laboratory accuracy was achieved?"}]
    assert first.output_path != second.output_path
    assert first.output_path.read_text(encoding="utf-8") == first.report_markdown + "\n"
    assert second.output_path.read_text(encoding="utf-8") == second.report_markdown + "\n"


def test_execution_error_is_safe_and_does_not_expose_internal_exception(tmp_path):
    def failing_factory(**_kwargs):
        raise RuntimeError(r"D:\internal\secret-path API_KEY=not-for-display")

    with pytest.raises(CrewExecutionError) as captured:
        execute_question(
            "Question", output_dir=tmp_path, crew_factory=failing_factory, mode="legacy"
        )

    message = str(captured.value)
    assert "could not complete" in message
    assert "secret-path" not in message
    assert "API_KEY" not in message


def test_retrieval_failure_is_classified_without_internal_details(tmp_path):
    class FailingCrew:
        def kickoff(self, inputs):
            raise RuntimeError("Could not connect to tenant default_tenant")

    with pytest.raises(RetrievalError) as captured:
        execute_question(
            "Question",
            output_dir=tmp_path,
            crew_factory=lambda **_kwargs: FakeBundle(crew=FailingCrew()),
            mode="legacy",
        )

    assert "document retrieval" in str(captured.value).lower()
    assert "default_tenant" not in str(captured.value)


def test_ollama_preflight_reports_connection_failure(monkeypatch):
    monkeypatch.setattr(
        execution_module,
        "check_ollama_health",
        lambda timeout: OllamaHealth(
            reachable=False,
            model="qwen3:8b",
            model_installed=False,
            base_url="http://localhost:11434",
            error_type="connection_refused",
        ),
    )

    with pytest.raises(OllamaConnectionError, match="Cannot reach the local Ollama"):
        check_ollama_preflight()


def test_ollama_preflight_reports_missing_model(monkeypatch):
    monkeypatch.setattr(
        execution_module,
        "check_ollama_health",
        lambda timeout: OllamaHealth(
            reachable=True,
            model="qwen3:8b",
            model_installed=False,
            base_url="http://localhost:11434",
            error_type="model_missing",
        ),
    )

    with pytest.raises(OllamaModelMissingError, match="ollama pull qwen3:8b"):
        check_ollama_preflight()
