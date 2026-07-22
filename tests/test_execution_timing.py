from dataclasses import dataclass
from types import SimpleNamespace

from greenloop_rag_crew import question_execution as execution_module
from greenloop_rag_crew.question_execution import execute_question


@dataclass
class FakeKickoffResult:
    raw: str = "# GreenLoop Document Analysis\n\nVerified report."


class FakeRetrievalService:
    def metrics_snapshot(self):
        return 0, 0.0


class FakeTask:
    callback = None


class FakeCrew:
    def __init__(self, tasks):
        self.tasks = tasks
        self.usage_metrics = SimpleNamespace(successful_requests=3)

    def kickoff(self, inputs):
        assert inputs == {"question": "What is the remote work policy?"}
        self.tasks.research_task.callback(object())
        self.tasks.fact_check_task.callback(object())
        self.tasks.report_task.callback(object())
        return FakeKickoffResult()


def test_execution_records_public_stage_timings_and_progress(monkeypatch, tmp_path):
    tasks = SimpleNamespace(
        research_task=FakeTask(),
        fact_check_task=FakeTask(),
        report_task=FakeTask(),
    )
    bundle = SimpleNamespace(crew=FakeCrew(tasks), tasks=tasks)
    progress = []

    monkeypatch.setattr(execution_module, "check_llm_preflight", lambda: None)

    result = execute_question(
        "What is the remote work policy?",
        output_dir=tmp_path,
        crew_factory=lambda **_kwargs: bundle,
        retrieval_service=FakeRetrievalService(),
        progress_callback=lambda stage, elapsed: progress.append((stage, elapsed)),
        mode="legacy",
    )

    expected_stages = {
        "application_initialization",
        "researcher_execution",
        "fact_checker_execution",
        "report_writer_execution",
        "total_crew_execution",
        "total_request_execution",
        "total_retrieval_execution",
    }
    assert expected_stages <= result.timings.keys()
    assert all(result.timings[stage] >= 0 for stage in expected_stages)
    assert [stage for stage, _elapsed in progress] == [
        "Preparing document index",
        "Researching documents",
        "Verifying claims",
        "Writing report",
        "Completed",
    ]
    assert progress[-1][1] == result.timings["report_writer_execution"]
    assert result.llm_calls == 3
