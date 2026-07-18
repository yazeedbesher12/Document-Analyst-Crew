import argparse
from types import SimpleNamespace
from pathlib import Path

import pytest
from crewai import LLM

from greenloop_rag_crew import main as main_module
from greenloop_rag_crew.runner import list_questions, run_all, run_question


def _safe_llm() -> LLM:
    return LLM(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.6,
        top_p=0.95,
        timeout=1,
        max_tokens=1000,
    )


def test_list_mode_does_not_call_ollama(monkeypatch):
    monkeypatch.setattr(
        "greenloop_rag_crew.runner.create_crew",
        lambda *args, **kwargs: pytest.fail("create_crew should not be called"),
    )

    questions = list_questions()

    assert len(questions) == 3


def test_dry_run_does_not_call_kickoff_and_selects_correct_question(monkeypatch):
    result = run_question(
        "remote_work_and_revenue",
        llm=_safe_llm(),
        overwrite=True,
        dry_run=True,
    )

    assert result.question_id == "remote_work_and_revenue"
    assert result.output_path == "output/report_01_remote_work_and_revenue.md"
    assert result.process_type == "sequential"
    assert result.agent_order == ["Document Researcher", "Fact Checker", "Report Writer"]


def test_unknown_question_id_fails():
    with pytest.raises(Exception, match="Unknown question_id"):
        run_question("missing", llm=_safe_llm(), dry_run=True)


def test_conflicting_cli_options_fail():
    args = argparse.Namespace(
        list=True,
        question_id="remote_work_and_revenue",
        all=False,
        dry_run=False,
        overwrite=False,
    )

    with pytest.raises(SystemExit, match="Choose exactly one"):
        main_module._validate_args(args)


def test_existing_output_is_protected(tmp_path, monkeypatch):
    output_file = tmp_path / "existing.md"
    output_file.write_text("already here", encoding="utf-8")

    fake_question = SimpleNamespace(
        id="remote_work_and_revenue",
        question="Question?",
        output_file=str(output_file),
        required_document_ids=["HR-HBK-2025-v1.4"],
    )

    monkeypatch.setattr("greenloop_rag_crew.runner.list_questions", lambda: [fake_question])

    with pytest.raises(FileExistsError):
        run_question("remote_work_and_revenue", llm=_safe_llm(), dry_run=True)


def test_each_question_gets_its_own_output_path():
    results = run_all(llm=_safe_llm(), overwrite=True, dry_run=True)
    output_paths = [result.output_path for result in results]

    assert output_paths == [
        "output/report_01_remote_work_and_revenue.md",
        "output/report_02_accuracy_comparison.md",
        "output/report_03_sla_and_revenue_loss.md",
    ]
    assert len(set(output_paths)) == 3


def test_all_three_dry_runs_use_sequential_process():
    results = run_all(llm=_safe_llm(), overwrite=True, dry_run=True)

    assert len(results) == 3
    assert all(result.process_type == "sequential" for result in results)
    assert all(
        result.task_order == ["research_task", "fact_check_task", "report_task"]
        for result in results
    )
