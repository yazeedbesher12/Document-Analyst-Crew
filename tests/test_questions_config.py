from pathlib import Path

import pytest

from greenloop_rag_crew.config_loader import (
    ConfigError,
    VALID_DOCUMENT_IDS,
    load_questions_config,
)


def test_exactly_three_questions_load():
    config = load_questions_config()

    assert [question.id for question in config.questions] == [
        "remote_work_and_revenue",
        "accuracy_comparison",
        "sla_and_revenue_loss",
    ]


def test_question_ids_are_unique():
    config = load_questions_config()
    ids = [question.id for question in config.questions]

    assert len(ids) == len(set(ids))


def test_output_files_are_unique_and_safe_markdown_paths():
    config = load_questions_config()
    output_files = [question.output_file for question in config.questions]

    assert len(output_files) == len(set(output_files))
    for output_file in output_files:
        path = Path(output_file)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert path.parts[0] == "output"
        assert path.suffix == ".md"


def test_required_document_ids_are_valid():
    config = load_questions_config()

    for question in config.questions:
        assert question.required_document_ids
        assert set(question.required_document_ids) <= VALID_DOCUMENT_IDS


def test_duplicate_output_paths_are_rejected(tmp_path):
    path = tmp_path / "questions.jsonc"
    path.write_text(
        """
        {
          questions: [
            {id:"remote_work_and_revenue", question:"q1", output_file:"output/a.md", required_document_ids:["HR-HBK-2025-v1.4"]},
            {id:"accuracy_comparison", question:"q2", output_file:"output/a.md", required_document_ids:["PRD-GLX1-2025-v2.1"]},
            {id:"sla_and_revenue_loss", question:"q3", output_file:"output/c.md", required_document_ids:["FIN-Q3-2025-v1.0"]},
          ],
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="output paths must be unique"):
        load_questions_config(path)


def test_unsafe_output_paths_are_rejected(tmp_path):
    path = tmp_path / "questions.jsonc"
    text = Path("src/greenloop_rag_crew/config/questions.jsonc").read_text(encoding="utf-8")
    path.write_text(text.replace("output/report_01_remote_work_and_revenue.md", "../bad.md"), encoding="utf-8")

    with pytest.raises(ConfigError, match="output_file"):
        load_questions_config(path)
