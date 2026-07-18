import runpy
from pathlib import Path


create_output_path = runpy.run_path(
    Path(__file__).resolve().parents[1] / "manual_run.py"
)["create_output_path"]


def test_each_question_gets_a_unique_markdown_report_path(tmp_path):
    first = create_output_path("What is the remote-work policy?", tmp_path)
    second = create_output_path("What laboratory accuracy was achieved?", tmp_path)
    repeated = create_output_path("What is the remote-work policy?", tmp_path)

    assert len({first, second, repeated}) == 3
    assert all(path.parent == tmp_path for path in (first, second, repeated))
    assert all(path.suffix == ".md" for path in (first, second, repeated))


def test_arabic_question_produces_a_readable_filename(tmp_path):
    path = create_output_path("ما هي سياسة العمل عن بعد؟", Path(tmp_path))

    assert "سياسة-العمل-عن-بعد" in path.name
