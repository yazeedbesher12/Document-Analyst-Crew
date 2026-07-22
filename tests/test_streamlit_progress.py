import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit_app import _StageProgress, _StreamingAnswer, _generation_metrics_text


class FakeStatus:
    def __init__(self):
        self.writes = []
        self.updates = []

    def write(self, value):
        self.writes.append(value)

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakePlaceholder:
    def __init__(self):
        self.markdowns = []

    def markdown(self, value):
        self.markdowns.append(value)


def test_stage_progress_displays_completed_stage_durations():
    status = FakeStatus()
    progress = _StageProgress(status)

    progress.advance("Researching documents", 1.2)
    progress.advance("Verifying claims", 2.3)
    progress.advance("Writing report", 3.4)
    progress.advance("Completed", 4.5)
    progress.complete(11.4)

    assert status.writes == [
        "Preparing document index: 1.2s",
        "Researching documents: 2.3s",
        "Verifying claims: 3.4s",
        "Writing report: 4.5s",
    ]
    assert status.updates == [
        {"label": "Researching documents", "state": "running"},
        {"label": "Verifying claims", "state": "running"},
        {"label": "Writing report", "state": "running"},
        {"label": "Completed (11.4s)", "state": "complete"},
    ]


def test_streamed_answer_updates_incrementally_and_keeps_completed_markdown():
    status = FakeStatus()
    progress = _StageProgress(status)
    progress.advance("Writing answer", 0.0)
    placeholder = FakePlaceholder()
    answer = _StreamingAnswer(placeholder, progress)

    answer.push("Hello")
    answer.push(" world")
    answer.replace_with_complete("## Direct Answer\nHello world")

    assert placeholder.markdowns[:2] == ["Hello\n\n...", "Hello world\n\n..."]
    assert placeholder.markdowns[-1] == "## Direct Answer\nHello world"
    assert any(update["label"].startswith("Writing answer (") for update in status.updates)


def test_generation_metrics_are_safe_and_displayable():
    text = _generation_metrics_text(
        {
            "time_to_first_token_seconds": 0.4,
            "generation_seconds": 2.0,
            "generated_output_tokens": 10,
            "tokens_per_second": 5.0,
            "model_already_loaded": True,
        },
        1,
    )

    assert "LLM requests: 1" in text
    assert "First token: 0.4s" in text
    assert "Model already loaded: yes" in text
