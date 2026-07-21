import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit_app import _StageProgress


class FakeStatus:
    def __init__(self):
        self.writes = []
        self.updates = []

    def write(self, value):
        self.writes.append(value)

    def update(self, **kwargs):
        self.updates.append(kwargs)


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
