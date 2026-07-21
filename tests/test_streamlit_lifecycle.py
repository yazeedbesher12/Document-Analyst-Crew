import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit_app


def test_streamlit_reruns_reuse_the_cached_retrieval_service(monkeypatch):
    calls = []
    service = object()
    streamlit_app._streamlit_retrieval_service.clear()
    monkeypatch.setattr(
        streamlit_app,
        "get_retrieval_service",
        lambda: calls.append("created") or service,
    )

    try:
        first = streamlit_app._streamlit_retrieval_service()
        second = streamlit_app._streamlit_retrieval_service()
    finally:
        streamlit_app._streamlit_retrieval_service.clear()

    assert first is service
    assert second is service
    assert calls == ["created"]
