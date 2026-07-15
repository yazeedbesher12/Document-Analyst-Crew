import json
import urllib.error

from greenloop_rag_crew.diagnostics import ollama_tool_check as check_module


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


def test_ollama_reachable_and_model_installed(monkeypatch):
    payload = json.dumps({"models": [{"name": "qwen3:8b"}]}).encode("utf-8")
    monkeypatch.setattr(
        check_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(payload),
    )
    monkeypatch.setenv("MODEL", "ollama/qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    health = check_module.check_ollama_health()

    assert health.reachable is True
    assert health.model_installed is True
    assert health.models == ["qwen3:8b"]


def test_ollama_unreachable(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(check_module.urllib.request, "urlopen", fail)

    health = check_module.check_ollama_health()

    assert health.reachable is False
    assert health.model_installed is False
    assert health.error_type == "connection_refused"


def test_ollama_model_missing(monkeypatch):
    payload = json.dumps({"models": [{"name": "other:latest"}]}).encode("utf-8")
    monkeypatch.setattr(
        check_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    health = check_module.check_ollama_health()

    assert health.reachable is True
    assert health.model_installed is False
    assert health.error_type == "model_missing"


def test_ollama_invalid_response(monkeypatch):
    monkeypatch.setattr(
        check_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(b"not-json"),
    )

    health = check_module.check_ollama_health()

    assert health.reachable is True
    assert health.error_type == "invalid_response"
