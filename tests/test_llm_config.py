import pytest

from greenloop_rag_crew import llm as llm_module


def test_valid_model_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("MODEL", "ollama/qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")

    model, base_url = llm_module.get_llm_settings()

    assert model == "ollama/qwen3:8b"
    assert base_url == "http://localhost:11434"


def test_missing_model_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "")
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    with pytest.raises(ValueError, match="OLLAMA_MODEL must not be empty"):
        llm_module.get_llm_settings()


def test_invalid_provider_prefix_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "openai/gpt-4")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    with pytest.raises(ValueError, match="OLLAMA_MODEL"):
        llm_module.get_llm_settings()


def test_valid_base_url_and_trailing_slash_handling(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://localhost:11434///")

    assert llm_module.get_llm_settings()[1] == "https://localhost:11434"


def test_invalid_base_url_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "localhost:11434")

    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        llm_module.get_llm_settings()


def test_create_llm_uses_local_ollama_settings(monkeypatch):
    created = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(llm_module, "LLM", FakeLLM)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm_module.create_llm()

    assert created["model"] == "ollama/qwen3:8b"
    assert created["base_url"] == "http://localhost:11434"
    assert created["temperature"] == 0.1
    assert created["top_p"] == 0.95
    assert created["max_tokens"] == 900
    assert created["timeout"] == llm_module.DEFAULT_TIMEOUT_SECONDS
    assert created["additional_params"] == {"extra_body": {"think": False}}


def test_latency_controls_are_read_from_the_environment(monkeypatch):
    created = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(llm_module, "LLM", FakeLLM)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("LLM_MAX_TOKENS", "512")
    monkeypatch.setenv("OLLAMA_THINK", "false")

    llm_module.create_llm()

    assert created["temperature"] == 0.1
    assert created["max_tokens"] == 512
    assert created["additional_params"] == {"extra_body": {"think": False}}
