import pytest

from greenloop_rag_crew import llm as llm_module
from greenloop_rag_crew import question_execution as execution_module


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch):
    """Keep provider-routing assertions independent from a developer's .env file."""

    monkeypatch.setattr(llm_module, "load_dotenv", lambda: False)


def test_default_provider_is_local_ollama(monkeypatch):
    for variable in ("LLM_PROVIDER", "OLLAMA_MODEL", "MODEL", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(variable, raising=False)

    settings = llm_module.get_provider_settings()

    assert settings.provider == "ollama"
    assert settings.model == "ollama/qwen3:8b"
    assert settings.base_url == "http://localhost:11434"


def test_ollama_supports_docker_host_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/")

    settings = llm_module.get_provider_settings()

    assert settings.model == "ollama/qwen3:8b"
    assert settings.base_url == "http://host.docker.internal:11434"


def test_ollama_is_selected_even_when_an_openrouter_key_exists(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OPENROUTER_API_KEY", "should-not-change-provider")

    settings = llm_module.get_provider_settings()

    assert settings.provider == "ollama"
    assert settings.model == "ollama/qwen3:8b"


def test_openrouter_factory_uses_only_openrouter_fields(monkeypatch):
    created = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(llm_module, "LLM", FakeLLM)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3-8b")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    llm_module.create_llm()

    assert created["model"] == "openrouter/qwen/qwen3-8b"
    assert created["base_url"] == "https://openrouter.ai/api/v1"
    assert created["api_key"] == "test-key"
    assert "additional_params" not in created
    assert created["max_tokens"] == 400


def test_openrouter_requires_key_without_network(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(llm_module.OpenRouterConfigurationError, match="OPENROUTER_API_KEY"):
        llm_module.get_provider_settings()


def test_openrouter_rejects_a_localhost_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://localhost:11434")

    with pytest.raises(llm_module.OpenRouterConfigurationError, match="not localhost"):
        llm_module.get_provider_settings()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")

    with pytest.raises(llm_module.LLMConfigurationError, match="LLM_PROVIDER"):
        llm_module.get_provider_settings()


def test_openrouter_preflight_does_not_contact_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3-8b")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(
        execution_module,
        "check_ollama_preflight",
        lambda: pytest.fail("OpenRouter preflight must not call Ollama"),
    )

    settings = execution_module.check_llm_preflight()

    assert settings.provider == "openrouter"
