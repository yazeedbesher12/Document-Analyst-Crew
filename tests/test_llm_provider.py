import pytest

from greenloop_rag_crew import llm as llm_module
from greenloop_rag_crew import question_execution as execution_module


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


def test_azure_factory_uses_crewai_supported_azure_fields(monkeypatch):
    created = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(llm_module, "LLM", FakeLLM)
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_LLM_MODEL", "azure/greenloop-deployment")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://greenloop.openai.azure.com/")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")

    llm_module.create_llm()

    assert created["model"] == "azure/greenloop-deployment"
    assert created["endpoint"] == "https://greenloop.openai.azure.com"
    assert created["api_key"] == "test-key"
    assert created["api_version"] == "2024-10-21"
    assert created["max_tokens"] <= 1000


@pytest.mark.parametrize("missing", ["AZURE_LLM_MODEL", "AZURE_API_KEY", "AZURE_ENDPOINT"])
def test_azure_required_variables_are_validated_without_network(monkeypatch, missing):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_LLM_MODEL", "azure/greenloop-deployment")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://greenloop.openai.azure.com")
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(llm_module.AzureConfigurationError, match=missing):
        llm_module.get_provider_settings()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")

    with pytest.raises(llm_module.LLMConfigurationError, match="LLM_PROVIDER"):
        llm_module.get_provider_settings()


def test_azure_preflight_does_not_contact_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_LLM_MODEL", "azure/greenloop-deployment")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://greenloop.openai.azure.com")
    monkeypatch.setattr(
        execution_module,
        "check_ollama_preflight",
        lambda: pytest.fail("Azure preflight must not call Ollama"),
    )

    settings = execution_module.check_llm_preflight()

    assert settings.provider == "azure"
