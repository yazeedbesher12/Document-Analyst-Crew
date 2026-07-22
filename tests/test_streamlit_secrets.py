import os

from streamlit.runtime.secrets import StreamlitSecretNotFoundError

from greenloop_rag_crew.streamlit_secrets import apply_streamlit_secrets


def test_root_level_streamlit_secrets_fill_missing_environment_values(monkeypatch):
    for name in ("LLM_PROVIDER", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
        monkeypatch.delenv(name, raising=False)

    apply_streamlit_secrets(
        {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test-secret",
            "OPENROUTER_MODEL": "qwen/qwen3-8b",
        }
    )

    assert os.environ["LLM_PROVIDER"] == "openrouter"
    assert os.environ["OPENROUTER_API_KEY"] == "test-secret"
    assert os.environ["OPENROUTER_MODEL"] == "qwen/qwen3-8b"


def test_existing_environment_value_overrides_streamlit_secret(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    apply_streamlit_secrets({"LLM_PROVIDER": "openrouter"})

    assert os.environ["LLM_PROVIDER"] == "ollama"


def test_missing_streamlit_secrets_are_treated_as_empty(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    class MissingSecrets:
        def get(self, _name):
            raise StreamlitSecretNotFoundError("No secrets found")

    apply_streamlit_secrets(MissingSecrets())

    assert "LLM_PROVIDER" not in os.environ
