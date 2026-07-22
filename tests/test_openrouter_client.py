import json

from greenloop_rag_crew import openrouter_client
from greenloop_rag_crew.llm import GenerationSettings, LLMSettings


class FakeStreamingResponse:
    def __iter__(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"content":"Green"}}]}\n',
                b'data: {"choices":[{"delta":{"content":"Loop"}}]}\n',
                b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n',
                b'data: [DONE]\n',
            ]
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_openrouter_stream_uses_openai_compatible_request_without_ollama_fields(monkeypatch):
    captured = {}
    tokens = []

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeStreamingResponse()

    monkeypatch.setattr(openrouter_client, "urlopen", fake_urlopen)
    response = openrouter_client.generate_chat_stream(
        settings=LLMSettings(
            provider="openrouter",
            model="qwen/qwen3-8b",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key-not-for-logs",
        ),
        generation=GenerationSettings(0.1, 400, 3072, False, "30m"),
        messages=[{"role": "user", "content": "Test"}],
        on_token=tokens.append,
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert "localhost" not in captured["url"]
    assert captured["authorization"] == "Bearer test-key-not-for-logs"
    assert captured["payload"] == {
        "model": "qwen/qwen3-8b",
        "messages": [{"role": "user", "content": "Test"}],
        "stream": True,
        "temperature": 0.1,
        "max_tokens": 400,
        "reasoning": {"enabled": False},
    }
    assert tokens == ["Green", "Loop"]
    assert response.content == "GreenLoop"
    assert response.request_count == 1
    assert response.output_tokens == 2


def test_openrouter_client_never_includes_api_key_in_safe_errors(monkeypatch):
    def fake_urlopen(*_args, **_kwargs):
        raise openrouter_client.URLError("connection failed")

    monkeypatch.setattr(openrouter_client, "urlopen", fake_urlopen)
    try:
        openrouter_client.generate_chat_stream(
            settings=LLMSettings(
                "openrouter",
                "qwen/qwen3-8b",
                "https://openrouter.ai/api/v1",
                api_key="never-display-this-key",
            ),
            generation=GenerationSettings(0.1, 400, 3072, False, "30m"),
            messages=[{"role": "user", "content": "Test"}],
            on_token=lambda _token: None,
        )
    except openrouter_client.OpenRouterGenerationError as exc:
        assert "never-display-this-key" not in str(exc)
        assert "OpenRouter" in str(exc)
    else:
        raise AssertionError("Expected the mocked connection failure.")
