import json

from greenloop_rag_crew.llm import GenerationSettings, LLMSettings
from greenloop_rag_crew import ollama_client


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeStreamingResponse(FakeResponse):
    def __iter__(self):
        return iter(
            [
                b'{"message":{"content":"hel"},"done":false}\n',
                b'{"message":{"content":"lo"},"done":false}\n',
                b'{"message":{"content":""},"done":true,"prompt_eval_count":4,"eval_count":2,"load_duration":0,"prompt_eval_duration":2,"eval_duration":3}\n',
            ]
        )


def test_direct_ollama_request_includes_think_false_and_keep_alive(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "message": {"content": "healthy"},
                "prompt_eval_count": 4,
                "eval_count": 1,
                "load_duration": 1,
                "prompt_eval_duration": 2,
                "eval_duration": 3,
            }
        )

    monkeypatch.setattr(ollama_client, "urlopen", fake_urlopen)
    result = ollama_client.generate_chat(
        settings=LLMSettings("ollama", "ollama/qwen3:8b", "http://localhost:11434"),
        generation=GenerationSettings(0.1, 400, 3072, False, "30m"),
        messages=[{"role": "user", "content": "safe test"}],
    )

    assert captured["payload"]["think"] is False
    assert captured["payload"]["keep_alive"] == "30m"
    assert captured["payload"]["options"]["num_predict"] == 400
    assert captured["payload"]["options"]["num_ctx"] == 3072
    assert result.request_count == 1
    assert result.thinking_present is False
    assert result.think_requested is False


def test_direct_ollama_stream_yields_tokens_incrementally_and_completes_once(monkeypatch):
    captured = {}
    tokens = []

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return FakeStreamingResponse({})

    monkeypatch.setattr(ollama_client, "urlopen", fake_urlopen)
    result = ollama_client.generate_chat_stream(
        settings=LLMSettings("ollama", "ollama/qwen3:8b", "http://localhost:11434"),
        generation=GenerationSettings(0.1, 400, 3072, False, "30m"),
        messages=[{"role": "user", "content": "safe test"}],
        on_token=tokens.append,
    )

    assert captured["payload"]["stream"] is True
    assert captured["payload"]["think"] is False
    assert tokens == ["hel", "lo"]
    assert result.content == "hello"
    assert result.request_count == 1
    assert result.output_tokens == 2
    assert result.model_already_loaded is True
