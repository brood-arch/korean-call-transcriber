import json


def test_get_llm_config_prefers_generic_env(monkeypatch):
    from src.extract.client import get_llm_config

    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("ZAI_API_KEY", "zai-key")
    monkeypatch.setenv("ZAI_BASE_URL", "https://zai.example/v1")

    assert get_llm_config() == {
        "api_key": "generic-key",
        "base_url": "https://llm.example/v1",
        "model": "custom-model",
    }


def test_call_extract_payload_uses_generic_model_without_glm_thinking(monkeypatch):
    from src.extract import client

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"summary": {"one_line": "ok"}}'}}]}).encode()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = client.call_llm_extract("key", "content")

    assert result["summary"]["one_line"] == "ok"
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-test"
    assert "thinking" not in captured["payload"]
