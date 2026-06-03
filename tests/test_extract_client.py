import json
import urllib.error
from unittest.mock import Mock


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
        "disable_thinking": "auto",
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


def test_call_llm_json_retries_after_rate_limit(monkeypatch):
    from src.extract import client

    calls = {"count": 0}
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode()

    def fake_urlopen(req, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "too many", {}, None)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(client.time, "sleep", lambda seconds: sleeps.append(seconds))

    result, usage = client.call_llm_json("prompt", api_key="key", response_format=False)

    assert result == {"ok": True}
    assert calls["count"] == 2
    assert sleeps == [30]
    assert usage == {}


def test_call_llm_json_returns_none_after_json_errors(monkeypatch):
    from src.extract import client

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "not-json"}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=FakeResponse()))
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)

    result, usage = client.call_llm_json("prompt", api_key="key", response_format=False)

    assert result is None
    assert "error" in usage or usage == {}


def test_call_llm_json_glm_disables_thinking(monkeypatch):
    from src.extract import client

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode()

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setenv("LLM_MODEL", "glm-test")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client.call_llm_json("prompt", api_key="key")

    assert captured["payload"]["thinking"] == {"type": "disabled"}
