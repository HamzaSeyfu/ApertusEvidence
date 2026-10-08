import io
import json
import urllib.request

import pytest

from apertus_evidence.backend import (
    BackendError,
    OpenAICompatibleJsonBackend,
    _chat_completions_url,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_chat_url_normalization() -> None:
    assert _chat_completions_url("http://localhost:8000") == (
        "http://localhost:8000/v1/chat/completions"
    )
    assert _chat_completions_url("http://localhost:8000/v1") == (
        "http://localhost:8000/v1/chat/completions"
    )


def test_backend_parses_json_model_output(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"label":"SUPPORT","score":0.9,"rationale":"Direct."}\n```'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleJsonBackend(
        base_url="http://localhost:8000",
        model="swiss-ai/Apertus-8B-Instruct",
        api_key="secret",
        timeout_seconds=7,
    )

    result = backend.generate_json(system="system", user="user")

    assert result["label"] == "SUPPORT"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["model"] == "swiss-ai/Apertus-8B-Instruct"
    assert captured["timeout"] == 7
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_backend_rejects_non_json_content(monkeypatch) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(
            {"choices": [{"message": {"content": "definitely true"}}]}
        ),
    )
    backend = OpenAICompatibleJsonBackend("http://localhost:8000", "apertus")

    with pytest.raises(BackendError):
        backend.generate_json(system="system", user="user")
