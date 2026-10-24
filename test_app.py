import json

import pytest
from fastapi.testclient import TestClient

import app as server


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.request = None

    async def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return FakeResponse(self.content)

    async def aclose(self):
        return None


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "test-key")
    with TestClient(server.app) as test_client:
        yield test_client


def test_raw_request_is_sent_and_model_response_returned(client):
    fake = FakeClient('{"status":201,"headers":{"X-Test":"yes"},"body":"created"}')
    client.app.state.client = fake
    response = client.post("/items?q=one", content=b"payload", headers={"X-Input": "value"})
    assert response.status_code == 201
    assert response.text == "created"
    assert response.headers["x-test"] == "yes"
    sent = fake.request[1]["json"]["messages"][1]["content"]
    assert sent.startswith("POST /items?q=one HTTP/")
    assert "x-input: value" in sent
    assert sent.endswith("base64:cGF5bG9hZA==")


def test_missing_key_returns_service_unavailable(client, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    response = client.get("/hello")
    assert response.status_code == 503


def test_malformed_model_output_returns_bad_gateway(client):
    client.app.state.client = FakeClient("not json")
    response = client.get("/hello")
    assert response.status_code == 502


def test_model_cannot_set_hop_by_hop_headers():
    result = server.parse_llm_response(json.dumps({
        "status": 200,
        "headers": {"Connection": "close", "X-Safe": "ok", "Content-Length": "999"},
        "body": "ok",
    }))
    assert result["headers"] == {"X-Safe": "ok"}
