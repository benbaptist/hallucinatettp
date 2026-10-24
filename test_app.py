import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

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
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    async def post(self, url, **kwargs):
        self.request = (url, kwargs)
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        with self.lock:
            self.active -= 1
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


def test_base_url_headers_are_omitted_for_paths_but_kept_at_root(client):
    fake = FakeClient('{"status":200,"headers":{},"body":"ok"}')
    client.app.state.client = fake
    headers = {
        "Host": "hallucinatettp.benbaptist.com",
        "Origin": "https://hallucinatettp.benbaptist.com",
        "Referer": "https://hallucinatettp.benbaptist.com/",
        "X-Forwarded-Host": "hallucinatettp.benbaptist.com",
        "X-Input": "preserved",
    }

    client.get("/cnn.com", headers=headers)
    path_request = fake.request[1]["json"]["messages"][1]["content"].lower()
    assert "get /cnn.com http/" in path_request
    assert "hallucinatettp.benbaptist.com" not in path_request
    assert "x-input: preserved" in path_request

    client.get("/", headers=headers)
    root_request = fake.request[1]["json"]["messages"][1]["content"].lower()
    assert "get / http/" in root_request
    assert "host: hallucinatettp.benbaptist.com" in root_request
    assert "origin: https://hallucinatettp.benbaptist.com" in root_request
    assert "referer: https://hallucinatettp.benbaptist.com/" in root_request
    assert "x-forwarded-host: hallucinatettp.benbaptist.com" in root_request


def test_missing_key_returns_service_unavailable(client, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    response = client.get("/hello")
    assert response.status_code == 503


def test_malformed_model_output_returns_bad_gateway(client):
    client.app.state.client = FakeClient("not json")
    response = client.get("/hello")
    assert response.status_code == 502


def test_concurrent_inference_limit(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_CONCURRENT_INFERENCES", 2)
    client.app.state.inference_semaphore = asyncio.Semaphore(2)
    fake = FakeClient('{"status":200,"headers":{},"body":"ok"}')
    client.app.state.client = fake

    def send_request(_):
        return client.get("/hello").status_code

    with ThreadPoolExecutor(max_workers=5) as executor:
        statuses = list(executor.map(send_request, range(5)))

    assert statuses == [200] * 5
    assert fake.max_active == 2


def test_model_cannot_set_hop_by_hop_headers():
    result = server.parse_llm_response(json.dumps({
        "status": 200,
        "headers": {"Connection": "close", "X-Safe": "ok", "Content-Length": "999"},
        "body": "ok",
    }))
    assert result["headers"] == {"X-Safe": "ok"}
