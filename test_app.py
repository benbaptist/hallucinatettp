import asyncio
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
    fake = FakeClient("Status: 201 Created\nContent-Type: text/plain\nX-Test: yes\n\ncreated")
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
    fake = FakeClient("Content-Type: text/plain\n\nok")
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


@pytest.mark.parametrize(
    ("content", "expected_status", "expected_body", "expected_content_type"),
    [
        ("Content-Type: text/html; charset=utf-8\n\n<h1>Hello</h1>", 200, "<h1>Hello</h1>", "text/html; charset=utf-8"),
        ("Status: 404 Not Found\nContent-Type: text/html\n\n<h1>Missing</h1>", 404, "<h1>Missing</h1>", "text/html"),
        ("<h1>Unformatted output</h1>", 200, "<h1>Unformatted output</h1>", "text/plain; charset=utf-8"),
        ("not json", 200, "not json", "text/plain; charset=utf-8"),
        ('{"status":201,"headers":{"X-Test":"yes"},"body":"created"}', 200, '{"status":201,"headers":{"X-Test":"yes"},"body":"created"}', "text/plain; charset=utf-8"),
    ],
)
def test_model_output_is_passed_through(client, content, expected_status, expected_body, expected_content_type):
    client.app.state.client = FakeClient(content)
    response = client.get("/hello")
    assert response.status_code == expected_status
    assert response.text == expected_body
    assert response.headers["content-type"] == expected_content_type


def test_head_response_has_no_body(client):
    client.app.state.client = FakeClient("Content-Type: text/html\n\nhello")
    response = client.head("/hello")
    assert response.status_code == 200
    assert response.content == b""


def test_concurrent_inference_limit(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_CONCURRENT_INFERENCES", 2)
    client.app.state.inference_semaphore = asyncio.Semaphore(2)
    fake = FakeClient("Content-Type: text/plain\n\nok")
    client.app.state.client = fake

    def send_request(_):
        return client.get("/hello").status_code

    with ThreadPoolExecutor(max_workers=5) as executor:
        statuses = list(executor.map(send_request, range(5)))

    assert statuses == [200] * 5
    assert fake.max_active == 2


def test_model_cannot_set_hop_by_hop_headers():
    response = server.generated_http_response(
        "Content-Type: text/plain\nConnection: close\nContent-Length: 999\nX-Safe: ok\n\nok",
        "GET",
    )
    assert response.headers["x-safe"] == "ok"
    assert "connection" not in response.headers
    assert response.headers["content-length"] == "2"
    assert response.body == b"ok"
