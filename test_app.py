import asyncio
import base64
import json
import os
import logging
import re
import threading

import httpx
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import app as server


class FakeResponse:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage or {"prompt_tokens": 12, "completion_tokens": 7}

    def raise_for_status(self):
        return None

    def json(self):
        result = {"choices": [{"message": {"content": self.content}}], "usage": self.usage}
        return result


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.request = None
        self.event_hooks = {
            "request": [server.archive_upstream_request],
            "response": [server.archive_upstream_response],
        }
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.usage = {"prompt_tokens": 12, "completion_tokens": 7}

    async def post(self, url, **kwargs):
        request = httpx.Request("POST", url, headers=kwargs["headers"], json=kwargs["json"])
        for hook in self.event_hooks["request"]:
            await hook(request)
        self.request = (url, kwargs)
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        with self.lock:
            self.active -= 1
        result = FakeResponse(self.content, self.usage)
        upstream_response = httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": self.content}}], "usage": self.usage},
        )
        for hook in self.event_hooks["response"]:
            await hook(upstream_response)
        return result

    async def aclose(self):
        return None


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "API_KEY", "test-key")
    monkeypatch.setattr(server, "HTTP_ARCHIVE_DIR", tmp_path / "http-archive")
    with TestClient(server.app) as test_client:
        yield test_client


def test_raw_request_is_sent_and_model_response_returned(client, caplog):
    caplog.set_level(logging.INFO, logger="hallucinatettp")
    fake = FakeClient("Status: 201 Created\nContent-Type: text/plain\nX-Test: yes\n\ncreated")
    client.app.state.client = fake
    response = client.post("/items?q=one", content=b"payload", headers={"X-Input": "value"})
    assert response.status_code == 201
    assert response.text == "created"
    assert response.headers["x-test"] == "yes"
    sent = fake.request[1]["json"]["messages"][1]["content"]
    assert sent.startswith("POST /?q=one HTTP/")
    assert "host: items" in sent
    assert "x-input: value" in sent
    assert sent.endswith("base64:cGF5bG9hZA==")
    assert 'input_tokens=12' in caplog.text
    assert 'output_tokens=7' in caplog.text
    assert 'input_tokens_per_sec=' in caplog.text
    assert 'output_tokens_per_sec=' in caplog.text
    assert 'input_tokens_total=12' in caplog.text
    assert 'output_tokens_total=7' in caplog.text

    response = client.post("/items", content=b"again")
    assert response.status_code == 201
    assert "input_tokens_total=24" in caplog.text
    assert "output_tokens_total=14" in caplog.text


def test_http_archive_saves_incoming_outgoing_and_upstream_messages(client):
    client.app.state.client = FakeClient("Content-Type: text/plain\nX-Result: ok\n\nreply")
    response = client.post("/archive.test?x=1", content=b"request\x00data", headers={"X-Test": "input"})
    assert response.status_code == 200

    archive_files = list(server.HTTP_ARCHIVE_DIR.glob("*.json"))
    assert len(archive_files) == 1
    archive_file = archive_files[0]
    payload = json.loads(archive_file.read_text())
    request_id = archive_file.stem
    assert payload["request_id"] == request_id
    records = payload["messages"]
    assert [record["direction"] for record in records] == [
        "incoming_request",
        "upstream_request",
        "upstream_response",
        "outgoing_response",
    ]
    incoming, upstream_request, upstream_response, outgoing = records
    assert incoming["method"] == "POST"
    assert incoming["target"] == "/archive.test?x=1"
    assert base64.b64decode(incoming["body_base64"]) == b"request\x00data"
    assert upstream_request["url"].endswith("/chat/completions")
    assert upstream_response["status_code"] == 200
    assert outgoing["status_code"] == 200
    assert base64.b64decode(outgoing["body_base64"]) == b"reply"
    assert os.stat(archive_file).st_mode & 0o777 == 0o600


def test_path_prefix_is_host_and_headers_are_simulated(client):
    fake = FakeClient("Content-Type: text/plain\n\nok")
    client.app.state.client = fake
    headers = {
        "Host": "hallucinatettp.benbaptist.com",
        "Origin": "https://hallucinatettp.benbaptist.com",
        "Referer": "https://hallucinatettp.benbaptist.com/",
        "Forwarded": "host=hallucinatettp.benbaptist.com;proto=https",
        "X-Forwarded-Host": "hallucinatettp.benbaptist.com",
        "X-Forwarded-Proto": "https",
        "X-Original-Host": "hallucinatettp.benbaptist.com",
        "X-Input": "preserved",
    }

    client.get("/benbaptist.com/account?tab=posts", headers=headers)
    sent = fake.request[1]["json"]["messages"][1]["content"].lower()
    assert sent.startswith("get /account?tab=posts http/")
    assert "host: benbaptist.com" in sent
    assert "origin: https://benbaptist.com" in sent
    assert "referer: https://benbaptist.com/account?tab=posts" in sent
    assert "forwarded: host=benbaptist.com;proto=https" in sent
    assert "x-forwarded-host: benbaptist.com" in sent
    assert "x-forwarded-proto: https" in sent
    assert "x-original-host: benbaptist.com" in sent
    assert "x-input: preserved" in sent
    assert "hallucinatettp.benbaptist.com" not in sent

    client.get("/", headers=headers)
    root_request = fake.request[1]["json"]["messages"][1]["content"].lower()
    assert "get / http/" in root_request
    assert "host: hallucinatettp.benbaptist.com" in root_request


def test_path_prefix_with_no_path_maps_to_origin_root(client):
    client.app.state.client = FakeClient("Content-Type: text/plain\n\nok")
    client.get("/cnn.com")
    sent = client.app.state.client.request[1]["json"]["messages"][1]["content"].lower()
    assert sent.startswith("get / http/")
    assert "host: cnn.com" in sent


@pytest.mark.parametrize("path", ["/bad_host/path", "/host:0/path", "/host:65536/path"])
def test_invalid_destination_host_is_bad_request(client, path):
    response = client.get(path)
    assert response.status_code == 400


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
