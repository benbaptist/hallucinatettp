from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import logging
import os
import re
import time
import uuid
from functools import partial
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

HTTP_ARCHIVE_PATH = Path(os.getenv("HTTP_ARCHIVE_PATH", "http-archive/http-archive.jsonl"))

ARCHIVE_REQUEST_ID: ContextVar[str | None] = ContextVar("archive_request_id", default=None)
API_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(1024 * 1024)))
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
if MAX_CONCURRENT_INFERENCES < 1:
    raise ValueError("MAX_CONCURRENT_INFERENCES must be at least 1")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("hallucinatettp")
SYSTEM_PROMPT = """PROMPT — LLM AS A CGI RESPONSE GENERATOR
========================================

ROLE
You are the response generator of a CGI program. The server runs you once per
request and uses your CGI header block and body to construct an HTTP response.

INPUT
You receive the raw HTTP request line and headers. For a browser URL such as
/example.com/path, the first path segment has already been converted into the
target Host and related origin headers; the request line contains only /path.
A URL with just / uses the actual incoming host. A non-empty request body is
provided as base64:<data> so arbitrary bytes are preserved.

TASK — produce a complete HTTP response in exactly this shape:

  1. CGI header block:
       Content-Type: text/html; charset=utf-8
       Status: <3-digit code>          (omit for 200)
     followed by EXACTLY ONE blank line.
  2. The response body, which may be whatever the server should send to the client. It could be literally anything, including HTML, JSON, binary data, or even another HTTP response. Just BE UNIQUE!
"""


def archive_http_message(request_id: str, direction: str, **fields) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "direction": direction,
        **fields,
    }
    encoded = (json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
    HTTP_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(HTTP_ARCHIVE_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.chmod(HTTP_ARCHIVE_PATH, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def headers_as_pairs(headers) -> list[list[str]]:
    return [[name.decode("latin-1"), value.decode("latin-1")] for name, value in headers]


class HTTPArchiveMiddleware:
    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        return getattr(self.app, name)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = scope.get("state", {}).get("request_id") or ARCHIVE_REQUEST_ID.get() or uuid.uuid4().hex[:12]
        scope.setdefault("state", {})["request_id"] = request_id
        token = ARCHIVE_REQUEST_ID.set(request_id)
        loop = asyncio.get_running_loop()
        request_body = bytearray()
        request_complete = False

        async def capture_request():
            nonlocal request_complete
            message = await receive()
            if message["type"] == "http.request":
                request_body.extend(message.get("body", b""))
                request_complete = not message.get("more_body", False)
            return message

        response_status = None
        response_headers = []
        response_body = bytearray()
        response_complete = False

        async def capture_response(message):
            nonlocal response_status, response_headers, response_complete
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = headers_as_pairs(message.get("headers", []))
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))
                response_complete = not message.get("more_body", False)
            await send(message)

        try:
            await self.app(scope, capture_request, capture_response)
        finally:
            raw_path = scope.get("raw_path", scope["path"].encode("utf-8")).decode("latin-1")
            query = scope.get("query_string", b"").decode("latin-1")
            target = raw_path + ("?" + query if query else "")
            version = scope.get("http_version", "1.1")
            await loop.run_in_executor(
                None,
                partial(
                    archive_http_message,
                    request_id,
                    "incoming_request",
                    protocol=f"HTTP/{version}",
                    method=scope["method"],
                    target=target,
                    scheme=scope.get("scheme"),
                    headers=headers_as_pairs(scope.get("headers", [])),
                    body_base64=base64.b64encode(request_body).decode("ascii"),
                    complete=request_complete,
                ),
            )
            if response_status is not None:
                await loop.run_in_executor(
                    None,
                    partial(
                        archive_http_message,
                        request_id,
                        "outgoing_response",
                        protocol=f"HTTP/{version}",
                        status_code=response_status,
                        headers=response_headers,
                        body_base64=base64.b64encode(response_body).decode("ascii"),
                        complete=response_complete,
                    ),
                )
            ARCHIVE_REQUEST_ID.reset(token)


async def archive_upstream_request(request: httpx.Request) -> None:
    await request.aread()
    archive_http_message(
        ARCHIVE_REQUEST_ID.get() or uuid.uuid4().hex[:12],
        "upstream_request",
        method=request.method,
        url=str(request.url),
        headers=headers_as_pairs(request.headers.raw),
        body_base64=base64.b64encode(request.content).decode("ascii"),
    )


async def archive_upstream_response(response: httpx.Response) -> None:
    await response.aread()
    archive_http_message(
        ARCHIVE_REQUEST_ID.get() or uuid.uuid4().hex[:12],
        "upstream_response",
        protocol=response.http_version,
        method=response.request.method,
        url=str(response.request.url),
        status_code=response.status_code,
        headers=headers_as_pairs(response.headers.raw),
        body_base64=base64.b64encode(response.content).decode("ascii"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        event_hooks={"request": [archive_upstream_request], "response": [archive_upstream_response]},
    )
    app.state.inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
    app.state.input_tokens_total = 0
    app.state.output_tokens_total = 0
    try:
        yield
    finally:
        await app.state.client.aclose()


fastapi_app = FastAPI(lifespan=lifespan)
app = fastapi_app
app.add_middleware(HTTPArchiveMiddleware)


@fastapi_app.middleware("http")
async def log_http_requests(request: Request, call_next):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    started_at = time.monotonic()
    path = request.url.path
    logger.info("request_started id=%s method=%s path=%s", request_id, request.method, path)
    response = None
    try:
        response = await call_next(request)
        return response
    except Exception:
        logger.exception("request_failed id=%s", request_id)
        raise
    finally:
        logger.info(
            "request_finished id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            path,
            response.status_code if response is not None else 500,
            (time.monotonic() - started_at) * 1000,
        )


HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
}
BODYLESS_STATUS_CODES = {204, 205, 304}
DESTINATION_HOST_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
    r"(?::([0-9]{1,5}))?\Z"
)


def request_as_http(request: Request, body: bytes) -> str:
    target = request.url.path
    destination_host = None
    if target != "/":
        destination_host, separator, remaining_path = target[1:].partition("/")
        host_match = DESTINATION_HOST_RE.fullmatch(destination_host)
        if (
            not host_match
            or len(destination_host.rsplit(":", 1)[0]) > 253
            or (host_match[1] and not 1 <= int(host_match[1]) <= 65535)
        ):
            raise ValueError("Invalid destination host")
        target = "/" + remaining_path if separator else "/"

    if request.url.query:
        target += "?" + request.url.query

    header_items = request.scope["headers"]
    headers = []
    if destination_host is None:
        headers = [
            f"{name.decode('latin-1')}: {value.decode('latin-1')}"
            for name, value in header_items
        ]
    else:
        scheme = request.url.scheme
        for name, value in header_items:
            if name.lower() == b"x-forwarded-proto":
                forwarded_scheme = value.decode("latin-1").split(",", 1)[0].strip().lower()
                if forwarded_scheme in {"http", "https"}:
                    scheme = forwarded_scheme
                break

        origin = f"{scheme}://{destination_host}"
        simulated_headers = {
            "host": destination_host,
            "origin": origin,
            "referer": origin + target,
            "forwarded": f"host={destination_host};proto={scheme}",
            "x-forwarded-host": destination_host,
            "x-forwarded-proto": scheme,
            "x-original-host": destination_host,
        }
        header_names_seen = set()
        for name, value in header_items:
            name_text = name.decode("latin-1")
            normalized_name = name_text.lower()
            if normalized_name in simulated_headers:
                header_names_seen.add(normalized_name)
                value_text = simulated_headers[normalized_name]
            else:
                value_text = value.decode("latin-1")
            headers.append(f"{name_text}: {value_text}")
        for name, value in simulated_headers.items():
            if name not in header_names_seen:
                headers.append(f"{name}: {value}")

    body_text = "base64:" + base64.b64encode(body).decode("ascii") if body else ""
    headers_text = "\r\n".join(headers)
    return f"{request.method} {target} HTTP/{request.scope.get('http_version', '1.1')}\r\n{headers_text}\r\n\r\n{body_text}"


def token_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def generated_http_response(content: str, request_method: str) -> Response:
    separator = re.search(r"\r?\n\r?\n", content)
    headers: dict[str, str] = {}
    status_code = 200
    if separator:
        header_block = content[:separator.start()]
        header_lines = header_block.splitlines()
        has_cgi_header = any(
            line.split(":", 1)[0].strip().lower() in {"content-type", "status"}
            for line in header_lines
            if ":" in line
        )
        if header_lines and has_cgi_header and all(":" in line for line in header_lines):
            for line in header_lines:
                name, value = line.split(":", 1)
                name, value = name.strip(), value.strip()
                if name.lower() == "status":
                    match = re.match(r"(\d{3})", value)
                    if match:
                        candidate = int(match.group(1))
                        if 200 <= candidate <= 599:
                            status_code = candidate
                    continue
                if (
                    re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name)
                    and all(char == "\t" or 32 <= ord(char) <= 255 for char in value)
                    and name.lower() not in HOP_BY_HOP_HEADERS
                ):
                    duplicate = next((existing for existing in headers if existing.lower() == name.lower()), None)
                    if duplicate:
                        del headers[duplicate]
                    headers[name] = value
            content = content[separator.end():]

    if request_method == "HEAD" or status_code in BODYLESS_STATUS_CODES:
        content = ""
    return Response(content=content, status_code=status_code, headers=headers, media_type=None if headers else "text/plain")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def generate_response(request: Request, path: str) -> Response:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])

    if not API_KEY:
        logger.warning("request_rejected id=%s reason=missing_api_key", request_id)
        response = Response("OPENAI_API_KEY is not configured", status_code=503, media_type="text/plain")
    else:
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            logger.warning("request_rejected id=%s reason=body_too_large bytes=%d", request_id, len(body))
            response = Response("Request body too large", status_code=413, media_type="text/plain")
        else:
            try:
                raw_request = request_as_http(request, body)
            except ValueError:
                logger.info("request_rejected id=%s reason=invalid_destination_host", request_id)
                response = Response("Invalid destination host", status_code=400, media_type="text/plain")
            else:
                semaphore: asyncio.Semaphore = request.app.state.inference_semaphore
                queued_at = time.monotonic()
                logger.info("inference_queued id=%s", request_id)
                try:
                    async with semaphore:
                        queue_duration_ms = (time.monotonic() - queued_at) * 1000
                        inference_started_at = time.monotonic()
                        logger.info("inference_started id=%s queue_ms=%.1f", request_id, queue_duration_ms)
                        upstream = await request.app.state.client.post(
                            API_URL,
                            headers={"Authorization": f"Bearer {API_KEY}"},
                            json={
                                "model": MODEL,
                                "temperature": 0.2,
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": raw_request},
                                ],
                            },
                        )
                        upstream.raise_for_status()
                        payload = upstream.json()
                        usage = payload.get("usage")
                        if not isinstance(usage, dict):
                            usage = {}
                        input_tokens = token_count(usage.get("prompt_tokens", usage.get("input_tokens")))
                        output_tokens = token_count(usage.get("completion_tokens", usage.get("output_tokens")))
                        generated = payload["choices"][0]["message"]["content"]
                        response = generated_http_response(generated, request.method)
                        inference_duration_ms = (time.monotonic() - inference_started_at) * 1000
                        request.app.state.input_tokens_total += input_tokens or 0
                        request.app.state.output_tokens_total += output_tokens or 0
                        input_rate = input_tokens / (inference_duration_ms / 1000) if input_tokens is not None and inference_duration_ms else None
                        output_rate = output_tokens / (inference_duration_ms / 1000) if output_tokens is not None and inference_duration_ms else None
                        logger.info(
                            "inference_finished id=%s duration_ms=%.1f status=%d input_tokens=%s output_tokens=%s input_tokens_per_sec=%s output_tokens_per_sec=%s input_tokens_total=%d output_tokens_total=%d",
                            request_id,
                            inference_duration_ms,
                            response.status_code,
                            input_tokens if input_tokens is not None else "unavailable",
                            output_tokens if output_tokens is not None else "unavailable",
                            f"{input_rate:.2f}" if input_rate is not None else "unavailable",
                            f"{output_rate:.2f}" if output_rate is not None else "unavailable",
                            request.app.state.input_tokens_total,
                            request.app.state.output_tokens_total,
                        )
                except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
                    logger.exception("inference_failed id=%s", request_id)
                    response = Response("Failed to generate a valid HTTP response", status_code=502, media_type="text/plain")

    return response
