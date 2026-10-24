from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

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
You receive the raw HTTP request line and headers. A non-empty request body is
provided as base64:<data> so arbitrary bytes are preserved.

TASK — produce a complete HTTP response in exactly this shape:

  1. CGI header block:
       Content-Type: text/html; charset=utf-8
       Status: <3-digit code>          (omit for 200)
     followed by EXACTLY ONE blank line.
  2. One complete, SELF-CONTAINED HTML5 document:
       <!DOCTYPE html><html><head>...
     - <meta charset=\"utf-8\"> and a <title> reflecting the request.
     - ALL styling in a single inline <style> block.
     - ALL behavior in inline <script>.
     - NO external references of any kind: no CDN links, no external CSS/JS,
       no fonts, no third-party fetches. Images only as inline data: URIs
       or omitted entirely.

RULES
  - Echo the interaction faithfully: show the method, path, and decoded query
    parameters in the page body where it makes sense.
  - HTML-escape EVERY user-supplied value (& < > \" ') before embedding it.
    Never place raw query-string or body data into the document.
  - Status codes: 200 normal replies; 404 unknown paths; 400 malformed input;
    500 only if you cannot produce a valid response at all. Error statuses
    still require the full header block and a human-readable HTML page.
  - Never emit a duplicate Content-Type; never send two header blocks.
  - Emit the response directly, not JSON or markdown fences.

STYLE
  Clean, minimal, readable typography via the inline stylesheet; the page
  must remain fully usable with JavaScript disabled.

EXAMPLE
  Input:  GET /hello.cgi?name=Ada
  Output:
    Content-Type: text/html; charset=utf-8

    <!DOCTYPE html>
    <html><head><meta charset=\"utf-8\"><title>Hello</title>
    <style>body{font-family:sans-serif;max-width:40em;margin:2em auto}</style>
    </head><body><h1>Hello, Ada</h1>
    <p>You sent: GET /hello.cgi?name=Ada</p></body></html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    app.state.inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
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
BASE_URL_HEADERS = {"host", "origin", "referer", "forwarded", "x-forwarded-host", "x-original-host"}
BODYLESS_STATUS_CODES = {204, 205, 304}


def request_as_http(request: Request, body: bytes) -> str:
    target = request.url.path
    if request.url.query:
        target += "?" + request.url.query
    include_base_url = request.url.path == "/"
    headers = "\r\n".join(
        f"{name.decode('latin-1')}: {value.decode('latin-1')}"
        for name, value in request.scope["headers"]
        if include_base_url or name.decode("latin-1").lower() not in BASE_URL_HEADERS
    )
    body_text = "base64:" + base64.b64encode(body).decode("ascii") if body else ""
    return f"{request.method} {target} HTTP/{request.scope.get('http_version', '1.1')}\r\n{headers}\r\n\r\n{body_text}"


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
    request_id = request.state.request_id

    if not API_KEY:
        logger.warning("request_rejected id=%s reason=missing_api_key", request_id)
        response = Response("OPENAI_API_KEY is not configured", status_code=503, media_type="text/plain")
    else:
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            logger.warning("request_rejected id=%s reason=body_too_large bytes=%d", request_id, len(body))
            response = Response("Request body too large", status_code=413, media_type="text/plain")
        else:
            raw_request = request_as_http(request, body)
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
                    generated = payload["choices"][0]["message"]["content"]
                    response = generated_http_response(generated, request.method)
                    inference_duration_ms = (time.monotonic() - inference_started_at) * 1000
                    logger.info(
                        "inference_finished id=%s duration_ms=%.1f status=%d",
                        request_id,
                        inference_duration_ms,
                        response.status_code,
                    )
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
                logger.exception("inference_failed id=%s", request_id)
                response = Response("Failed to generate a valid HTTP response", status_code=502, media_type="text/plain")

    return response
