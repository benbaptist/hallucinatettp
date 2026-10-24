# HallucinateTTP

A tiny HTTP server that forwards each raw incoming request to an OpenAI-compatible chat-completions API and turns the model's JSON response into an HTTP response.

## Run with Docker Compose

```sh
cp .env.example .env
# Set OPENAI_API_KEY in .env
docker compose up --build
```

The server listens on port `8000` by default. Set `PORT` in `.env` to change both the container listener and published host port (for example, `PORT=9000` serves at `http://localhost:9000`). Check `/healthz` for health. Other requests are sent to the model. Configure `OPENAI_MODEL`, `OPENAI_BASE_URL`, `MAX_BODY_BYTES`, and `LOG_LEVEL` in `.env` as needed. `MAX_CONCURRENT_INFERENCES` limits simultaneous model calls per server process (defaults to `4`); requests above the limit wait in a queue. Logs include a request ID, request start/finish, inference start/finish or failure, queue wait, and elapsed time.

## Run locally

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
export PORT=9000  # optional; defaults to 8000
uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
```

The model must return a JSON object with exactly `status` (integer), `headers` (string-to-string object), and `body` (string). The system prompt requests this format, and the server validates the response and strips unsafe framing headers. Request bodies are sent as base64 to preserve arbitrary bytes. Raw requests can contain credentials; only use this service in a trusted environment. The API key is not forwarded to the model as request data.
