# MLX Local Inference Server

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![Platform](https://img.shields.io/badge/platform-macOS%20(M1/M2/M3/M4)-green) ![API](https://img.shields.io/badge/API-Chat%20Completions-blue) ![API](https://img.shields.io/badge/API-Responses-green) ![Privacy](https://img.shields.io/badge/Privacy-Local%20Inference-orange) ![Built with](https://img.shields.io/badge/Built%20with-Claude-7B3F00?logo=anthropic)

An OpenAI-compatible local inference server for Apple Silicon using [MLX](https://github.com/ml-explore/mlx). Drop-in replacement for any app that speaks the OpenAI protocol, serving Chat Completions and Responses APIs.

## What's in here

| File | Purpose |
|---|---|
| `server.py` | OpenAI-compatible chat server with Chat Completions and Responses APIs |
| `test_server.py` | Integration test client for the server (health, models, chat, streaming, responses, multi-turn) |

## server.py — the main event

A FastAPI server that loads any MLX model and exposes OpenAI-compatible endpoints. Any app or library that works with the OpenAI API works with this server out of the box.



<img src="screen.png" width="50%">

### Features

- **OpenAI Chat Completions API** (`/v1/chat/completions`) — streaming and non-streaming
- **OpenAI Responses API** (`/v1/responses`) — streaming and non-streaming, with multi-turn conversation via `previous_response_id`
- **`/v1/models`** — model discovery endpoint
- **`/health`** — readiness probe
- **Multi-turn conversation store** — in-memory with 1-hour TTL, auto-archiving expired conversations to `conversation_logs/` as JSON
- **Think-tag stripping** — removes `<think>...</think>` reasoning blocks from model output
- **Inference lock** — serializes requests so concurrent callers queue cleanly instead of corrupting MLX GPU state
- **CORS enabled** — browser-based apps can call it directly
- **Pydantic validation** — malformed requests get clear 422 errors
- **Structured error responses** — OpenAI-style error JSON on failures
- **Chat templates** — automatically applied from the model's tokenizer
- **Configurable** — temperature, top_p, max_tokens, stop sequences, repetition_penalty

### Chat Completions parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | loaded model | Ignored for routing; echoed back in the response |
| `messages` | required | Standard role/content message list (string, list-of-parts, or null content supported) |
| `temperature` | 0.7 | 0.0–2.0 |
| `top_p` | 0.95 | 0.0–1.0 |
| `max_tokens` | 4096 | |
| `stream` | false | SSE streaming |
| `stop` | none | String or list of stop sequences |
| `repetition_penalty` | 1.0 | |

### Responses API parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | loaded model | Ignored for routing; echoed back in the response |
| `input` | required | String or message list (supports inline conversation history and multi-part content) |
| `instructions` | none | System prompt |
| `previous_response_id` | none | Chain multi-turn conversations |
| `temperature` | 0.7 | 0.0–2.0 |
| `top_p` | 0.95 | 0.0–1.0 |
| `max_output_tokens` | 4096 | |
| `stream` | false | SSE streaming |

Extra fields sent by OpenAI clients (like `presence_penalty`, `frequency_penalty`, etc.) are silently ignored so nothing breaks.

### Client compatibility

Tested with:
- **OpenAI Python SDK** — Chat Completions and Responses APIs
- **OpenClaw** — via `"api": "openai-responses"` custom provider config
- **curl** — direct HTTP requests
- **Browser-based apps** — via CORS

The server accepts `content` as a plain string, a list of content parts (`"text"` or `"input_text"` types), or `null` — covering all formats used by OpenAI-compatible clients.

## 

🚀 [Roadmap](ROADMAP.md)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- See `requirements.txt` for dependencies

## License

Personal project — use as you like.
