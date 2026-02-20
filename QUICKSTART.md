# Quickstart

Get the server running in under 2 minutes.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

This installs MLX, mlx-lm, FastAPI, Uvicorn, and Pydantic.

## 2. Start the server

```bash
python server.py --model mlx-community/MiniMax-M2.5-8bit --port 8080
```

The first run downloads the model weights from HuggingFace (~8 GB for MiniMax 8-bit). Subsequent runs load from cache.

You'll see:

```
14:32:01  INFO      Loading model mlx-community/MiniMax-M2.5-8bit …
14:32:13  INFO      Model ready in 12.3s
14:32:13  INFO      Starting server on 127.0.0.1:8080
```

### Other models

```bash
# Qwen 3.5 (large MoE — needs ~64 GB RAM)
python server.py --model mlx-community/Qwen3.5-397B-A17B-4bit --port 8080

# Llama 3
python server.py --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8080

# Any mlx-community model works
python server.py --model mlx-community/<model-name> --port 8080
```

### Server options

| Flag | Default | Description |
|---|---|---|
| `--model` | `MiniMax-M2.5-8bit` | HuggingFace model ID or local path |
| `--host` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on LAN) |
| `--port` | `8080` | Listen port |
| `--workers` | `1` | Uvicorn workers (keep at 1 for MLX) |

## 3. Talk to it

### Python (OpenAI SDK)

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

# Non-streaming
response = client.chat.completions.create(
    model="minimax",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="minimax",
    messages=[{"role": "user", "content": "Write a haiku about coding."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

### Responses API

```python
# Non-streaming
response = client.responses.create(
    model="minimax",
    instructions="Reply in one short sentence.",
    input="What is the capital of France?",
)
print(response.output_text)

# Multi-turn conversation
follow_up = client.responses.create(
    model="minimax",
    input="And what about Germany?",
    previous_response_id=response.id,
)
print(follow_up.output_text)
```

### curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

### Health check

```bash
curl http://localhost:8080/health
# {"status":"ok","model":"mlx-community/MiniMax-M2.5-8bit","loaded":true}
```

## 4. Run the test suite

In a second terminal:

```bash
python test_server.py
```

This runs seven checks (health, model listing, non-streaming chat, streaming chat, Responses API, Responses streaming, and multi-turn conversation) and reports pass/fail.

## Troubleshooting

**"Model not loaded yet" (503)** — The server is still downloading or loading the model. Wait for the "Model ready" log line.

**Out of memory** — Try a smaller quantization. The `4bit` variants use roughly half the RAM of `8bit`.

**Port already in use** — Pick a different port: `--port 9090`.

**Slow first response** — The first inference compiles MLX kernels. Subsequent requests are much faster.
