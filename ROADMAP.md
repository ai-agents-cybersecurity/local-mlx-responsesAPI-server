# Roadmap

## ~~Tool/Function Calling~~ (Done)

Non-streaming and streaming tool calling across both Chat Completions and Responses APIs. Supports Qwen3 JSON and Qwen3.5 XML tool call formats. Full round-trip: model emits structured tool calls, client executes, results fed back for final response. Tested end-to-end with OpenClaw.

## Streaming Tool Call Optimization

Currently, when tools are provided in streaming mode, the full output is buffered until generation completes so tool call tags can be detected. A future optimization could parse tool call tags incrementally to emit structured chunks sooner.

## Conversation State Management

Currently using an in-memory dict (`_conversation_store`) — ephemeral, lost on restart.

| Approach | Persistence | Complexity | Use case |
|----------|------------|------------|----------|
| **In-memory dict** | None | Trivial | Local dev |
| **In-memory + TTL** (current) | None, but bounded | Low | Local dev, prevents memory leak |
| **JSON file** | Survives restarts | Low | Single-user local |
| **SQLite** | Survives restarts | Medium | Single-node, multi-user |
| **Redis** | Configurable | Medium | Multi-worker, TTL built-in |
| **Postgres/DB** | Full durability | Higher | Production |

## ~~Anthropic Messages API (`/v1/messages`)~~ (Done)

Streaming and non-streaming Messages API compatible with the Anthropic Python SDK. System prompt as top-level `system` field, `user`/`assistant` message array, structured content blocks, and Anthropic SSE event types (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`).

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="local")
resp = client.messages.create(
    model="local",
    max_tokens=256,
    system="Reply in one short sentence.",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
```
