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

## Anthropic Messages API (`/v1/messages`)

Add a `/v1/messages` endpoint that speaks the Anthropic/Claude native protocol, so clients using the Anthropic SDK can hit this server directly without translation.

**Key differences from OpenAI Chat Completions:**
- System prompt is a top-level `system` field, not a message in the array
- `messages` array only contains `user` and `assistant` roles
- Content blocks are structured (`type: "text"`, `type: "image"`, etc.) rather than plain strings
- Streaming uses SSE with typed events (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`)
- Response includes `stop_reason` instead of `finish_reason`
- Token usage split across `message_start` (input) and `message_delta` (output)

**Endpoint:** `POST /v1/messages`

**Client usage would look like:**
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
