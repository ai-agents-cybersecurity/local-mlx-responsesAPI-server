# Anthropic Messages API — Compatibility Test Suite

<img src="images/anthropictest.png" width="100%">

Tests your MLX server's `/v1/messages` endpoint against the real Anthropic Claude API spec, the official Anthropic Python SDK, LangChain's `ChatAnthropic`, and LangGraph agents.

Use this to verify your server is compatible with tools like **Shannon**, **Strix**, **CAI**, and any app built on the Anthropic or LangChain ecosystems.

## Quick start

```bash
# 1. Start your MLX server
python server.py --model mlx-community/Qwen3.5-9B-bf16 --port 8080

# 2. Run the base tests (no extra dependencies)
python test_anthropic.py

# 3. Run everything (SDK + LangChain + LangGraph)
pip install -r requirements_anthropic_test.txt
python test_anthropic.py --use-sdk --use-langchain
```

## CLI options

| Flag | Description |
|---|---|
| `--base-url URL` | Server base URL (default: `http://localhost:8080`) |
| `--use-sdk` | Enable Anthropic Python SDK tests |
| `--use-langchain` | Enable LangChain and LangGraph tests |
| `--verbose` / `-v` | Print full response payloads |
| `--category CAT` / `-c CAT` | Run only one test category (see below) |

## Test categories

### Core tests (no extra dependencies)

These use only `urllib` from the standard library to make raw HTTP requests, so they work with zero pip installs beyond the server itself.

| Category | Tests | What it checks |
|---|---|---|
| **connection** | 1 | Server health / model loaded |
| **structure** | 2 | Response JSON shape: `id`, `type`, `role`, `content` blocks, `usage`, `stop_reason`, `msg_` prefix |
| **system** | 2 | System prompt as string and as content-block list |
| **streaming** | 2 | Full SSE lifecycle (`message_start` → `content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta` → `message_stop`) and correct event ordering |
| **conversation** | 2 | Multi-turn context retention, user content as list-of-blocks |
| **parameters** | 4 | `max_tokens`, `temperature=0`, `stop_sequences`, `metadata` passthrough |
| **headers** | 2 | `x-api-key` and `anthropic-version` headers accepted |
| **errors** | 3 | Empty messages, missing `max_tokens`, consecutive user roles |
| **compatibility** | 1 | Unknown/extra fields silently ignored (SDK forward-compat) |
| **pentesting** | 2 | Long security-analysis prompts, structured JSON output |
| **performance** | 1 | Tokens-per-second throughput benchmark |

### Anthropic SDK tests (`--use-sdk`)

Requires `pip install anthropic`.

| Category | Tests | What it checks |
|---|---|---|
| **sdk** | 4 | Basic message, streaming, system prompt, multi-turn — all through the official `anthropic.Anthropic` client |

### LangChain / LangGraph tests (`--use-langchain`)

Requires `pip install langchain-anthropic langgraph`.

| Category | Tests | What it checks |
|---|---|---|
| **langchain** | 7 | `ChatAnthropic.invoke()`, `.stream()`, `SystemMessage`/`HumanMessage`/`AIMessage`, `bind_tools()`, LCEL chains (`prompt \| model \| parser`), batch invocation |
| **langgraph** | 3 | `create_react_agent` with single tool, multi-step agent (two tools, sequential reasoning), conversation memory via `MemorySaver` |

## Dependencies

**Base tests** — Python 3.10+, nothing else.

**Full suite** — install with:

```bash
pip install -r requirements_anthropic_test.txt
```

Contents of `requirements_anthropic_test.txt`:

```
anthropic>=0.39.0
langchain-anthropic>=0.3.0
langgraph>=0.2.0
```

## Output

The script prints color-coded results with PASS/FAIL/WARN per test and ends with a summary:

```
══════════════════════════════════════════════════════════════════
  RESULTS: 32/32 passed  (48.3s total)
══════════════════════════════════════════════════════════════════

  Compatibility score: 100%
  Your server is ready for Shannon / Strix / CAI!
══════════════════════════════════════════════════════════════════
```

Exit code is `0` if all tests pass, `1` if any fail.

## Running a single category

```bash
python test_anthropic.py -c streaming
python test_anthropic.py --use-langchain -c langchain
python test_anthropic.py --use-langchain -c langgraph
```

## What this tests against

This suite validates compatibility with the **Anthropic Messages API** (`POST /v1/messages`) as documented at [docs.anthropic.com](https://docs.anthropic.com). Specifically:

- Request/response JSON schema (content blocks, usage, stop reasons)
- SSE streaming event types and ordering
- Header requirements (`x-api-key`, `anthropic-version`)
- Parameter handling (`max_tokens`, `temperature`, `stop_sequences`, `metadata`)
- Error responses for malformed requests
- SDK-level compatibility (Anthropic, LangChain, LangGraph)
- Agentic workflows (tool calling, ReAct loops, multi-step reasoning, conversation memory)
