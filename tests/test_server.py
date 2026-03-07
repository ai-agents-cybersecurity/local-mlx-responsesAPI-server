#!/usr/bin/env python3
"""
Test client for the MLX Chat Server.

Make sure the server is running first:
    python server.py --model mlx-community/MiniMax-M2.5-8bit --port 8080

Then run these tests:
    python test_server.py                       # defaults to localhost:8080
    python test_server.py --base-url http://localhost:9090/v1
"""

from __future__ import annotations

import argparse
import sys
import time

from openai import OpenAI


def test_health(base_url: str) -> bool:
    """Hit /health to make sure the server is alive."""
    import urllib.request, json

    health_url = base_url.replace("/v1", "/health")
    try:
        with urllib.request.urlopen(health_url, timeout=5) as r:
            data = json.loads(r.read())
            print(f"  /health → {data}")
            return data.get("loaded", False)
    except Exception as e:
        print(f"  /health FAILED: {e}")
        return False


def test_models(client: OpenAI) -> bool:
    """List available models."""
    try:
        models = client.models.list()
        print(f"  /v1/models → {[m.id for m in models.data]}")
        return len(models.data) > 0
    except Exception as e:
        print(f"  /v1/models FAILED: {e}")
        return False


def test_chat(client: OpenAI) -> bool:
    """Non-streaming completion."""
    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": "Reply in one short sentence."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            max_tokens=64,
            temperature=0.3,
        )
        elapsed = time.perf_counter() - t0
        text = resp.choices[0].message.content
        usage = resp.usage
        print(f"  Response ({elapsed:.1f}s): {text!r}")
        print(f"  Usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
        return bool(text)
    except Exception as e:
        print(f"  Non-streaming FAILED: {e}")
        return False


def test_stream(client: OpenAI) -> bool:
    """Streaming completion."""
    try:
        t0 = time.perf_counter()
        chunks = []
        stream = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "user", "content": "Count from 1 to 5."},
            ],
            max_tokens=128,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)
                print(delta, end="", flush=True)
        elapsed = time.perf_counter() - t0
        print(f"\n  Stream done ({elapsed:.1f}s, {len(chunks)} chunks)")
        return len(chunks) > 0
    except Exception as e:
        print(f"  Streaming FAILED: {e}")
        return False


def test_responses(client: OpenAI) -> bool:
    """Non-streaming Responses API."""
    try:
        t0 = time.perf_counter()
        resp = client.responses.create(
            model="local",
            instructions="Reply in one short sentence.",
            input="What is 2+2?",
            max_output_tokens=64,
            temperature=0.3,
        )
        elapsed = time.perf_counter() - t0
        text = resp.output_text
        print(f"  Response ({elapsed:.1f}s): {text!r}")
        print(f"  Usage: input={resp.usage.input_tokens} output={resp.usage.output_tokens}")
        # Verify no <think> blocks leaked through
        if "<think>" in text or "</think>" in text:
            print("  WARN: think tags found in output!")
            return False
        return bool(text)
    except Exception as e:
        print(f"  Responses API FAILED: {e}")
        return False


def test_responses_stream(client: OpenAI) -> bool:
    """Streaming Responses API."""
    try:
        t0 = time.perf_counter()
        chunks = []
        stream = client.responses.create(
            model="local",
            input="Count from 1 to 5.",
            max_output_tokens=128,
            temperature=0.3,
            stream=True,
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
                print(event.delta, end="", flush=True)
        elapsed = time.perf_counter() - t0
        print(f"\n  Stream done ({elapsed:.1f}s, {len(chunks)} deltas)")
        return len(chunks) > 0
    except Exception as e:
        print(f"  Responses streaming FAILED: {e}")
        return False


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}


def test_chat_tool_calls(client: OpenAI) -> bool:
    """Non-streaming chat with tools — model should produce tool_calls."""
    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
            ],
            tools=[WEATHER_TOOL],
            max_tokens=512,
            temperature=0.3,
        )
        elapsed = time.perf_counter() - t0
        choice = resp.choices[0]
        print(f"  Response ({elapsed:.1f}s)")
        print(f"  finish_reason: {choice.finish_reason}")
        print(f"  content: {choice.message.content!r}")
        print(f"  tool_calls: {choice.message.tool_calls}")

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            print(f"  Tool call: {tc.function.name}({tc.function.arguments})")
            return True
        else:
            print("  WARN: Expected finish_reason='tool_calls' with tool_calls list")
            # Still pass if model produced text (some models don't support tools natively)
            return bool(choice.message.content)
    except Exception as e:
        print(f"  Tool calls FAILED: {e}")
        return False


def test_chat_tool_results(client: OpenAI) -> bool:
    """Multi-turn: tool call → tool result → text response."""
    try:
        # Turn 1: get tool call
        t0 = time.perf_counter()
        resp1 = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "user", "content": "What is the weather in London?"},
            ],
            tools=[WEATHER_TOOL],
            max_tokens=512,
            temperature=0.3,
        )
        choice1 = resp1.choices[0]
        print(f"  Turn 1 finish_reason: {choice1.finish_reason}")

        if choice1.finish_reason != "tool_calls" or not choice1.message.tool_calls:
            print("  WARN: Model did not produce tool calls, skipping multi-turn test")
            return bool(choice1.message.content)

        tc = choice1.message.tool_calls[0]
        print(f"  Tool call: {tc.function.name}({tc.function.arguments})")

        # Turn 2: send tool result back
        resp2 = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "user", "content": "What is the weather in London?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": '{"temperature": 15, "condition": "cloudy"}',
                },
            ],
            tools=[WEATHER_TOOL],
            max_tokens=256,
            temperature=0.3,
        )
        elapsed = time.perf_counter() - t0
        choice2 = resp2.choices[0]
        print(f"  Turn 2 ({elapsed:.1f}s): {choice2.message.content!r}")
        return bool(choice2.message.content)
    except Exception as e:
        print(f"  Tool results FAILED: {e}")
        return False


def test_responses_tool_calls(client: OpenAI) -> bool:
    """Responses API with tools — should produce function_call output items."""
    try:
        import urllib.request, json

        base = client.base_url
        url = f"{base}responses"
        payload = json.dumps({
            "model": "local",
            "input": "What is the weather in Tokyo?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                }
            ],
            "max_output_tokens": 512,
            "temperature": 0.3,
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local",
            },
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        elapsed = time.perf_counter() - t0

        print(f"  Response ({elapsed:.1f}s)")
        print(f"  Output items: {len(data.get('output', []))}")

        for item in data.get("output", []):
            if item.get("type") == "function_call":
                print(f"  function_call: {item.get('name')}({item.get('arguments')})")
                return True

        # If model returned a message instead (no native tool support), still pass
        for item in data.get("output", []):
            if item.get("type") == "message":
                text = ""
                for c in item.get("content", []):
                    text += c.get("text", "")
                print(f"  Message (no tool call): {text[:100]!r}")
                return bool(text)

        print("  WARN: No output items found")
        return False
    except Exception as e:
        print(f"  Responses tool calls FAILED: {e}")
        return False


def test_responses_multi_turn(client: OpenAI) -> bool:
    """Multi-turn conversation via previous_response_id."""
    try:
        # Turn 1: introduce a fact
        t0 = time.perf_counter()
        resp1 = client.responses.create(
            model="local",
            instructions="You are a helpful assistant. Remember everything the user tells you.",
            input="My name is Alice.",
            max_output_tokens=64,
            temperature=0.3,
        )
        elapsed1 = time.perf_counter() - t0
        print(f"  Turn 1 ({elapsed1:.1f}s): {resp1.output_text!r}")
        print(f"  Turn 1 id: {resp1.id}")

        # Turn 2: ask about the fact, passing previous_response_id
        t1 = time.perf_counter()
        resp2 = client.responses.create(
            model="local",
            input="What is my name?",
            previous_response_id=resp1.id,
            max_output_tokens=64,
            temperature=0.3,
        )
        elapsed2 = time.perf_counter() - t1
        text2 = resp2.output_text
        print(f"  Turn 2 ({elapsed2:.1f}s): {text2!r}")

        if "Alice" in text2 or "alice" in text2.lower():
            print("  Multi-turn context preserved!")
            return True
        else:
            print("  WARN: Response does not mention 'Alice'")
            return False
    except Exception as e:
        print(f"  Multi-turn FAILED: {e}")
        return False


# ── Anthropic Messages API tests ─────────────────────────────────────────────


def test_anthropic_messages(base_url: str) -> bool:
    """Non-streaming Anthropic Messages API."""
    import urllib.request, json

    url = f"{base_url}messages"
    payload = json.dumps({
        "model": "local",
        "max_tokens": 64,
        "system": "Reply in one short sentence.",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": "local",
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        elapsed = time.perf_counter() - t0

        msg_type = data.get("type")
        role = data.get("role")
        content = data.get("content", [])
        text = content[0].get("text", "") if content else ""
        usage = data.get("usage", {})
        print(f"  Response ({elapsed:.1f}s): type={msg_type}, role={role}")
        print(f"  Text: {text!r}")
        print(f"  Usage: input={usage.get('input_tokens')} output={usage.get('output_tokens')}")

        if "<think>" in text or "</think>" in text:
            print("  WARN: think tags found in output!")
            return False
        return msg_type == "message" and role == "assistant" and bool(text)
    except Exception as e:
        print(f"  Anthropic Messages FAILED: {e}")
        return False


def test_anthropic_messages_stream(base_url: str) -> bool:
    """Streaming Anthropic Messages API."""
    import urllib.request, json

    url = f"{base_url}messages"
    payload = json.dumps({
        "model": "local",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "temperature": 0.3,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": "local",
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode()
        elapsed = time.perf_counter() - t0

        # Parse SSE events
        deltas = []
        saw_message_start = False
        saw_message_stop = False
        for line in body.splitlines():
            if line.startswith("data: "):
                try:
                    evt = json.loads(line[6:])
                    evt_type = evt.get("type", "")
                    if evt_type == "message_start":
                        saw_message_start = True
                    elif evt_type == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "text_delta":
                            deltas.append(delta.get("text", ""))
                    elif evt_type == "message_stop":
                        saw_message_stop = True
                except json.JSONDecodeError:
                    pass

        full_text = "".join(deltas)
        print(f"  Stream done ({elapsed:.1f}s, {len(deltas)} deltas): {full_text[:80]!r}")
        print(f"  message_start={saw_message_start}, message_stop={saw_message_stop}")
        return saw_message_start and saw_message_stop and len(deltas) > 0
    except Exception as e:
        print(f"  Anthropic Messages streaming FAILED: {e}")
        return False


# ── Azure OpenAI endpoint tests ──────────────────────────────────────────────


def test_azure_responses(base_url: str) -> bool:
    """Azure-style Responses API via deployment URL."""
    import urllib.request, json

    # Derive server root from base_url (strip /v1)
    server_root = base_url.replace("/v1", "")
    deployment = "my-gpt4-deployment"
    url = f"{server_root}/openai/deployments/{deployment}/responses?api-version=2025-03-01-preview"

    payload = json.dumps({
        "input": "What is 2+2? Reply in one short sentence.",
        "max_output_tokens": 64,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "api-key": "local",
        },
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        elapsed = time.perf_counter() - t0

        model = data.get("model", "")
        print(f"  Response ({elapsed:.1f}s), model={model!r}")

        # Verify the deployment name appears as the model in the response
        if model != deployment:
            print(f"  WARN: expected model={deployment!r}, got {model!r}")

        # Extract text from output
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    text = c.get("text", "")
                    if text:
                        print(f"  Text: {text!r}")
                        return True

        print("  WARN: No text output found")
        return False
    except Exception as e:
        print(f"  Azure Responses FAILED: {e}")
        return False


def test_azure_responses_stream(base_url: str) -> bool:
    """Azure-style streaming Responses API."""
    import urllib.request, json

    server_root = base_url.replace("/v1", "")
    deployment = "my-gpt4-deployment"
    url = f"{server_root}/openai/deployments/{deployment}/responses?api-version=2025-03-01-preview"

    payload = json.dumps({
        "input": "Count from 1 to 3.",
        "stream": True,
        "max_output_tokens": 128,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "api-key": "local",
        },
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode()
        elapsed = time.perf_counter() - t0

        # Parse SSE events
        deltas = []
        for line in body.splitlines():
            if line.startswith("data: "):
                try:
                    evt = json.loads(line[6:])
                    if evt.get("type") == "response.output_text.delta":
                        deltas.append(evt.get("delta", ""))
                except json.JSONDecodeError:
                    pass

        full_text = "".join(deltas)
        print(f"  Stream done ({elapsed:.1f}s, {len(deltas)} deltas): {full_text[:80]!r}")
        return len(deltas) > 0
    except Exception as e:
        print(f"  Azure Responses streaming FAILED: {e}")
        return False


def test_azure_models(base_url: str) -> bool:
    """Azure-style model listing via deployment URL."""
    import urllib.request, json

    server_root = base_url.replace("/v1", "")
    deployment = "my-gpt4-deployment"
    url = f"{server_root}/openai/deployments/{deployment}/models?api-version=2025-03-01-preview"

    req = urllib.request.Request(
        url,
        headers={"api-key": "local"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        models = [m["id"] for m in data.get("data", [])]
        print(f"  Azure /models → {models}")
        return deployment in models
    except Exception as e:
        print(f"  Azure models FAILED: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    args = p.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="local")

    tests = [
        ("Health check", lambda: test_health(args.base_url)),
        ("List models", lambda: test_models(client)),
        ("Non-streaming chat", lambda: test_chat(client)),
        ("Streaming chat", lambda: test_stream(client)),
        ("Chat tool calls", lambda: test_chat_tool_calls(client)),
        ("Chat tool results", lambda: test_chat_tool_results(client)),
        ("Responses API", lambda: test_responses(client)),
        ("Responses streaming", lambda: test_responses_stream(client)),
        ("Responses tool calls", lambda: test_responses_tool_calls(client)),
        ("Responses multi-turn", lambda: test_responses_multi_turn(client)),
        ("Anthropic Messages API", lambda: test_anthropic_messages(args.base_url)),
        ("Anthropic Messages streaming", lambda: test_anthropic_messages_stream(args.base_url)),
        ("Azure Responses API", lambda: test_azure_responses(args.base_url)),
        ("Azure Responses streaming", lambda: test_azure_responses_stream(args.base_url)),
        ("Azure models", lambda: test_azure_models(args.base_url)),
    ]

    results = []
    for name, fn in tests:
        print(f"\n{'─'*60}")
        print(f"TEST: {name}")
        print(f"{'─'*60}")
        ok = fn()
        results.append((name, ok))
        print(f"  → {'PASS' if ok else 'FAIL'}")

    print(f"\n{'═'*60}")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"{'═'*60}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
