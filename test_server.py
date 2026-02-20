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
        ("Responses API", lambda: test_responses(client)),
        ("Responses streaming", lambda: test_responses_stream(client)),
        ("Responses multi-turn", lambda: test_responses_multi_turn(client)),
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
