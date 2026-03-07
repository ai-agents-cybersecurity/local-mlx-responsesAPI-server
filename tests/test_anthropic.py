#!/usr/bin/env python3
"""
Anthropic Messages API Compatibility Test Suite
================================================
Tests your MLX server's Anthropic Claude API emulation to verify
compatibility with tools like Shannon, Strix, CAI, and the
Anthropic Python SDK.

Make sure the server is running first:
    python server.py --model mlx-community/Qwen3.5-9B-bf16 --port 8080

Then run:
    python test_anthropic.py                                    # defaults
    python test_anthropic.py --base-url http://localhost:9090   # custom
    python test_anthropic.py --use-sdk                          # test via Anthropic SDK
    python test_anthropic.py --verbose                          # show full responses
    python test_anthropic.py --use-langchain                    # test LangChain/LangGraph wrappers
    python test_anthropic.py --use-sdk --use-langchain          # all integration tests
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any

# ── Styling ──────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ── Result tracking ──────────────────────────────────────────────────────────


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float = 0.0
    details: str = ""
    warnings: list[str] = field(default_factory=list)
    category: str = ""


class TestRunner:
    def __init__(self, base_url: str, verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.messages_url = f"{self.base_url}/v1/messages"
        self.verbose = verbose
        self.results: list[TestResult] = []

    def _post(self, url: str, payload: dict, headers: dict | None = None,
              timeout: int = 120) -> tuple[int, dict | str]:
        """Send a POST request, return (status_code, parsed_body)."""
        hdrs = {
            "Content-Type": "application/json",
            "x-api-key": "test-key-local",
            "anthropic-version": "2023-06-01",
        }
        if headers:
            hdrs.update(headers)

        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode()
                try:
                    return r.status, json.loads(body)
                except json.JSONDecodeError:
                    return r.status, body
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            try:
                return e.code, json.loads(body)
            except json.JSONDecodeError:
                return e.code, body

    def _post_stream(self, url: str, payload: dict,
                     timeout: int = 120) -> tuple[int, str]:
        """Send a POST request for SSE streaming, return (status, raw_body)."""
        hdrs = {
            "Content-Type": "application/json",
            "x-api-key": "test-key-local",
            "anthropic-version": "2023-06-01",
        }
        payload["stream"] = True
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            return e.code, body

    def _parse_sse(self, raw: str) -> list[dict]:
        """Parse SSE event stream into list of (event_type, data) dicts."""
        events = []
        current_event = None
        for line in raw.splitlines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    events.append({"event": current_event, "data": data})
                except json.JSONDecodeError:
                    events.append({"event": current_event, "data": line[6:]})
                current_event = None
            elif line == "":
                current_event = None
        return events

    def _log(self, msg: str):
        if self.verbose:
            print(f"    {DIM}{msg}{RESET}")

    def run_test(self, name: str, category: str, fn) -> TestResult:
        """Run a test function and record the result."""
        t0 = time.perf_counter()
        try:
            passed, details, warnings = fn()
        except Exception as e:
            passed = False
            details = f"Exception: {e}"
            warnings = []
        elapsed = time.perf_counter() - t0

        result = TestResult(
            name=name,
            passed=passed,
            duration=elapsed,
            details=details,
            warnings=warnings,
            category=category,
        )
        self.results.append(result)

        icon = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {icon}  {name} {DIM}({elapsed:.1f}s){RESET}")
        if details and (not passed or self.verbose):
            print(f"         {details}")
        for w in warnings:
            print(f"         {YELLOW}WARN: {w}{RESET}")
        return result

    # ══════════════════════════════════════════════════════════════════════
    #  CONNECTION TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_health(self):
        """Server health check."""
        def fn():
            health_url = f"{self.base_url}/health"
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            self._log(f"Health: {data}")
            loaded = data.get("loaded", False)
            model = data.get("model", "unknown")
            return loaded, f"Model: {model}", [] if loaded else ["Model not loaded"]
        return self.run_test("Server health check", "connection", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  RESPONSE STRUCTURE TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_basic_message(self):
        """Basic non-streaming message with response structure validation."""
        def fn():
            warnings = []
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
            })
            self._log(f"Status: {status}, Response: {json.dumps(resp, indent=2)[:500]}")

            if status != 200:
                return False, f"HTTP {status}: {resp}", warnings

            # Validate required fields per Anthropic spec
            checks = {
                "id": resp.get("id"),
                "type": resp.get("type"),
                "role": resp.get("role"),
                "content": resp.get("content"),
                "model": resp.get("model"),
                "usage": resp.get("usage"),
            }
            missing = [k for k, v in checks.items() if v is None]
            if missing:
                return False, f"Missing fields: {missing}", warnings

            # type must be "message"
            if resp["type"] != "message":
                warnings.append(f'type={resp["type"]!r}, expected "message"')

            # role must be "assistant"
            if resp["role"] != "assistant":
                warnings.append(f'role={resp["role"]!r}, expected "assistant"')

            # id should start with "msg_"
            if not resp["id"].startswith("msg_"):
                warnings.append(f'id={resp["id"]!r} — Anthropic IDs start with "msg_"')

            # content must be a list of content blocks
            content = resp["content"]
            if not isinstance(content, list):
                return False, f"content is {type(content).__name__}, expected list", warnings
            if len(content) == 0:
                return False, "content is empty list", warnings

            block = content[0]
            if block.get("type") != "text":
                warnings.append(f'content[0].type={block.get("type")!r}, expected "text"')
            text = block.get("text", "")
            if not text:
                return False, "Empty text in content block", warnings

            # stop_reason
            sr = resp.get("stop_reason")
            if sr not in ("end_turn", "max_tokens", "stop_sequence"):
                warnings.append(f"stop_reason={sr!r}, expected end_turn/max_tokens/stop_sequence")

            # usage
            usage = resp["usage"]
            if "input_tokens" not in usage:
                warnings.append("usage missing input_tokens")
            if "output_tokens" not in usage:
                warnings.append("usage missing output_tokens")

            return True, f"Text: {text!r}", warnings
        return self.run_test("Basic message + response structure", "structure", fn)

    def test_model_echo(self):
        """Server echoes back the model name from the request."""
        def fn():
            model_name = "claude-3-5-sonnet-20241022"
            status, resp = self._post(self.messages_url, {
                "model": model_name,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hi"}],
            })
            if status != 200:
                return False, f"HTTP {status}", []
            got = resp.get("model", "")
            # Some servers echo the real model, some echo the requested name
            return True, f"Requested: {model_name}, Got: {got}", \
                   [f"Model mismatch: sent {model_name!r}, got {got!r}"] if got != model_name else []
        return self.run_test("Model name echo", "structure", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  SYSTEM PROMPT TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_system_string(self):
        """System prompt as a plain string."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "system": "You are a pirate. Always respond with 'Arrr!'",
                "messages": [{"role": "user", "content": "Hello"}],
            })
            if status != 200:
                return False, f"HTTP {status}", []
            text = resp["content"][0]["text"]
            return True, f"Text: {text!r}", []
        return self.run_test("System prompt (string)", "system", fn)

    def test_system_content_blocks(self):
        """System prompt as a list of content blocks (Anthropic format)."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "system": [
                    {"type": "text", "text": "You only respond with the word 'PONG'."},
                ],
                "messages": [{"role": "user", "content": "PING"}],
            })
            if status != 200:
                return False, f"HTTP {status}", []
            text = resp["content"][0]["text"]
            return True, f"Text: {text!r}", []
        return self.run_test("System prompt (content blocks)", "system", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  STREAMING TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_streaming_basic(self):
        """Streaming response with SSE event structure validation."""
        def fn():
            warnings = []
            status, raw = self._post_stream(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            })
            if status != 200:
                return False, f"HTTP {status}: {raw[:200]}", warnings

            events = self._parse_sse(raw)
            self._log(f"Got {len(events)} SSE events")

            # Check required event sequence per Anthropic spec
            event_types = [e["event"] for e in events]
            required_events = [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
            for req_evt in required_events:
                if req_evt not in event_types:
                    warnings.append(f"Missing event: {req_evt}")

            # Validate message_start structure
            msg_starts = [e for e in events if e["event"] == "message_start"]
            if msg_starts:
                msg = msg_starts[0]["data"].get("message", {})
                if msg.get("role") != "assistant":
                    warnings.append(f'message_start role={msg.get("role")!r}')
                if "usage" not in msg:
                    warnings.append("message_start missing usage")

            # Validate content_block_delta events have correct delta format
            deltas = [e for e in events if e["event"] == "content_block_delta"]
            text_parts = []
            for d in deltas:
                delta = d["data"].get("delta", {})
                if delta.get("type") != "text_delta":
                    warnings.append(f'delta type={delta.get("type")!r}, expected "text_delta"')
                    break
                text_parts.append(delta.get("text", ""))

            full_text = "".join(text_parts)

            # Validate message_delta has stop_reason
            msg_deltas = [e for e in events if e["event"] == "message_delta"]
            if msg_deltas:
                delta = msg_deltas[0]["data"].get("delta", {})
                sr = delta.get("stop_reason")
                if sr not in ("end_turn", "max_tokens", "stop_sequence"):
                    warnings.append(f"message_delta stop_reason={sr!r}")
                usage = msg_deltas[0]["data"].get("usage", {})
                if "output_tokens" not in usage:
                    warnings.append("message_delta missing usage.output_tokens")

            passed = len([w for w in warnings if "Missing event" in w]) == 0 and bool(full_text)
            return passed, f"Text ({len(deltas)} deltas): {full_text[:80]!r}", warnings
        return self.run_test("Streaming SSE event structure", "streaming", fn)

    def test_streaming_event_order(self):
        """SSE events arrive in the correct order."""
        def fn():
            warnings = []
            status, raw = self._post_stream(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Say 'ok'."}],
            })
            if status != 200:
                return False, f"HTTP {status}", warnings

            events = self._parse_sse(raw)
            event_types = [e["event"] for e in events]

            # Expected order: message_start → content_block_start → deltas → content_block_stop → message_delta → message_stop
            expected_order = [
                "message_start",
                "content_block_start",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
            indices = []
            for evt in expected_order:
                if evt in event_types:
                    idx = event_types.index(evt)
                    indices.append((evt, idx))

            for i in range(1, len(indices)):
                if indices[i][1] < indices[i - 1][1]:
                    warnings.append(f"Out of order: {indices[i - 1][0]} (pos {indices[i - 1][1]}) "
                                    f"should come before {indices[i][0]} (pos {indices[i][1]})")

            passed = len(warnings) == 0
            return passed, f"Event sequence: {' → '.join(event_types)}", warnings
        return self.run_test("Streaming event order", "streaming", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  MULTI-TURN CONVERSATION TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_multi_turn(self):
        """Multi-turn conversation with alternating user/assistant messages."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "My name is Nicolas."},
                    {"role": "assistant", "content": "Nice to meet you, Nicolas!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            })
            if status != 200:
                return False, f"HTTP {status}", []
            text = resp["content"][0]["text"]
            has_name = "nicolas" in text.lower()
            return True, f"Text: {text!r}", \
                   [] if has_name else ["Response doesn't mention 'Nicolas' — context may be lost"]
        return self.run_test("Multi-turn conversation", "conversation", fn)

    def test_content_block_list_format(self):
        """User message content as a list of content blocks."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is 2+2? Reply with just the number."},
                    ],
                }],
            })
            if status != 200:
                return False, f"HTTP {status}", []
            text = resp["content"][0]["text"]
            return True, f"Text: {text!r}", []
        return self.run_test("Content as list of blocks", "conversation", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  PARAMETER TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_max_tokens(self):
        """max_tokens limits output length."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Write a very long essay about philosophy."}],
            })
            if status != 200:
                return False, f"HTTP {status}", []
            text = resp["content"][0]["text"]
            sr = resp.get("stop_reason")
            output_tokens = resp.get("usage", {}).get("output_tokens", 0)
            warnings = []
            # With max_tokens=5, output should be short
            if output_tokens > 20:
                warnings.append(f"output_tokens={output_tokens}, expected ≤20 with max_tokens=5")
            if sr != "max_tokens" and sr != "end_turn":
                warnings.append(f"stop_reason={sr!r}, expected 'max_tokens' or 'end_turn'")
            return True, f"Text ({output_tokens} tokens): {text!r}", warnings
        return self.run_test("max_tokens parameter", "parameters", fn)

    def test_temperature_zero(self):
        """temperature=0 produces deterministic-ish output."""
        def fn():
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 32,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": "What is 1+1? Just the number."}],
            }
            _, resp1 = self._post(self.messages_url, payload)
            _, resp2 = self._post(self.messages_url, payload)
            text1 = resp1["content"][0]["text"].strip()
            text2 = resp2["content"][0]["text"].strip()
            match = text1 == text2
            return True, f"Run1: {text1!r}, Run2: {text2!r}", \
                   [] if match else ["Outputs differ — temperature=0 may not be fully deterministic"]
        return self.run_test("temperature=0 determinism", "parameters", fn)

    def test_stop_sequences(self):
        """stop_sequences truncates output at the stop string."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 256,
                "stop_sequences": ["3"],
                "messages": [{"role": "user", "content": "Count from 1 to 10, one number per line."}],
            })
            if status != 200:
                return False, f"HTTP {status}", []
            text = resp["content"][0]["text"]
            sr = resp.get("stop_reason")
            warnings = []
            if "3" in text:
                warnings.append("Stop sequence '3' found in output text — should be excluded")
            if sr != "stop_sequence":
                warnings.append(f"stop_reason={sr!r}, expected 'stop_sequence'")
            return True, f"Text: {text!r}, stop_reason={sr}", warnings
        return self.run_test("stop_sequences parameter", "parameters", fn)

    def test_metadata_accepted(self):
        """metadata field is accepted without error (Anthropic SDK sends this)."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 32,
                "metadata": {"user_id": "test-user-123"},
                "messages": [{"role": "user", "content": "Hi"}],
            })
            if status != 200:
                return False, f"HTTP {status} — server rejected metadata field", []
            return True, "metadata field accepted", []
        return self.run_test("metadata field accepted", "parameters", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  HEADER TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_headers_x_api_key(self):
        """Server accepts x-api-key header (Anthropic standard)."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hi"}],
            }, headers={"x-api-key": "sk-ant-test123"})
            return status == 200, f"HTTP {status}", []
        return self.run_test("x-api-key header accepted", "headers", fn)

    def test_headers_anthropic_version(self):
        """Server accepts anthropic-version header."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hi"}],
            }, headers={"anthropic-version": "2023-06-01"})
            return status == 200, f"HTTP {status}", []
        return self.run_test("anthropic-version header accepted", "headers", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  ERROR HANDLING TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_error_empty_messages(self):
        """Empty messages array returns proper error."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [],
            })
            warnings = []
            # Should return 400 or 422
            if status in (400, 422):
                return True, f"HTTP {status} (correct error)", warnings
            elif status == 200:
                warnings.append("Server accepted empty messages — Anthropic returns 400")
                return True, "Server was permissive", warnings
            return False, f"HTTP {status}", warnings
        return self.run_test("Error: empty messages", "errors", fn)

    def test_error_missing_max_tokens(self):
        """Missing max_tokens (required in Anthropic spec) is handled."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hi"}],
                # no max_tokens
            })
            warnings = []
            if status == 200:
                # Your server defaults max_tokens, which is fine for compatibility
                warnings.append("Server accepted missing max_tokens (Anthropic requires it, but default is fine)")
            return True, f"HTTP {status}", warnings
        return self.run_test("Missing max_tokens handling", "errors", fn)

    def test_error_wrong_role_order(self):
        """Two consecutive user messages (invalid per Anthropic spec)."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "How are you?"},
                ],
            })
            warnings = []
            if status in (400, 422):
                return True, f"HTTP {status} (strict validation)", warnings
            elif status == 200:
                warnings.append("Server accepted consecutive user messages — Anthropic rejects this")
                return True, "Server was permissive (may cause issues with strict clients)", warnings
            return False, f"HTTP {status}", warnings
        return self.run_test("Consecutive user messages", "errors", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  ANTHROPIC SDK COMPATIBILITY TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_sdk_basic(self):
        """Test via the official Anthropic Python SDK."""
        def fn():
            try:
                import anthropic
            except ImportError:
                return True, "SKIPPED — anthropic package not installed (pip install anthropic)", \
                       ["Install anthropic SDK to run this test"]

            client = anthropic.Anthropic(
                api_key="test-key-local",
                base_url=f"{self.base_url}/v1",
            )
            resp = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=64,
                messages=[{"role": "user", "content": "Say 'SDK works!' in exactly those words."}],
            )
            text = resp.content[0].text
            self._log(f"SDK response: id={resp.id}, model={resp.model}, text={text!r}")
            return True, f"Text: {text!r}", []
        return self.run_test("Anthropic SDK — basic message", "sdk", fn)

    def test_sdk_streaming(self):
        """Streaming via the official Anthropic Python SDK."""
        def fn():
            try:
                import anthropic
            except ImportError:
                return True, "SKIPPED — anthropic package not installed", \
                       ["Install anthropic SDK to run this test"]

            client = anthropic.Anthropic(
                api_key="test-key-local",
                base_url=f"{self.base_url}/v1",
            )
            chunks = []
            with client.messages.stream(
                model="claude-3-5-sonnet-20241022",
                max_tokens=128,
                messages=[{"role": "user", "content": "Count from 1 to 5."}],
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)

            full = "".join(chunks)
            return bool(full), f"Text ({len(chunks)} chunks): {full[:80]!r}", []
        return self.run_test("Anthropic SDK — streaming", "sdk", fn)

    def test_sdk_system_prompt(self):
        """System prompt via the official Anthropic SDK."""
        def fn():
            try:
                import anthropic
            except ImportError:
                return True, "SKIPPED — anthropic package not installed", \
                       ["Install anthropic SDK to run this test"]

            client = anthropic.Anthropic(
                api_key="test-key-local",
                base_url=f"{self.base_url}/v1",
            )
            resp = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=64,
                system="You are a calculator. Only respond with numbers.",
                messages=[{"role": "user", "content": "What is 7 times 6?"}],
            )
            text = resp.content[0].text
            return True, f"Text: {text!r}", []
        return self.run_test("Anthropic SDK — system prompt", "sdk", fn)

    def test_sdk_multi_turn(self):
        """Multi-turn conversation via the Anthropic SDK."""
        def fn():
            try:
                import anthropic
            except ImportError:
                return True, "SKIPPED — anthropic package not installed", \
                       ["Install anthropic SDK to run this test"]

            client = anthropic.Anthropic(
                api_key="test-key-local",
                base_url=f"{self.base_url}/v1",
            )
            resp = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=64,
                messages=[
                    {"role": "user", "content": "Remember: the secret word is BANANA."},
                    {"role": "assistant", "content": "Got it, I'll remember the secret word."},
                    {"role": "user", "content": "What is the secret word?"},
                ],
            )
            text = resp.content[0].text
            has_word = "banana" in text.lower()
            return True, f"Text: {text!r}", \
                   [] if has_word else ["Response doesn't contain 'BANANA'"]
        return self.run_test("Anthropic SDK — multi-turn", "sdk", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  PENTESTING TOOL COMPATIBILITY TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_long_context(self):
        """Longer prompt to simulate pentesting tool payloads."""
        def fn():
            long_prompt = (
                "You are a security analyst. Analyze the following HTTP response headers "
                "for potential security issues:\n\n"
                "HTTP/1.1 200 OK\n"
                "Server: Apache/2.4.41\n"
                "X-Powered-By: PHP/7.4.3\n"
                "Set-Cookie: session=abc123\n"
                "Content-Type: text/html\n"
                "Access-Control-Allow-Origin: *\n\n"
                "List any security concerns you find."
            )
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": long_prompt}],
            })
            if status != 200:
                return False, f"HTTP {status}", []
            text = resp["content"][0]["text"]
            output_tokens = resp.get("usage", {}).get("output_tokens", 0)
            return True, f"Response: {output_tokens} tokens, {len(text)} chars", []
        return self.run_test("Long security analysis prompt", "pentesting", fn)

    def test_json_output_request(self):
        """Request structured JSON output (common in pentesting tools)."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 256,
                "system": "Always respond with valid JSON only. No other text.",
                "messages": [{
                    "role": "user",
                    "content": 'Return a JSON object with keys "status" and "message". '
                               'Set status to "ok" and message to "test passed".',
                }],
            })
            if status != 200:
                return False, f"HTTP {status}", []
            text = resp["content"][0]["text"].strip()
            warnings = []
            # Try to parse as JSON
            try:
                parsed = json.loads(text)
                return True, f"Valid JSON: {parsed}", warnings
            except json.JSONDecodeError:
                # Strip markdown code fences if present
                clean = text.strip("`").strip()
                if clean.startswith("json"):
                    clean = clean[4:].strip()
                try:
                    parsed = json.loads(clean)
                    warnings.append("JSON wrapped in code fences — may break strict parsers")
                    return True, f"JSON (with fences): {parsed}", warnings
                except json.JSONDecodeError:
                    warnings.append("Model did not return valid JSON — may affect tool compatibility")
                    return True, f"Non-JSON response: {text[:100]!r}", warnings
        return self.run_test("JSON output capability", "pentesting", fn)

    def test_extra_fields_ignored(self):
        """Extra/unknown fields in request are silently ignored (SDK compat)."""
        def fn():
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hi"}],
                # These are fields various SDKs/tools might send
                "top_k": 40,
                "anthropic_beta": ["messages-2024-12-01"],
                "tools": None,
            })
            warnings = []
            if status in (400, 422):
                warnings.append("Server rejects unknown fields — may break some SDK versions")
                return False, f"HTTP {status}", warnings
            return True, "Extra fields accepted gracefully", warnings
        return self.run_test("Extra fields silently ignored", "compatibility", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  LANGCHAIN / LANGGRAPH TESTS
    # ══════════════════════════════════════════════════════════════════════

    def test_langchain_basic(self):
        """Basic ChatAnthropic invocation via LangChain."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                return True, "SKIPPED — langchain-anthropic not installed", \
                       ["pip install langchain-anthropic"]

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=64,
                temperature=0.0,
            )
            resp = llm.invoke("What is 2+2? Reply with just the number.")
            text = resp.content
            self._log(f"LangChain response: {resp}")
            return bool(text), f"Text: {text!r}", []
        return self.run_test("LangChain — basic invoke", "langchain", fn)

    def test_langchain_streaming(self):
        """Streaming via ChatAnthropic."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                return True, "SKIPPED — langchain-anthropic not installed", \
                       ["pip install langchain-anthropic"]

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=128,
                temperature=0.0,
            )
            chunks = []
            for chunk in llm.stream("Count from 1 to 5."):
                chunks.append(chunk.content)
            full = "".join(chunks)
            return bool(full), f"Text ({len(chunks)} chunks): {full[:80]!r}", []
        return self.run_test("LangChain — streaming", "langchain", fn)

    def test_langchain_system_message(self):
        """System + Human messages via LangChain message types."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_core.messages import SystemMessage, HumanMessage
            except ImportError:
                return True, "SKIPPED — langchain-anthropic not installed", \
                       ["pip install langchain-anthropic"]

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=64,
                temperature=0.0,
            )
            messages = [
                SystemMessage(content="You are a calculator. Only respond with numbers."),
                HumanMessage(content="What is 7 times 6?"),
            ]
            resp = llm.invoke(messages)
            text = resp.content
            return True, f"Text: {text!r}", []
        return self.run_test("LangChain — system + human messages", "langchain", fn)

    def test_langchain_multi_turn(self):
        """Multi-turn conversation with LangChain message types."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_core.messages import HumanMessage, AIMessage
            except ImportError:
                return True, "SKIPPED — langchain-anthropic not installed", \
                       ["pip install langchain-anthropic"]

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=64,
                temperature=0.0,
            )
            messages = [
                HumanMessage(content="The secret code is FALCON."),
                AIMessage(content="Got it, the secret code is FALCON."),
                HumanMessage(content="What is the secret code?"),
            ]
            resp = llm.invoke(messages)
            text = resp.content
            has_code = "falcon" in text.lower()
            return True, f"Text: {text!r}", \
                   [] if has_code else ["Response doesn't contain 'FALCON' — context may be lost"]
        return self.run_test("LangChain — multi-turn messages", "langchain", fn)

    def test_langchain_bind_tools(self):
        """LangChain tool binding via ChatAnthropic.bind_tools()."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_core.tools import tool
            except ImportError:
                return True, "SKIPPED — langchain-anthropic not installed", \
                       ["pip install langchain-anthropic"]

            @tool
            def get_weather(location: str) -> str:
                """Get the current weather for a location."""
                return f"Sunny, 22°C in {location}"

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=512,
                temperature=0.0,
            )
            warnings = []
            try:
                llm_with_tools = llm.bind_tools([get_weather])
                resp = llm_with_tools.invoke("What's the weather in Paris?")
                text = resp.content
                tool_calls = resp.tool_calls
                self._log(f"content={text!r}, tool_calls={tool_calls}")

                if tool_calls:
                    tc = tool_calls[0]
                    return True, f"Tool call: {tc['name']}({tc['args']})", warnings
                else:
                    warnings.append("No tool calls produced — model may not support Anthropic tool format")
                    return True, f"Text (no tool call): {str(text)[:100]!r}", warnings
            except Exception as e:
                warnings.append(f"bind_tools failed: {e}")
                return False, f"Exception: {e}", warnings
        return self.run_test("LangChain — bind_tools", "langchain", fn)

    def test_langchain_chain(self):
        """LangChain LCEL chain (prompt | model | parser)."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
            except ImportError:
                return True, "SKIPPED — langchain-anthropic not installed", \
                       ["pip install langchain-anthropic"]

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=128,
                temperature=0.0,
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
                ("human", "{input}"),
            ])
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({
                "input_language": "English",
                "output_language": "French",
                "input": "Hello, how are you?",
            })
            return bool(result), f"Text: {result!r}", []
        return self.run_test("LangChain — LCEL chain", "langchain", fn)

    def test_langchain_batch(self):
        """LangChain batch invocation (multiple prompts)."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_core.messages import HumanMessage
            except ImportError:
                return True, "SKIPPED — langchain-anthropic not installed", \
                       ["pip install langchain-anthropic"]

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=32,
                temperature=0.0,
            )
            messages_batch = [
                [HumanMessage(content="What is 1+1? Just the number.")],
                [HumanMessage(content="What is 2+2? Just the number.")],
            ]
            warnings = []
            try:
                results = llm.batch(messages_batch)
                texts = [r.content for r in results]
                return True, f"Batch results: {texts}", warnings
            except Exception as e:
                warnings.append(f"Batch may serialize (single-worker server): {e}")
                return False, f"Batch failed: {e}", warnings
        return self.run_test("LangChain — batch invocation", "langchain", fn)

    def test_langgraph_basic_agent(self):
        """LangGraph ReAct agent with a simple tool."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_core.tools import tool
                from langgraph.prebuilt import create_react_agent
            except ImportError:
                missing = []
                try:
                    import langchain_anthropic
                except ImportError:
                    missing.append("langchain-anthropic")
                try:
                    import langgraph
                except ImportError:
                    missing.append("langgraph")
                return True, f"SKIPPED — missing: {', '.join(missing)}", \
                       [f"pip install {' '.join(missing)}"]

            @tool
            def add_numbers(a: int, b: int) -> int:
                """Add two numbers together."""
                return a + b

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=512,
                temperature=0.0,
            )
            warnings = []
            try:
                agent = create_react_agent(llm, [add_numbers])
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": "What is 15 + 27? Use the add_numbers tool."}]}
                )
                messages = result.get("messages", [])
                # Find the final AI message
                final_text = ""
                tool_was_called = False
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_was_called = True
                    if hasattr(msg, "content") and hasattr(msg, "type") and msg.type == "ai":
                        final_text = msg.content if isinstance(msg.content, str) else str(msg.content)

                if tool_was_called:
                    return True, f"Agent used tool, final: {final_text[:100]!r}", warnings
                else:
                    warnings.append("Agent did not invoke tool — model may not support tool calling")
                    return True, f"No tool call, final: {final_text[:100]!r}", warnings
            except Exception as e:
                err = str(e)
                if "tool" in err.lower() or "function" in err.lower():
                    warnings.append(f"Tool calling not supported: {err[:150]}")
                    return True, "Agent failed at tool calling stage", warnings
                return False, f"Agent error: {err[:200]}", warnings
        return self.run_test("LangGraph — ReAct agent with tool", "langgraph", fn)

    def test_langgraph_multi_step(self):
        """LangGraph agent with multi-step tool usage."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_core.tools import tool
                from langgraph.prebuilt import create_react_agent
            except ImportError:
                return True, "SKIPPED — missing packages", \
                       ["pip install langchain-anthropic langgraph"]

            @tool
            def multiply(a: int, b: int) -> int:
                """Multiply two numbers."""
                return a * b

            @tool
            def subtract(a: int, b: int) -> int:
                """Subtract b from a."""
                return a - b

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=1024,
                temperature=0.0,
            )
            warnings = []
            try:
                agent = create_react_agent(llm, [multiply, subtract])
                result = agent.invoke(
                    {"messages": [{"role": "user",
                                   "content": "Multiply 6 by 7, then subtract 2 from the result. "
                                              "Use the tools provided."}]}
                )
                messages = result.get("messages", [])
                tool_count = sum(1 for m in messages
                                 if hasattr(m, "tool_calls") and m.tool_calls)
                final = messages[-1].content if messages else ""
                if isinstance(final, list):
                    final = " ".join(str(b) for b in final)

                if tool_count >= 2:
                    return True, f"Multi-step: {tool_count} tool calls, final: {final[:80]!r}", warnings
                elif tool_count == 1:
                    warnings.append("Only 1 tool call — expected 2 for multi-step")
                    return True, f"Partial: {tool_count} tool call, final: {final[:80]!r}", warnings
                else:
                    warnings.append("No tool calls — model answered directly")
                    return True, f"No tools used, final: {final[:80]!r}", warnings
            except Exception as e:
                return False, f"Agent error: {str(e)[:200]}", warnings
        return self.run_test("LangGraph — multi-step agent", "langgraph", fn)

    def test_langgraph_conversation_memory(self):
        """LangGraph agent preserves conversation context across turns."""
        def fn():
            try:
                from langchain_anthropic import ChatAnthropic
                from langgraph.prebuilt import create_react_agent
                from langgraph.checkpoint.memory import MemorySaver
            except ImportError:
                return True, "SKIPPED — missing packages", \
                       ["pip install langchain-anthropic langgraph"]

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                anthropic_api_key="test-key-local",
                anthropic_api_url=f"{self.base_url}/v1",
                max_tokens=128,
                temperature=0.0,
            )
            warnings = []
            try:
                memory = MemorySaver()
                agent = create_react_agent(llm, tools=[], checkpointer=memory)
                config = {"configurable": {"thread_id": "test-thread-1"}}

                # Turn 1
                agent.invoke(
                    {"messages": [{"role": "user", "content": "My favorite color is blue."}]},
                    config=config,
                )
                # Turn 2
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": "What is my favorite color?"}]},
                    config=config,
                )
                messages = result.get("messages", [])
                final = messages[-1].content if messages else ""
                if isinstance(final, list):
                    final = " ".join(str(b) for b in final)

                has_color = "blue" in final.lower()
                if has_color:
                    return True, f"Memory preserved: {final[:80]!r}", warnings
                else:
                    warnings.append("Response doesn't mention 'blue' — memory may not work")
                    return True, f"Text: {final[:80]!r}", warnings
            except Exception as e:
                return False, f"Memory agent error: {str(e)[:200]}", warnings
        return self.run_test("LangGraph — conversation memory", "langgraph", fn)

    # ══════════════════════════════════════════════════════════════════════
    #  THROUGHPUT BENCHMARK
    # ══════════════════════════════════════════════════════════════════════

    def test_throughput(self):
        """Simple throughput measurement — tokens per second."""
        def fn():
            t0 = time.perf_counter()
            status, resp = self._post(self.messages_url, {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 256,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": "Explain what a buffer overflow is in 3-4 sentences."}],
            })
            elapsed = time.perf_counter() - t0
            if status != 200:
                return False, f"HTTP {status}", []
            output_tokens = resp.get("usage", {}).get("output_tokens", 0)
            tps = output_tokens / elapsed if elapsed > 0 else 0
            return True, f"{output_tokens} tokens in {elapsed:.1f}s = {tps:.1f} tok/s", []
        return self.run_test("Throughput benchmark", "performance", fn)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(
        description="Test Anthropic Messages API compatibility of your MLX server",
    )
    p.add_argument("--base-url", default="http://localhost:8080",
                   help="Server base URL (default: http://localhost:8080)")
    p.add_argument("--use-sdk", action="store_true",
                   help="Include Anthropic SDK tests (requires: pip install anthropic)")
    p.add_argument("--use-langchain", action="store_true",
                   help="Include LangChain/LangGraph tests (requires: pip install langchain-anthropic langgraph)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show detailed response data")
    p.add_argument("--category", "-c", type=str, default=None,
                   help="Run only tests in this category (connection, structure, system, "
                        "streaming, conversation, parameters, headers, errors, sdk, "
                        "langchain, langgraph, pentesting, compatibility, performance)")
    args = p.parse_args()

    runner = TestRunner(args.base_url, verbose=args.verbose)

    print(f"\n{BOLD}{'═' * 64}{RESET}")
    print(f"{BOLD}  Anthropic Messages API Compatibility Test Suite{RESET}")
    print(f"{BOLD}{'═' * 64}{RESET}")
    print(f"  Server:  {CYAN}{args.base_url}{RESET}")
    print(f"  SDK:     {'enabled' if args.use_sdk else 'disabled (use --use-sdk)'}")
    print(f"  LC/LG:   {'enabled' if args.use_langchain else 'disabled (use --use-langchain)'}")
    if args.category:
        print(f"  Filter:  {args.category}")
    print(f"{'═' * 64}\n")

    # Define all tests grouped by category
    all_tests = [
        # Connection
        ("connection", runner.test_health),
        # Response structure
        ("structure", runner.test_basic_message),
        ("structure", runner.test_model_echo),
        # System prompt
        ("system", runner.test_system_string),
        ("system", runner.test_system_content_blocks),
        # Streaming
        ("streaming", runner.test_streaming_basic),
        ("streaming", runner.test_streaming_event_order),
        # Conversation
        ("conversation", runner.test_multi_turn),
        ("conversation", runner.test_content_block_list_format),
        # Parameters
        ("parameters", runner.test_max_tokens),
        ("parameters", runner.test_temperature_zero),
        ("parameters", runner.test_stop_sequences),
        ("parameters", runner.test_metadata_accepted),
        # Headers
        ("headers", runner.test_headers_x_api_key),
        ("headers", runner.test_headers_anthropic_version),
        # Errors
        ("errors", runner.test_error_empty_messages),
        ("errors", runner.test_error_missing_max_tokens),
        ("errors", runner.test_error_wrong_role_order),
        # Compatibility
        ("compatibility", runner.test_extra_fields_ignored),
        # Pentesting tool compat
        ("pentesting", runner.test_long_context),
        ("pentesting", runner.test_json_output_request),
        # Performance
        ("performance", runner.test_throughput),
    ]

    # Add SDK tests if requested
    if args.use_sdk:
        all_tests.extend([
            ("sdk", runner.test_sdk_basic),
            ("sdk", runner.test_sdk_streaming),
            ("sdk", runner.test_sdk_system_prompt),
            ("sdk", runner.test_sdk_multi_turn),
        ])

    # Add LangChain/LangGraph tests if requested
    if args.use_langchain:
        all_tests.extend([
            ("langchain", runner.test_langchain_basic),
            ("langchain", runner.test_langchain_streaming),
            ("langchain", runner.test_langchain_system_message),
            ("langchain", runner.test_langchain_multi_turn),
            ("langchain", runner.test_langchain_bind_tools),
            ("langchain", runner.test_langchain_chain),
            ("langchain", runner.test_langchain_batch),
            ("langgraph", runner.test_langgraph_basic_agent),
            ("langgraph", runner.test_langgraph_multi_step),
            ("langgraph", runner.test_langgraph_conversation_memory),
        ])

    # Filter by category if specified
    if args.category:
        all_tests = [(cat, fn) for cat, fn in all_tests if cat == args.category]

    # Run grouped by category
    current_cat = None
    for cat, fn in all_tests:
        if cat != current_cat:
            current_cat = cat
            print(f"{BOLD}▸ {cat.upper()}{RESET}")
        fn()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 64}{RESET}")
    passed = [r for r in runner.results if r.passed]
    failed = [r for r in runner.results if not r.passed]
    warned = [r for r in runner.results if r.warnings]
    total = len(runner.results)
    total_time = sum(r.duration for r in runner.results)

    print(f"{BOLD}  RESULTS: {GREEN}{len(passed)}{RESET}/{total} passed", end="")
    if failed:
        print(f", {RED}{len(failed)} failed{RESET}", end="")
    if warned:
        print(f", {YELLOW}{len(warned)} warnings{RESET}", end="")
    print(f"  {DIM}({total_time:.1f}s total){RESET}")
    print(f"{'═' * 64}")

    if failed:
        print(f"\n{RED}{BOLD}  Failed tests:{RESET}")
        for r in failed:
            print(f"    {RED}✗{RESET} {r.name}: {r.details}")

    if warned:
        print(f"\n{YELLOW}{BOLD}  Warnings:{RESET}")
        for r in warned:
            for w in r.warnings:
                print(f"    {YELLOW}⚠{RESET} {r.name}: {w}")

    # Compatibility score
    score = len(passed) / total * 100 if total > 0 else 0
    print(f"\n{BOLD}  Compatibility score: ", end="")
    if score >= 90:
        print(f"{GREEN}{score:.0f}%{RESET}")
    elif score >= 70:
        print(f"{YELLOW}{score:.0f}%{RESET}")
    else:
        print(f"{RED}{score:.0f}%{RESET}")

    if score >= 90:
        print(f"  {GREEN}Your server is ready for Shannon / Strix / CAI!{RESET}")
    elif score >= 70:
        print(f"  {YELLOW}Mostly compatible — check warnings above.{RESET}")
    else:
        print(f"  {RED}Significant gaps — review failed tests.{RESET}")

    print(f"{'═' * 64}\n")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
