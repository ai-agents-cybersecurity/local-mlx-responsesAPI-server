#!/usr/bin/env python3
"""
OpenAI-Compatible MLX Chat Server
==================================
A production-quality local inference server for MLX models that speaks
the OpenAI Chat Completions protocol.

Usage:
    python server.py                                          # default model
    python server.py --model mlx-community/MiniMax-M2.5-8bit  # pick a model
    python server.py --port 8888 --host 0.0.0.0               # custom bind

Then hit it from any OpenAI client:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
    r = client.chat.completions.create(
        model="minimax",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*device_info is deprecated.*")

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import re
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import mlx.core as mx
from mlx_lm import load, stream_generate, generate
from mlx_lm.utils import load_model, load_tokenizer, _download
from mlx_lm.sample_utils import make_sampler

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mlx-server")

# ── Pydantic request / response schemas ─────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    model_config = {"extra": "ignore"}

    def text(self) -> str:
        """Normalise content to a plain string."""
        if self.content is None:
            return ""
        if isinstance(self.content, list):
            return _normalise_content(self.content)
        return self.content


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=4096, ge=1)
    stream: bool = False
    stop: list[str] | str | None = None
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    # extra fields silently ignored so any OpenAI client works
    model_config = {"extra": "ignore"}


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[dict] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo


# ── Responses API schemas ────────────────────────────────────────────────────


class ResponsesRequest(BaseModel):
    model: str | None = None
    input: str | list = ""
    instructions: str | None = None
    previous_response_id: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=4096, ge=1)
    stream: bool = False
    tools: list[dict] | None = None
    model_config = {"extra": "ignore"}


class ResponsesOutputText(BaseModel):
    type: str = "output_text"
    text: str
    annotations: list = []


class ResponsesOutputMessage(BaseModel):
    type: str = "message"
    id: str
    status: str = "completed"
    role: str = "assistant"
    content: list[ResponsesOutputText] = []


class ResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponsesApiResponse(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    model: str
    status: str = "completed"
    output: list = []
    usage: ResponsesUsage | None = None


# ── Model holder ─────────────────────────────────────────────────────────────


@dataclass
class ModelHolder:
    """Holds the loaded model + tokenizer in a thread-safe-ish way."""

    model_path: str = ""
    model: object = None
    tokenizer: object = None
    loaded: bool = False

    @staticmethod
    def _resolve_path(model_path: str) -> Path:
        """Return a local Path, using the HF cache if available to avoid network calls."""
        p = Path(model_path)
        if p.exists():
            return p
        # Check HF cache before hitting the network
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache_dir / ("models--" + model_path.replace("/", "--"))
        if repo_dir.exists():
            snapshots = repo_dir / "snapshots"
            if snapshots.exists():
                revisions = sorted(snapshots.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
                for rev in revisions:
                    # Only use cache if snapshot has weight files, not just metadata
                    has_weights = any(rev.glob("*.safetensors"))
                    has_tokenizer = any(rev.glob("tokenizer*"))
                    if has_weights and has_tokenizer:
                        log.info("Using cached model at %s", rev)
                        return rev
            log.info("Incomplete cache for %s, downloading missing files…", model_path)
        return Path(_download(model_path))

    def load(self, model_path: str) -> None:
        log.info("Loading model %s …", model_path)
        t0 = time.perf_counter()
        local_path = self._resolve_path(model_path)
        model, config = load_model(local_path, lazy=False, strict=False)
        tokenizer = load_tokenizer(
            local_path, eos_token_ids=config.get("eos_token_id", None)
        )
        self.model, self.tokenizer = model, tokenizer
        self.model_path = model_path
        self.loaded = True
        elapsed = time.perf_counter() - t0
        log.info("Model ready in %.1fs", elapsed)


holder = ModelHolder()

# Serialize all inference — MLX models are NOT thread-safe.
# Concurrent requests queue up behind this lock instead of corrupting state.
_inference_lock = asyncio.Lock()

# Maps response_id → (timestamp, session_id, full message history)
_conversation_store: dict[str, tuple[float, str, list[dict]]] = {}
_CONVERSATION_TTL: float = 3600.0  # 1 hour
_CONVERSATION_MAX: int = 1000      # max entries before forced eviction
_CONVERSATION_LOG_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation_logs")

# Ensure log directory exists at import time
os.makedirs(_CONVERSATION_LOG_DIR, exist_ok=True)


def _sess_id() -> str:
    return "sess_" + uuid.uuid4().hex[:12]


def _archive_conversation(resp_id: str, ts: float, session_id: str, messages: list[dict]) -> None:
    """Save an expired conversation to disk as a timestamped JSON file."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    filename = f"{dt.strftime('%Y-%m-%dT%H-%M-%S')}_{session_id}_{resp_id}.json"
    path = os.path.join(_CONVERSATION_LOG_DIR, filename)
    record = {
        "session_id": session_id,
        "response_id": resp_id,
        "created_at": dt.isoformat(),
        "evicted_at": datetime.now(timezone.utc).isoformat(),
        "turns": len(messages),
        "messages": messages,
    }
    try:
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        log.info("Archived conversation %s (session %s) → %s", resp_id, session_id, filename)
    except OSError:
        log.exception("Failed to archive conversation %s", resp_id)


def _evict_conversations() -> None:
    """Remove expired entries; if still over limit, drop oldest."""
    now = time.time()
    expired = [k for k, (ts, _, _) in _conversation_store.items() if now - ts > _CONVERSATION_TTL]
    for k in expired:
        ts, session_id, messages = _conversation_store.pop(k)
        _archive_conversation(k, ts, session_id, messages)
    # If still over capacity, drop oldest entries
    if len(_conversation_store) > _CONVERSATION_MAX:
        by_age = sorted(_conversation_store, key=lambda k: _conversation_store[k][0])
        for k in by_age[: len(_conversation_store) - _CONVERSATION_MAX]:
            ts, session_id, messages = _conversation_store.pop(k)
            _archive_conversation(k, ts, session_id, messages)


def _store_conversation(resp_id: str, session_id: str, messages: list[dict]) -> None:
    """Store a conversation and evict stale entries."""
    _conversation_store[resp_id] = (time.time(), session_id, messages)
    log.info("Stored conversation %s (session %s, %d messages, store size=%d)",
             resp_id, session_id, len(messages), len(_conversation_store))
    _evict_conversations()


def _get_conversation(resp_id: str) -> tuple[str, list[dict]] | None:
    """Retrieve a conversation if it exists and hasn't expired.
    Returns (session_id, messages) or None."""
    entry = _conversation_store.get(resp_id)
    if entry is None:
        log.warning("Conversation lookup MISS for %s (store has %d entries: %s)",
                    resp_id, len(_conversation_store), list(_conversation_store.keys()))
        return None
    ts, session_id, messages = entry
    if time.time() - ts > _CONVERSATION_TTL:
        log.warning("Conversation %s expired (age=%.0fs)", resp_id, time.time() - ts)
        _conversation_store.pop(resp_id)
        _archive_conversation(resp_id, ts, session_id, messages)
        return None
    log.info("Conversation lookup HIT for %s (session %s, %d messages)", resp_id, session_id, len(messages))
    return session_id, messages


# ── Prompt helpers ───────────────────────────────────────────────────────────


def _normalize_tools_for_template(tools: list[dict]) -> list[dict]:
    """Normalize tool definitions to the format expected by apply_chat_template(tools=...).

    Accepts both OpenAI Chat format (nested: {type, function: {name, ...}})
    and Responses API format (flat: {type, name, description, parameters}).
    Returns list of {type: "function", function: {name, description, parameters}}.
    """
    normalized = []
    for tool in tools:
        if "function" in tool:
            # Already in OpenAI Chat format
            normalized.append({
                "type": "function",
                "function": {
                    "name": tool["function"].get("name", ""),
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {}),
                },
            })
        else:
            # Responses API flat format: {type, name, description, parameters}
            normalized.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            })
    return normalized


def _build_prompt(tokenizer, messages: list[dict], tools: list[dict] | None = None) -> tuple[str, int]:
    """Apply the tokenizer's chat template and return (prompt_text, token_count)."""
    # Preserve tool-related fields in message dicts
    msg_dicts = []
    for m in messages:
        d = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            # Jinja templates (Qwen3.5 etc.) expect arguments as a dict,
            # but OpenAI wire format stores them as a JSON string.
            # Deserialize here so the template can iterate with .items().
            template_tcs = []
            for tc in m["tool_calls"]:
                tc_copy = {**tc}
                if "function" in tc_copy:
                    func = {**tc_copy["function"]}
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            func["arguments"] = {}
                    tc_copy["function"] = func
                template_tcs.append(tc_copy)
            d["tool_calls"] = template_tcs
        if m.get("tool_call_id"):
            d["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            d["name"] = m["name"]
        msg_dicts.append(d)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        template_kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        if tools:
            template_kwargs["tools"] = tools
        try:
            prompt_text = tokenizer.apply_chat_template(
                msg_dicts,
                **template_kwargs,
            )
        except Exception:
            # Some templates don't support tools kwarg — fall back without it
            template_kwargs.pop("tools", None)
            try:
                prompt_text = tokenizer.apply_chat_template(
                    msg_dicts,
                    **template_kwargs,
                )
            except Exception:
                # Template can't handle tool messages — flatten them into
                # plain user/assistant messages so generation can proceed
                log.warning("Chat template failed with tool messages, flattening")
                flat = []
                for d in msg_dicts:
                    if d["role"] == "tool":
                        # Convert tool result to a user message
                        flat.append({"role": "user", "content": f"[Tool result]: {d.get('content', '')}"})
                    elif d.get("tool_calls"):
                        # Convert assistant tool call to a plain assistant message
                        tc_text = ", ".join(
                            f"{tc.get('function', {}).get('name', '?')}(...)"
                            for tc in d["tool_calls"]
                        )
                        flat.append({"role": "assistant", "content": f"[Called tools: {tc_text}]"})
                    else:
                        flat.append({"role": d["role"], "content": d.get("content") or ""})
                prompt_text = tokenizer.apply_chat_template(
                    flat,
                    **template_kwargs,
                )
    else:
        # Fallback: plain concatenation
        prompt_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in msg_dicts
        )
        prompt_text += "\nassistant:"

    prompt_len = len(tokenizer.encode(prompt_text))
    return prompt_text, prompt_len


# ── Request ID helper ────────────────────────────────────────────────────────


def _call_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def _req_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:12]


_THINK_RE = re.compile(r"(<think>)?[\s\S]*?</think>\s*", re.DOTALL)

# GPT-OSS channel format: extract only <|channel|>final<|message|>...<|end|>
_CHANNEL_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\|end\|>|$)"
)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.
    Handles cases where the opening <think> tag is missing."""
    return _THINK_RE.sub("", text)


def _strip_channels(text: str) -> str:
    """Extract final-channel content from GPT-OSS channel format.
    If the text doesn't use channel format, returns it unchanged."""
    if "<|channel|>" not in text:
        return text
    matches = _CHANNEL_FINAL_RE.findall(text)
    if matches:
        return "\n".join(m.strip() for m in matches)
    return text


def _postprocess(text: str) -> str:
    """Apply all model-output cleanup: think-tags, channel format, etc."""
    text = _strip_think(text)
    text = _strip_channels(text)
    return text


# Matches the outer <tool_call>...</tool_call> wrapper (any content inside)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.DOTALL)

# MiniMax format: <PREFIX:tool_call>...<invoke>...</invoke>...</PREFIX:tool_call>
_MINIMAX_BLOCK_RE = re.compile(r"<[^>]*:tool_call>\s*([\s\S]*?)\s*</[^>]*:tool_call>", re.DOTALL)

# JSON-style inner content: {"name": ..., "arguments": ...}
_TOOL_CALL_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# XML-style inner content: <function=NAME> <parameter=KEY> VALUE </parameter> ... </function>
_TOOL_CALL_FUNC_RE = re.compile(r"<function=([^>]+)>([\s\S]*?)</function>", re.DOTALL)
_TOOL_CALL_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>", re.DOTALL)

# MiniMax invoke format: <invoke name="NAME"><parameter name="KEY">VALUE</parameter></invoke>
_INVOKE_RE = re.compile(r'<invoke\s+name="([^"]+)">([\s\S]*?)</invoke>', re.DOTALL)
_INVOKE_PARAM_RE = re.compile(r'<parameter\s+name="([^"]+)">\s*([\s\S]*?)\s*</parameter>', re.DOTALL)


def _parse_xml_tool_call(inner: str) -> dict | None:
    """Parse XML-style tool call: <function=NAME><parameter=K>V</parameter>...</function>."""
    func_match = _TOOL_CALL_FUNC_RE.search(inner)
    if not func_match:
        return None

    name = func_match.group(1).strip()
    body = func_match.group(2)

    params = {}
    for pm in _TOOL_CALL_PARAM_RE.finditer(body):
        key = pm.group(1).strip()
        value = pm.group(2).strip()
        # Try to parse as JSON value (number, bool, etc.), otherwise keep as string
        try:
            params[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            params[key] = value

    return {"name": name, "arguments": json.dumps(params)}


def _parse_invoke_tool_call(inner: str) -> dict | None:
    """Parse MiniMax invoke format: <invoke name="NAME"><parameter name="K">V</parameter></invoke>."""
    invoke_match = _INVOKE_RE.search(inner)
    if not invoke_match:
        return None

    name = invoke_match.group(1).strip()
    body = invoke_match.group(2)

    params = {}
    for pm in _INVOKE_PARAM_RE.finditer(body):
        key = pm.group(1).strip()
        value = pm.group(2).strip()
        try:
            params[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            params[key] = value

    return {"name": name, "arguments": json.dumps(params)}


def _parse_json_tool_call(inner: str) -> dict | None:
    """Parse JSON-style tool call: {"name": "...", "arguments": {...}}."""
    json_match = _TOOL_CALL_JSON_RE.search(inner)
    if not json_match:
        return None
    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    # Must have a "name" key to be a valid tool call — reject random JSON
    # fragments that the greedy regex might pick up from XML parameter values
    if "name" not in parsed:
        return None

    name = parsed["name"]
    arguments = parsed.get("arguments", {})
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {"name": name, "arguments": arguments}


def _parse_tool_calls(raw_text: str) -> tuple[str | None, list[dict]]:
    """Extract tool call blocks from model output.

    Supported outer wrappers:
      - <tool_call>...</tool_call>          (Qwen3, Qwen3.5, Hermes)
      - <PREFIX:tool_call>...</PREFIX:tool_call>  (MiniMax)

    Supported inner formats:
      - JSON:   {"name":"fn","arguments":{...}}
      - XML:    <function=fn><parameter=k>v</parameter></function>
      - Invoke: <invoke name="fn"><parameter name="k">v</parameter></invoke>

    Returns (remaining_text_or_None, list_of_tool_call_dicts).
    Each tool call dict has {id, type, function: {name, arguments}}.
    """
    # Try standard <tool_call> blocks first, then MiniMax format
    blocks = _TOOL_CALL_BLOCK_RE.findall(raw_text)
    strip_re = _TOOL_CALL_BLOCK_RE
    if not blocks:
        blocks = _MINIMAX_BLOCK_RE.findall(raw_text)
        strip_re = _MINIMAX_BLOCK_RE
    if not blocks:
        return raw_text, []

    tool_calls = []
    for inner in blocks:
        # Try JSON, then XML (<function=...>), then invoke (<invoke name=...>)
        result = (
            _parse_json_tool_call(inner)
            or _parse_xml_tool_call(inner)
            or _parse_invoke_tool_call(inner)
        )
        if result is None:
            log.warning("Failed to parse tool_call content: %s", inner[:200])
            continue

        tool_calls.append({
            "id": _call_id(),
            "type": "function",
            "function": result,
        })

    if not tool_calls:
        return raw_text, []

    # Strip tool_call blocks from text; remaining text is the non-tool content
    remaining = strip_re.sub("", raw_text).strip()
    return remaining or None, tool_calls


def _postprocess_with_tools(raw_text: str, tools_were_provided: bool) -> tuple[str | None, list[dict] | None, str]:
    """Post-process model output with tool call extraction.

    Returns (content, tool_calls, finish_reason).
    """
    text = _strip_think(raw_text)
    text = _strip_channels(text)

    if tools_were_provided:
        content, tool_calls = _parse_tool_calls(text)
        if tool_calls:
            return content, tool_calls, "tool_calls"

    return text, None, "stop"


class _StreamFilter:
    """Streaming filter that buffers tokens until think-tags and channel
    markers are resolved, then yields only user-facing text."""

    def __init__(self):
        self._buf = ""
        self._thinking = True  # assume output may start inside <think>
        self._in_final = False  # inside <|channel|>final<|message|>...
        self._uses_channels = False

    def feed(self, token: str) -> str:
        """Feed a token, return text to emit (may be empty)."""
        # Phase 1: buffer while inside <think> block
        if self._thinking:
            self._buf += token
            end = self._buf.find("</think>")
            if end != -1:
                self._thinking = False
                after = self._buf[end + len("</think>"):]
                self._buf = ""
                if not after.strip():
                    return ""
                # Fall through to channel check with accumulated text
                token = after.lstrip("\n")
            elif "<|channel|>" in self._buf or "<|start|>" in self._buf:
                # Not a think-tag model — switch to channel mode
                self._thinking = False
                self._uses_channels = True
                token = self._buf
                self._buf = ""
            else:
                return ""

        # Phase 2: handle <|channel|> format
        self._buf += token

        # Detect channel format on first occurrence
        if not self._uses_channels and "<|channel|>" in self._buf:
            self._uses_channels = True

        if not self._uses_channels:
            # No channel format — emit directly
            out = self._buf
            self._buf = ""
            return out

        # Buffer until we can resolve channel boundaries
        emit = ""
        while True:
            if self._in_final:
                # Look for end marker
                end_pos = self._buf.find("<|end|>")
                start_pos = self._buf.find("<|start|>")
                # Find the earliest boundary
                boundary = -1
                if end_pos != -1 and start_pos != -1:
                    boundary = min(end_pos, start_pos)
                elif end_pos != -1:
                    boundary = end_pos
                elif start_pos != -1:
                    boundary = start_pos

                if boundary != -1:
                    emit += self._buf[:boundary]
                    # Skip past the marker
                    if self._buf[boundary:].startswith("<|end|>"):
                        self._buf = self._buf[boundary + len("<|end|>"):]
                    else:
                        self._buf = self._buf[boundary:]
                    self._in_final = False
                else:
                    # Could be partial marker at end — hold back potential partial
                    safe, self._buf = self._safe_emit(self._buf)
                    emit += safe
                    break
            else:
                # Look for <|channel|>final<|message|>
                marker = "<|channel|>final<|message|>"
                pos = self._buf.find(marker)
                if pos != -1:
                    self._buf = self._buf[pos + len(marker):]
                    self._in_final = True
                    continue
                # Check for other channel starts to skip
                ch_pos = self._buf.find("<|channel|>")
                if ch_pos != -1:
                    # Non-final channel — discard up to next boundary and keep looking
                    rest = self._buf[ch_pos + len("<|channel|>"):]
                    end_pos = rest.find("<|end|>")
                    if end_pos != -1:
                        self._buf = rest[end_pos + len("<|end|>"):]
                        continue
                    start_pos = rest.find("<|start|>")
                    if start_pos != -1:
                        self._buf = rest[start_pos:]
                        continue
                    # No end found yet — keep buffering
                    break
                else:
                    # No channel markers — might be partial, hold back
                    safe, self._buf = self._safe_emit(self._buf)
                    # But don't emit non-channel text when we know format uses channels
                    break

            if not self._buf:
                break

        return emit

    def flush(self) -> str:
        """Flush any remaining buffered text at end of stream."""
        if not self._buf:
            return ""
        if self._uses_channels:
            # Only emit if we were in the final channel
            if self._in_final:
                out = self._buf
                self._buf = ""
                return out
            # Discard leftover non-final content
            self._buf = ""
            return ""
        # No channel format — emit whatever is left (e.g. no think block)
        out = _strip_think(self._buf)
        self._buf = ""
        return out

    @staticmethod
    def _safe_emit(buf: str) -> tuple[str, str]:
        """Split buf into safe-to-emit prefix and remainder that could be
        the start of a special token like <|channel|>, <|end|>, <|start|>."""
        # Hold back anything from the last '<' onwards (could be partial tag)
        last_lt = buf.rfind("<")
        if last_lt != -1 and last_lt > len(buf) - 30:
            return buf[:last_lt], buf[last_lt:]
        return buf, ""


def _resp_id() -> str:
    return "resp_" + uuid.uuid4().hex[:12]


def _msg_id() -> str:
    return "msg_" + uuid.uuid4().hex[:12]


def _normalise_content(content) -> str:
    """Normalise content to a plain string (handles str, list-of-parts, None)."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type", "")
                # Handle both "text" and "input_text" content part types
                if ptype in ("text", "input_text"):
                    parts.append(part.get("text", ""))
                elif "text" in part:
                    parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def _responses_input_to_messages(
    input_data: str | list,
    instructions: str | None,
    previous_response_id: str | None = None,
) -> tuple[str, list[dict]]:
    """Convert Responses API input (string or message list) to internal format.

    When *previous_response_id* is given, the prior conversation is prepended
    so the model sees the full multi-turn context.  When not given but input
    contains multiple roles (user + assistant), treat it as inline history.

    Returns (session_id, messages).
    """
    # -- build the *new* turn's messages (without system) --
    new_messages: list[dict] = []
    if isinstance(input_data, str):
        new_messages.append({"role": "user", "content": input_data})
    else:
        log.info("Responses input list (%d items): %r", len(input_data), input_data)
        for item in input_data:
            if isinstance(item, dict):
                item_type = item.get("type", "")

                # Handle function_call → convert to assistant message with tool_calls
                if item_type == "function_call":
                    call_id = item.get("call_id") or item.get("id", _call_id())
                    fn_name = item.get("name", "")
                    arguments = item.get("arguments", "{}")
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    # Merge into the previous assistant message if it already has tool_calls
                    tc_entry = {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": fn_name, "arguments": arguments},
                    }
                    if new_messages and new_messages[-1].get("role") == "assistant" and "tool_calls" in new_messages[-1]:
                        new_messages[-1]["tool_calls"].append(tc_entry)
                    else:
                        new_messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tc_entry],
                        })
                    continue

                # Handle function_call_output → convert to tool role message
                if item_type == "function_call_output":
                    new_messages.append({
                        "role": "tool",
                        "content": item.get("output", ""),
                        "tool_call_id": item.get("call_id", ""),
                    })
                    continue

                role = item.get("role", "user")
                content = _normalise_content(item.get("content", ""))
                # Some clients send {"type": "message", ...} wrapper
                if item_type == "message" and not content and "content" not in item:
                    # Skip structural items without content
                    continue
                new_messages.append({"role": role, "content": content})
            elif isinstance(item, str):
                new_messages.append({"role": "user", "content": item})

    # -- resolve prior context --
    prior_result = _get_conversation(previous_response_id) if previous_response_id else None
    if prior_result is not None:
        session_id, prior_messages = prior_result
        prior = list(prior_messages)  # shallow copy

        if instructions:
            # Replace any existing system message with the new instructions
            if prior and prior[0]["role"] == "system":
                prior[0] = {"role": "system", "content": instructions}
            else:
                prior.insert(0, {"role": "system", "content": instructions})
        return session_id, prior + new_messages

    # No prior context — start fresh with a new session
    session_id = _sess_id()
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.extend(new_messages)
    return session_id, messages


# ── FastAPI app ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model is loaded before uvicorn starts (see main())
    yield


app = FastAPI(title="MLX Chat Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "model": holder.model_path, "loaded": holder.loaded}


@app.get("/v1/models")
async def list_models():
    """Minimal /v1/models so clients can discover the loaded model."""
    return {
        "object": "list",
        "data": [
            {
                "id": holder.model_path,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not holder.loaded:
        raise HTTPException(503, "Model not loaded yet")

    # Convert ChatMessage objects to dicts, preserving tool-related fields
    messages = []
    for m in req.messages:
        d: dict = {"role": m.role, "content": m.text()}
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        messages.append(d)

    # Normalize tools for the chat template
    template_tools = None
    if req.tools and req.tool_choice != "none":
        template_tools = _normalize_tools_for_template(req.tools)

    prompt_text, prompt_len = _build_prompt(holder.tokenizer, messages, tools=template_tools)

    max_tokens = req.max_tokens or 4096
    completion_id = _req_id()
    created = int(time.time())
    model_name = req.model or holder.model_path

    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)

    gen_kwargs = dict(
        model=holder.model,
        tokenizer=holder.tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    # ── Streaming ────────────────────────────────────────────────────────
    if req.stream:
        has_tools = bool(template_tools)

        async def event_stream() -> AsyncIterator[str]:
            async with _inference_lock:
                comp_tokens = 0
                filt = _StreamFilter()
                # When tools are provided, buffer everything so we can
                # detect tool_call tags before emitting content.
                raw_buf = "" if has_tools else None

                for resp in stream_generate(**gen_kwargs):
                    comp_tokens += 1
                    text = filt.feed(resp.text)
                    if not text:
                        continue

                    if raw_buf is not None:
                        # Tools mode: accumulate, don't emit yet
                        raw_buf += text
                    else:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Flush any remaining buffered text from StreamFilter
                remaining = filt.flush()

                if raw_buf is not None:
                    # Tools mode: post-process the full output
                    raw_buf += (remaining or "")
                    content, tool_calls_list = _parse_tool_calls(raw_buf)

                    if tool_calls_list:
                        # Emit tool calls as structured chunks
                        for i, tc in enumerate(tool_calls_list):
                            tc_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "tool_calls": [{
                                                "index": i,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": tc["function"],
                                            }],
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(tc_chunk)}\n\n"

                        final = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        }
                        yield f"data: {json.dumps(final)}\n\n"
                    else:
                        # No tool calls found — emit accumulated text as content
                        if raw_buf:
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant", "content": raw_buf},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                        final = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(final)}\n\n"
                else:
                    # No tools mode: emit remaining + final stop
                    if remaining:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": remaining},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    final = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming ────────────────────────────────────────────────────
    async with _inference_lock:
        result = generate(**gen_kwargs)

    content, tool_calls, finish_reason = _postprocess_with_tools(result, tools_were_provided=bool(template_tools))
    result_text = content or ""
    comp_len = len(holder.tokenizer.encode(result_text)) if result_text else 0

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[
            Choice(
                message=ChoiceMessage(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_len,
            completion_tokens=comp_len,
            total_tokens=prompt_len + comp_len,
        ),
    )


# ── Responses API ────────────────────────────────────────────────────────────


@app.post("/v1/responses")
async def responses_create(req: ResponsesRequest):
    log.info("Responses request: model=%s, prev_id=%s, stream=%s, input_type=%s",
             req.model, req.previous_response_id, req.stream, type(req.input).__name__)
    if not holder.loaded:
        raise HTTPException(503, "Model not loaded yet")

    session_id, messages = _responses_input_to_messages(req.input, req.instructions, req.previous_response_id)

    # Normalize tools for the chat template
    template_tools = None
    if req.tools:
        template_tools = _normalize_tools_for_template(req.tools)

    prompt_text, prompt_len = _build_prompt(holder.tokenizer, messages, tools=template_tools)

    max_tokens = req.max_output_tokens or 4096
    resp_id = _resp_id()
    msg_id = _msg_id()
    created = int(time.time())
    model_name = req.model or holder.model_path

    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)

    gen_kwargs = dict(
        model=holder.model,
        tokenizer=holder.tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    # ── Streaming ────────────────────────────────────────────────────────
    if req.stream:
        has_tools = bool(template_tools)

        async def _stream_response() -> AsyncIterator[str]:
            def _evt(event_type: str, data: dict) -> str:
                payload = {"type": event_type, **data}
                return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

            stub_response = {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "model": model_name,
                "status": "in_progress",
                "output": [],
                "usage": None,
            }

            yield _evt("response.created", stub_response)
            yield _evt("response.in_progress", stub_response)

            # Generate tokens (buffer when tools are present)
            full_text = ""
            comp_tokens = 0
            filt = _StreamFilter()
            # When tools provided, defer all output items until generation completes
            items_emitted = False

            if not has_tools:
                # No tools: emit message item + content part immediately for streaming
                msg_item = {
                    "type": "message",
                    "id": msg_id,
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
                yield _evt(
                    "response.output_item.added",
                    {"output_index": 0, "item": msg_item},
                )
                content_part = {"type": "output_text", "text": "", "annotations": []}
                yield _evt(
                    "response.content_part.added",
                    {"output_index": 0, "content_index": 0, "part": content_part},
                )
                items_emitted = True

            async with _inference_lock:
                for resp in stream_generate(**gen_kwargs):
                    comp_tokens += 1
                    text = filt.feed(resp.text)
                    if not text:
                        continue

                    full_text += text
                    if not has_tools:
                        yield _evt(
                            "response.output_text.delta",
                            {"output_index": 0, "content_index": 0, "delta": text},
                        )

            # Flush any remaining buffered text from StreamFilter
            remaining = filt.flush()
            if remaining:
                full_text += remaining
                if not has_tools:
                    yield _evt(
                        "response.output_text.delta",
                        {"output_index": 0, "content_index": 0, "delta": remaining},
                    )

            # Post-process for tool calls when tools were provided
            output_items_done = []
            if has_tools:
                text_content, tool_calls_list = _parse_tool_calls(full_text)
                if tool_calls_list:
                    # Emit function_call output items
                    for i, tc in enumerate(tool_calls_list):
                        fc_item = {
                            "type": "function_call",
                            "id": tc["id"],
                            "call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                            "status": "completed",
                        }
                        yield _evt(
                            "response.output_item.added",
                            {"output_index": i, "item": fc_item},
                        )
                        yield _evt(
                            "response.output_item.done",
                            {"output_index": i, "item": fc_item},
                        )
                        output_items_done.append(fc_item)

                    # Store with tool calls for multi-turn
                    assistant_msg: dict = {"role": "assistant", "content": text_content}
                    assistant_msg["tool_calls"] = tool_calls_list
                    _store_conversation(resp_id, session_id, messages + [assistant_msg])
                else:
                    # No tool calls found — emit as normal message
                    msg_item = {
                        "type": "message",
                        "id": msg_id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    yield _evt(
                        "response.output_item.added",
                        {"output_index": 0, "item": msg_item},
                    )
                    content_part = {"type": "output_text", "text": "", "annotations": []}
                    yield _evt(
                        "response.content_part.added",
                        {"output_index": 0, "content_index": 0, "part": content_part},
                    )
                    if full_text:
                        yield _evt(
                            "response.output_text.delta",
                            {"output_index": 0, "content_index": 0, "delta": full_text},
                        )
                    yield _evt(
                        "response.output_text.done",
                        {"output_index": 0, "content_index": 0, "text": full_text},
                    )
                    yield _evt(
                        "response.content_part.done",
                        {"output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": full_text, "annotations": []}},
                    )
                    msg_item_done = {
                        "type": "message",
                        "id": msg_id,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": full_text, "annotations": []}],
                    }
                    yield _evt(
                        "response.output_item.done",
                        {"output_index": 0, "item": msg_item_done},
                    )
                    output_items_done.append(msg_item_done)
                    _store_conversation(resp_id, session_id, messages + [{"role": "assistant", "content": full_text}])
            else:
                # No tools path — finalize the already-streaming message
                _store_conversation(resp_id, session_id, messages + [{"role": "assistant", "content": full_text}])

                yield _evt(
                    "response.output_text.done",
                    {"output_index": 0, "content_index": 0, "text": full_text},
                )
                yield _evt(
                    "response.content_part.done",
                    {"output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": full_text, "annotations": []}},
                )
                msg_item_done = {
                    "type": "message",
                    "id": msg_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": full_text, "annotations": []}],
                }
                yield _evt(
                    "response.output_item.done",
                    {"output_index": 0, "item": msg_item_done},
                )
                output_items_done.append(msg_item_done)

            # response.completed
            usage = {
                "input_tokens": prompt_len,
                "output_tokens": comp_tokens,
                "total_tokens": prompt_len + comp_tokens,
            }
            final_response = {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "model": model_name,
                "status": "completed",
                "output": output_items_done,
                "usage": usage,
            }
            yield _evt("response.completed", final_response)

        return StreamingResponse(
            _stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming ────────────────────────────────────────────────────
    async with _inference_lock:
        result = generate(**gen_kwargs)

    content, tool_calls, finish_reason = _postprocess_with_tools(result, tools_were_provided=bool(template_tools))
    result_text = content or ""
    comp_len = len(holder.tokenizer.encode(result_text)) if result_text else 0

    # Build output items
    output_items: list = []
    if tool_calls:
        # Add function_call output items for each tool call
        for tc in tool_calls:
            output_items.append({
                "type": "function_call",
                "id": tc["id"],
                "call_id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
                "status": "completed",
            })
        # Store tool calls in conversation for multi-turn
        assistant_msg: dict = {"role": "assistant", "content": content}
        assistant_msg["tool_calls"] = tool_calls
        _store_conversation(resp_id, session_id, messages + [assistant_msg])
    else:
        output_items.append(
            ResponsesOutputMessage(
                id=msg_id,
                content=[ResponsesOutputText(text=result_text)],
            )
        )
        _store_conversation(resp_id, session_id, messages + [{"role": "assistant", "content": result_text}])

    return ResponsesApiResponse(
        id=resp_id,
        created_at=created,
        model=model_name,
        output=output_items,
        usage=ResponsesUsage(
            input_tokens=prompt_len,
            output_tokens=comp_len,
            total_tokens=prompt_len + comp_len,
        ),
    )


# ── Error handler ────────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": "internal_error",
            }
        },
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible MLX chat server")
    p.add_argument(
        "--model",
        default="mlx-community/MiniMax-M2.5-8bit",
        help="HuggingFace model ID or local path (default: MiniMax-M2.5-8bit)",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    p.add_argument("--workers", type=int, default=1, help="Uvicorn workers (default: 1)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load model before starting the server so it's immediately available
    holder.load(args.model)

    log.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
