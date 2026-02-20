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
import re
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import mlx.core as mx
from mlx_lm import load, stream_generate, generate
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
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=4096, ge=1)
    stream: bool = False
    stop: list[str] | str | None = None
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    # extra fields silently ignored so any OpenAI client works
    model_config = {"extra": "ignore"}


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


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
    output: list[ResponsesOutputMessage] = []
    usage: ResponsesUsage | None = None


# ── Model holder ─────────────────────────────────────────────────────────────


@dataclass
class ModelHolder:
    """Holds the loaded model + tokenizer in a thread-safe-ish way."""

    model_path: str = ""
    model: object = None
    tokenizer: object = None
    loaded: bool = False

    def load(self, model_path: str) -> None:
        log.info("Loading model %s …", model_path)
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_path)
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
    _evict_conversations()


def _get_conversation(resp_id: str) -> tuple[str, list[dict]] | None:
    """Retrieve a conversation if it exists and hasn't expired.
    Returns (session_id, messages) or None."""
    entry = _conversation_store.get(resp_id)
    if entry is None:
        return None
    ts, session_id, messages = entry
    if time.time() - ts > _CONVERSATION_TTL:
        _conversation_store.pop(resp_id)
        _archive_conversation(resp_id, ts, session_id, messages)
        return None
    return session_id, messages


# ── Prompt helpers ───────────────────────────────────────────────────────────


def _build_prompt(tokenizer, messages: list[dict]) -> tuple[str, int]:
    """Apply the tokenizer's chat template and return (prompt_text, token_count)."""
    msg_dicts = [{"role": m["role"], "content": m["content"]} for m in messages]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            msg_dicts,
            tokenize=False,
            add_generation_prompt=True,
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


def _req_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:12]


_THINK_RE = re.compile(r"(<think>)?[\s\S]*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.
    Handles cases where the opening <think> tag is missing."""
    return _THINK_RE.sub("", text)


def _resp_id() -> str:
    return "resp_" + uuid.uuid4().hex[:12]


def _msg_id() -> str:
    return "msg_" + uuid.uuid4().hex[:12]


def _responses_input_to_messages(
    input_data: str | list,
    instructions: str | None,
    previous_response_id: str | None = None,
) -> tuple[str, list[dict]]:
    """Convert Responses API input (string or message list) to internal format.

    When *previous_response_id* is given, the prior conversation is prepended
    so the model sees the full multi-turn context.

    Returns (session_id, messages).
    """
    # -- build the *new* turn's messages (without system) --
    new_messages: list[dict] = []
    if isinstance(input_data, str):
        new_messages.append({"role": "user", "content": input_data})
    else:
        for item in input_data:
            if isinstance(item, dict):
                new_messages.append(
                    {"role": item.get("role", "user"), "content": item.get("content", "")}
                )

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

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt_text, prompt_len = _build_prompt(holder.tokenizer, messages)

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

        async def event_stream() -> AsyncIterator[str]:
            async with _inference_lock:
                comp_tokens = 0
                buf = ""
                thinking = True  # assume output starts inside <think>

                for resp in stream_generate(**gen_kwargs):
                    comp_tokens += 1

                    if thinking:
                        buf += resp.text
                        # Check if we've exited the think block
                        end = buf.find("</think>")
                        if end != -1:
                            thinking = False
                            # Grab any text after </think>
                            after = buf[end + len("</think>"):]
                            buf = ""
                            text = after.lstrip("\n")
                            if not text:
                                continue
                        else:
                            continue
                    else:
                        text = resp.text

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

                # Final chunk with finish_reason
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
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

    result = _strip_think(result)
    comp_len = len(holder.tokenizer.encode(result))

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[
            Choice(
                message=ChoiceMessage(content=result),
                finish_reason="stop",
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
    if not holder.loaded:
        raise HTTPException(503, "Model not loaded yet")

    session_id, messages = _responses_input_to_messages(req.input, req.instructions, req.previous_response_id)
    prompt_text, prompt_len = _build_prompt(holder.tokenizer, messages)

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

            # output_item.added
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

            # content_part.added
            content_part = {"type": "output_text", "text": "", "annotations": []}
            yield _evt(
                "response.content_part.added",
                {
                    "output_index": 0,
                    "content_index": 0,
                    "part": content_part,
                },
            )

            # Generate tokens
            full_text = ""
            comp_tokens = 0
            buf = ""
            thinking = True

            async with _inference_lock:
                for resp in stream_generate(**gen_kwargs):
                    comp_tokens += 1

                    if thinking:
                        buf += resp.text
                        end = buf.find("</think>")
                        if end != -1:
                            thinking = False
                            after = buf[end + len("</think>") :]
                            buf = ""
                            text = after.lstrip("\n")
                            if not text:
                                continue
                        else:
                            continue
                    else:
                        text = resp.text

                    full_text += text
                    yield _evt(
                        "response.output_text.delta",
                        {
                            "output_index": 0,
                            "content_index": 0,
                            "delta": text,
                        },
                    )

            # Store conversation for multi-turn
            _store_conversation(resp_id, session_id, messages + [{"role": "assistant", "content": full_text}])

            # output_text.done
            yield _evt(
                "response.output_text.done",
                {
                    "output_index": 0,
                    "content_index": 0,
                    "text": full_text,
                },
            )

            # content_part.done
            yield _evt(
                "response.content_part.done",
                {
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": full_text, "annotations": []},
                },
            )

            # output_item.done
            msg_item_done = {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": full_text, "annotations": []}
                ],
            }
            yield _evt(
                "response.output_item.done",
                {"output_index": 0, "item": msg_item_done},
            )

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
                "output": [msg_item_done],
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

    result = _strip_think(result)
    comp_len = len(holder.tokenizer.encode(result))

    # Store conversation for multi-turn
    _store_conversation(resp_id, session_id, messages + [{"role": "assistant", "content": result}])

    return ResponsesApiResponse(
        id=resp_id,
        created_at=created,
        model=model_name,
        output=[
            ResponsesOutputMessage(
                id=msg_id,
                content=[ResponsesOutputText(text=result)],
            )
        ],
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
