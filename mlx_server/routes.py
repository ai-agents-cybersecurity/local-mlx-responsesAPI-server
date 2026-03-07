"""API route handlers for all endpoints."""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from mlx_lm import stream_generate, generate
from mlx_lm.sample_utils import make_sampler

from mlx_server.model import holder, inference_lock
from mlx_server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    UsageInfo,
    ResponsesRequest,
    ResponsesApiResponse,
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesUsage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicContentBlock,
    AnthropicUsage,
)
from mlx_server.prompt import build_prompt, normalize_tools_for_template, responses_input_to_messages
from mlx_server.tool_parsing import parse_tool_calls, postprocess_with_tools
from mlx_server.streaming import StreamFilter
from mlx_server.conversation import store_conversation

import logging

log = logging.getLogger("mlx-server")

router = APIRouter()


# ── ID helpers ──────────────────────────────────────────────────────────────

def _req_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:12]

def _resp_id() -> str:
    return "resp_" + uuid.uuid4().hex[:12]

def _msg_id() -> str:
    return "msg_" + uuid.uuid4().hex[:12]


# ── Health & Models ─────────────────────────────────────────────────────────


@router.get("/health")
async def health():
    return {"status": "ok", "model": holder.model_path, "loaded": holder.loaded}


@router.get("/v1/models")
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


# ── Chat Completions ────────────────────────────────────────────────────────


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not holder.loaded:
        raise HTTPException(503, "Model not loaded yet")

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

    template_tools = None
    if req.tools and req.tool_choice != "none":
        template_tools = normalize_tools_for_template(req.tools)

    prompt_text, prompt_len = build_prompt(holder.tokenizer, messages, tools=template_tools)

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
            async with inference_lock:
                comp_tokens = 0
                filt = StreamFilter()
                raw_buf = "" if has_tools else None

                for resp in stream_generate(**gen_kwargs):
                    comp_tokens += 1
                    text = filt.feed(resp.text)
                    if not text:
                        continue

                    if raw_buf is not None:
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

                remaining = filt.flush()

                if raw_buf is not None:
                    raw_buf += (remaining or "")
                    log.info("[CHAT STREAM] raw_buf for tool parsing (%d chars): %s", len(raw_buf), repr(raw_buf[:1000]))
                    content, tool_calls_list = parse_tool_calls(raw_buf)
                    log.info("[CHAT STREAM] parse result: content=%s, tool_calls=%s", repr(content)[:200] if content else None, tool_calls_list)

                    if tool_calls_list:
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
    async with inference_lock:
        result = generate(**gen_kwargs)

    content, tool_calls, finish_reason = postprocess_with_tools(result, tools_were_provided=bool(template_tools))
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


# ── Responses API ───────────────────────────────────────────────────────────


@router.post("/v1/responses")
async def responses_create(req: ResponsesRequest):
    log.info("Responses request: model=%s, prev_id=%s, stream=%s, input_type=%s",
             req.model, req.previous_response_id, req.stream, type(req.input).__name__)
    if not holder.loaded:
        raise HTTPException(503, "Model not loaded yet")

    session_id, messages = responses_input_to_messages(req.input, req.instructions, req.previous_response_id)

    template_tools = None
    if req.tools:
        template_tools = normalize_tools_for_template(req.tools)

    prompt_text, prompt_len = build_prompt(holder.tokenizer, messages, tools=template_tools)

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

            full_text = ""
            comp_tokens = 0
            filt = StreamFilter()
            items_emitted = False

            if not has_tools:
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

            async with inference_lock:
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

            remaining = filt.flush()
            if remaining:
                full_text += remaining
                if not has_tools:
                    yield _evt(
                        "response.output_text.delta",
                        {"output_index": 0, "content_index": 0, "delta": remaining},
                    )

            output_items_done = []
            if has_tools:
                log.info("[RESP STREAM] full_text for tool parsing (%d chars): %s", len(full_text), repr(full_text[:1000]))
                text_content, tool_calls_list = parse_tool_calls(full_text)
                log.info("[RESP STREAM] parse result: text=%s, tool_calls=%s", repr(text_content)[:200] if text_content else None, tool_calls_list)
                if tool_calls_list:
                    for i, tc in enumerate(tool_calls_list):
                        fn = tc["function"]
                        fc_item_initial = {
                            "type": "function_call",
                            "id": tc["id"],
                            "call_id": tc["id"],
                            "name": fn["name"],
                            "arguments": "",
                            "status": "in_progress",
                        }
                        yield _evt(
                            "response.output_item.added",
                            {"output_index": i, "item": fc_item_initial},
                        )
                        yield _evt(
                            "response.function_call_arguments.delta",
                            {"output_index": i, "item_id": tc["id"], "delta": fn["arguments"]},
                        )
                        yield _evt(
                            "response.function_call_arguments.done",
                            {"output_index": i, "item_id": tc["id"], "arguments": fn["arguments"]},
                        )
                        fc_item_done = {
                            "type": "function_call",
                            "id": tc["id"],
                            "call_id": tc["id"],
                            "name": fn["name"],
                            "arguments": fn["arguments"],
                            "status": "completed",
                        }
                        yield _evt(
                            "response.output_item.done",
                            {"output_index": i, "item": fc_item_done},
                        )
                        output_items_done.append(fc_item_done)

                    assistant_msg: dict = {"role": "assistant", "content": text_content}
                    assistant_msg["tool_calls"] = tool_calls_list
                    store_conversation(resp_id, session_id, messages + [assistant_msg])
                else:
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
                    store_conversation(resp_id, session_id, messages + [{"role": "assistant", "content": full_text}])
            else:
                store_conversation(resp_id, session_id, messages + [{"role": "assistant", "content": full_text}])

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
    async with inference_lock:
        result = generate(**gen_kwargs)

    content, tool_calls, finish_reason = postprocess_with_tools(result, tools_were_provided=bool(template_tools))
    result_text = content or ""
    comp_len = len(holder.tokenizer.encode(result_text)) if result_text else 0

    output_items: list = []
    if tool_calls:
        for tc in tool_calls:
            output_items.append({
                "type": "function_call",
                "id": tc["id"],
                "call_id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
                "status": "completed",
            })
        assistant_msg: dict = {"role": "assistant", "content": content}
        assistant_msg["tool_calls"] = tool_calls
        store_conversation(resp_id, session_id, messages + [assistant_msg])
    else:
        output_items.append(
            ResponsesOutputMessage(
                id=msg_id,
                content=[ResponsesOutputText(text=result_text)],
            )
        )
        store_conversation(resp_id, session_id, messages + [{"role": "assistant", "content": result_text}])

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


# ── Anthropic Messages API ──────────────────────────────────────────────────


@router.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest):
    """Anthropic-compatible Messages API (POST /v1/messages).

    Speaks the Anthropic/Claude native protocol so clients using the
    Anthropic SDK can hit this server directly.
    """
    log.info("Anthropic Messages request: model=%s, stream=%s, messages=%d",
             req.model, req.stream, len(req.messages))
    if not holder.loaded:
        raise HTTPException(503, "Model not loaded yet")

    if not req.messages:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "messages: at least one message is required",
                },
            },
        )

    # Build internal messages list: system is top-level, not in messages array
    messages: list[dict] = []
    if req.system:
        if isinstance(req.system, list):
            # system can be a list of content blocks
            sys_text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in req.system
            )
        else:
            sys_text = req.system
        messages.append({"role": "system", "content": sys_text})

    for m in req.messages:
        messages.append({"role": m.role, "content": m.text()})

    prompt_text, prompt_len = build_prompt(holder.tokenizer, messages)

    max_tokens = req.max_tokens
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    model_name = req.model or holder.model_path

    sampler = make_sampler(
        temp=req.temperature if req.temperature is not None else 0.7,
        top_p=req.top_p if req.top_p is not None else 0.95,
    )

    # Build stop sequences list
    stop_seqs = req.stop_sequences or []

    gen_kwargs = dict(
        model=holder.model,
        tokenizer=holder.tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    # ── Streaming ────────────────────────────────────────────────────────
    if req.stream:

        async def _stream_anthropic() -> AsyncIterator[str]:
            def _sse(event_type: str, data: dict) -> str:
                return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            # message_start — includes input usage
            yield _sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": prompt_len, "output_tokens": 0},
                },
            })

            # content_block_start
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            })

            comp_tokens = 0
            full_text = ""
            filt = StreamFilter()
            hit_stop_seq = None

            async with inference_lock:
                for resp in stream_generate(**gen_kwargs):
                    comp_tokens += 1
                    text = filt.feed(resp.text)
                    if not text:
                        continue
                    full_text += text

                    # Check for stop sequences in accumulated text
                    if stop_seqs:
                        for seq in stop_seqs:
                            idx = full_text.find(seq)
                            if idx != -1:
                                hit_stop_seq = seq
                                # Trim to just before the stop sequence
                                full_text = full_text[:idx]
                                break
                        if hit_stop_seq:
                            break

                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    })

            if not hit_stop_seq:
                remaining = filt.flush()
                if remaining:
                    comp_tokens += 1
                    full_text += remaining

                    # Check stop sequences in remaining text too
                    if stop_seqs:
                        for seq in stop_seqs:
                            idx = full_text.find(seq)
                            if idx != -1:
                                hit_stop_seq = seq
                                full_text = full_text[:idx]
                                break

                    if not hit_stop_seq and remaining:
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": remaining},
                        })

            stop_reason = "stop_sequence" if hit_stop_seq else "end_turn"

            # content_block_stop
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            })

            # message_delta — includes output usage and stop_reason
            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": hit_stop_seq},
                "usage": {"output_tokens": comp_tokens},
            })

            # message_stop
            yield _sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(
            _stream_anthropic(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming ────────────────────────────────────────────────────
    async with inference_lock:
        result = generate(**gen_kwargs)

    from mlx_server.postprocess import postprocess
    result_text = postprocess(result)

    # Check for stop sequences and truncate
    hit_stop_seq = None
    if stop_seqs and result_text:
        for seq in stop_seqs:
            idx = result_text.find(seq)
            if idx != -1:
                if hit_stop_seq is None or idx < result_text.find(hit_stop_seq):
                    hit_stop_seq = seq
        if hit_stop_seq:
            result_text = result_text[:result_text.find(hit_stop_seq)]

    comp_len = len(holder.tokenizer.encode(result_text)) if result_text else 0
    stop_reason = "stop_sequence" if hit_stop_seq else "end_turn"

    return AnthropicMessagesResponse(
        id=msg_id,
        content=[AnthropicContentBlock(text=result_text)],
        model=model_name,
        stop_reason=stop_reason,
        stop_sequence=hit_stop_seq,
        usage=AnthropicUsage(
            input_tokens=prompt_len,
            output_tokens=comp_len,
        ),
    )


@router.post("/v1/v1/messages")
async def anthropic_messages_sdk_compat(req: AnthropicMessagesRequest):
    """Compatibility route for Anthropic SDK clients.

    The Anthropic SDK posts to path '/v1/messages' relative to base_url.
    When base_url is 'http://host:port/v1', httpx resolves this to
    '/v1/v1/messages'. This route catches that doubled prefix.
    """
    return await anthropic_messages(req)


# ── Azure OpenAI-compatible routes ──────────────────────────────────────────


@router.post("/openai/deployments/{deployment_id}/responses")
async def azure_responses_create(
    deployment_id: str,
    req: ResponsesRequest,
    api_version: str | None = Query(None, alias="api-version"),
):
    """Azure OpenAI Responses API -- delegates to the standard handler."""
    log.info("Azure Responses request: deployment=%s, api-version=%s", deployment_id, api_version)
    req.model = deployment_id
    return await responses_create(req)


@router.post("/openai/deployments/{deployment_id}/chat/completions")
async def azure_chat_completions(
    deployment_id: str,
    req: ChatCompletionRequest,
    api_version: str | None = Query(None, alias="api-version"),
):
    """Azure OpenAI Chat Completions API -- delegates to the standard handler."""
    log.info("Azure Chat Completions request: deployment=%s, api-version=%s", deployment_id, api_version)
    req.model = deployment_id
    return await chat_completions(req)


@router.get("/openai/deployments/{deployment_id}/models")
async def azure_list_models(
    deployment_id: str,
    api_version: str | None = Query(None, alias="api-version"),
):
    """Azure-style model info -- returns the loaded model tagged with the
    deployment id the client asked about."""
    return {
        "object": "list",
        "data": [
            {
                "id": deployment_id,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }
