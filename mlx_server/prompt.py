"""Prompt building and input conversion helpers."""

from __future__ import annotations

import json
import logging

from mlx_server.conversation import get_conversation, sess_id
from mlx_server.schemas import _normalise_content
from mlx_server.tool_parsing import call_id

log = logging.getLogger("mlx-server")


def normalize_tools_for_template(tools: list[dict]) -> list[dict]:
    """Normalize tool definitions to the format expected by apply_chat_template(tools=...).

    Accepts both OpenAI Chat format (nested: {type, function: {name, ...}})
    and Responses API format (flat: {type, name, description, parameters}).
    Returns list of {type: "function", function: {name, description, parameters}}.
    """
    normalized = []
    for tool in tools:
        if "function" in tool:
            normalized.append({
                "type": "function",
                "function": {
                    "name": tool["function"].get("name", ""),
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {}),
                },
            })
        else:
            normalized.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            })
    return normalized


def build_prompt(tokenizer, messages: list[dict], tools: list[dict] | None = None) -> tuple[str, int]:
    """Apply the tokenizer's chat template and return (prompt_text, token_count)."""
    msg_dicts = []
    for m in messages:
        d = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
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
            template_kwargs.pop("tools", None)
            try:
                prompt_text = tokenizer.apply_chat_template(
                    msg_dicts,
                    **template_kwargs,
                )
            except Exception:
                log.warning("Chat template failed with tool messages, flattening")
                flat = []
                for d in msg_dicts:
                    if d["role"] == "tool":
                        flat.append({"role": "user", "content": f"[Tool result]: {d.get('content', '')}"})
                    elif d.get("tool_calls"):
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
        prompt_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in msg_dicts
        )
        prompt_text += "\nassistant:"

    prompt_len = len(tokenizer.encode(prompt_text))
    return prompt_text, prompt_len


def responses_input_to_messages(
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
    new_messages: list[dict] = []
    if isinstance(input_data, str):
        new_messages.append({"role": "user", "content": input_data})
    else:
        log.info("Responses input list (%d items): %r", len(input_data), input_data)
        for item in input_data:
            if isinstance(item, dict):
                item_type = item.get("type", "")

                if item_type == "function_call":
                    cid = item.get("call_id") or item.get("id", call_id())
                    fn_name = item.get("name", "")
                    arguments = item.get("arguments", "{}")
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    tc_entry = {
                        "id": cid,
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

                if item_type == "function_call_output":
                    new_messages.append({
                        "role": "tool",
                        "content": item.get("output", ""),
                        "tool_call_id": item.get("call_id", ""),
                    })
                    continue

                role = item.get("role", "user")
                content = _normalise_content(item.get("content", ""))
                if item_type == "message" and not content and "content" not in item:
                    continue
                new_messages.append({"role": role, "content": content})
            elif isinstance(item, str):
                new_messages.append({"role": "user", "content": item})

    prior_result = get_conversation(previous_response_id) if previous_response_id else None
    if prior_result is not None:
        session_id_val, prior_messages = prior_result
        prior = list(prior_messages)

        if instructions:
            if prior and prior[0]["role"] == "system":
                prior[0] = {"role": "system", "content": instructions}
            else:
                prior.insert(0, {"role": "system", "content": instructions})
        return session_id_val, prior + new_messages

    session_id_val = sess_id()
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.extend(new_messages)
    return session_id_val, messages
