"""Tool call parsing for multiple model output formats."""

from __future__ import annotations

import json
import logging
import re
import uuid

log = logging.getLogger("mlx-server")


def call_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


# ── Regex patterns ──────────────────────────────────────────────────────────

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
# Flexible quoting: name="X", name='X', name=X all accepted
_INVOKE_RE = re.compile(r'<invoke\s+name\s*=\s*["\']?([^"\'>\s]+)["\']?\s*>([\s\S]*?)</invoke>', re.DOTALL)
_INVOKE_PARAM_RE = re.compile(r'<parameter\s+name\s*=\s*["\']?([^"\'>\s]+)["\']?\s*>\s*([\s\S]*?)\s*</parameter>', re.DOTALL)

# Bare <invoke> blocks not wrapped in any tool_call container
_BARE_INVOKE_RE = re.compile(r'<invoke\s+name\s*=\s*["\']?([^"\'>\s]+)["\']?\s*>([\s\S]*?)</invoke>', re.DOTALL)


# ── Parsers ─────────────────────────────────────────────────────────────────


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
        try:
            params[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            params[key] = value

    return {"name": name, "arguments": json.dumps(params)}


def _parse_invoke_body(name: str, body: str) -> dict | None:
    """Parse parameters from the body of an <invoke> block.

    Tries multiple parameter formats and JSON fallback.
    Returns {"name": ..., "arguments": "..."} or None.
    """
    params = {}
    # Try <parameter name="key">value</parameter> (with flexible quoting)
    for pm in _INVOKE_PARAM_RE.finditer(body):
        key = pm.group(1).strip()
        value = pm.group(2).strip()
        try:
            params[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            params[key] = value

    # Also try <parameter=key>value</parameter> (Qwen-style)
    if not params:
        for pm in _TOOL_CALL_PARAM_RE.finditer(body):
            key = pm.group(1).strip()
            value = pm.group(2).strip()
            try:
                params[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[key] = value

    # Fallback: try to parse the body as JSON arguments
    if not params:
        json_match = _TOOL_CALL_JSON_RE.search(body)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict):
                    params = parsed
            except json.JSONDecodeError:
                pass

    if not params:
        log.warning("Invoke tool call '%s' has no parseable arguments. Body: %s", name, body[:300])

    return {"name": name, "arguments": json.dumps(params)}


def _parse_invoke_tool_call(inner: str) -> dict | None:
    """Parse MiniMax invoke format: <invoke name="NAME"><parameter name="K">V</parameter></invoke>."""
    invoke_match = _INVOKE_RE.search(inner)
    if not invoke_match:
        return None

    name = invoke_match.group(1).strip()
    body = invoke_match.group(2)
    return _parse_invoke_body(name, body)


def _parse_json_tool_call(inner: str) -> dict | None:
    """Parse JSON-style tool call: {"name": "...", "arguments": {...}}."""
    json_match = _TOOL_CALL_JSON_RE.search(inner)
    if not json_match:
        return None
    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    if "name" not in parsed:
        return None

    name = parsed["name"]
    arguments = parsed.get("arguments", {})
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {"name": name, "arguments": arguments}


def parse_tool_calls(raw_text: str) -> tuple[str | None, list[dict]]:
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
    # Try standard <tool_call> blocks first, then MiniMax wrapper, then bare <invoke>
    blocks = _TOOL_CALL_BLOCK_RE.findall(raw_text)
    strip_re = _TOOL_CALL_BLOCK_RE
    if not blocks:
        blocks = _MINIMAX_BLOCK_RE.findall(raw_text)
        strip_re = _MINIMAX_BLOCK_RE
    if not blocks:
        # Last resort: look for bare <invoke> blocks directly in the text
        invoke_matches = _BARE_INVOKE_RE.finditer(raw_text)
        bare_calls = []
        for m in invoke_matches:
            name = m.group(1).strip()
            body = m.group(2)
            result = _parse_invoke_body(name, body)
            if result:
                log.info("Bare invoke tool call: %s", result)
                bare_calls.append({
                    "id": call_id(),
                    "type": "function",
                    "function": result,
                })
        if bare_calls:
            remaining = _BARE_INVOKE_RE.sub("", raw_text).strip()
            remaining = re.sub(r'\S*:tool_call\s*', '', remaining).strip()
            return remaining or None, bare_calls
        return raw_text, []

    tool_calls = []
    for inner in blocks:
        log.info("Tool call block inner (%d chars): %s", len(inner), repr(inner[:500]))
        json_result = _parse_json_tool_call(inner)
        xml_result = _parse_xml_tool_call(inner) if not json_result else None
        invoke_result = _parse_invoke_tool_call(inner) if not json_result and not xml_result else None
        result = json_result or xml_result or invoke_result
        log.info("Parse results -- json=%s xml=%s invoke=%s -> %s",
                 json_result, xml_result, invoke_result, result)
        if result is None:
            log.warning("Failed to parse tool_call content: %s", inner[:200])
            continue

        tool_calls.append({
            "id": call_id(),
            "type": "function",
            "function": result,
        })

    if not tool_calls:
        return raw_text, []

    remaining = strip_re.sub("", raw_text).strip()
    return remaining or None, tool_calls


def postprocess_with_tools(raw_text: str, tools_were_provided: bool) -> tuple[str | None, list[dict] | None, str]:
    """Post-process model output with tool call extraction.

    Returns (content, tool_calls, finish_reason).
    """
    from mlx_server.postprocess import strip_think, strip_channels

    text = strip_think(raw_text)
    text = strip_channels(text)

    if tools_were_provided:
        content, tool_calls = parse_tool_calls(text)
        if tool_calls:
            return content, tool_calls, "tool_calls"

    return text, None, "stop"
