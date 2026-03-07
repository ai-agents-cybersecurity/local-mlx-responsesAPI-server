"""Output postprocessing: think-tag stripping, channel format handling."""

from __future__ import annotations

import re

# ── Think-tag stripping ─────────────────────────────────────────────────────

_THINK_RE = re.compile(r"(<think>)?[\s\S]*?</think>\s*", re.DOTALL)

# GPT-OSS channel format: extract only <|channel|>final<|message|>...<|end|>
_CHANNEL_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\|end\|>|$)"
)


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.
    Handles cases where the opening <think> tag is missing."""
    return _THINK_RE.sub("", text)


def strip_channels(text: str) -> str:
    """Extract final-channel content from GPT-OSS channel format.
    If the text doesn't use channel format, returns it unchanged."""
    if "<|channel|>" not in text:
        return text
    matches = _CHANNEL_FINAL_RE.findall(text)
    if matches:
        return "\n".join(m.strip() for m in matches)
    return text


def postprocess(text: str) -> str:
    """Apply all model-output cleanup: think-tags, channel format, etc."""
    text = strip_think(text)
    text = strip_channels(text)
    return text
