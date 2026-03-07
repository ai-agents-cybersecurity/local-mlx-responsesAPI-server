"""Pydantic request / response schemas for all API surfaces."""

from __future__ import annotations

from pydantic import BaseModel, Field


def _normalise_content(content) -> str:
    """Normalise content to a plain string (handles str, list-of-parts, None)."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type", "")
                if ptype in ("text", "input_text"):
                    parts.append(part.get("text", ""))
                elif "text" in part:
                    parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


# ── Chat Completions schemas ────────────────────────────────────────────────


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


# ── Responses API schemas ───────────────────────────────────────────────────


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


# ── Anthropic Messages API schemas ──────────────────────────────────────────


class AnthropicMessage(BaseModel):
    role: str
    content: str | list = ""
    model_config = {"extra": "ignore"}

    def text(self) -> str:
        """Normalise content to a plain string."""
        if isinstance(self.content, list):
            parts = []
            for block in self.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(self.content)


class AnthropicMessagesRequest(BaseModel):
    model: str | None = None
    messages: list[AnthropicMessage]
    max_tokens: int = Field(default=4096, ge=1)
    system: str | list | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False
    stop_sequences: list[str] | None = None
    metadata: dict | None = None
    model_config = {"extra": "ignore"}


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str = ""


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage
