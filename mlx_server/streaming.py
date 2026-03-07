"""Streaming token filter for think-tags and channel markers."""

from __future__ import annotations

from mlx_server.postprocess import strip_think


class StreamFilter:
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
                token = after.lstrip("\n")
            elif "<|channel|>" in self._buf or "<|start|>" in self._buf:
                self._thinking = False
                self._uses_channels = True
                token = self._buf
                self._buf = ""
            else:
                return ""

        # Phase 2: handle <|channel|> format
        self._buf += token

        if not self._uses_channels and "<|channel|>" in self._buf:
            self._uses_channels = True

        if not self._uses_channels:
            out = self._buf
            self._buf = ""
            return out

        # Buffer until we can resolve channel boundaries
        emit = ""
        while True:
            if self._in_final:
                end_pos = self._buf.find("<|end|>")
                start_pos = self._buf.find("<|start|>")
                boundary = -1
                if end_pos != -1 and start_pos != -1:
                    boundary = min(end_pos, start_pos)
                elif end_pos != -1:
                    boundary = end_pos
                elif start_pos != -1:
                    boundary = start_pos

                if boundary != -1:
                    emit += self._buf[:boundary]
                    if self._buf[boundary:].startswith("<|end|>"):
                        self._buf = self._buf[boundary + len("<|end|>"):]
                    else:
                        self._buf = self._buf[boundary:]
                    self._in_final = False
                else:
                    safe, self._buf = self._safe_emit(self._buf)
                    emit += safe
                    break
            else:
                marker = "<|channel|>final<|message|>"
                pos = self._buf.find(marker)
                if pos != -1:
                    self._buf = self._buf[pos + len(marker):]
                    self._in_final = True
                    continue
                ch_pos = self._buf.find("<|channel|>")
                if ch_pos != -1:
                    rest = self._buf[ch_pos + len("<|channel|>"):]
                    end_pos = rest.find("<|end|>")
                    if end_pos != -1:
                        self._buf = rest[end_pos + len("<|end|>"):]
                        continue
                    start_pos = rest.find("<|start|>")
                    if start_pos != -1:
                        self._buf = rest[start_pos:]
                        continue
                    break
                else:
                    safe, self._buf = self._safe_emit(self._buf)
                    break

            if not self._buf:
                break

        return emit

    def flush(self) -> str:
        """Flush any remaining buffered text at end of stream."""
        if not self._buf:
            return ""
        if self._uses_channels:
            if self._in_final:
                out = self._buf
                self._buf = ""
                return out
            self._buf = ""
            return ""
        out = strip_think(self._buf)
        self._buf = ""
        return out

    @staticmethod
    def _safe_emit(buf: str) -> tuple[str, str]:
        """Split buf into safe-to-emit prefix and remainder that could be
        the start of a special token like <|channel|>, <|end|>, <|start|>."""
        last_lt = buf.rfind("<")
        if last_lt != -1 and last_lt > len(buf) - 30:
            return buf[:last_lt], buf[last_lt:]
        return buf, ""
