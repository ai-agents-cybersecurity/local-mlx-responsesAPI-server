"""In-memory conversation store with TTL and archiving."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

log = logging.getLogger("mlx-server")

# Maps response_id -> (timestamp, session_id, full message history)
_conversation_store: dict[str, tuple[float, str, list[dict]]] = {}
_CONVERSATION_TTL: float = 3600.0  # 1 hour
_CONVERSATION_MAX: int = 1000      # max entries before forced eviction
_CONVERSATION_LOG_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversation_logs")

# Ensure log directory exists at import time
os.makedirs(_CONVERSATION_LOG_DIR, exist_ok=True)


def sess_id() -> str:
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
        log.info("Archived conversation %s (session %s) -> %s", resp_id, session_id, filename)
    except OSError:
        log.exception("Failed to archive conversation %s", resp_id)


def _evict_conversations() -> None:
    """Remove expired entries; if still over limit, drop oldest."""
    now = time.time()
    expired = [k for k, (ts, _, _) in _conversation_store.items() if now - ts > _CONVERSATION_TTL]
    for k in expired:
        ts, session_id, messages = _conversation_store.pop(k)
        _archive_conversation(k, ts, session_id, messages)
    if len(_conversation_store) > _CONVERSATION_MAX:
        by_age = sorted(_conversation_store, key=lambda k: _conversation_store[k][0])
        for k in by_age[: len(_conversation_store) - _CONVERSATION_MAX]:
            ts, session_id, messages = _conversation_store.pop(k)
            _archive_conversation(k, ts, session_id, messages)


def store_conversation(resp_id: str, session_id: str, messages: list[dict]) -> None:
    """Store a conversation and evict stale entries."""
    _conversation_store[resp_id] = (time.time(), session_id, messages)
    log.info("Stored conversation %s (session %s, %d messages, store size=%d)",
             resp_id, session_id, len(messages), len(_conversation_store))
    _evict_conversations()


def get_conversation(resp_id: str) -> tuple[str, list[dict]] | None:
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
