"""
ChainMind Short-Term Memory — Sliding window conversation buffer.

Maintains recent conversation context and tool call history
for the ReAct loop. In-memory with configurable max tokens.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from chainmind.core.interfaces import IMemoryStore
from chainmind.core.types import MemoryEntry

logger = logging.getLogger(__name__)


class ShortTermMemory(IMemoryStore):
    """
    In-memory sliding window buffer for conversation context.

    Keeps the most recent N entries per session, evicting oldest
    when capacity is exceeded.
    """

    def __init__(self, max_entries_per_session: int = 50, max_total_chars: int = 16000):
        self._sessions: dict[str, deque[MemoryEntry]] = {}
        self._max_entries = max_entries_per_session
        self._max_chars = max_total_chars

    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry in the session buffer."""
        session_id = entry.session_id

        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self._max_entries)

        self._sessions[session_id].append(entry)

        # Enforce character limit by evicting oldest
        total_chars = sum(len(e.content) for e in self._sessions[session_id])
        while total_chars > self._max_chars and len(self._sessions[session_id]) > 1:
            self._sessions[session_id].popleft()
            total_chars = sum(len(e.content) for e in self._sessions[session_id])

        return entry.entry_id

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        Retrieve recent memories.

        For short-term memory, 'query' is treated as session_id.
        Returns most recent entries for that session.
        """
        session_id = query  # In STM, query = session_id
        if session_id not in self._sessions:
            return []

        entries = list(self._sessions[session_id])
        return entries[-top_k:]  # Most recent

    async def clear(self, session_id: str | None = None) -> int:
        """Clear memories for a session or all sessions."""
        if session_id:
            count = len(self._sessions.get(session_id, []))
            self._sessions.pop(session_id, None)
            return count
        else:
            total = sum(len(q) for q in self._sessions.values())
            self._sessions.clear()
            return total

    @property
    def session_count(self) -> int:
        return len(self._sessions)
