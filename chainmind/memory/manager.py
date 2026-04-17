"""
ChainMind Memory Manager — Lifecycle manager for short-term and long-term memory.

Coordinates reads/writes across both memory tiers:
- Short-term: fast, volatile, per-session
- Long-term: persistent, cross-session, importance-weighted
"""

from __future__ import annotations

import logging
from typing import Any

from chainmind.config.settings import Settings
from chainmind.core.interfaces import IMemoryStore
from chainmind.core.types import MemoryEntry
from chainmind.memory.long_term import LongTermMemory
from chainmind.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


class MemoryManager(IMemoryStore):
    """
    Unified memory manager that coordinates STM and LTM.

    Write path: Store in STM always. Promote to LTM if importance >= threshold.
    Read path: Query STM first (recent), then LTM (historical).
    """

    def __init__(self, settings: Settings):
        self._stm = ShortTermMemory(
            max_total_chars=settings.memory_short_term_max_tokens * 4,  # ~chars
        )
        self._ltm = LongTermMemory(db_path=str(settings.memory_db_path))
        self._ltm_promotion_threshold = 0.6  # Importance threshold for LTM storage

    @property
    def short_term(self) -> ShortTermMemory:
        return self._stm

    @property
    def long_term(self) -> LongTermMemory:
        return self._ltm

    async def store(self, entry: MemoryEntry) -> str:
        """Store in STM, and promote to LTM if important enough."""
        # Always store in short-term
        await self._stm.store(entry)

        # Promote to long-term if important
        if entry.importance >= self._ltm_promotion_threshold:
            await self._ltm.store(entry)
            logger.debug(
                f"Memory promoted to LTM: {entry.entry_id} (importance={entry.importance})"
            )

        return entry.entry_id

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        Retrieve from both STM and LTM, deduplicated.

        STM results come first (more recent), then LTM fills the gap.
        """
        stm_results = await self._stm.retrieve(query, top_k=top_k)

        # If STM has enough, return immediately
        if len(stm_results) >= top_k:
            return stm_results[:top_k]

        # Fill remaining from LTM
        remaining = top_k - len(stm_results)
        ltm_results = await self._ltm.retrieve(query, top_k=remaining)

        # Deduplicate by entry_id
        seen_ids = {e.entry_id for e in stm_results}
        combined = list(stm_results)
        for entry in ltm_results:
            if entry.entry_id not in seen_ids:
                combined.append(entry)
                seen_ids.add(entry.entry_id)

        return combined[:top_k]

    async def clear(self, session_id: str | None = None) -> int:
        """Clear both STM and LTM."""
        stm_count = await self._stm.clear(session_id)
        ltm_count = await self._ltm.clear(session_id)
        return stm_count + ltm_count
