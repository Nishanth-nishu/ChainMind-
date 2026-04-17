"""
ChainMind Long-Term Memory — Persistent episodic memory using SQLite.

Stores past decisions, outcomes, and user preferences.
No external database required — uses aiosqlite for async SQLite.
Supports both exact-match SQL queries and semantic search via
integration with the dense retriever.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from chainmind.core.interfaces import IMemoryStore
from chainmind.core.types import MemoryEntry

logger = logging.getLogger(__name__)


class LongTermMemory(IMemoryStore):
    """
    SQLite-backed persistent episodic memory.

    Schema:
    - entry_id (PK)
    - session_id
    - agent_id
    - content
    - memory_type (episodic, semantic, procedural)
    - importance (0.0 - 1.0)
    - created_at
    - metadata (JSON)
    """

    def __init__(self, db_path: str = "./data/memory.db"):
        self._db_path = db_path
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Create the database and table if not exists."""
        if self._initialized:
            return

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    entry_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT DEFAULT 'episodic',
                    importance REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent ON memories(agent_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC)
            """)
            await db.commit()

        self._initialized = True
        logger.info(f"Long-term memory initialized at {self._db_path}")

    async def store(self, entry: MemoryEntry) -> str:
        """Persist a memory entry to SQLite."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO memories
                   (entry_id, session_id, agent_id, content, memory_type, importance, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.entry_id,
                    entry.session_id,
                    entry.agent_id,
                    entry.content,
                    entry.memory_type,
                    entry.importance,
                    entry.created_at.isoformat(),
                    json.dumps(entry.metadata, default=str),
                ),
            )
            await db.commit()

        return entry.entry_id

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        Retrieve memories by keyword search.

        Uses SQLite FTS-like LIKE queries for simple keyword matching.
        For semantic search, integrate with the dense retriever.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM memories
                   WHERE content LIKE ?
                   ORDER BY importance DESC, created_at DESC
                   LIMIT ?""",
                (f"%{query}%", top_k),
            )
            rows = await cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def retrieve_by_session(
        self, session_id: str, top_k: int = 10
    ) -> list[MemoryEntry]:
        """Retrieve all memories for a specific session."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM memories
                   WHERE session_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (session_id, top_k),
            )
            rows = await cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def retrieve_important(self, min_importance: float = 0.7, top_k: int = 10) -> list[MemoryEntry]:
        """Retrieve high-importance memories across all sessions."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM memories
                   WHERE importance >= ?
                   ORDER BY importance DESC, created_at DESC
                   LIMIT ?""",
                (min_importance, top_k),
            )
            rows = await cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def clear(self, session_id: str | None = None) -> int:
        """Clear memories for a session or all."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            if session_id:
                cursor = await db.execute(
                    "DELETE FROM memories WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = await db.execute("DELETE FROM memories")
            await db.commit()
            return cursor.rowcount

    def _row_to_entry(self, row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        return MemoryEntry(
            entry_id=row["entry_id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
