"""Shared memory operations mixin for MemoryEnhancedRobot and EpisodeLogger."""

from __future__ import annotations

from typing import Any

import memme


class MemoryMixin:
    """Provides common store/recall/stats operations. Subclasses must set `self.memory` and `self.user_id`."""

    memory: memme.MemoryStore
    user_id: str

    def store(self, content: str, **kwargs: Any) -> dict:
        """Explicitly store a memory."""
        return self.memory.add(content, user_id=self.user_id, **kwargs)

    def recall(self, query: str, *, limit: int = 5, **kwargs: Any) -> list[dict]:
        """Search memories by semantic similarity."""
        return self.memory.search(query, user_id=self.user_id, limit=limit, **kwargs)

    def recall_context(self, query: str, *, limit: int = 3) -> str:
        """Search and format memories as a text context string."""
        results = self.recall(query, limit=limit)
        if not results:
            return ""
        return "\n".join(
            f"[Memory {i}] {r.get('content', '')}"
            for i, r in enumerate(results, 1)
        )

    def list_memories(self, *, limit: int = 20) -> list[dict]:
        """List all stored memories."""
        return self.memory.list(user_id=self.user_id, limit=limit)

    def list_episodes(self, *, limit: int = 20) -> list[dict]:
        """List stored episodes."""
        return self.memory.list_episodes(user_id=self.user_id, limit=limit)

    def stats(self) -> dict:
        """Get memory statistics."""
        return self.memory.user_stats(self.user_id)
