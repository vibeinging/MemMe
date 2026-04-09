"""EpisodeLogger — standalone episode recording without wrapping Robot."""

from __future__ import annotations

import json
import time
from typing import Any

import memme

from memme_lerobot.base import MemoryMixin


class EpisodeLogger(MemoryMixin):
    """Record robot episodes into MemMe without wrapping the Robot class.

    Use this when you want explicit control over what gets stored::

        from memme_lerobot import EpisodeLogger

        logger = EpisodeLogger(user_id="robot-001")

        logger.begin("pick-and-place task")
        logger.observe("Saw red cup at position (0.3, 0.1, 0.05)")
        logger.act("Moved gripper to (0.3, 0.1, 0.1)")
        logger.observe("Gripper aligned with red cup")
        logger.act("Closed gripper")
        logger.result("success", "Picked up red cup")
        episode = logger.end()

        # Later: recall relevant experiences
        memories = logger.recall("how to pick up a cup")
    """

    def __init__(
        self,
        *,
        user_id: str = "robot",
        db_path: str = "robot_memory.db",
        embedder: str = "onnx",
        api_key: str | None = None,
        llm_provider: str = "openai",
        llm_api_key: str | None = None,
        llm_model: str | None = None,
    ):
        self.user_id = user_id
        self.memory = memme.MemoryStore(
            db_path=db_path,
            embedder=embedder,
            api_key=api_key,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
        )
        self._session_id: str | None = None
        self._task_desc: str | None = None
        self._events: list[tuple[str, str]] = []

    def begin(self, task_description: str = "") -> str:
        """Begin a new episode."""
        self._session_id = f"ep_{int(time.time() * 1000)}"
        self._task_desc = task_description
        self._events = []
        if task_description:
            self._events.append(("system", f"Task: {task_description}"))
        return self._session_id

    def observe(self, description: str) -> None:
        """Record an observation."""
        self._events.append(("user", f"[Observe] {description}"))

    def act(self, description: str) -> None:
        """Record an action taken."""
        self._events.append(("assistant", f"[Act] {description}"))

    def result(self, outcome: str, details: str = "") -> None:
        """Record the outcome of an action."""
        text = f"[Result] {outcome}"
        if details:
            text += f": {details}"
        self._events.append(("user", text))

    def note(self, text: str) -> None:
        """Record a free-form note."""
        self._events.append(("user", f"[Note] {text}"))

    def end(self) -> dict | None:
        """End episode, compact into memories, return result."""
        if not self._events:
            self._session_id = None
            return None

        try:
            result = self.memory.add_smart_messages(
                self._events,
                user_id=self.user_id,
                run_id=self._session_id,
            )
            self._events = []
            self._session_id = None
            return result[0] if result else None
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("compact failed, falling back to simple storage: %s", e)
            combined = "\n".join(f"{role}: {text}" for role, text in self._events)
            if self._task_desc:
                combined = f"Task: {self._task_desc}\n{combined}"
            self.memory.add(combined, user_id=self.user_id)
            self._events = []
            self._session_id = None
            return None

    # store, recall, recall_context, list_memories, list_episodes, stats
    # inherited from MemoryMixin
