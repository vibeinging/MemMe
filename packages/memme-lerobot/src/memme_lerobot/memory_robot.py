"""MemoryEnhancedRobot — wraps a LeRobot Robot with persistent memory."""

from __future__ import annotations

import json
import time
from typing import Any

import memme

from memme_lerobot.base import MemoryMixin


class MemoryEnhancedRobot(MemoryMixin):
    """Wraps a LeRobot Robot to automatically record and recall experiences.

    Usage::

        from lerobot.robots import make_robot
        from memme_lerobot import MemoryEnhancedRobot

        robot = MemoryEnhancedRobot(
            robot=make_robot("so100"),
            user_id="robot-001",
            db_path="robot_memory.duckdb",
        )

        robot.connect()
        obs = robot.get_observation()          # auto-recalls relevant memories
        robot.send_action(action)              # auto-stores experience
        memories = robot.recall("red cup")     # explicit memory search
        robot.disconnect()                     # auto-compacts episode
    """

    def __init__(
        self,
        robot: Any = None,
        *,
        user_id: str = "robot",
        db_path: str = "robot_memory.duckdb",
        embedder: str = "onnx",
        api_key: str | None = None,
        llm_provider: str = "openai",
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        auto_store: bool = True,
        auto_recall: bool = False,
        max_buffer_size: int = 100,
    ):
        self.robot = robot
        self.user_id = user_id
        self.auto_store = auto_store
        self.auto_recall = auto_recall
        self.max_buffer_size = max_buffer_size

        self.memory = memme.MemoryStore(
            db_path=db_path,
            embedder=embedder,
            api_key=api_key,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
        )

        self._session_id: str | None = None
        self._episode_events: list[dict] = []
        self._last_observation: dict | None = None
        self._last_recall: list[dict] | None = None

    # ------------------------------------------------------------------
    # Robot interface passthrough
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to robot and start a new memory session."""
        if self.robot is not None:
            self.robot.connect()
        self._session_id = f"session_{int(time.time() * 1000)}"
        self._episode_events = []

    def disconnect(self) -> None:
        """Disconnect from robot and compact the episode."""
        self._compact_episode()
        if self.robot is not None:
            self.robot.disconnect()
        self._session_id = None

    def get_observation(self) -> dict:
        """Get observation from robot, optionally recalling relevant memories."""
        if self.robot is None:
            raise RuntimeError("No robot connected")

        obs = self.robot.get_observation()
        self._last_observation = obs

        obs_text = _obs_to_text(obs) if (self.auto_store or self.auto_recall) else None

        if self.auto_store and self._session_id and obs_text:
            self._log_event("observation", obs_text)

        if self.auto_recall and obs_text:
            self._last_recall = self.memory.search(
                obs_text, user_id=self.user_id, limit=3
            )

        return obs

    def send_action(self, action: Any) -> None:
        """Send action to robot, optionally storing the experience."""
        if self.robot is not None:
            self.robot.send_action(action)

        if self.auto_store and self._session_id:
            self._log_event("action", _action_to_text(action))

    @property
    def last_recall(self) -> list[dict] | None:
        """Get the most recent auto-recall results."""
        return self._last_recall

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def store_experience(
        self,
        observation: str,
        action: str,
        result: str | None = None,
    ) -> dict:
        """Store a structured (observation, action, result) experience."""
        parts = [f"Observed: {observation}", f"Action: {action}"]
        if result:
            parts.append(f"Result: {result}")
        content = ". ".join(parts)
        metadata = json.dumps(
            {"type": "experience", "observation": observation, "action": action}
        )
        return self.memory.add(content, user_id=self.user_id, metadata=metadata)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def start_episode(self, session_id: str | None = None) -> str:
        """Manually start a new episode/session."""
        self._compact_episode()  # compact previous if any
        self._session_id = session_id or f"session_{int(time.time() * 1000)}"
        self._episode_events = []
        return self._session_id

    def end_episode(self) -> dict | None:
        """Manually end the current episode and compact into memories."""
        return self._compact_episode()

    # ------------------------------------------------------------------
    # Knowledge graph
    # ------------------------------------------------------------------

    def add_knowledge(self, text: str) -> dict:
        """Extract entities/relationships from text into the knowledge graph."""
        return self.memory.add_graph(text, user_id=self.user_id)

    def search_knowledge(self, query: str, *, depth: int = 2) -> dict:
        """Search the knowledge graph."""
        return self.memory.search_graph(query, user_id=self.user_id, depth=depth)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, content: str) -> None:
        """Log an event to the current episode buffer. Auto-flushes when full."""
        if not content:
            return
        self._episode_events.append(
            {"event_type": event_type, "content": content}
        )
        if len(self._episode_events) >= self.max_buffer_size:
            self._compact_episode()

    def _compact_episode(self) -> dict | None:
        """Compact buffered events into an episode with extracted memories."""
        if not self._episode_events or not self._session_id:
            return None

        # Ingest all events via add_smart_messages
        messages = []
        for ev in self._episode_events:
            role = "user" if ev["event_type"] == "observation" else "assistant"
            messages.append((role, ev["content"]))

        if not messages:
            return None

        try:
            result = self.memory.add_smart_messages(
                messages, user_id=self.user_id, run_id=self._session_id
            )
            self._episode_events = []
            return result[0] if result else None
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("compact failed, falling back to simple storage: %s", e)
            combined = "\n".join(
                f"[{ev['event_type']}] {ev['content']}"
                for ev in self._episode_events
            )
            self.memory.add(combined, user_id=self.user_id)
            self._episode_events = []
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs_to_text(obs: Any) -> str:
    """Convert a robot observation to a text description.
    Skips large tensors (e.g. images) that produce no useful search signal."""
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        parts = []
        for k, v in obs.items():
            if isinstance(v, (int, float, str, bool)):
                parts.append(f"{k}={v}")
            elif hasattr(v, "shape") and hasattr(v, "numel"):
                if v.numel() <= 64:
                    parts.append(f"{k}={v.tolist()}")
                # Skip large tensors (images, point clouds)
        return ", ".join(parts) if parts else ""
    return str(obs)


def _action_to_text(action: Any) -> str:
    """Convert a robot action to a text description."""
    if isinstance(action, str):
        return action
    if hasattr(action, "tolist"):
        return f"action={action.tolist()}"
    return str(action)
