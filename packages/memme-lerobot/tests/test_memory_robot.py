"""Tests for MemoryEnhancedRobot and EpisodeLogger.

Uses mock embedder (no API key needed). Run with:
    cd crates/memme-python && maturin develop --release
    cd ../.. && python -m pytest packages/memme-lerobot/tests/ -v
"""

import pytest

from memme_lerobot import MemoryEnhancedRobot, EpisodeLogger


class MockRobot:
    """Minimal mock of a LeRobot Robot."""

    def __init__(self):
        self.connected = False
        self.actions_sent = []
        self._obs_counter = 0

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        self._obs_counter += 1
        return {
            "joint_positions": [0.1 * self._obs_counter, 0.2, 0.3],
            "gripper_open": self._obs_counter % 2 == 0,
        }

    def send_action(self, action):
        self.actions_sent.append(action)


# ---------------------------------------------------------------------------
# MemoryEnhancedRobot tests
# ---------------------------------------------------------------------------


class TestMemoryEnhancedRobot:
    def make_robot(self, **kwargs):
        return MemoryEnhancedRobot(
            robot=MockRobot(),
            user_id="test-robot",
            db_path=":memory:",
            embedder="mock",
            **kwargs,
        )

    def test_connect_disconnect(self):
        r = self.make_robot()
        r.connect()
        assert r.robot.connected
        r.disconnect()
        assert not r.robot.connected

    def test_store_and_recall(self):
        r = self.make_robot()
        r.connect()

        r.store("The red cup is on the kitchen table")
        r.store("The blue plate is in the cabinet")

        results = r.recall("cup")
        assert len(results) > 0
        assert any("red cup" in m.get("content", "") for m in results)

        r.disconnect()

    def test_recall_context(self):
        r = self.make_robot()
        r.connect()

        r.store("Battery level is 85%")
        r.store("Obstacle at coordinates (2, 3)")

        ctx = r.recall_context("battery")
        assert isinstance(ctx, str)
        assert "[Memory 1]" in ctx

        r.disconnect()

    def test_store_experience(self):
        r = self.make_robot()
        r.connect()

        result = r.store_experience(
            observation="Red cup at (0.3, 0.1)",
            action="Move gripper to (0.3, 0.1, 0.1)",
            result="Gripper aligned",
        )
        assert "id" in result

        memories = r.list_memories()
        assert len(memories) == 1
        assert "Red cup" in memories[0]["content"]

        r.disconnect()

    def test_auto_store_observation_and_action(self):
        r = self.make_robot(auto_store=True)
        r.connect()

        obs = r.get_observation()
        assert obs is not None
        r.send_action([0.1, 0.2, 0.3])

        # Events should be buffered
        assert len(r._episode_events) == 2

        r.disconnect()

    def test_manual_episode(self):
        r = self.make_robot(auto_store=False)
        r.connect()

        sid = r.start_episode()
        assert sid is not None

        # Manually store some experience
        r.store("Task started: pick red cup")
        r.store("Task completed: cup placed in box")

        r.end_episode()
        r.disconnect()

    def test_stats(self):
        r = self.make_robot()
        r.connect()

        r.store("Memory 1")
        r.store("Memory 2")

        stats = r.stats()
        assert stats["total_memories"] == 2

        r.disconnect()

    def test_no_robot_mode(self):
        """MemoryEnhancedRobot without a physical robot — memory-only mode."""
        r = MemoryEnhancedRobot(
            robot=None,
            user_id="memory-only",
            db_path=":memory:",
            embedder="mock",
        )
        r.connect()

        r.store("standalone memory")
        results = r.recall("memory")
        assert len(results) > 0

        r.disconnect()

    def test_no_robot_get_observation_raises(self):
        r = MemoryEnhancedRobot(
            robot=None,
            user_id="test",
            db_path=":memory:",
            embedder="mock",
        )
        with pytest.raises(RuntimeError, match="No robot"):
            r.get_observation()


# ---------------------------------------------------------------------------
# EpisodeLogger tests
# ---------------------------------------------------------------------------


class TestEpisodeLogger:
    def make_logger(self):
        return EpisodeLogger(
            user_id="test-logger",
            db_path=":memory:",
            embedder="mock",
        )

    def test_basic_episode(self):
        logger = self.make_logger()

        sid = logger.begin("pick and place task")
        assert sid is not None

        logger.observe("Red cup at position (0.3, 0.1)")
        logger.act("Move gripper to target")
        logger.observe("Gripper aligned")
        logger.act("Close gripper")
        logger.result("success", "Cup picked up")

        # End compacts (fallback to simple storage without LLM)
        logger.end()

        # Should have stored something
        memories = logger.list_memories()
        assert len(memories) >= 1

    def test_recall_after_episodes(self):
        logger = self.make_logger()

        # Episode 1
        logger.begin("grasp red cup")
        logger.observe("Red cup on table")
        logger.act("Picked up red cup")
        logger.result("success")
        logger.end()

        # Episode 2
        logger.begin("grasp blue box")
        logger.observe("Blue box on shelf")
        logger.act("Picked up blue box")
        logger.result("success")
        logger.end()

        results = logger.recall("cup")
        assert len(results) > 0

    def test_empty_episode(self):
        logger = self.make_logger()
        logger.begin("empty task")
        result = logger.end()
        assert result is None

    def test_note(self):
        logger = self.make_logger()
        logger.begin("task with notes")
        logger.note("Important: handle fragile objects carefully")
        logger.end()

        memories = logger.list_memories()
        assert len(memories) >= 1

    def test_direct_store(self):
        logger = self.make_logger()
        result = logger.store("Direct memory entry")
        assert "id" in result

    def test_recall_context(self):
        logger = self.make_logger()
        logger.store("Robot arm calibrated at position zero")

        ctx = logger.recall_context("calibration")
        assert "[Memory 1]" in ctx

    def test_stats(self):
        logger = self.make_logger()
        logger.store("Memory A")
        logger.store("Memory B")

        stats = logger.stats()
        assert stats["total_memories"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
