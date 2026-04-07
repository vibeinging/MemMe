# memme-lerobot

> Give your LeRobot long-term memory.

A Python package that adds persistent, searchable memory to [LeRobot](https://github.com/huggingface/lerobot) robots, powered by [MemMe](https://github.com/vibeinging/MemMe).

## Features

- **Auto-record**: wraps `Robot` to automatically store observations and actions
- **Auto-recall**: retrieves relevant memories during inference
- **Episode compaction**: compresses raw events into structured memories
- **Knowledge graph**: extracts entities and relationships from experiences
- **Forgetting curve**: naturally decays unimportant memories over time
- **Edge-first**: runs locally with ONNX embeddings, no cloud required

## Install

```bash
# Build memme Python bindings first
cd crates/memme-python && maturin develop --release

# Install memme-lerobot
pip install -e packages/memme-lerobot
```

## Quick Start

### Option 1: Wrap your Robot

```python
from lerobot.robots import make_robot
from memme_lerobot import MemoryEnhancedRobot

robot = MemoryEnhancedRobot(
    robot=make_robot("so100"),
    user_id="robot-001",
    db_path="robot_memory.duckdb",
)

robot.connect()

# Normal LeRobot loop — memory happens automatically
for step in range(100):
    obs = robot.get_observation()    # auto-stores observation
    action = policy.select_action(obs)
    robot.send_action(action)        # auto-stores action

robot.disconnect()  # auto-compacts episode into memories

# Later: recall relevant experiences
memories = robot.recall("how to pick up red objects")
context = robot.recall_context("grasping from table")
```

### Option 2: Standalone Episode Logger

```python
from memme_lerobot import EpisodeLogger

logger = EpisodeLogger(user_id="robot-001")

logger.begin("pick-and-place red cup")
logger.observe("Red cup detected at position (0.3, 0.1, 0.05)")
logger.act("Moved gripper to (0.3, 0.1, 0.1)")
logger.observe("Gripper aligned with cup")
logger.act("Closed gripper")
logger.result("success", "Picked up red cup")
episode = logger.end()  # compacts into memories

# Recall for future tasks
memories = logger.recall("how to pick up a cup")
```

### Option 3: Direct Memory API

```python
import memme

store = memme.MemoryStore(db_path="robot_memory.duckdb")

# Store
store.add("Robot successfully grasped red cup from table", user_id="robot-001")

# Search
results = store.search("cup grasping", user_id="robot-001")

# Knowledge graph
store.add_graph("Robot picked up red cup from kitchen table", user_id="robot-001")
graph = store.search_graph("red cup", user_id="robot-001")
```

## API Reference

### MemoryEnhancedRobot

| Method | Description |
|--------|-------------|
| `connect()` | Connect to robot, start memory session |
| `disconnect()` | Disconnect, auto-compact episode |
| `get_observation()` | Get observation (auto-stores, optional auto-recall) |
| `send_action(action)` | Send action (auto-stores) |
| `recall(query, limit=5)` | Search memories |
| `recall_context(query, limit=3)` | Search and format as text context |
| `store(content)` | Explicitly store a memory |
| `store_experience(obs, action, result)` | Store structured experience |
| `start_episode()` | Manually start new episode |
| `end_episode()` | Manually end and compact episode |
| `add_knowledge(text)` | Extract entities into knowledge graph |
| `search_knowledge(query)` | Search knowledge graph |
| `stats()` | Get memory statistics |

### EpisodeLogger

| Method | Description |
|--------|-------------|
| `begin(task)` | Start recording episode |
| `observe(desc)` | Log observation |
| `act(desc)` | Log action |
| `result(outcome, details)` | Log result |
| `note(text)` | Log free-form note |
| `end()` | End episode, compact into memories |
| `recall(query)` | Search memories |

## License

Apache-2.0
