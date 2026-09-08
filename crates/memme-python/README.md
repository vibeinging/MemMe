# MemMe Python SDK

Local long-term memory for AI pets and companion devices. MemMe stores owner
memory, per-pet relationship memory, and fresh events in one SQLite file.

[![PyPI](https://img.shields.io/pypi/v/memme.svg)](https://pypi.org/project/memme/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/vibeinging/MemMe/blob/main/LICENSE)

## Install

> **Status**: PyPI currently serves `memme 0.1.1`, which is the legacy DuckDB
> build. The current SQLite + VexDB-Lite version is `0.1.2`; its wheels are
> built and verified but not yet uploaded. Until then, build from source:

```bash
git clone https://github.com/vibeinging/MemMe.git
cd MemMe
pip install maturin
maturin develop --release
```

The Python wheels support CPython 3.8+ on macOS and Linux, for x64 and arm64.
MemMe loads a trusted VexDB-Lite v0.0.17 SQLite extension at runtime. Set its
absolute path before opening a store:

```bash
export MEMME_VEXDB_LITE_EXTENSION=/absolute/path/to/vexdb_lite.so
```

The repository helper downloads the pinned library for your architecture and
verifies it with SHA-256:

```bash
export MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)"
```

Only load a library you trust because SQLite extensions run as native code in
the Python process.

## AI-pet memory

```python
from memme import MemoryStore

store = MemoryStore("momo-memory.db", embedder="mock")

# Owner-global memory can be found while any of this owner's pets is active.
store.add(
    "The owner has a severe peanut allergy.",
    user_id="owner-001",
)

# Relationship memory belongs only to Momo and this owner.
store.add(
    "Momo and the owner first met under the ginkgo tree.",
    user_id="owner-001",
    agent_id="momo",
)

memories = store.search(
    "What should I remember for Momo's birthday snack?",
    user_id="owner-001",
    agent_id="momo",
    limit=5,
)
```

Use a real embedding backend in production:

```python
store = MemoryStore(
    "momo-memory.db",
    embedder="openai",
    api_key="...",
    base_url="https://api.openai.com/v1/embeddings",
    embed_model="text-embedding-3-small",
    dims=1536,
)
```

`embedder="mock"` is deterministic and intended only for tests.

## Storage and retrieval

- SQLite single-file authoritative storage
- VexDB-Lite vector index
- SQLite FTS full-text search
- Owner and pet relationship isolation
- Expired and superseded fact filtering
- History, immutable safety memory, backup, and export/import
- Optional LLM fact extraction and consolidation

## PetMemBench

The `0.1.2` release passed all 11 required scenarios, all 3 extended scenarios,
and reached 100% Recall@10 on the 2,000-memory product benchmark. The benchmark
uses deterministic local test embeddings so it measures the storage and
retrieval path, not a remote embedding service.

See the [repository](https://github.com/vibeinging/MemMe) for the complete
benchmark, architecture, Rust API, and Node.js SDK.

## License

Apache-2.0.
