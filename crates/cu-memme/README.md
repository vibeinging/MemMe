# cu-memme

> Give your Copper-rs robot long-term memory.

A [Copper-rs](https://github.com/copper-project/copper-rs) task powered by [MemMe](https://github.com/vibeinging/MemMe) — the edge-first AI memory engine.

**Status: Experimental** — Copper-rs (cu29) API is evolving rapidly. This crate provides the architecture and message types; the CuTask trait implementation needs adjustment per cu29 version.

## Features

- **Store & Search** — semantic memory with vector + BM25 hybrid search
- **Session/Episode** — structured event ingestion and compaction
- **ONNX embeddings** — local inference, no network required
- **DuckDB single-file** — edge-friendly, survives power loss
- **Background task** — designed for copper's best-effort path

## RON Configuration

```ron
(
    tasks: [
        (id: "perception", type: "my_app::Perception"),
        (id: "memme", type: "cu_memme::MemMeTask", background: true, config: {
            "db_path": "robot_memory.duckdb",
            "embedding_dims": "384",
        }),
        (id: "planner", type: "my_app::Planner"),
    ],
    cnx: [
        (src: "perception", dst: "memme", msg: "cu_memme::MemMeRequest"),
        (src: "memme", dst: "planner", msg: "cu_memme::MemMeResponse"),
    ],
)
```

## Message Types

### MemMeRequest

```rust
enum MemMeRequest {
    Store { content: String, user_id: String, metadata: Option<String> },
    Search { query: String, user_id: String, top_k: u32 },
    IngestEvent { content: String, user_id: String, session_id: Option<String>, event_type: Option<String> },
    Compact { session_id: String },
    Noop,
}
```

### MemMeResponse

```rust
enum MemMeResponse {
    Stored { id: String, content: String },
    SearchResults { results: Vec<MemoryHit> },
    EventAck { event_id: String },
    Compacted { episode_id: String, memory_count: u32 },
    Empty,
    Error { message: String },
}
```

## License

Apache-2.0
