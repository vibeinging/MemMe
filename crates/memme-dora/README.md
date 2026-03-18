# memme-dora

> Give your dora-rs robot long-term memory.

A [dora-rs](https://dora-rs.ai/) custom node powered by [MemMe](https://github.com/vibeinging/MemMe) — the edge-first AI memory engine.

## Features

- **Store & Search** — semantic memory with vector + BM25 hybrid search
- **Session/Episode** — structured event ingestion and compaction
- **Knowledge Graph** — automatic entity-relation extraction
- **Forgetting Curve** — FSRS-based memory decay for natural recall
- **Edge-first** — DuckDB single-file, no external services required

## Quick Start

```bash
# Build
cargo build -p memme-dora --release

# Set your OpenAI API key (for embeddings)
export OPENAI_API_KEY=sk-xxx

# Run with dora
dora start examples/dataflow.yml
```

## Inputs

| Input ID | Format | Description |
|----------|--------|-------------|
| `store` | JSON `{"content": "...", "user_id": "..."}` | Store a memory |
| `search` | JSON `{"query": "...", "user_id": "...", "top_k": 5}` | Search memories |
| `session_event` | JSON `{"content": "...", "user_id": "...", "event_type": "user_message"}` | Ingest session event |
| `compact` | JSON `{"session_id": "..."}` | Compact session into episode + memories |

## Outputs

| Output ID | Format | Description |
|-----------|--------|-------------|
| `stored` | JSON | Stored memory result |
| `results` | JSON array | Search results with scores |
| `event_ack` | JSON | Ingested event acknowledgment |
| `compacted` | JSON | Compact result (episode + extracted memories) |
| `error` | JSON `{"input": "...", "error": "..."}` | Error details |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for embeddings |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | Custom API endpoint |
| `MEMME_DB_PATH` | No | `robot_memory.duckdb` | DuckDB file path |
| `MEMME_COLLECTION` | No | `default` | Collection name |
| `MEMME_EMBEDDING_DIMS` | No | `1536` | Embedding dimensions |
| `MEMME_LLM_MODEL` | No | `gpt-4.1-nano` | LLM model for fact extraction |

## Dataflow Example

```yaml
nodes:
  - id: memme
    custom:
      build: cargo build -p memme-dora --release
      source: target/release/memme-dora
      inputs:
        store: speech/text
        search: llm/memory_query
      outputs:
        - stored
        - results
        - error
      env:
        MEMME_DB_PATH: ./robot_memory.duckdb
        OPENAI_API_KEY: env:OPENAI_API_KEY
```

## License

Apache-2.0
