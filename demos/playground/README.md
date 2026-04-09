# MemMe Playground

Interactive local web demo. All data stays on your device.

## Quick Start

```bash
pip install memme
python demos/playground/server.py
```

Browser opens automatically at `http://localhost:7860`.

## Configuration

1. **Embedding** — Choose ONNX (local, no API key) or OpenAI
2. **LLM** — Optional. Required for Smart Mode, Graph extraction, and Conversation compaction
3. Click **Connect**

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMME_PORT` | `7860` | Server port |
| `MEMME_DB` | `playground.db` | Database file path |

## Features

| Tab | Description | Requires LLM |
|-----|-------------|:---:|
| **Memories** | Add and browse stored memories | No |
| **Search** | Vector or hybrid (vector + keyword) search | No |
| **Conversation** | Build a dialogue, compact into episode | Yes |
| **Graph** | Extract entities and relationships from text | Yes |
| **Stats** | View memory analytics | No |
