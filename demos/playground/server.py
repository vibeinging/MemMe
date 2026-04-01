#!/usr/bin/env python3
"""
MemMe Playground — local web demo.

Usage:
    pip install memme
    python demos/playground/server.py

Opens a browser with an interactive memory playground.
All data stays on your device in a local .duckdb file.
"""

import http.server
import json
import os
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

try:
    import memme
except ImportError:
    print("memme not installed. Run: pip install memme")
    sys.exit(1)

PORT = int(os.environ.get("MEMME_PORT", 7860))
STATIC_DIR = Path(__file__).parent
DB_PATH = os.environ.get("MEMME_DB", str(STATIC_DIR / "playground.duckdb"))

# Global store — initialized lazily after user configures models.
store = None
llm_config = {}  # Saved LLM config for chat endpoint


def init_store(config: dict):
    """Create or reconfigure the MemoryStore from user-provided config."""
    global store

    # Allow re-configuration by resetting the store.
    store = None

    kwargs = {"db_path": DB_PATH}

    # Embedding
    embedder = config.get("embedder", "onnx")
    kwargs["embedder"] = embedder
    if embedder == "openai":
        kwargs["api_key"] = config.get("embed_api_key", "")
        if config.get("embed_base_url"):
            kwargs["base_url"] = config["embed_base_url"]
        if config.get("embed_model"):
            kwargs["embed_model"] = config["embed_model"]
        if config.get("embed_dims"):
            kwargs["dims"] = int(config["embed_dims"])

    # LLM
    global llm_config
    llm_provider = config.get("llm_provider", "")
    if llm_provider:
        kwargs["llm_provider"] = llm_provider
        if config.get("llm_api_key"):
            kwargs["llm_api_key"] = config["llm_api_key"]
        if config.get("llm_model"):
            kwargs["llm_model"] = config["llm_model"]
        if config.get("llm_base_url"):
            kwargs["llm_base_url"] = config["llm_base_url"]
        llm_config = {
            "provider": llm_provider,
            "api_key": config.get("llm_api_key", ""),
            "model": config.get("llm_model", "gpt-4o-mini"),
            "base_url": config.get("llm_base_url", ""),
        }
    else:
        llm_config = {}

    try:
        store = memme.MemoryStore(**kwargs)
    except RuntimeError as e:
        if "WAL" in str(e):
            # Corrupted WAL file — remove and retry.
            wal = DB_PATH + ".wal"
            if os.path.exists(wal):
                os.remove(wal)
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            store = memme.MemoryStore(**kwargs)
        else:
            raise

    return {"status": "ok", "db_path": DB_PATH, "embedder": embedder}


def chat_with_memory(query: str, user_id: str = "playground") -> dict:
    """Search memories, then ask LLM to answer based on them."""
    if not llm_config:
        return {"error": "LLM not configured. Set an LLM provider in the sidebar."}

    # 1. Search for relevant memories
    results = store.search(query, user_id=user_id, limit=5)
    memories_text = "\n".join(
        f"- {r.get('content', r.get('memory', ''))}" for r in results
    )

    # 2. Build prompt
    system = (
        "You are a helpful assistant with access to a memory database. "
        "The database contains facts about various people and things. "
        "Answer the question based ONLY on the memories provided below. "
        "Treat each memory as a factual record — refer to people by their names, not as 'you'. "
        "If the memories don't contain relevant information, say so honestly. "
        "Be concise and direct.\n\n"
        f"## Memories\n{memories_text if memories_text else '(no relevant memories found)'}"
    )

    # 3. Call LLM (OpenAI-compatible API)
    provider = llm_config["provider"]
    api_key = llm_config["api_key"]
    model = llm_config["model"] or "gpt-4o-mini"

    if provider == "ollama":
        base = llm_config.get("base_url") or "http://localhost:11434"
        url = f"{base}/api/chat"
        resp = requests.post(url, json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "stream": False,
        }, timeout=60)
        resp.raise_for_status()
        answer = resp.json().get("message", {}).get("content", "")
    else:
        # OpenAI / Anthropic / Gemini — all use OpenAI-compatible endpoint
        if provider == "anthropic":
            base = llm_config.get("base_url") or "https://api.anthropic.com"
        elif provider == "gemini":
            base = llm_config.get("base_url") or "https://generativelanguage.googleapis.com/v1beta/openai"
        else:
            base = llm_config.get("base_url") or "https://api.openai.com"

        # Ensure base URL ends with a chat completions path
        chat_url = f"{base.rstrip('/')}/v1/chat/completions"
        if "/v1/v1/" in chat_url:
            chat_url = chat_url.replace("/v1/v1/", "/v1/")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.post(chat_url, headers=headers, json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    return {
        "answer": answer,
        "memories_used": results,
        "memories_count": len(results),
    }


def handle_api(path: str, body: dict) -> dict:
    """Route API calls to the memme store."""
    global store

    if path == "/api/config":
        return init_store(body)

    if store is None:
        return {"error": "Store not configured. Send POST /api/config first."}

    try:
        if path == "/api/add":
            result = store.add(
                body["content"],
                user_id=body.get("user_id", "playground"),
                metadata=body.get("metadata"),
            )
            return {"memory": result}

        elif path == "/api/add_smart":
            result = store.add_smart(
                body["content"],
                user_id=body.get("user_id", "playground"),
            )
            return {"memories": result}

        elif path == "/api/add_messages":
            messages = [(m["role"], m["content"]) for m in body["messages"]]
            result = store.add_smart_messages(
                messages,
                user_id=body.get("user_id", "playground"),
            )
            return {"result": result}

        elif path == "/api/search":
            results = store.search(
                body["query"],
                user_id=body.get("user_id", "playground"),
                limit=body.get("limit", 10),
            )
            return {"results": results}

        elif path == "/api/hybrid_search":
            results = store.hybrid_search(
                body["query"],
                user_id=body.get("user_id", "playground"),
                limit=body.get("limit", 10),
            )
            return {"results": results}

        elif path == "/api/list":
            results = store.list(
                user_id=body.get("user_id", "playground"),
                limit=body.get("limit", 50),
            )
            return {"memories": results}

        elif path == "/api/batch_add":
            items = body.get("items", [])
            count = 0
            for content in items:
                store.add(
                    content,
                    user_id=body.get("user_id", "playground"),
                )
                count += 1
            return {"count": count}

        elif path == "/api/delete":
            store.delete(body["id"])
            return {"status": "deleted"}

        elif path == "/api/add_graph":
            result = store.add_graph(
                body["content"],
                user_id=body.get("user_id", "playground"),
            )
            return {"graph": result}

        elif path == "/api/search_graph":
            result = store.search_graph(
                body["query"],
                user_id=body.get("user_id", "playground"),
            )
            return {"graph": result}

        elif path == "/api/chat":
            return chat_with_memory(
                body["query"],
                user_id=body.get("user_id", "playground"),
            )

        elif path == "/api/stats":
            result = store.user_stats(body.get("user_id", "playground"))
            return {"stats": result}

        elif path == "/api/episodes":
            result = store.list_episodes(
                user_id=body.get("user_id", "playground"),
                limit=body.get("limit", 20),
            )
            return {"episodes": result}

        else:
            return {"error": f"Unknown endpoint: {path}"}

    except Exception as e:
        return {"error": str(e)}


class PlaygroundHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static files + JSON API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}

        result = handle_api(self.path, body)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, format, *args):
        # Quiet logging — only errors
        if args and "404" in str(args[0]):
            super().log_message(format, *args)


def main():
    server = http.server.HTTPServer(("127.0.0.1", PORT), PlaygroundHandler)
    url = f"http://localhost:{PORT}"

    print(f"""
  ╔══════════════════════════════════════════╗
  ║         MemMe Playground                 ║
  ║                                          ║
  ║  {url:<40s}║
  ║  Data: {DB_PATH:<33s}║
  ║                                          ║
  ║  Your memories, truly yours.             ║
  ╚══════════════════════════════════════════╝
""")

    # Open browser after a short delay
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
