"""
MemMe Benchmark Base — 统一的 API 配置管理

4 组独立配置，互不干扰：
1. Embedding   — MemMe Rust engine 的向量生成
2. Engine LLM  — MemMe Rust engine 的 compact/meditate
3. Answer LLM  — Benchmark Python 脚本的回答生成
4. Judge LLM   — Benchmark Python 脚本的评判

用法:
    from base_benchmark import add_api_args, build_config, create_store, llm_chat

    # 在 argparse 中:
    add_api_args(parser)

    # 解析后:
    config = build_config(args)
    store = create_store(config, db_path)
    answer = llm_chat(config, "answer", messages)
    judge = llm_chat(config, "judge", messages)
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# ── Load .env ──

def _load_env():
    """Load .env from project root."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()


# ── Config ──

@dataclass
class BenchmarkAPIConfig:
    """4 组独立 API 配置."""

    # Embedding (MemMe Rust engine)
    embed_api_key: str = ""
    embed_base_url: str = ""
    embed_model: str = "text-embedding-3-small"
    embed_dims: int = 1536

    # Engine LLM (MemMe Rust compact/meditate)
    engine_provider: str = "openai"  # openai / ollama / anthropic
    engine_api_key: str = ""
    engine_base_url: str = ""
    engine_model: str = "gpt-4o-mini"

    # Answer LLM (Python benchmark — generate answers)
    answer_api_key: str = ""
    answer_base_url: str = ""
    answer_model: str = "gpt-4o-mini"

    # Judge LLM (Python benchmark — evaluate answers)
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_model: str = "gpt-4o-mini"

    # MemMe features
    enable_lattice: bool = False
    enable_splade: bool = False
    enable_forgetting_curve: bool = False
    top_k: int = 30

    # Rerank
    rerank_api_key: str = ""
    rerank_base_url: str = ""
    rerank_model: str = ""

    # Benchmark
    workers: int = 4
    reuse_cache: bool = False


def add_api_args(parser: argparse.ArgumentParser):
    """Add all 4 API groups to argparse."""

    g = parser.add_argument_group("Embedding (MemMe vector generation)")
    g.add_argument("--embed-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    g.add_argument("--embed-base-url", default=os.environ.get("OPENAI_BASE_URL", "") + "/embeddings")
    g.add_argument("--embed-model", default="text-embedding-3-small")
    g.add_argument("--embed-dims", type=int, default=1536)

    g = parser.add_argument_group("Engine LLM (MemMe compact/meditate)")
    g.add_argument("--engine-provider", default="openai", choices=["openai", "ollama", "anthropic", "gemini"])
    g.add_argument("--engine-api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    g.add_argument("--engine-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    g.add_argument("--engine-model", default="qwen-turbo")

    g = parser.add_argument_group("Answer LLM (benchmark answer generation)")
    g.add_argument("--answer-api-key", default=os.environ.get("GETGOAPI_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    g.add_argument("--answer-base-url", default=os.environ.get("GETGOAPI_BASE_URL", "https://api.getgoapi.com/v1") + "/chat/completions")
    g.add_argument("--answer-model", default="gpt-4o-mini")

    g = parser.add_argument_group("Judge LLM (benchmark evaluation)")
    g.add_argument("--judge-api-key", default="")  # empty = same as answer
    g.add_argument("--judge-base-url", default="")  # empty = same as answer
    g.add_argument("--judge-model", default="gpt-4o-mini")

    g = parser.add_argument_group("MemMe features")
    g.add_argument("--enable-lattice", action="store_true", default=False)
    g.add_argument("--enable-splade", action="store_true", default=False)
    g.add_argument("--enable-forgetting-curve", action="store_true", default=False)
    g.add_argument("--top-k", type=int, default=30)

    g = parser.add_argument_group("Rerank")
    g.add_argument("--rerank-api-key", default=os.environ.get("RERANK_API_KEY", ""))
    g.add_argument("--rerank-base-url", default=os.environ.get("RERANK_BASE_URL", ""))
    g.add_argument("--rerank-model", default=os.environ.get("RERANK_MODEL", ""))

    g = parser.add_argument_group("Benchmark")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--reuse-cache", action="store_true")


def build_config(args) -> BenchmarkAPIConfig:
    """Build config from parsed args with defaults filled in."""
    return BenchmarkAPIConfig(
        embed_api_key=args.embed_api_key,
        embed_base_url=args.embed_base_url,
        embed_model=args.embed_model,
        embed_dims=args.embed_dims,
        engine_provider=args.engine_provider,
        engine_api_key=args.engine_api_key,
        engine_base_url=args.engine_base_url,
        engine_model=args.engine_model,
        answer_api_key=args.answer_api_key,
        answer_base_url=args.answer_base_url,
        answer_model=args.answer_model,
        judge_api_key=args.judge_api_key or args.answer_api_key,
        judge_base_url=args.judge_base_url or args.answer_base_url,
        judge_model=args.judge_model,
        enable_lattice=args.enable_lattice,
        enable_splade=args.enable_splade,
        enable_forgetting_curve=getattr(args, 'enable_forgetting_curve', False),
        top_k=args.top_k,
        rerank_api_key=args.rerank_api_key,
        rerank_base_url=args.rerank_base_url,
        rerank_model=args.rerank_model,
        workers=args.workers,
        reuse_cache=args.reuse_cache,
    )


def print_config(config: BenchmarkAPIConfig):
    """Print config summary for verification."""
    print(f"  Embedding:  {config.embed_model} ({config.embed_dims}d) via {_mask_url(config.embed_base_url)}")
    print(f"  Engine LLM: {config.engine_model} ({config.engine_provider}) via {_mask_url(config.engine_base_url)}")
    print(f"  Answer LLM: {config.answer_model} via {_mask_url(config.answer_base_url)}")
    print(f"  Judge LLM:  {config.judge_model} via {_mask_url(config.judge_base_url)}")
    print(f"  Lattice: {config.enable_lattice} | SPLADE: {config.enable_splade} | Top-K: {config.top_k}")
    if config.rerank_model:
        print(f"  Rerank: {config.rerank_model} via {_mask_url(config.rerank_base_url)}")


def _mask_url(url: str) -> str:
    """Shorten URL for display."""
    if "getgoapi" in url: return "GetGoAPI"
    if "openai.com" in url: return "OpenAI"
    if "dashscope" in url: return "DashScope"
    if "localhost" in url or "127.0.0.1" in url: return "Ollama(local)"
    return url[:40]


# ── MemoryStore creation ──

def create_store(config: BenchmarkAPIConfig, db_path: str):
    """Create a MemMe MemoryStore with correct API separation."""
    import memme

    kwargs = dict(
        db_path=db_path,
        embedder="openai",
        api_key=config.embed_api_key,
        base_url=config.embed_base_url,
        embed_model=config.embed_model,
        dims=config.embed_dims,
        enable_forgetting_curve=config.enable_forgetting_curve,
        enable_lattice=config.enable_lattice,
        enable_splade=config.enable_splade,
    )

    # Engine LLM (only if key provided)
    if config.engine_api_key:
        kwargs["llm_provider"] = config.engine_provider
        kwargs["llm_api_key"] = config.engine_api_key
        kwargs["llm_model"] = config.engine_model
        if config.engine_base_url:
            kwargs["llm_base_url"] = config.engine_base_url

    # Rerank
    if config.rerank_api_key:
        kwargs["rerank_api_key"] = config.rerank_api_key
        kwargs["rerank_base_url"] = config.rerank_base_url
        kwargs["rerank_model"] = config.rerank_model

    return memme.MemoryStore(**kwargs)


# ── LLM chat ──

def llm_chat(
    config: BenchmarkAPIConfig,
    role: str,  # "answer" or "judge"
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Call LLM for answer or judge, using the correct API config."""
    if role == "judge":
        api_key = config.judge_api_key
        base_url = config.judge_base_url
        model = config.judge_model
    else:
        api_key = config.answer_api_key
        base_url = config.answer_base_url
        model = config.answer_model

    for attempt in range(3):
        try:
            r = requests.post(
                base_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=60,
            )
            data = r.json()
            if "error" in data:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return ""
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return ""
    return ""


# ── Validation ──

def validate_config(config: BenchmarkAPIConfig) -> bool:
    """Quick validation: test each API endpoint."""
    ok = True

    # Test embedding
    try:
        r = requests.post(
            config.embed_base_url,
            headers={"Authorization": f"Bearer {config.embed_api_key}", "Content-Type": "application/json"},
            json={"model": config.embed_model, "input": "test"},
            timeout=10,
        )
        d = r.json()
        if "data" in d:
            print(f"  ✓ Embedding: {config.embed_model} ({len(d['data'][0]['embedding'])}d)")
        else:
            print(f"  ✗ Embedding: {d.get('error', {}).get('message', 'unknown error')[:60]}")
            ok = False
    except Exception as e:
        print(f"  ✗ Embedding: {e}")
        ok = False

    # Test answer LLM
    try:
        resp = llm_chat(config, "answer", [{"role": "user", "content": "say ok"}], max_tokens=5)
        if resp:
            print(f"  ✓ Answer LLM: {config.answer_model}")
        else:
            print(f"  ✗ Answer LLM: empty response")
            ok = False
    except Exception as e:
        print(f"  ✗ Answer LLM: {e}")
        ok = False

    # Engine LLM tested implicitly when MemoryStore.compact() is called

    return ok
