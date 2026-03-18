#!/usr/bin/env python3
"""Measure token consumption of MemMe search results used as LLM context.

Compares with mem0's reported numbers:
  - mem0:  1,764 tokens (average)
  - Zep:   3,911 tokens
  - mem0g: 3,616 tokens

Usage:
    cd benchmarks/locomo
    python ../token_consumption_bench.py
"""

import json
import os
import sys
import time
import statistics
from pathlib import Path

import tiktoken
import memme


# ── Config ──

JSONL_PATH = "locomo/results_rerank_full/run_20260326_162109.jsonl"
DATA_PATH = "locomo/locomo10.json"
CACHE_DIR = "locomo/cache"
TOP_K = 30

COMMON_STORE_KWARGS = dict(
    embedder="openai",
    api_key="sk-your-dashscope-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    embed_model="text-embedding-v3",
    dims=1024,
    enable_forgetting_curve=False,
)

RERANK_KWARGS = dict(
    rerank_api_key="sk-your-dashscope-key",
    rerank_base_url="https://dashscope.aliyuncs.com",
    rerank_model="gte-rerank-v2",
)


# ── Answer prompt template (same as benchmark) ──

ANSWER_PROMPT = """You are answering a question based on the memories below.

RULES:
1. Use information from the memories as your primary source. You may make reasonable inferences from the memories.
2. For factual questions (what/where/who): Answer with specific details from the memories. Prefer exact names, places, dates.
3. For list questions ("what activities/books/events/items"): Scan ALL memories carefully. List EVERY matching item found, separated by commas. Do not omit any.
4. For "how many" questions: If an exact number is stated, use it. Otherwise count the distinct items explicitly mentioned.
5. For time questions ("when"): Look for specific dates, relative time references ("last Friday", "2 weeks ago"), or temporal context.
6. For inference questions ("would...?", "likely...?", "might...?"): Reason based on the person's known traits, values, and behaviors from the memories. Give a clear answer (e.g. "Likely yes/no") with brief reasoning.
7. For open-ended questions about preferences, opinions, or characteristics: Synthesize from all relevant memories to form a complete picture.
8. Answer concisely — no explanation needed for factual answers. Brief reasoning is OK for inference questions.
9. If memories contain relevant information, ALWAYS attempt an answer. Only say "Unknown" if the memories have absolutely no relevant information.

Speaker 1 ({speaker_1}) memories:
{speaker_1_memories}

Speaker 2 ({speaker_2}) memories:
{speaker_2_memories}

Question: {question}

Answer:"""


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def load_conversations(data_path: str) -> dict:
    """Load locomo data and return {sample_id: {speaker_a, speaker_b, uid_a, uid_b}}."""
    with open(data_path) as f:
        data = json.load(f)

    convs = {}
    for conv in data:
        sample_id = conv["sample_id"]
        conversation = conv["conversation"]
        session_nums = sorted(set(
            int(k.replace("session_", ""))
            for k in conversation.keys()
            if k.startswith("session_") and k.replace("session_", "").isdigit()
        ))
        if not session_nums:
            continue
        first_session = conversation.get(f"session_{session_nums[0]}", [])
        speakers = list(dict.fromkeys(t["speaker"] for t in first_session))
        if len(speakers) < 2:
            speakers = speakers + ["unknown"]
        convs[sample_id] = {
            "speaker_a": speakers[0],
            "speaker_b": speakers[1],
            "uid_a": f"{sample_id}_{speakers[0]}",
            "uid_b": f"{sample_id}_{speakers[1]}",
        }
    return convs


def measure_tokens(use_rerank: bool, label: str):
    """Run token measurement across all questions."""
    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    # Load JSONL results to get the exact questions that were benchmarked
    questions_by_conv = {}
    with open(JSONL_PATH) as f:
        for line in f:
            d = json.loads(line)
            sid = d["sample_id"]
            q = d["question"]
            if sid not in questions_by_conv:
                questions_by_conv[sid] = []
            questions_by_conv[sid].append(q)

    convs = load_conversations(DATA_PATH)

    # Per-question token counts (for the memory context portion only)
    memory_tokens = []
    # Full prompt tokens
    prompt_tokens = []

    total_questions = sum(len(qs) for qs in questions_by_conv.values())
    done = 0

    for sample_id, questions in sorted(questions_by_conv.items()):
        db_path = os.path.join(CACHE_DIR, f"{sample_id}.duckdb")
        if not os.path.exists(db_path):
            print(f"  [SKIP] {db_path} not found")
            done += len(questions)
            continue

        info = convs.get(sample_id)
        if not info:
            print(f"  [SKIP] {sample_id} not in locomo data")
            done += len(questions)
            continue

        store_kwargs = {**COMMON_STORE_KWARGS, "db_path": db_path}
        if use_rerank:
            store_kwargs.update(RERANK_KWARGS)

        store = memme.MemoryStore(**store_kwargs)

        for q in questions:
            try:
                results_a = store.search(q, user_id=info["uid_a"], limit=TOP_K)
            except Exception:
                results_a = []
            try:
                results_b = store.search(q, user_id=info["uid_b"], limit=TOP_K)
            except Exception:
                results_b = []

            memories_a = json.dumps([r["content"] for r in results_a], ensure_ascii=False)
            memories_b = json.dumps([r["content"] for r in results_b], ensure_ascii=False)

            # Count tokens in memory context only
            mem_text = memories_a + memories_b
            mem_tok = count_tokens(mem_text, enc)
            memory_tokens.append(mem_tok)

            # Count tokens in full prompt
            full_prompt = ANSWER_PROMPT.format(
                speaker_1=info["speaker_a"],
                speaker_2=info["speaker_b"],
                speaker_1_memories=memories_a,
                speaker_2_memories=memories_b,
                question=q,
            )
            prompt_tok = count_tokens(full_prompt, enc)
            prompt_tokens.append(prompt_tok)

            done += 1
            if done % 100 == 0 or done == total_questions:
                print(f"  [{done}/{total_questions}] processed", flush=True)

    # Report
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Questions measured: {len(memory_tokens)}")
    print()

    if memory_tokens:
        sorted_mem = sorted(memory_tokens)
        n = len(sorted_mem)
        print("  Memory context tokens (just the retrieved memories):")
        print(f"    Mean:   {statistics.mean(memory_tokens):,.0f}")
        print(f"    Median: {statistics.median(memory_tokens):,.0f}")
        print(f"    P5:     {sorted_mem[int(n * 0.05)]:,}")
        print(f"    P25:    {sorted_mem[int(n * 0.25)]:,}")
        print(f"    P50:    {sorted_mem[int(n * 0.50)]:,}")
        print(f"    P75:    {sorted_mem[int(n * 0.75)]:,}")
        print(f"    P95:    {sorted_mem[int(n * 0.95)]:,}")
        print(f"    Min:    {min(memory_tokens):,}")
        print(f"    Max:    {max(memory_tokens):,}")
        print()
        print("  Full prompt tokens (memories + template + question):")
        print(f"    Mean:   {statistics.mean(prompt_tokens):,.0f}")
        print(f"    Median: {statistics.median(prompt_tokens):,.0f}")
        print(f"    P95:    {sorted(prompt_tokens)[int(n * 0.95)]:,}")

    print()
    return memory_tokens


def main():
    os.chdir(Path(__file__).resolve().parent)

    print("=" * 60)
    print("  MemMe Token Consumption Benchmark")
    print("  Comparing with mem0 (1,764), Zep (3,911), mem0g (3,616)")
    print("=" * 60)

    # Check tiktoken
    try:
        import tiktoken
    except ImportError:
        print("Installing tiktoken...")
        os.system(f"{sys.executable} -m pip install tiktoken")
        import tiktoken

    # Measure WITH rerank
    print("\n[1/2] Measuring with rerank (gte-rerank-v2)...")
    rerank_tokens = measure_tokens(use_rerank=True, label="WITH RERANK (gte-rerank-v2)")

    # Measure WITHOUT rerank
    print("\n[2/2] Measuring without rerank...")
    no_rerank_tokens = measure_tokens(use_rerank=False, label="WITHOUT RERANK (RRF only)")

    # Comparison
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)

    systems = [
        ("mem0", 1764),
        ("Zep", 3911),
        ("mem0g (GraphRAG)", 3616),
    ]

    if rerank_tokens:
        rerank_mean = statistics.mean(rerank_tokens)
        systems.append(("MemMe (rerank)", round(rerank_mean)))
    if no_rerank_tokens:
        no_rerank_mean = statistics.mean(no_rerank_tokens)
        systems.append(("MemMe (no rerank)", round(no_rerank_mean)))

    print(f"\n  {'System':<25} {'Avg Memory Tokens':>18}")
    print(f"  {'-' * 25} {'-' * 18}")
    for name, tokens in sorted(systems, key=lambda x: x[1]):
        print(f"  {name:<25} {tokens:>18,}")

    print()
    if rerank_tokens:
        rerank_mean = statistics.mean(rerank_tokens)
        ratio = rerank_mean / 1764
        print(f"  MemMe (rerank) vs mem0: {ratio:.2f}x ({rerank_mean:,.0f} vs 1,764)")
    if no_rerank_tokens:
        no_rerank_mean = statistics.mean(no_rerank_tokens)
        ratio = no_rerank_mean / 1764
        print(f"  MemMe (no rerank) vs mem0: {ratio:.2f}x ({no_rerank_mean:,.0f} vs 1,764)")

    print()


if __name__ == "__main__":
    main()
