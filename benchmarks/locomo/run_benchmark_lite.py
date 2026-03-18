#!/usr/bin/env python3
"""MemMe LOCOMO Benchmark — No-LLM / Local-LLM Ingestion

Tests retrieval quality with different ingestion strategies:
- raw:    add() raw conversation chunks (no LLM at ingestion)
- ollama: add_smart() with local Ollama model (local LLM at ingestion)

In both modes, answer generation + judge use cloud LLM for evaluation consistency.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import aiohttp
import memme


# ── Configuration ──

@dataclass
class BenchConfig:
    api_key: str  # DashScope key (embedding)
    llm_api_key: str = ""  # LLM key (answer + judge), defaults to api_key
    mode: str = "raw"  # "raw" or "ollama"
    chat_base_url: str = ""
    embed_base_url: str = ""
    chat_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-v3"
    embed_dims: int = 1024
    top_k: int = 30
    data_path: str = "locomo10.json"
    output_dir: str = "results_lite_baseline"
    conversations: Optional[list] = None
    categories: Optional[list] = None
    judge_runs: int = 1
    max_llm_concurrent: int = 5
    # Ollama settings (for ollama mode)
    ollama_model: str = "qwen2.5:3b"
    ollama_host: str = "http://localhost:11434"
    # Chunking settings (for raw mode)
    chunk_turns: int = 3  # turns per chunk
    # RRF weights
    rrf_vector_weight: float = 0.5
    rrf_fts_weight: float = 0.3
    rrf_entity_weight: float = 0.0  # no entity graph in lite mode
    rrf_k: int = 30
    rrf_temporal_weight: float = 0.15


# ── Async API Client ──

class AsyncAPIClient:
    def __init__(self, config: BenchConfig):
        self.config = config
        self.llm_sem = asyncio.Semaphore(config.max_llm_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None
        self._stats = {"llm_calls": 0, "llm_errors": 0}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            headers={
                "Authorization": f"Bearer {self.config.llm_api_key or self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    @property
    def stats(self):
        return self._stats

    async def chat_completion(self, messages: list, temperature: float = 0.0,
                              max_tokens: int = 512, model: str = None) -> str:
        url = f"{self.config.chat_base_url}/chat/completions"
        payload = {
            "model": model or self.config.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with self.llm_sem:
            for attempt in range(5):
                try:
                    async with self.session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        self._stats["llm_calls"] += 1
                        content = data["choices"][0]["message"]["content"].strip()
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        return content
                except Exception as e:
                    self._stats["llm_errors"] += 1
                    if attempt < 4:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Chat API failed after 5 attempts: {e}")


# ── Scoring ──

def normalize_answer(text) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not ref_tokens: return 1.0 if not pred_tokens else 0.0
    if not pred_tokens: return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)
    if num_common == 0: return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_bleu1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens: return 0.0
    ref_count = defaultdict(int)
    for t in ref_tokens: ref_count[t] += 1
    clipped = 0
    for t in pred_tokens:
        if ref_count[t] > 0:
            clipped += 1
            ref_count[t] -= 1
    return clipped / len(pred_tokens)


# ── Answer + Judge ──

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

CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}

@dataclass
class QuestionResult:
    sample_id: str
    question: str
    reference: str
    prediction: str
    category: int
    category_name: str
    f1: float
    bleu1: float
    judge_scores: list = field(default_factory=list)
    judge_mean: float = 0.0


async def answer_and_judge(config, client, qa, sample_id,
                           speaker_a, speaker_b, uid_a, uid_b,
                           pre_searched) -> QuestionResult:
    question = qa["question"]
    reference = str(qa["answer"])
    category = qa["category"]

    try:
        memories_a = json.dumps(pre_searched.get((uid_a, question), []), ensure_ascii=False)
        memories_b = json.dumps(pre_searched.get((uid_b, question), []), ensure_ascii=False)

        prompt = ANSWER_PROMPT.format(
            speaker_1=speaker_a, speaker_2=speaker_b,
            speaker_1_memories=memories_a, speaker_2_memories=memories_b,
            question=question,
        )
        prediction = await client.chat_completion(
            [{"role": "user", "content": prompt}], max_tokens=200
        )
    except Exception as e:
        print(f"    [WARN] Answer failed: {str(e)[:60]}", flush=True)
        prediction = "I don't know"

    f1 = compute_f1(prediction, reference)
    bleu1 = compute_bleu1(prediction, reference)

    judge_scores = []
    for _ in range(config.judge_runs):
        judge_prompt = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {reference}
Generated answer: {prediction}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""
        try:
            response = await client.chat_completion(
                [{"role": "user", "content": judge_prompt}], temperature=0.0, max_tokens=100,
                model=config.judge_model,
            )
            judge_scores.append(1.0 if "CORRECT" in response.upper() else 0.0)
        except Exception:
            judge_scores.append(0.0)

    judge_mean = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

    return QuestionResult(
        sample_id=sample_id, question=question, reference=reference,
        prediction=prediction, category=category,
        category_name=CATEGORY_NAMES.get(category, "unknown"),
        f1=f1, bleu1=bleu1, judge_scores=judge_scores, judge_mean=judge_mean,
    )


# ── Ingestion ──

def ingest_raw(config: BenchConfig, conv: dict):
    """Ingest using add() — raw text chunks, no LLM extraction."""
    conversation = conv["conversation"]
    sample_id = conv.get("sample_id", "unknown")

    session_nums = sorted(set(
        int(k.replace("session_", ""))
        for k in conversation.keys()
        if k.startswith("session_") and k.replace("session_", "").isdigit()
    ))
    if not session_nums:
        return None

    first_session = conversation.get(f"session_{session_nums[0]}", [])
    speakers = list(dict.fromkeys(t["speaker"] for t in first_session))
    if len(speakers) < 2:
        speakers = speakers + ["unknown"]
    speaker_a, speaker_b = speakers[0], speakers[1]
    uid_a = f"{sample_id}_{speaker_a}"
    uid_b = f"{sample_id}_{speaker_b}"

    cache_dir = f"cache_lite_{config.mode}"
    db_path = f"{cache_dir}/{sample_id}.duckdb"
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(db_path) and getattr(config, 'reuse_cache', False):
        store = memme.MemoryStore(
            db_path=db_path,
            embedder="openai",
            api_key=config.api_key,
            base_url=config.embed_base_url,
            embed_model=config.embed_model,
            dims=config.embed_dims,
            enable_forgetting_curve=False,
            rrf_vector_weight=config.rrf_vector_weight,
            rrf_fts_weight=config.rrf_fts_weight,
            rrf_entity_weight=config.rrf_entity_weight,
            rrf_k=config.rrf_k,
            rrf_temporal_weight=config.rrf_temporal_weight,
        )
        print(f"  Using cached db: {db_path}", flush=True)
        return store, speaker_a, speaker_b, uid_a, uid_b

    # Fresh store — no LLM configured
    store = memme.MemoryStore(
        db_path=db_path,
        embedder="openai",
        api_key=config.api_key,
        base_url=config.embed_base_url,
        embed_model=config.embed_model,
        dims=config.embed_dims,
        enable_forgetting_curve=False,
        rrf_vector_weight=config.rrf_vector_weight,
        rrf_fts_weight=config.rrf_fts_weight,
        rrf_entity_weight=config.rrf_entity_weight,
        rrf_k=config.rrf_k,
        rrf_temporal_weight=config.rrf_temporal_weight,
    )

    # Chunk conversation into small groups of turns
    chunk_size = config.chunk_turns
    total_chunks = 0
    for num in session_nums:
        session_key = f"session_{num}"
        turns = conversation.get(session_key, [])
        if not turns:
            continue
        date_key = f"session_{num}_date_time"
        session_date = conversation.get(date_key, "")
        date_prefix = f"[Date: {session_date}] " if session_date else ""

        for i in range(0, len(turns), chunk_size):
            batch = turns[i:i + chunk_size]
            text = date_prefix + " ".join(f"{t['speaker']}: {t['text']}" for t in batch)

            for uid in [uid_a, uid_b]:
                try:
                    store.add(content=text, user_id=uid)
                except Exception as e:
                    print(f"      [WARN] add() failed: {str(e)[:60]}", flush=True)
            total_chunks += 1

    # Build FTS index
    store.rebuild_fts_index()
    print(f" {total_chunks} chunks", end="", flush=True)

    return store, speaker_a, speaker_b, uid_a, uid_b


def ingest_ollama(config: BenchConfig, conv: dict):
    """Ingest using add_smart() with Ollama local LLM."""
    conversation = conv["conversation"]
    sample_id = conv.get("sample_id", "unknown")

    session_nums = sorted(set(
        int(k.replace("session_", ""))
        for k in conversation.keys()
        if k.startswith("session_") and k.replace("session_", "").isdigit()
    ))
    if not session_nums:
        return None

    first_session = conversation.get(f"session_{session_nums[0]}", [])
    speakers = list(dict.fromkeys(t["speaker"] for t in first_session))
    if len(speakers) < 2:
        speakers = speakers + ["unknown"]
    speaker_a, speaker_b = speakers[0], speakers[1]
    uid_a = f"{sample_id}_{speaker_a}"
    uid_b = f"{sample_id}_{speaker_b}"

    cache_dir = f"cache_lite_{config.mode}_{config.ollama_model.replace(':', '_')}"
    db_path = f"{cache_dir}/{sample_id}.duckdb"
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(db_path) and getattr(config, 'reuse_cache', False):
        store = memme.MemoryStore(
            db_path=db_path,
            embedder="openai",
            api_key=config.api_key,
            base_url=config.embed_base_url,
            embed_model=config.embed_model,
            dims=config.embed_dims,
            enable_forgetting_curve=False,
            rrf_vector_weight=config.rrf_vector_weight,
            rrf_fts_weight=config.rrf_fts_weight,
            rrf_entity_weight=config.rrf_entity_weight,
            rrf_k=config.rrf_k,
            rrf_temporal_weight=config.rrf_temporal_weight,
            # Ollama via OpenAI-compatible API
            llm_api_key="ollama",
            llm_model=config.ollama_model,
            llm_base_url=config.ollama_host,
        )
        print(f"  Using cached db: {db_path}", flush=True)
        return store, speaker_a, speaker_b, uid_a, uid_b

    store = memme.MemoryStore(
        db_path=db_path,
        embedder="openai",
        api_key=config.api_key,
        base_url=config.embed_base_url,
        embed_model=config.embed_model,
        dims=config.embed_dims,
        enable_forgetting_curve=False,
        rrf_vector_weight=config.rrf_vector_weight,
        rrf_fts_weight=config.rrf_fts_weight,
        rrf_entity_weight=config.rrf_entity_weight,
        rrf_k=config.rrf_k,
        rrf_temporal_weight=config.rrf_temporal_weight,
        llm_api_key="ollama",
        llm_model=config.ollama_model,
        llm_base_url=config.ollama_host,
    )

    batch_size = 10
    all_batches = []
    for num in session_nums:
        session_key = f"session_{num}"
        turns = conversation.get(session_key, [])
        if not turns:
            continue
        date_key = f"session_{num}_date_time"
        session_date = conversation.get(date_key, "")
        date_prefix = f"[Date: {session_date}]\n" if session_date else ""
        for i in range(0, len(turns), batch_size):
            batch = turns[i:i + batch_size]
            text = date_prefix + "\n".join(f"{t['speaker']}: {t['text']}" for t in batch)
            all_batches.append(text)

    done = 0
    total = len(all_batches) * 2
    for text in all_batches:
        for uid in [uid_a, uid_b]:
            for attempt in range(3):
                try:
                    store.add_smart(text, user_id=uid)
                    done += 1
                    if done % 10 == 0 or done == total:
                        print(f"      [{done}/{total}]", flush=True)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(3 * (attempt + 1))
                    else:
                        done += 1
                        print(f"      [WARN] add_smart failed: {str(e)[:60]}", flush=True)

    store.rebuild_fts_index()
    return store, speaker_a, speaker_b, uid_a, uid_b


# ── Benchmark Runner ──

async def run_benchmark(config: BenchConfig):
    data_path = Path(config.data_path)
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)

    with open(data_path) as f:
        conversations = json.load(f)

    if config.conversations is not None:
        conversations = [conversations[i] for i in config.conversations if i < len(conversations)]

    os.makedirs(config.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_file = Path(config.output_dir) / f"run_{ts}.jsonl"
    summary_file = Path(config.output_dir) / f"summary_{ts}.json"

    mode_desc = {
        "raw": "add() raw chunks → vector + FTS + temporal (no LLM extraction)",
        "ollama": f"add_smart() Ollama {config.ollama_model} → vector + FTS + temporal",
    }

    print(f"=== MemMe LOCOMO Benchmark — Lite Mode ({config.mode}) ===")
    print(f"Mode: {mode_desc.get(config.mode, config.mode)}")
    print(f"Conversations: {len(conversations)}")
    print(f"Answer Model: {config.chat_model} (cloud, for evaluation only)")
    print(f"Judge Model: {config.judge_model}")
    print(f"Embedding: {config.embed_model} ({config.embed_dims}d)")
    print(f"Top-K: {config.top_k}")
    if config.mode == "raw":
        print(f"Chunk size: {config.chunk_turns} turns/chunk")
    print(f"RRF weights: vec={config.rrf_vector_weight} fts={config.rrf_fts_weight} entity={config.rrf_entity_weight} temporal={config.rrf_temporal_weight}")
    print(f"Output: {results_file}")
    print()

    ingest_fn = ingest_raw if config.mode == "raw" else ingest_ollama
    all_results = []

    async with AsyncAPIClient(config) as client:
        for conv_idx, conv in enumerate(conversations):
            sample_id = conv.get("sample_id", f"conv_{conv_idx}")
            print(f"\n--- Conversation {conv_idx + 1}/{len(conversations)}: {sample_id} ---")

            # Phase 1: Ingest
            print(f"  Ingesting ({config.mode})...", end="", flush=True)
            t0 = time.time()
            result = ingest_fn(config, conv)
            if result is None:
                print(" SKIPPED")
                continue
            store, speaker_a, speaker_b, uid_a, uid_b = result
            print(f" done in {time.time() - t0:.0f}s")

            # Phase 2: Search
            qa_pairs = [qa for qa in conv.get("qa", []) if qa["category"] != 5]
            if config.categories is not None:
                qa_pairs = [qa for qa in qa_pairs if qa["category"] in config.categories]

            print(f"  Searching {len(qa_pairs)} questions...", end="", flush=True)
            t_search = time.time()
            pre_searched = {}
            for qa in qa_pairs:
                q = qa["question"]
                try:
                    results_a = store.search(q, user_id=uid_a, limit=config.top_k)
                    pre_searched[(uid_a, q)] = [r["content"] for r in results_a]
                except Exception:
                    pre_searched[(uid_a, q)] = []
                try:
                    results_b = store.search(q, user_id=uid_b, limit=config.top_k)
                    pre_searched[(uid_b, q)] = [r["content"] for r in results_b]
                except Exception:
                    pre_searched[(uid_b, q)] = []
            print(f" done in {time.time() - t_search:.0f}s")

            # Phase 3: Answer + Judge (async)
            print(f"  Answering {len(qa_pairs)} questions...", flush=True)
            t1 = time.time()

            batch_size = config.max_llm_concurrent
            for batch_start in range(0, len(qa_pairs), batch_size):
                batch = qa_pairs[batch_start:batch_start + batch_size]
                tasks = [
                    answer_and_judge(config, client, qa, sample_id,
                                     speaker_a, speaker_b, uid_a, uid_b, pre_searched)
                    for qa in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in batch_results:
                    if isinstance(r, Exception):
                        print(f"    [ERROR] {str(r)[:60]}", flush=True)
                        continue
                    all_results.append(r)
                    with open(results_file, "a") as f:
                        f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

                done = min(batch_start + batch_size, len(qa_pairs))
                valid = [r for r in batch_results if not isinstance(r, Exception)]
                if valid:
                    last = valid[-1]
                    print(f"    [{done}/{len(qa_pairs)}] J={last.judge_mean:.1f} | {last.prediction[:50]}")

            print(f"  Done in {time.time() - t1:.0f}s")

            # Interim scores
            interim = generate_summary(all_results, config.mode)
            if interim.get("overall"):
                o = interim["overall"]
                cats = interim.get("by_category", {})
                parts = [f"{c}={cats[c]['judge_mean']:.1f}" for c in
                         ['single-hop', 'multi-hop', 'temporal', 'open-domain'] if c in cats]
                print(f"  >>> Cumulative ({len(all_results)} Q): Judge={o['judge_mean']:.1f}% | {' | '.join(parts)}")

        print(f"\n  API stats: {client.stats}")

    summary = generate_summary(all_results, config.mode)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"\nResults: {results_file}")
    print(f"Summary: {summary_file}")


def generate_summary(results, mode="raw"):
    by_category = defaultdict(list)
    for r in results:
        by_category[r.category_name].append(r)

    summary = {
        "total_questions": len(results),
        "protocol": f"lite_mode_{mode}",
        "overall": {},
        "by_category": {},
        "baselines": {
            "memme_smart": {"overall": 70.78, "single-hop": 80.50, "multi-hop": 67.13, "temporal": 55.51, "open-domain": 72.93},
            "mem0":        {"single-hop": 67.13, "multi-hop": 51.15, "temporal": 55.51, "open-domain": 72.93},
        },
    }

    all_f1, all_b1, all_j = [], [], []
    for cat_name, cat_results in sorted(by_category.items()):
        f1s = [r.f1 for r in cat_results]
        b1s = [r.bleu1 for r in cat_results]
        js = [r.judge_mean for r in cat_results]
        summary["by_category"][cat_name] = {
            "count": len(cat_results),
            "f1_mean": sum(f1s) / len(f1s) * 100,
            "bleu1_mean": sum(b1s) / len(b1s) * 100,
            "judge_mean": sum(js) / len(js) * 100,
        }
        all_f1.extend(f1s); all_b1.extend(b1s); all_j.extend(js)

    if all_f1:
        summary["overall"] = {
            "f1_mean": sum(all_f1) / len(all_f1) * 100,
            "bleu1_mean": sum(all_b1) / len(all_b1) * 100,
            "judge_mean": sum(all_j) / len(all_j) * 100,
        }
    return summary


def print_summary(summary):
    print("\n" + "=" * 90)
    print(f"  MemMe LOCOMO Benchmark — Lite Mode ({summary.get('protocol', '')})")
    print("=" * 90)

    if summary.get("overall"):
        o = summary["overall"]
        print(f"\n  Overall ({summary['total_questions']} questions):")
        print(f"    F1:    {o['f1_mean']:.2f}")
        print(f"    BLEU1: {o['bleu1_mean']:.2f}")
        print(f"    Judge: {o['judge_mean']:.2f}")

    baselines = summary.get("baselines", {})
    smart = baselines.get("memme_smart", {})
    mem0 = baselines.get("mem0", {})

    print(f"\n  {'Category':<15} {'N':>5} {'F1':>8} {'B1':>8} {'Judge':>8} | {'Smart':>8} {'Mem0':>8} {'Delta':>8}")
    print("  " + "-" * 75)

    for cat in ['single-hop', 'multi-hop', 'temporal', 'open-domain']:
        scores = summary.get("by_category", {}).get(cat)
        if not scores: continue
        s = smart.get(cat, 0)
        m = mem0.get(cat, 0)
        j = scores["judge_mean"]
        delta = j - s if s else 0
        print(f"  {cat:<15} {scores['count']:>5} {scores['f1_mean']:>8.2f} {scores['bleu1_mean']:>8.2f} {j:>8.2f} | {s:>8.2f} {m:>8.2f} {delta:>+8.2f}")

    if summary.get("overall"):
        o = summary["overall"]
        so = smart.get("overall", 0)
        print(f"\n  Overall Judge: {o['judge_mean']:.2f}  (Smart: {so:.2f}, delta: {o['judge_mean'] - so:+.2f})")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="MemMe LOCOMO Benchmark — Lite Mode")
    parser.add_argument("--mode", choices=["raw", "ollama"], default="raw",
                        help="raw: add() chunks, no LLM | ollama: add_smart() with local LLM")
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("EMBED_BASE_URL", ""))
    parser.add_argument("--chat-base-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--chat-model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--embed-model", default="text-embedding-v3")
    parser.add_argument("--embed-dims", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--judge-runs", type=int, default=1)
    parser.add_argument("--conversations", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--data-path", default="locomo10.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-llm-concurrent", type=int, default=5)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--chunk-turns", type=int, default=3, help="Turns per chunk in raw mode")
    # Ollama settings
    parser.add_argument("--ollama-model", default="qwen2.5:3b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    # RRF weights
    parser.add_argument("--rrf-vector-weight", type=float, default=0.5)
    parser.add_argument("--rrf-fts-weight", type=float, default=0.3)
    parser.add_argument("--rrf-entity-weight", type=float, default=0.0)
    parser.add_argument("--rrf-k", type=int, default=30)
    parser.add_argument("--rrf-temporal-weight", type=float, default=0.15)
    # Results-only mode
    parser.add_argument("--results-only", type=str, default=None)

    args = parser.parse_args()

    if args.results_only:
        results = []
        with open(args.results_only) as f:
            for line in f:
                results.append(QuestionResult(**json.loads(line)))
        summary = generate_summary(results, args.mode)
        print_summary(summary)
        return

    output_dir = args.output_dir or f"results_lite_{args.mode}"

    config = BenchConfig(
        api_key=args.api_key,
        llm_api_key=args.llm_api_key,
        mode=args.mode,
        chat_base_url=args.chat_base_url,
        embed_base_url=args.base_url,
        chat_model=args.chat_model,
        judge_model=args.judge_model,
        embed_model=args.embed_model,
        embed_dims=args.embed_dims,
        top_k=args.top_k,
        judge_runs=args.judge_runs,
        data_path=args.data_path,
        output_dir=output_dir,
        max_llm_concurrent=args.max_llm_concurrent,
        conversations=[int(x) for x in args.conversations.split(",")] if args.conversations else None,
        categories=[int(x) for x in args.categories.split(",")] if args.categories else None,
        chunk_turns=args.chunk_turns,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        rrf_vector_weight=args.rrf_vector_weight,
        rrf_fts_weight=args.rrf_fts_weight,
        rrf_entity_weight=args.rrf_entity_weight,
        rrf_k=args.rrf_k,
        rrf_temporal_weight=args.rrf_temporal_weight,
    )
    config.reuse_cache = args.reuse_cache
    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()
