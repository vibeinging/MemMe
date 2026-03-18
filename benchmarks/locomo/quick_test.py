#!/usr/bin/env python3
"""
Quick answer-only test — skips ingestion by loading cached memories.

Usage:
  # First time: ingest and save cache (one-time cost, ~70min)
  python3 quick_test.py --build-cache --conversations 0

  # Then iterate on prompts instantly (~5-10min per run)
  python3 quick_test.py --conversations 0
  python3 quick_test.py --conversations 0 --categories 1    # single-hop only
  python3 quick_test.py --conversations 0 --categories 3    # temporal only
  python3 quick_test.py --conversations 0 --questions 5     # first 5 questions only
"""

import argparse
import asyncio
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import aiohttp

# ── Config ──

@dataclass
class Config:
    api_key: str = os.environ.get("DASHSCOPE_API_KEY", "")
    chat_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embed_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen3.5-plus"
    embed_model: str = "text-embedding-v3"
    top_k: int = 30
    dedup_threshold: float = 0.92
    max_llm_concurrent: int = 5
    max_embed_concurrent: int = 10
    batch_size: int = 10
    data_path: str = "locomo10.json"
    cache_dir: str = "cache"


# ── Async API Client ──

class APIClient:
    def __init__(self, config: Config):
        self.config = config
        self.llm_sem = asyncio.Semaphore(config.max_llm_concurrent)
        self.embed_sem = asyncio.Semaphore(config.max_embed_concurrent)
        self.session = None
        self.stats = {"llm": 0, "embed": 0}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, *args):
        if self.session: await self.session.close()

    async def chat(self, messages, temperature=0.0, max_tokens=512):
        url = f"{self.config.chat_base_url}/chat/completions"
        payload = {"model": self.config.chat_model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        async with self.llm_sem:
            for attempt in range(5):
                try:
                    async with self.session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        self.stats["llm"] += 1
                        content = data["choices"][0]["message"]["content"].strip()
                        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Chat failed: {e}")

    async def embed(self, text):
        url = f"{self.config.embed_base_url}/embeddings"
        payload = {"model": self.config.embed_model, "input": text[:2000]}
        async with self.embed_sem:
            for attempt in range(5):
                try:
                    async with self.session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        self.stats["embed"] += 1
                        return data["data"][0]["embedding"]
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Embed failed: {e}")

    async def embed_batch(self, texts):
        return await asyncio.gather(*[self.embed(t) for t in texts])


# ── Scoring ──

def normalize_answer(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def compute_f1(pred, ref):
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not r: return 1.0 if not p else 0.0
    if not p: return 0.0
    common = set(p) & set(r)
    n = sum(min(p.count(t), r.count(t)) for t in common)
    if n == 0: return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a)**0.5
    nb = sum(x*x for x in b)**0.5
    return dot / (na * nb) if na and nb else 0.0

def bm25_score(qt, dt, N, avgdl, k1=1.5, b=0.75):
    if not dt or not qt: return 0.0
    dl = len(dt)
    tf = defaultdict(int)
    for t in dt: tf[t] += 1
    s = 0.0
    for q in set(qt):
        idf = math.log((N - 1 + 0.5) / (1 + 0.5) + 1)
        if tf.get(q, 0) > 0:
            s += idf * (tf[q] * (k1+1)) / (tf[q] + k1 * (1 - b + b * dl / avgdl))
    return s


# ── Memory Cache ──

class MemoryCache:
    """Cached memory store — can be saved/loaded from disk."""

    def __init__(self):
        self.memories = {}   # user_id -> list of {content, embedding, tokens, metadata}
        self.embeddings = {} # text -> embedding
        self.speakers = {}   # conv_id -> (speaker_a, speaker_b, uid_a, uid_b)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'memories': self.memories,
                'embeddings': self.embeddings,
                'speakers': self.speakers,
            }, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Cache saved: {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path):
        cache = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cache.memories = data['memories']
        cache.embeddings = data['embeddings']
        cache.speakers = data['speakers']
        total = sum(len(v) for v in cache.memories.values())
        print(f"  Cache loaded: {total} memories from {path}")
        return cache

    def search_hybrid(self, query_embedding, query_tokens, user_id, top_k=20):
        mems = self.memories.get(user_id, [])
        if not mems: return []

        vec = [(cosine_similarity(query_embedding, m["embedding"]), m) for m in mems]
        vec.sort(key=lambda x: x[0], reverse=True)

        N = len(mems)
        avgdl = sum(len(m["tokens"]) for m in mems) / max(N, 1)
        bm = [(bm25_score(query_tokens, m["tokens"], N, avgdl), m) for m in mems]
        bm.sort(key=lambda x: x[0], reverse=True)

        k = 60
        rrf = defaultdict(float)
        cmap = {}
        for rank, (_, m) in enumerate(vec):
            key = id(m); rrf[key] += 0.6/(k+rank); cmap[key] = m
        for rank, (_, m) in enumerate(bm):
            key = id(m); rrf[key] += 0.4/(k+rank); cmap[key] = m

        sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)
        results = []
        for mid in sorted_ids[:top_k]:
            m = cmap[mid]
            ts = m["metadata"].get("timestamp", "")
            results.append({"content": f"[{ts}] {m['content']}" if ts else m["content"], "score": rrf[mid]})
        return results


# ── Query Variants (local, no LLM) ──

def generate_variants(question):
    variants = [question]
    stop = {'what','when','where','who','how','why','which','does','did','has','have',
            'is','are','was','were','do','the','a','an','in','on','at','to','for','of',
            'and','or','would','could','should','will','can','may','might','still',
            'also','been','being','about','after','before','during','some','many',
            'much','more','most','than','that','this','with','from','into','not',
            'if','she','he','her','his','they','their','it','its'}
    words = question.rstrip('?').split()
    names = [w for w in words if w[0].isupper() and w.lower() not in stop and len(w) > 1]
    terms = [w for w in words if w.lower() not in stop and not w[0].isupper() and len(w) > 2]

    if names and terms:
        variants.append(f"{' '.join(names)} {' '.join(terms[:3])}")
    q = question.lower()
    if names:
        if 'activit' in q: variants.append(f"{names[0]} hobbies interests activities enjoys")
        elif 'book' in q or 'read' in q: variants.append(f"{names[0]} book reading title")
        elif 'how many' in q: variants.append(f"{names[0]} {' '.join(terms[:3])} number count")
        elif 'where' in q: variants.append(f"{names[0]} {' '.join(terms[:2])} location place")
        elif 'would' in q: variants.append(f"{names[0]} preference personality likes values")
        elif 'event' in q: variants.append(f"{names[0]} event attended participated")
    return variants[:3]


# ── Answer Prompt ──

# mem0's exact answer prompt from their LOCOMO evaluation
# Source: https://github.com/mem0ai/mem0/blob/main/evaluation/prompts.py
ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to timestamps
3. If memories contain contradictory information, prioritize the most recent memory
4. Convert relative time references to specific dates (e.g., "last year" -> actual year based on memory timestamp)
5. Focus only on content of memories from both speakers
6. The answer should be less than 5-6 words.

APPROACH (Think step by step):
1. Examine all memories related to question
2. Examine timestamps carefully
3. Look for explicit mentions of dates, times, locations, events
4. If calculation needed, show your work
5. Formulate precise, concise answer
6. Double-check answer directly addresses question
7. Ensure final answer avoids vague time references

Memories for user {speaker_1}:
{speaker_1_memories}

Memories for user {speaker_2}:
{speaker_2_memories}

Question: {question}

Answer:"""


# ── Build Cache ──

def get_extraction_prompt(timestamp=""):
    """mem0's LOCOMO-specific extraction prompt (narrative style).
    Source: https://github.com/mem0ai/mem0/blob/main/evaluation/src/memzero/add.py
    """
    time_ctx = f"\nThe conversation timestamp is: {timestamp}" if timestamp else ""
    return f"""You are a Personal Information Organizer. Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred
2. Include meaningful personal narratives
3. Make each memory rich with specific details rather than general statements
4. Extract memories only from user messages
5. Format each memory as a paragraph with clear narrative structure
{time_ctx}

Return the memories in JSON format: {{"facts": ["memory1", "memory2", ...]}}

Remember:
- Do not use pronouns. Always use the person's actual name.
- Detect the language and record in the same language.
- If no relevant information found, return empty list."""


async def build_cache(config: Config, conv_indices: list):
    with open(config.data_path) as f:
        conversations = json.load(f)

    cache = MemoryCache()

    async with APIClient(config) as client:
        for ci in conv_indices:
            conv = conversations[ci]
            sample_id = conv.get("sample_id", f"conv_{ci}")
            conversation = conv["conversation"]

            session_nums = sorted(set(
                int(k.replace("session_", ""))
                for k in conversation.keys()
                if k.startswith("session_") and k.replace("session_", "").isdigit()
            ))
            if not session_nums: continue

            first_session = conversation.get(f"session_{session_nums[0]}", [])
            speakers = list(dict.fromkeys(t["speaker"] for t in first_session))
            if len(speakers) < 2: speakers += ["unknown"]
            sa, sb = speakers[0], speakers[1]
            uid_a, uid_b = f"{sample_id}_{sa}", f"{sample_id}_{sb}"
            cache.speakers[ci] = (sa, sb, uid_a, uid_b)

            print(f"  Ingesting conv {ci} ({sample_id})...", flush=True)
            t0 = time.time()

            for num in session_nums:
                turns = conversation.get(f"session_{num}", [])
                timestamp = conversation.get(f"session_{num}_date_time", "")
                if not turns: continue

                msgs_a, msgs_b = [], []
                for turn in turns:
                    if turn["speaker"] == sa:
                        msgs_a.append({"role": "user", "content": turn["text"]})
                        msgs_b.append({"role": "assistant", "content": turn["text"]})
                    else:
                        msgs_a.append({"role": "assistant", "content": turn["text"]})
                        msgs_b.append({"role": "user", "content": turn["text"]})

                meta = {"timestamp": timestamp} if timestamp else {}

                for i in range(0, len(msgs_a), config.batch_size):
                    for batch, uid in [(msgs_a[i:i+config.batch_size], uid_a),
                                       (msgs_b[i:i+config.batch_size], uid_b)]:
                        text = "\n".join(f"{m['role']}: {m['content']}" for m in batch)
                        prompt = get_extraction_prompt(timestamp)
                        messages = [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": f"Extract facts:\n\n{text}"},
                        ]
                        try:
                            raw = await client.chat(messages, temperature=0.1, max_tokens=2048)
                            raw = raw.strip()
                            if raw.startswith("```"):
                                raw = re.sub(r'^```\w*\n?', '', raw)
                                raw = re.sub(r'\n?```$', '', raw)
                            data = json.loads(raw)
                            facts = [str(f) for f in data.get("facts", []) if f and len(str(f).strip()) > 3]
                        except Exception as e:
                            print(f"    [WARN] {str(e)[:50]}", flush=True)
                            facts = []

                        # Embed and dedup
                        new_facts = [f for f in facts if f not in cache.embeddings]
                        if new_facts:
                            embs = await client.embed_batch(new_facts)
                            for f, e in zip(new_facts, embs):
                                cache.embeddings[f] = e

                        if uid not in cache.memories:
                            cache.memories[uid] = []

                        for fact in facts:
                            emb = cache.embeddings[fact]
                            dup = any(cosine_similarity(emb, m["embedding"]) > config.dedup_threshold
                                     for m in cache.memories[uid])
                            if not dup:
                                cache.memories[uid].append({
                                    "content": fact, "embedding": emb,
                                    "tokens": normalize_answer(fact).split(),
                                    "metadata": meta,
                                })

            n_a = len(cache.memories.get(uid_a, []))
            n_b = len(cache.memories.get(uid_b, []))
            print(f"    Done in {time.time()-t0:.0f}s ({n_a}+{n_b} memories)", flush=True)

    os.makedirs(config.cache_dir, exist_ok=True)
    cache_path = Path(config.cache_dir) / f"conv_{'_'.join(str(c) for c in conv_indices)}.pkl"
    cache.save(cache_path)
    return cache


# ── Answer Phase Only ──

CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}

@dataclass
class QResult:
    question: str
    reference: str
    prediction: str
    category: int
    category_name: str
    f1: float
    judge_mean: float = 0.0


async def run_answers(config: Config, cache: MemoryCache, conv_indices: list,
                      categories: list = None, max_questions: int = 0):
    with open(config.data_path) as f:
        conversations = json.load(f)

    all_results = []

    async with APIClient(config) as client:
        for ci in conv_indices:
            conv = conversations[ci]
            sa, sb, uid_a, uid_b = cache.speakers[ci]

            qa_pairs = [qa for qa in conv.get("qa", []) if qa["category"] != 5]
            if categories:
                qa_pairs = [qa for qa in qa_pairs if qa["category"] in categories]
            if max_questions > 0:
                qa_pairs = qa_pairs[:max_questions]

            print(f"\n  Conv {ci}: {len(qa_pairs)} questions", flush=True)

            batch_size = config.max_llm_concurrent
            for bs in range(0, len(qa_pairs), batch_size):
                batch = qa_pairs[bs:bs+batch_size]

                async def answer_one(qa):
                    q = qa["question"]
                    ref = str(qa["answer"])
                    cat = qa["category"]

                    # Multi-query search
                    queries = generate_variants(q)
                    to_embed = [qr for qr in queries if qr not in cache.embeddings]
                    if to_embed:
                        embs = await client.embed_batch(to_embed)
                        for qr, e in zip(to_embed, embs):
                            cache.embeddings[qr] = e

                    all_r = defaultdict(lambda: {"score": 0.0, "content": ""})
                    for qi, qr in enumerate(queries):
                        qe = cache.embeddings[qr]
                        qt = normalize_answer(qr).split()
                        for uid in [uid_a, uid_b]:
                            for rank, r in enumerate(cache.search_hybrid(qe, qt, uid, config.top_k)):
                                w = 1.0 if qi == 0 else 0.7
                                all_r[r["content"]]["content"] = r["content"]
                                all_r[r["content"]]["score"] += w / (60 + rank)

                    merged = sorted(all_r.values(), key=lambda x: x["score"], reverse=True)[:config.top_k]

                    # Split back for prompt
                    mem_a = [r["content"] for r in cache.search_hybrid(
                        cache.embeddings[q], normalize_answer(q).split(), uid_a, config.top_k)]
                    mem_b = [r["content"] for r in cache.search_hybrid(
                        cache.embeddings[q], normalize_answer(q).split(), uid_b, config.top_k)]

                    prompt = ANSWER_PROMPT.format(
                        speaker_1=sa, speaker_2=sb,
                        speaker_1_memories=json.dumps(mem_a, ensure_ascii=False),
                        speaker_2_memories=json.dumps(mem_b, ensure_ascii=False),
                        question=q,
                    )
                    try:
                        pred = await client.chat([{"role": "user", "content": prompt}], max_tokens=50)
                    except:
                        pred = "I don't know"

                    f1 = compute_f1(pred, ref)

                    # Judge
                    jp = f"""Question: {q}\nReference: {ref}\nPredicted: {pred}\n\nIs the prediction correct? Reply ONLY "correct" or "wrong"."""
                    try:
                        jr = await client.chat([{"role": "user", "content": jp}], temperature=0.3, max_tokens=10)
                        judge = 1.0 if "correct" in jr.lower() else 0.0
                    except:
                        judge = 0.0

                    return QResult(q, ref, pred, cat, CATEGORY_NAMES.get(cat, "?"), f1, judge)

                tasks = [answer_one(qa) for qa in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in batch_results:
                    if isinstance(r, Exception):
                        print(f"    [ERR] {str(r)[:50]}", flush=True)
                        continue
                    all_results.append(r)

                done = min(bs + batch_size, len(qa_pairs))
                print(f"    [{done}/{len(qa_pairs)}]", end="", flush=True)

        print(f"\n  API: {client.stats}")

    # Summary
    print("\n" + "=" * 70)
    by_cat = defaultdict(list)
    for r in all_results:
        by_cat[r.category_name].append(r)

    print(f"  {'Category':<15} {'N':>5} {'F1':>8} {'Judge':>8} | {'mem0':>8}")
    print("  " + "-" * 55)
    for cat in ['single-hop', 'multi-hop', 'temporal', 'open-domain']:
        cr = by_cat.get(cat, [])
        if not cr: continue
        f1 = sum(r.f1 for r in cr) / len(cr) * 100
        j = sum(r.judge_mean for r in cr) / len(cr) * 100
        m0 = {"single-hop": 67.13, "multi-hop": 51.15, "temporal": 55.51, "open-domain": 72.93}[cat]
        print(f"  {cat:<15} {len(cr):>5} {f1:>8.2f} {j:>8.2f} | {m0:>8.2f}")

    total_j = sum(r.judge_mean for r in all_results) / len(all_results) * 100 if all_results else 0
    print(f"\n  Overall Judge: {total_j:.2f}")
    print("=" * 70)

    return all_results


# ── Main ──

async def main_async(args):
    config = Config()

    conv_indices = [int(x) for x in args.conversations.split(",")] if args.conversations else [0]
    categories = [int(x) for x in args.categories.split(",")] if args.categories else None
    cache_path = Path(config.cache_dir) / f"conv_{'_'.join(str(c) for c in conv_indices)}.pkl"

    if args.build_cache:
        print("=== Building memory cache (one-time) ===")
        cache = await build_cache(config, conv_indices)
    elif cache_path.exists():
        cache = MemoryCache.load(cache_path)
    else:
        print(f"No cache found at {cache_path}. Run with --build-cache first.")
        sys.exit(1)

    print(f"\n=== Quick Answer Test ===")
    await run_answers(config, cache, conv_indices, categories, args.questions)


def main():
    parser = argparse.ArgumentParser(description="Quick benchmark test (answer-only)")
    parser.add_argument("--build-cache", action="store_true", help="Build memory cache (one-time)")
    parser.add_argument("--conversations", type=str, default="0")
    parser.add_argument("--categories", type=str, default=None, help="e.g. '1' for single-hop, '3' for temporal")
    parser.add_argument("--questions", type=int, default=0, help="Max questions (0=all)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
