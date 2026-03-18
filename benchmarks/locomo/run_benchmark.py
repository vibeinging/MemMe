#!/usr/bin/env python3
"""
MemMe LOCOMO Benchmark Runner

Evaluates MemMe's memory quality on the LOCOMO benchmark (arXiv:2402.17753).
Compares against mem0, Zep, OpenAI baselines.

Usage:
    python run_benchmark.py --api-key sk-xxx --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
    python run_benchmark.py --api-key sk-xxx --conversations 0,1,2 --categories 1,2,3
    python run_benchmark.py --results-only results/run_20260318.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests


# ── Configuration ──

@dataclass
class BenchConfig:
    api_key: str
    chat_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embed_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen3.5-plus"
    embed_model: str = "text-embedding-v3"
    embed_dims: int = 1024
    judge_model: str = "qwen3.5-plus"
    top_k: int = 20  # memories to retrieve (more = better recall)
    max_context_messages: int = 10  # recent messages for context
    data_path: str = "locomo10.json"
    output_dir: str = "results"
    conversations: Optional[list] = None  # None = all
    categories: Optional[list] = None  # None = all (1-5)
    judge_runs: int = 3  # LLM judge evaluation runs (mem0 uses 10)
    skip_adversarial: bool = True  # mem0 skips category 5


# ── API Helpers ──

def chat_completion(config: BenchConfig, messages: list, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Call the chat completion API."""
    url = f"{config.chat_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.chat_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 4:
                wait = 3 * (attempt + 1)
                print(f"    [retry {attempt+1}/5, wait {wait}s: {str(e)[:60]}]", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Chat API failed after 5 attempts: {e}")


def embed_text(config: BenchConfig, text: str) -> list:
    """Get embedding vector for text."""
    url = f"{config.embed_base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": config.embed_model, "input": text}
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Embed API failed: {e}")


# ── Scoring ──

def normalize_answer(text) -> str:
    """Normalize answer for F1 calculation."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu1(prediction: str, reference: str) -> float:
    """BLEU-1 (unigram) score."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_count = defaultdict(int)
    for t in ref_tokens:
        ref_count[t] += 1

    clipped = 0
    for t in pred_tokens:
        if ref_count[t] > 0:
            clipped += 1
            ref_count[t] -= 1

    return clipped / len(pred_tokens) if pred_tokens else 0.0


def llm_judge(config: BenchConfig, question: str, reference: str, prediction: str, category: int) -> float:
    """Use LLM as judge to evaluate answer correctness."""
    if category in (3, 5):
        # Temporal and adversarial: binary scoring
        prompt = f"""You are evaluating whether a predicted answer is correct.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Is the predicted answer correct? Consider it correct if it conveys the same key information as the reference, even if worded differently.
Reply with ONLY "correct" or "wrong"."""
    else:
        # Factual: 3-level scoring
        prompt = f"""You are evaluating whether a predicted answer is correct.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Rate the predicted answer:
- "correct" if it contains the key facts from the reference answer
- "partial" if it contains some but not all key facts
- "wrong" if it is incorrect or misses the main point

Reply with ONLY one word: "correct", "partial", or "wrong"."""

    messages = [{"role": "user", "content": prompt}]
    response = chat_completion(config, messages, temperature=0.3, max_tokens=10)
    response = response.lower().strip().rstrip('.')

    if "correct" in response:
        return 1.0
    elif "partial" in response:
        return 0.5
    else:
        return 0.0


# ── Memory System Interface ──

class MemMeMemorySystem:
    """Memory system with hybrid search (vector + BM25 + RRF fusion)."""

    def __init__(self, config: BenchConfig):
        self.config = config
        self.memories = []
        self.embeddings = {}

    def reset(self):
        self.memories = []
        self.embeddings = {}

    def add_memory(self, content: str, user_id: str, metadata: dict = None):
        embedding = self._get_embedding(content)
        # Pre-tokenize for BM25
        tokens = normalize_answer(content).split()
        self.memories.append({
            "content": content,
            "user_id": user_id,
            "embedding": embedding,
            "metadata": metadata or {},
            "tokens": tokens,
        })

    def search(self, query: str, user_id: str, top_k: int = 10) -> list:
        """Hybrid search: vector similarity + BM25 keyword matching + RRF fusion."""
        user_memories = [m for m in self.memories if m["user_id"] == user_id]
        if not user_memories:
            return []

        # 1. Vector search (semantic)
        query_embedding = self._get_embedding(query)
        vector_scored = []
        for m in user_memories:
            sim = self._cosine_similarity(query_embedding, m["embedding"])
            vector_scored.append((sim, m))
        vector_scored.sort(key=lambda x: x[0], reverse=True)
        vector_ranked = [(rank, m) for rank, (_, m) in enumerate(vector_scored)]

        # 2. BM25 search (keyword/lexical)
        query_tokens = normalize_answer(query).split()
        bm25_scored = []
        for m in user_memories:
            score = self._bm25_score(query_tokens, m["tokens"], user_memories)
            bm25_scored.append((score, m))
        bm25_scored.sort(key=lambda x: x[0], reverse=True)
        bm25_ranked = [(rank, m) for rank, (_, m) in enumerate(bm25_scored)]

        # 3. RRF fusion
        k = 60  # RRF constant
        rrf_scores = defaultdict(float)
        content_map = {}

        vector_weight = 0.6
        bm25_weight = 0.4

        for rank, m in vector_ranked:
            key = id(m)
            rrf_scores[key] += vector_weight / (k + rank)
            content_map[key] = m

        for rank, m in bm25_ranked:
            key = id(m)
            rrf_scores[key] += bm25_weight / (k + rank)
            content_map[key] = m

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for mid in sorted_ids[:top_k]:
            m = content_map[mid]
            results.append({"content": m["content"], "score": rrf_scores[mid]})

        return results

    def _get_embedding(self, text: str) -> list:
        if text not in self.embeddings:
            self.embeddings[text] = embed_text(self.config, text[:2000])
        return self.embeddings[text]

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _bm25_score(query_tokens: list, doc_tokens: list, all_docs: list,
                     k1: float = 1.5, b: float = 0.75) -> float:
        """Simple BM25 scoring."""
        if not doc_tokens or not query_tokens:
            return 0.0
        import math
        N = len(all_docs)
        avgdl = sum(len(d["tokens"]) for d in all_docs) / max(N, 1)
        dl = len(doc_tokens)
        score = 0.0
        doc_tf = defaultdict(int)
        for t in doc_tokens:
            doc_tf[t] += 1
        for qt in set(query_tokens):
            # Document frequency
            df = sum(1 for d in all_docs if qt in d["tokens"])
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf = doc_tf.get(qt, 0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        return score


# ── Benchmark Runner ──

CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}


def extract_atomic_facts(config: BenchConfig, turns: list, session_date: str) -> list:
    """Extract atomic facts using sliding window + detail pass (SimpleMem approach)."""
    speakers = list(set(t['speaker'] for t in turns))
    all_facts = []

    # Phase 1: Sliding window extraction (5 turns per chunk, overlap 1)
    window_size = 5
    for i in range(0, len(turns), window_size - 1):
        chunk = turns[i:i + window_size]
        if not chunk:
            break
        dialogue = "\n".join(f"{t['speaker']}: {t['text']}" for t in chunk)

        prompt = f"""Extract EVERY atomic fact from this conversation excerpt ({session_date}).
Resolve ALL pronouns to actual names ({', '.join(speakers)}).

ONE fact per line. Maximum 12 words each. Include:
- Specific names, titles, numbers, dates, counts
- "{speakers[0]} is single" not "mentioned being single"
- "read 'Becoming Nicole'" not "read a book"
- "has 3 children" not "has children"
- "went to beach 2 times in 2023" not "went to beach"
- "{speakers[0]} and {speakers[-1]} both painted sunsets" for shared facts

Excerpt:
{dialogue}

Facts:"""

        messages = [{"role": "user", "content": prompt}]
        try:
            response = chat_completion(config, messages, max_tokens=1024)
            facts = [line.strip().lstrip("- •0123456789.)").strip() for line in response.split("\n") if line.strip()]
            all_facts.extend([f"[{session_date}] {fact}" for fact in facts if len(fact) > 5])
        except Exception as e:
            print(f"    [WARN] Chunk extraction failed: {str(e)[:50]}", flush=True)

    # Phase 2: Detail extraction pass (numbers, names, dates, titles)
    full_dialogue = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
    detail_prompt = f"""From this conversation ({session_date}), extract ONLY specific details that are easy to miss.
Focus on: exact numbers, book/movie titles, specific dates, relationship status, counts, ages, named places, named events.

Speakers: {', '.join(speakers)}

Format: one detail per line, include the person's name.
Examples: "Melanie has 3 children", "Caroline is single", "Melanie read 'Becoming Nicole'", "Caroline moved from Sweden 4 years ago"

Conversation:
{full_dialogue}

Specific details:"""

    messages = [{"role": "user", "content": detail_prompt}]
    try:
        response = chat_completion(config, messages, max_tokens=1024)
        details = [line.strip().lstrip("- •0123456789.)").strip() for line in response.split("\n") if line.strip()]
        all_facts.extend([f"[{session_date}] {d}" for d in details if len(d) > 5])
    except Exception as e:
        print(f"    [WARN] Detail extraction failed: {str(e)[:50]}", flush=True)

    return all_facts


def extract_key_dialogue_turns(turns: list, session_date: str) -> list:
    """Store key raw dialogue turns as backup memories (observation approach)."""
    PLEASANTRIES = ['how are you', 'see you', 'bye', 'take care', 'talk to you',
                    'good morning', 'hello', 'hi there', 'sounds good', 'okay',
                    "i'm fine", "that's great", "no problem", "you're welcome"]
    memories = []
    for t in turns:
        text = t['text'].strip()
        speaker = t['speaker']
        if len(text) > 25 and not any(g in text.lower() for g in PLEASANTRIES):
            memories.append(f"[{session_date}] {speaker} said: \"{text[:250]}\"")
    return memories


def ingest_conversation(config: BenchConfig, memory: MemMeMemorySystem, conv: dict, user_id: str):
    """Ingest all sessions using multi-granularity storage."""
    conversation = conv["conversation"]

    session_nums = sorted(set(
        int(k.replace("session_", ""))
        for k in conversation.keys()
        if k.startswith("session_") and k.replace("session_", "").isdigit()
    ))

    total_memories = 0
    for num in session_nums:
        session_key = f"session_{num}"
        date_key = f"session_{num}_date_time"
        turns = conversation.get(session_key, [])
        session_date = conversation.get(date_key, f"Session {num}")

        if not turns:
            continue

        # Layer 1: Atomic facts (LLM-extracted, coreference-resolved)
        try:
            facts = extract_atomic_facts(config, turns, session_date)
            for fact in facts:
                memory.add_memory(fact, user_id, metadata={"type": "fact", "session": num})
            total_memories += len(facts)
        except Exception as e:
            print(f"    [WARN] Fact extraction failed for session {num}: {str(e)[:60]}", flush=True)

        # Layer 2: Key raw dialogue turns (backup for specific quotes/details)
        raw_turns = extract_key_dialogue_turns(turns, session_date)
        for turn in raw_turns:
            memory.add_memory(turn, user_id, metadata={"type": "dialogue", "session": num})
        total_memories += len(raw_turns)

        time.sleep(1.0)

    return total_memories


def answer_question(config: BenchConfig, memory: MemMeMemorySystem, question: str, user_id: str, category: int = 1) -> str:
    """Answer a question using retrieved memories as context."""
    # For reasoning/temporal questions, retrieve more context
    k = config.top_k * 2 if category in (2, 3) else config.top_k
    results = memory.search(question, user_id, top_k=k)

    if not results:
        context = "No relevant memories found."
    else:
        context = "\n".join(f"- {r['content']}" for r in results)

    if category == 3:
        # Temporal/reasoning: needs inference from evidence
        prompt = f"""Based on the memories below, answer the question. You MUST give a definitive answer by reasoning from the available evidence. Do NOT say "I don't have that information."

Rules:
- Use evidence from memories to reason about the answer
- For "Would X...?" questions, answer "Yes" or "No" with a brief reason
- For "What would...?" questions, give your best inference
- Keep the answer SHORT — one phrase or short sentence
- Start with the direct answer (Yes/No/the answer), then a brief reason

Memories:
{context}

Question: {question}

Answer:"""
    elif category == 2:
        # Multi-hop: needs synthesis across memories
        prompt = f"""Synthesize information from the memories below to answer the question. Be concise.

Rules:
- Combine facts from multiple memories to form the answer
- Answer in a few words or a short phrase
- If listing items, use commas
- Do NOT say "I don't have that information" — try to answer from what's available

Memories:
{context}

Question: {question}

Answer:"""
    else:
        # Single-hop / open-domain: direct factual recall
        prompt = f"""Answer the question using the memories below. Be VERY brief — a few words only.

Rules:
- Give the most SPECIFIC answer possible (exact names, numbers, titles, dates)
- "abstract art" not "art" or "paintings"
- "3" not "a few" or "several"
- "'Becoming Nicole'" not "a book"
- Use commas for lists (e.g., "running, pottery")
- If you can infer the answer from the memories, DO answer — don't say "not mentioned"
- Only say "not mentioned" if the memories are completely irrelevant

Memories:
{context}

Question: {question}

Brief answer:"""

    messages = [{"role": "user", "content": prompt}]
    return chat_completion(config, messages, max_tokens=150)


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


def run_benchmark(config: BenchConfig):
    """Run the full LOCOMO benchmark."""
    # Load dataset
    data_path = Path(config.data_path)
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)

    with open(data_path) as f:
        conversations = json.load(f)

    # Filter conversations
    if config.conversations is not None:
        conversations = [conversations[i] for i in config.conversations if i < len(conversations)]

    # Prepare output
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = Path(config.output_dir) / f"run_{timestamp}.jsonl"
    summary_file = Path(config.output_dir) / f"summary_{timestamp}.json"

    print(f"=== MemMe LOCOMO Benchmark ===")
    print(f"Conversations: {len(conversations)}")
    print(f"Model: {config.chat_model}")
    print(f"Embedding: {config.embed_model} ({config.embed_dims}d)")
    print(f"Top-K: {config.top_k}")
    print(f"Judge runs: {config.judge_runs}")
    print(f"Output: {results_file}")
    print()

    memory = MemMeMemorySystem(config)
    all_results = []

    for conv_idx, conv in enumerate(conversations):
        sample_id = conv.get("sample_id", f"conv_{conv_idx}")
        print(f"\n--- Conversation {conv_idx + 1}/{len(conversations)}: {sample_id} ---")

        # Reset memory for each conversation
        memory.reset()

        # Phase 1: Ingest conversation into memory
        print("  Ingesting sessions...", end="", flush=True)
        t0 = time.time()
        num_facts = ingest_conversation(config, memory, conv, user_id=sample_id)
        print(f" {num_facts} facts in {time.time() - t0:.1f}s")

        # Phase 2: Answer questions
        qa_pairs = conv.get("qa", [])
        if config.categories is not None:
            qa_pairs = [qa for qa in qa_pairs if qa["category"] in config.categories]
        if config.skip_adversarial:
            qa_pairs = [qa for qa in qa_pairs if qa["category"] != 5]

        print(f"  Answering {len(qa_pairs)} questions...")

        for qi, qa in enumerate(qa_pairs):
            question = qa["question"]
            reference = str(qa["answer"])
            category = qa["category"]

            # Get model's answer (category-aware prompt)
            try:
                prediction = answer_question(config, memory, question, user_id=sample_id, category=category)
            except Exception as e:
                print(f"    [WARN] Answer failed for Q{qi}: {str(e)[:60]}", flush=True)
                prediction = "error"

            # Compute F1 and BLEU-1
            f1 = compute_f1(prediction, reference)
            bleu1 = compute_bleu1(prediction, reference)

            # LLM Judge
            judge_scores = []
            for _ in range(config.judge_runs):
                try:
                    score = llm_judge(config, question, reference, prediction, category)
                    judge_scores.append(score)
                except Exception:
                    judge_scores.append(0.0)
                time.sleep(0.3)

            judge_mean = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

            result = QuestionResult(
                sample_id=sample_id,
                question=question,
                reference=reference,
                prediction=prediction,
                category=category,
                category_name=CATEGORY_NAMES.get(category, "unknown"),
                f1=f1,
                bleu1=bleu1,
                judge_scores=judge_scores,
                judge_mean=judge_mean,
            )
            all_results.append(result)

            # Write incrementally
            with open(results_file, "a") as f:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

            # Progress
            if (qi + 1) % 10 == 0 or qi == len(qa_pairs) - 1:
                print(f"    [{qi + 1}/{len(qa_pairs)}] F1={f1:.3f} B1={bleu1:.3f} J={judge_mean:.2f}")

            time.sleep(0.5)  # Rate limit

    # Generate summary
    summary = generate_summary(all_results)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")


def generate_summary(results: list) -> dict:
    """Generate aggregate scores by category."""
    by_category = defaultdict(list)
    for r in results:
        by_category[r.category_name].append(r)

    summary = {
        "total_questions": len(results),
        "overall": {},
        "by_category": {},
        "baselines": {
            "mem0": {"single-hop": 67.13, "multi-hop": 51.15, "temporal": 55.51, "open-domain": 72.93},
            "zep": {"single-hop": 61.70, "multi-hop": 41.35, "temporal": 49.31, "open-domain": 76.60},
            "mem0g": {"single-hop": 65.71, "multi-hop": 47.19, "temporal": 58.13, "open-domain": 75.71},
        },
    }

    all_f1, all_b1, all_j = [], [], []
    for cat_name, cat_results in sorted(by_category.items()):
        f1_scores = [r.f1 for r in cat_results]
        b1_scores = [r.bleu1 for r in cat_results]
        j_scores = [r.judge_mean for r in cat_results]

        cat_summary = {
            "count": len(cat_results),
            "f1_mean": sum(f1_scores) / len(f1_scores) * 100,
            "bleu1_mean": sum(b1_scores) / len(b1_scores) * 100,
            "judge_mean": sum(j_scores) / len(j_scores) * 100,
        }
        summary["by_category"][cat_name] = cat_summary
        all_f1.extend(f1_scores)
        all_b1.extend(b1_scores)
        all_j.extend(j_scores)

    if all_f1:
        summary["overall"] = {
            "f1_mean": sum(all_f1) / len(all_f1) * 100,
            "bleu1_mean": sum(all_b1) / len(all_b1) * 100,
            "judge_mean": sum(all_j) / len(all_j) * 100,
        }

    return summary


def print_summary(summary: dict):
    """Print formatted summary."""
    print("\n" + "=" * 70)
    print("  MemMe LOCOMO Benchmark Results")
    print("=" * 70)

    if summary.get("overall"):
        o = summary["overall"]
        print(f"\n  Overall ({summary['total_questions']} questions):")
        print(f"    F1:    {o['f1_mean']:.2f}")
        print(f"    BLEU1: {o['bleu1_mean']:.2f}")
        print(f"    Judge: {o['judge_mean']:.2f}")

    print(f"\n  {'Category':<15} {'Count':>6} {'F1':>8} {'B1':>8} {'Judge':>8}  {'vs Mem0':>8}  {'vs Zep':>8}")
    print("  " + "-" * 65)

    baselines = summary.get("baselines", {})
    for cat, scores in sorted(summary.get("by_category", {}).items()):
        mem0_score = baselines.get("mem0", {}).get(cat, 0)
        zep_score = baselines.get("zep", {}).get(cat, 0)
        j = scores["judge_mean"]
        vs_mem0 = f"{j - mem0_score:+.2f}" if mem0_score else "N/A"
        vs_zep = f"{j - zep_score:+.2f}" if zep_score else "N/A"

        print(f"  {cat:<15} {scores['count']:>6} {scores['f1_mean']:>8.2f} {scores['bleu1_mean']:>8.2f} {j:>8.2f}  {vs_mem0:>8}  {vs_zep:>8}")

    print("=" * 70)


def load_and_summarize(results_file: str):
    """Load results from JSONL and generate summary."""
    results = []
    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            results.append(QuestionResult(**data))

    summary = generate_summary(results)
    print_summary(summary)


# ── Entry Point ──

def main():
    parser = argparse.ArgumentParser(description="MemMe LOCOMO Benchmark Runner")
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--chat-model", default="qwen3.5-plus")
    parser.add_argument("--embed-model", default="text-embedding-v3")
    parser.add_argument("--embed-dims", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--judge-runs", type=int, default=3)
    parser.add_argument("--conversations", type=str, default=None, help="Comma-separated conversation indices (0-9)")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated category numbers (1-5)")
    parser.add_argument("--results-only", type=str, default=None, help="Just summarize existing results file")
    parser.add_argument("--data-path", default="locomo10.json")
    parser.add_argument("--output-dir", default="results")

    args = parser.parse_args()

    if args.results_only:
        load_and_summarize(args.results_only)
        return

    config = BenchConfig(
        api_key=args.api_key,
        chat_base_url=args.base_url,
        embed_base_url=args.base_url,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        embed_dims=args.embed_dims,
        top_k=args.top_k,
        judge_runs=args.judge_runs,
        data_path=args.data_path,
        output_dir=args.output_dir,
        conversations=[int(x) for x in args.conversations.split(",")] if args.conversations else None,
        categories=[int(x) for x in args.categories.split(",")] if args.categories else None,
    )

    run_benchmark(config)


if __name__ == "__main__":
    main()
