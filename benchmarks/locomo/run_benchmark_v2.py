#!/usr/bin/env python3
"""
MemMe LOCOMO Benchmark Runner v2 — Fair Evaluation

Matches mem0's exact evaluation protocol:
1. Raw dialogue passed to memory system (no pre-extraction tricks)
2. Dual-speaker perspective (both speakers stored separately)
3. Dual-speaker retrieval (search from both speakers)
4. Unified answer prompt (≤5-6 words, same for all categories)
5. Skip adversarial (category 5), same as mem0

Reference: https://github.com/mem0ai/mem0/tree/main/evaluation
"""

import argparse
import json
import math
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
    top_k: int = 10  # same as mem0 default
    data_path: str = "locomo10.json"
    output_dir: str = "results_fair"
    conversations: Optional[list] = None
    categories: Optional[list] = None
    judge_runs: int = 1
    batch_size: int = 10  # messages per mem0.add() call, same as mem0


# ── API Helpers ──

def chat_completion(config: BenchConfig, messages: list, temperature: float = 0.0, max_tokens: int = 512) -> str:
    url = f"{config.chat_base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {"model": config.chat_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 4:
                wait = 3 * (attempt + 1)
                print(f"      [retry {attempt+1}/5, wait {wait}s]", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Chat API failed: {e}")


def embed_text(config: BenchConfig, text: str) -> list:
    url = f"{config.embed_base_url}/embeddings"
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {"model": config.embed_model, "input": text[:2000]}
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Embed API failed: {e}")


# ── Scoring (same as mem0) ──

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

def llm_judge(config: BenchConfig, question: str, reference: str, prediction: str, category: int) -> float:
    """Same judge logic as mem0's evaluation."""
    prompt = f"""You are evaluating whether a predicted answer is correct.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Is the predicted answer correct? Consider it correct if it conveys the same key information as the reference, even if worded differently.
Reply with ONLY "correct" or "wrong"."""

    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(config, messages, temperature=0.3, max_tokens=10)
        return 1.0 if "correct" in response.lower() else 0.0
    except Exception:
        return 0.0


# ── Memory System (simulates mem0's add/search behavior) ──

class MemorySystem:
    """
    Simulates mem0's memory pipeline:
    - add(): takes raw dialogue messages, extracts facts via LLM, stores with embeddings
    - search(): retrieves by embedding similarity + BM25 hybrid

    Key difference from mem0: mem0 uses their own extraction prompt internally.
    We use our framework's extraction prompt (the thing being benchmarked).
    """
    def __init__(self, config: BenchConfig):
        self.config = config
        self.memories = {}  # user_id -> list of memories
        self.embeddings = {}  # text -> embedding (cache)

    def reset(self):
        self.memories = {}
        self.embeddings = {}

    def add(self, messages: list, user_id: str, metadata: dict = None):
        """
        Add memories from raw dialogue messages.
        Mimics mem0.add(): passes raw messages to LLM for fact extraction,
        then stores extracted facts with embeddings.
        """
        # Format messages as conversation text (same as mem0)
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        # Extract facts using LLM (this is what mem0 does internally)
        facts = self._extract_facts(text)

        if user_id not in self.memories:
            self.memories[user_id] = []

        for fact in facts:
            embedding = self._get_embedding(fact)
            tokens = normalize_answer(fact).split()
            self.memories[user_id].append({
                "content": fact,
                "embedding": embedding,
                "tokens": tokens,
                "metadata": metadata or {},
            })

    def _extract_facts(self, text: str) -> list:
        """Extract facts from conversation text using LLM."""
        prompt = f"""You are a Personal Information Organizer. Extract relevant facts and preferences from this conversation.

Rules:
- Return JSON format: {{"facts": ["fact1", "fact2", ...]}}
- Each fact should be atomic — one piece of information per fact
- Include specific details: names, dates, numbers, titles
- Use actual names, not pronouns
- Detect the language and record facts in the same language

Conversation:
{text}"""

        messages = [{"role": "user", "content": prompt}]
        try:
            raw = chat_completion(self.config, messages, temperature=0.1, max_tokens=2048)
            # Parse JSON response
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```\w*\n?', '', raw)
                raw = re.sub(r'\n?```$', '', raw)
            data = json.loads(raw)
            facts = data.get("facts", [])
            return [str(f) for f in facts if f]
        except Exception as e:
            print(f"      [WARN] Fact extraction failed: {str(e)[:60]}", flush=True)
            return []

    def search(self, query: str, user_id: str, top_k: int = 10) -> list:
        """Search memories using hybrid (vector + BM25 + RRF)."""
        user_memories = self.memories.get(user_id, [])
        if not user_memories:
            return []

        query_embedding = self._get_embedding(query)
        query_tokens = normalize_answer(query).split()

        # Vector scores
        vector_scored = []
        for m in user_memories:
            sim = cosine_similarity(query_embedding, m["embedding"])
            vector_scored.append((sim, m))
        vector_scored.sort(key=lambda x: x[0], reverse=True)

        # BM25 scores
        bm25_scored = []
        N = len(user_memories)
        avgdl = sum(len(m["tokens"]) for m in user_memories) / max(N, 1)
        for m in user_memories:
            score = bm25_score(query_tokens, m["tokens"], N, avgdl)
            bm25_scored.append((score, m))
        bm25_scored.sort(key=lambda x: x[0], reverse=True)

        # RRF fusion
        k = 60
        rrf_scores = defaultdict(float)
        content_map = {}
        for rank, (_, m) in enumerate(vector_scored):
            key = id(m)
            rrf_scores[key] += 0.6 / (k + rank)
            content_map[key] = m
        for rank, (_, m) in enumerate(bm25_scored):
            key = id(m)
            rrf_scores[key] += 0.4 / (k + rank)
            content_map[key] = m

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for mid in sorted_ids[:top_k]:
            m = content_map[mid]
            ts = m["metadata"].get("timestamp", "")
            results.append({
                "content": f"{ts}: {m['content']}" if ts else m["content"],
                "score": rrf_scores[mid],
            })
        return results

    def _get_embedding(self, text: str) -> list:
        if text not in self.embeddings:
            self.embeddings[text] = embed_text(self.config, text[:2000])
        return self.embeddings[text]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

def bm25_score(query_tokens, doc_tokens, N, avgdl, k1=1.5, b=0.75):
    if not doc_tokens or not query_tokens: return 0.0
    dl = len(doc_tokens)
    doc_tf = defaultdict(int)
    for t in doc_tokens: doc_tf[t] += 1
    score = 0.0
    for qt in set(query_tokens):
        df = 1  # approximate
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        tf = doc_tf.get(qt, 0)
        if tf > 0:
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return score


# ── Answer Prompt (matches mem0's protocol) ──

ANSWER_PROMPT = """You are a memory-based assistant. Answer the question using ONLY the provided memories.

Instructions:
- Carefully analyze all provided memories from both speakers
- Most recent memories (by timestamp) take precedence when there are contradictions
- Convert relative time references (like "last year") to specific dates based on memory timestamps
- Focus only on the content of the memories provided
- Your response MUST be less than 5-6 words
- If the information is not available in the memories, respond with "I don't know"

Speaker 1 ({speaker_1}) memories:
{speaker_1_memories}

Speaker 2 ({speaker_2}) memories:
{speaker_2_memories}

Question: {question}

Answer (5-6 words max):"""


# ── Benchmark Runner ──

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


def ingest_conversation(config: BenchConfig, memory: MemorySystem, conv: dict):
    """
    Ingest conversation exactly as mem0 does:
    1. Process session by session
    2. For each session, create messages from both speaker perspectives
    3. Add memories for each speaker separately
    4. Attach timestamp metadata
    """
    conversation = conv["conversation"]
    sample_id = conv.get("sample_id", "unknown")

    # Find speakers from first session
    session_nums = sorted(set(
        int(k.replace("session_", ""))
        for k in conversation.keys()
        if k.startswith("session_") and k.replace("session_", "").isdigit()
    ))

    if not session_nums:
        return 0

    # Identify the two speakers
    first_session = conversation.get(f"session_{session_nums[0]}", [])
    speakers = list(dict.fromkeys(t["speaker"] for t in first_session))
    if len(speakers) < 2:
        speakers = speakers + ["unknown"]
    speaker_a, speaker_b = speakers[0], speakers[1]

    # User IDs: same convention as mem0 (speaker name as user_id)
    user_id_a = f"{sample_id}_{speaker_a}"
    user_id_b = f"{sample_id}_{speaker_b}"

    total_added = 0

    for num in session_nums:
        session_key = f"session_{num}"
        date_key = f"session_{num}_date_time"
        turns = conversation.get(session_key, [])
        timestamp = conversation.get(date_key, "")

        if not turns:
            continue

        # Format messages from speaker A's perspective (A=user, B=assistant)
        messages_a = []
        messages_b = []
        for turn in turns:
            if turn["speaker"] == speaker_a:
                messages_a.append({"role": "user", "content": turn["text"]})
                messages_b.append({"role": "assistant", "content": turn["text"]})
            else:
                messages_a.append({"role": "assistant", "content": turn["text"]})
                messages_b.append({"role": "user", "content": turn["text"]})

        # Batch messages (mem0 uses batch_size=10)
        batch_size = config.batch_size
        metadata = {"timestamp": timestamp} if timestamp else {}

        for i in range(0, len(messages_a), batch_size):
            batch_a = messages_a[i:i + batch_size]
            batch_b = messages_b[i:i + batch_size]

            try:
                memory.add(batch_a, user_id=user_id_a, metadata=metadata)
            except Exception as e:
                print(f"      [WARN] Add failed for {speaker_a}: {str(e)[:50]}", flush=True)

            try:
                memory.add(batch_b, user_id=user_id_b, metadata=metadata)
            except Exception as e:
                print(f"      [WARN] Add failed for {speaker_b}: {str(e)[:50]}", flush=True)

            total_added += 2
            time.sleep(0.5)

    return total_added, speaker_a, speaker_b, user_id_a, user_id_b


def answer_question(config: BenchConfig, memory: MemorySystem, question: str,
                    speaker_a: str, speaker_b: str,
                    user_id_a: str, user_id_b: str) -> str:
    """
    Answer question using dual-speaker search, same as mem0.
    Search both speakers' memories, combine, let LLM answer.
    """
    # Search from both speakers
    results_a = memory.search(question, user_id=user_id_a, top_k=config.top_k)
    results_b = memory.search(question, user_id=user_id_b, top_k=config.top_k)

    memories_a = json.dumps([r["content"] for r in results_a], ensure_ascii=False)
    memories_b = json.dumps([r["content"] for r in results_b], ensure_ascii=False)

    prompt = ANSWER_PROMPT.format(
        speaker_1=speaker_a,
        speaker_2=speaker_b,
        speaker_1_memories=memories_a,
        speaker_2_memories=memories_b,
        question=question,
    )

    messages = [{"role": "user", "content": prompt}]
    return chat_completion(config, messages, max_tokens=50)


def run_benchmark(config: BenchConfig):
    data_path = Path(config.data_path)
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)

    with open(data_path) as f:
        conversations = json.load(f)

    if config.conversations is not None:
        conversations = [conversations[i] for i in config.conversations if i < len(conversations)]

    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = Path(config.output_dir) / f"run_{timestamp}.jsonl"
    summary_file = Path(config.output_dir) / f"summary_{timestamp}.json"

    print(f"=== MemMe LOCOMO Benchmark v2 (Fair) ===")
    print(f"Conversations: {len(conversations)}")
    print(f"Model: {config.chat_model}")
    print(f"Embedding: {config.embed_model} ({config.embed_dims}d)")
    print(f"Top-K: {config.top_k}")
    print(f"Protocol: dual-speaker, unified prompt, ≤5-6 words")
    print(f"Output: {results_file}")
    print()

    memory = MemorySystem(config)
    all_results = []

    for conv_idx, conv in enumerate(conversations):
        sample_id = conv.get("sample_id", f"conv_{conv_idx}")
        print(f"\n--- Conversation {conv_idx + 1}/{len(conversations)}: {sample_id} ---")

        memory.reset()

        # Phase 1: Ingest (dual-speaker, same as mem0)
        print("  Ingesting (dual-speaker)...", end="", flush=True)
        t0 = time.time()
        result = ingest_conversation(config, memory, conv)
        if result is None:
            print(" SKIPPED (no sessions)")
            continue
        total_added, speaker_a, speaker_b, uid_a, uid_b = result
        n_memories_a = len(memory.memories.get(uid_a, []))
        n_memories_b = len(memory.memories.get(uid_b, []))
        print(f" done in {time.time()-t0:.0f}s ({n_memories_a}+{n_memories_b} memories)")

        # Phase 2: Answer questions
        qa_pairs = conv.get("qa", [])
        # Filter categories (skip adversarial=5, same as mem0)
        qa_pairs = [qa for qa in qa_pairs if qa["category"] != 5]
        if config.categories is not None:
            qa_pairs = [qa for qa in qa_pairs if qa["category"] in config.categories]

        print(f"  Answering {len(qa_pairs)} questions...")

        for qi, qa in enumerate(qa_pairs):
            question = qa["question"]
            reference = str(qa["answer"])
            category = qa["category"]

            try:
                prediction = answer_question(
                    config, memory, question,
                    speaker_a, speaker_b, uid_a, uid_b
                )
            except Exception as e:
                print(f"    [WARN] Answer failed: {str(e)[:60]}", flush=True)
                prediction = "I don't know"

            f1 = compute_f1(prediction, reference)
            bleu1 = compute_bleu1(prediction, reference)

            judge_scores = []
            for _ in range(config.judge_runs):
                score = llm_judge(config, question, reference, prediction, category)
                judge_scores.append(score)
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

            with open(results_file, "a") as f:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

            if (qi + 1) % 10 == 0 or qi == len(qa_pairs) - 1:
                print(f"    [{qi+1}/{len(qa_pairs)}] F1={f1:.3f} B1={bleu1:.3f} J={judge_mean:.1f} | {prediction[:50]}")

            time.sleep(0.3)

    summary = generate_summary(all_results)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"\nResults: {results_file}")
    print(f"Summary: {summary_file}")


def generate_summary(results):
    by_category = defaultdict(list)
    for r in results:
        by_category[r.category_name].append(r)

    summary = {
        "total_questions": len(results),
        "protocol": "fair_v2_dual_speaker",
        "overall": {},
        "by_category": {},
        "baselines": {
            "mem0":  {"single-hop": 67.13, "multi-hop": 51.15, "temporal": 55.51, "open-domain": 72.93},
            "zep":   {"single-hop": 61.70, "multi-hop": 41.35, "temporal": 49.31, "open-domain": 76.60},
            "mem0g": {"single-hop": 65.71, "multi-hop": 47.19, "temporal": 58.13, "open-domain": 75.71},
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
    print("\n" + "=" * 80)
    print("  MemMe LOCOMO Benchmark v2 (Fair Protocol)")
    print("=" * 80)

    if summary.get("overall"):
        o = summary["overall"]
        print(f"\n  Overall ({summary['total_questions']} questions):")
        print(f"    F1:    {o['f1_mean']:.2f}")
        print(f"    BLEU1: {o['bleu1_mean']:.2f}")
        print(f"    Judge: {o['judge_mean']:.2f}")

    baselines = summary.get("baselines", {})
    print(f"\n  {'Category':<15} {'N':>5} {'F1':>8} {'B1':>8} {'Judge':>8} | {'Mem0':>8} {'Zep':>8} {'Mem0g':>8}")
    print("  " + "-" * 75)

    for cat in ['single-hop', 'multi-hop', 'temporal', 'open-domain']:
        scores = summary.get("by_category", {}).get(cat)
        if not scores: continue
        m0 = baselines.get("mem0", {}).get(cat, 0)
        zp = baselines.get("zep", {}).get(cat, 0)
        mg = baselines.get("mem0g", {}).get(cat, 0)
        j = scores["judge_mean"]
        print(f"  {cat:<15} {scores['count']:>5} {scores['f1_mean']:>8.2f} {scores['bleu1_mean']:>8.2f} {j:>8.2f} | {m0:>8.2f} {zp:>8.2f} {mg:>8.2f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="MemMe LOCOMO Benchmark v2 (Fair)")
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--chat-model", default="qwen3.5-plus")
    parser.add_argument("--embed-model", default="text-embedding-v3")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--judge-runs", type=int, default=1)
    parser.add_argument("--conversations", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--data-path", default="locomo10.json")
    parser.add_argument("--output-dir", default="results_fair")
    parser.add_argument("--results-only", type=str, default=None)

    args = parser.parse_args()

    if args.results_only:
        results = []
        with open(args.results_only) as f:
            for line in f:
                data = json.loads(line)
                results.append(QuestionResult(**data))
        summary = generate_summary(results)
        print_summary(summary)
        return

    config = BenchConfig(
        api_key=args.api_key,
        chat_base_url=args.base_url,
        embed_base_url=args.base_url,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
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
