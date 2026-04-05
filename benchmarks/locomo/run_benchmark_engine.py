#!/usr/bin/env python3
"""MemMe LOCOMO Benchmark — Engine Mode

Pipeline:
- Ingestion: append_events → compact → meditate (Rust engine)
- Search: MemMe Rust engine four-channel (vector + BM25 + entity spreading + RRF)
- Answer+Judge: async aiohttp
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import aiohttp
import memme


# ── Configuration ──

@dataclass
class BenchConfig:
    api_key: str  # DashScope key (for embedding)
    llm_api_key: str = ""  # key for LLM calls (engine extraction + answer + judge), defaults to api_key
    chat_base_url: str = ""   # for aiohttp (answer+judge); e.g. "https://api.openai.com/v1"
    llm_base_url: str = ""    # for Rust engine extraction (adds /v1/chat/completions); e.g. "https://api.openai.com"
    embed_base_url: str = ""  # for embedding; e.g. "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"  # answer model
    judge_model: str = "gpt-4o-mini"  # judge model (mem0 uses gpt-4o-mini for fair comparison)
    engine_llm_model: str = "gpt-4o-mini"  # engine extraction model
    embed_model: str = "text-embedding-v3"
    embed_dims: int = 1024
    top_k: int = 30
    data_path: str = "locomo10.json"
    output_dir: str = "results_engine"
    conversations: Optional[list] = None
    categories: Optional[list] = None
    judge_runs: int = 1
    # Concurrency settings
    max_llm_concurrent: int = 5   # async answer+judge
    # RRF weight tuning
    enable_forgetting_curve: bool = False  # disable for benchmark (all memories same age)
    rrf_vector_weight: float = 0.5
    rrf_fts_weight: float = 0.3
    rrf_entity_weight: float = 0.2
    rrf_k: int = 30
    rrf_temporal_weight: float = 0.15
    # Rerank settings
    rerank_api_key: str = ""
    rerank_base_url: str = ""
    rerank_model: str = ""


# ── Async API Client (for answer + judge only) ──

class AsyncAPIClient:
    """Async API client with rate limiting via semaphores."""

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
        url = self.config.chat_base_url
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
                        # Strip <think>...</think> blocks
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        return content
                except Exception as e:
                    self._stats["llm_errors"] += 1
                    if attempt < 4:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Chat API failed after 5 attempts: {e}")


# ── Scoring (same as v3_fast / mem0) ──

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


# ── Answer Prompt (strict factual, from v7_iter) ──

ANSWER_PROMPT = """You are answering a question based on the memories below.

RULES:
1. Use information from the memories as your primary source. You may also use well-known world knowledge to make reasonable inferences (e.g., "Tampa is in Florida", "Xenoblade Chronicles is a Nintendo Switch game").
2. For factual questions (what/where/who): Answer with specific details from the memories. Prefer exact names, places, dates.
3. For list questions ("what activities/books/events/items"): Scan ALL memories carefully. List EVERY matching item found, separated by commas. Do not omit any.
4. For "how many" / counting questions: Multiple memories may describe the SAME event in different words. First identify each DISTINCT event by its unique date or specific detail, then count. Do not equate the number of memory entries with the number of events.
5. For time questions ("when"): Look for specific dates, relative time references ("last Friday", "2 weeks ago"), or temporal context.
6. For inference questions ("would...?", "likely...?", "might...?"): Reason based on the person's known traits, values, and behaviors from the memories. Give a clear answer (e.g. "Likely yes/no") with brief reasoning.
7. For open-ended questions about preferences, opinions, or characteristics: Synthesize from all relevant memories to form a complete picture.
8. When memories conflict (e.g., "likes turtles" vs "is allergic to turtles"), prefer the more specific and restrictive fact. Allergies, medical conditions, and negative constraints override general positive preferences.
9. Think step-by-step for complex questions: briefly identify the relevant memories, reason over them, then state your answer.
10. Answer concisely — keep reasoning brief (1-2 sentences). Only say "Unknown" if the memories have absolutely no relevant information.

Speaker 1 ({speaker_1}) memories:
{speaker_1_memories}

Speaker 2 ({speaker_2}) memories:
{speaker_2_memories}

Question: {question}

Answer:"""


# ── Multi-Query Expansion (no LLM, local rules) ──

def _generate_query_variants(question: str) -> list:
    """Generate generic query variants for better recall — no answer leakage."""
    variants = [question]
    stop_words = {'what', 'when', 'where', 'who', 'how', 'why', 'which', 'does',
                  'did', 'has', 'have', 'is', 'are', 'was', 'were', 'do', 'the',
                  'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
                  'would', 'could', 'should', 'will', 'can', 'may', 'might',
                  'still', 'also', 'been', 'being', 'about', 'after', 'before',
                  'during', 'some', 'many', 'much', 'more', 'most', 'than',
                  'that', 'this', 'with', 'from', 'into', 'not', 'if', 'she',
                  'he', 'her', 'his', 'they', 'their', 'it', 'its'}
    words = question.rstrip('?').split()
    names = [w for w in words if len(w) > 1 and w[0].isupper() and w.lower() not in stop_words]
    key_terms = [w for w in words if w.lower() not in stop_words and not w[0].isupper() and len(w) > 2]

    # Variant 1: person + key terms
    if names and key_terms:
        variants.append(f"{' '.join(names)} {' '.join(key_terms[:4])}")

    # Variant 2: category-aware rephrasing (generic, no answer terms)
    q_lower = question.lower()
    if names:
        name = names[0]
        if 'activities' in q_lower or 'partake' in q_lower or 'hobbies' in q_lower:
            variants.append(f"{name} hobbies activities interests enjoys does")
        elif 'destress' in q_lower or 'relax' in q_lower:
            variants.append(f"{name} relax calm peace self-care hobby")
        elif 'book' in q_lower or 'read' in q_lower:
            variants.append(f"{name} book read title novel story")
        elif 'event' in q_lower or 'participated' in q_lower or 'attended' in q_lower:
            variants.append(f"{name} event attended participated joined organized")
        elif 'pet' in q_lower or 'animal' in q_lower:
            variants.append(f"{name} pet animal cat dog name")
        elif 'symbol' in q_lower:
            variants.append(f"{name} symbol meaning important special significant")
        elif 'paint' in q_lower or 'art' in q_lower:
            variants.append(f"{name} painted art artwork created made")
        elif 'pottery' in q_lower:
            variants.append(f"{name} pottery made created workshop clay")
        elif 'bought' in q_lower or 'item' in q_lower or 'purchase' in q_lower:
            variants.append(f"{name} bought purchased item gift acquired")
        elif 'move' in q_lower or 'from' in q_lower and 'where' in q_lower:
            variants.append(f"{name} moved from country origin hometown home")
        elif 'music' in q_lower or 'artist' in q_lower or 'band' in q_lower:
            variants.append(f"{name} music concert band artist seen heard")
        elif 'identity' in q_lower:
            variants.append(f"{name} identity who gender")
        elif 'career' in q_lower:
            variants.append(f"{name} career job work aspiration goal profession")
        elif 'family' in q_lower:
            variants.append(f"{name} family together kids children activity")
        elif 'support' in q_lower:
            variants.append(f"{name} support help encourage mentor friend")
        elif 'change' in q_lower or 'transition' in q_lower:
            variants.append(f"{name} change transition journey experience challenge")
        elif 'children' in q_lower or 'many' in q_lower:
            variants.append(f"{name} children kids family son daughter")
        elif 'relationship' in q_lower:
            variants.append(f"{name} relationship status partner")
        else:
            if key_terms:
                variants.append(f"{name} {' '.join(key_terms[:5])}")

    return variants[:3]


# ── Benchmark Data Structures ──

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


# ── Engine Ingestion (sync) ──

def ingest_conversation_engine(config: BenchConfig, conv: dict):
    """Ingest a conversation using the full MemMe pipeline:
    append_events → compact → meditate.

    Returns (store, speaker_a, speaker_b, uid_a, uid_b) or None.
    """
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
    user_id_a = f"{sample_id}_{speaker_a}"
    user_id_b = f"{sample_id}_{speaker_b}"

    # Use file-based db so we can reuse ingested data
    cache_dir = os.environ.get("MEMME_CACHE_DIR", "cache")
    db_path = f"{cache_dir}/{sample_id}.duckdb"
    os.makedirs(cache_dir, exist_ok=True)
    skip_ingest = os.path.exists(db_path) and getattr(config, 'reuse_cache', False)

    store = memme.MemoryStore(
        db_path=db_path,
        embedder="openai",
        api_key=config.api_key,
        base_url=config.embed_base_url,
        embed_model=config.embed_model,
        dims=config.embed_dims,
        llm_api_key=config.llm_api_key or config.api_key,
        llm_model=config.engine_llm_model,
        llm_base_url=config.llm_base_url,
        enable_forgetting_curve=config.enable_forgetting_curve,
        rrf_vector_weight=config.rrf_vector_weight,
        rrf_fts_weight=config.rrf_fts_weight,
        rrf_entity_weight=config.rrf_entity_weight,
        rrf_k=config.rrf_k,
        rrf_temporal_weight=config.rrf_temporal_weight,
        rerank_api_key=config.rerank_api_key or None,
        rerank_base_url=config.rerank_base_url or None,
        rerank_model=config.rerank_model or None,
    )

    if skip_ingest:
        print(f"  Using cached db: {db_path}", flush=True)
        return store, speaker_a, speaker_b, user_id_a, user_id_b

    # Full pipeline: append_events → compact → meditate
    # Each LoCoMo session maps to one MemMe session per user.
    total_sessions = len(session_nums) * 2  # x2 for both speakers
    done_sessions = 0

    for num in session_nums:
        session_key = f"session_{num}"
        turns = conversation.get(session_key, [])
        if not turns:
            continue

        # Get session date for temporal grounding
        # Normalize incomplete dates: "2023-06" → "2023-06-01", "2023" → "2023-01-01"
        date_key = f"session_{num}_date_time"
        session_date = conversation.get(date_key, "")
        if session_date:
            import re
            if re.match(r'^\d{4}$', session_date.strip()):
                session_date = session_date.strip() + "-01-01"
            elif re.match(r'^\d{4}-\d{2}$', session_date.strip()):
                session_date = session_date.strip() + "-01"

        # Convert turns to ChatMessage format
        # Use "user" role for all speakers; include speaker name in content
        # so compact's purification can resolve coreference properly.
        date_prefix = f"[Date: {session_date}] " if session_date else ""
        messages = []
        for turn in turns:
            messages.append(("user", f"{date_prefix}{turn['speaker']}: {turn['text']}"))

        # Ingest for both speakers (each has their own memory)
        for uid in [user_id_a, user_id_b]:
            session_id = f"{sample_id}_s{num}_{uid}"
            try:
                # 1. Append events
                store.append_events(messages, session_id=session_id, user_id=uid)

                # 2. Compact → create episode
                store.compact(session_id)
            except Exception as e:
                print(f"      [WARN] session {session_id} failed: {str(e)[:200]}", flush=True)

            done_sessions += 1
            if done_sessions % 10 == 0 or done_sessions == total_sessions:
                print(f"      [{done_sessions}/{total_sessions}] sessions processed", flush=True)

    # 3. Meditate — reconcile facts for each user
    for uid in [user_id_a, user_id_b]:
        try:
            record = store.meditate(user_id=uid, triggered_by="benchmark")
            created = record.get("memories_created", 0)
            updated = record.get("memories_updated", 0)
            deleted = record.get("conflicts_found", 0)
            print(f"      Meditate {uid}: +{created} ~{updated} -{deleted}", flush=True)
        except Exception as e:
            print(f"      [WARN] meditate failed for {uid}: {str(e)[:80]}", flush=True)

    # Build FTS index after all processing
    store.rebuild_fts_index()

    return store, speaker_a, speaker_b, user_id_a, user_id_b


# ── Answer + Judge (async) ──

async def answer_and_judge_question(
    config: BenchConfig, client: AsyncAPIClient,
    qa: dict, sample_id: str,
    speaker_a: str, speaker_b: str, uid_a: str, uid_b: str,
    pre_searched: dict,
) -> QuestionResult:
    """Answer one question using engine search, then judge — can run concurrently."""
    question = qa["question"]
    reference = str(qa.get("answer", qa.get("adversarial_answer", "")))
    category = qa["category"]

    try:
        # Use pre-computed search results (search is sync, done before async Q&A)
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

    # Judge
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
        sample_id=sample_id,
        question=question,
        reference=reference,
        prediction=prediction,
        category=category,
        category_name=CATEGORY_NAMES.get(category, "unknown"),
        f1=f1, bleu1=bleu1,
        judge_scores=judge_scores,
        judge_mean=judge_mean,
    )


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

    print(f"=== MemMe LOCOMO Benchmark — Engine Mode ===")
    print(f"Conversations: {len(conversations)}")
    print(f"Answer Model: {config.chat_model}")
    print(f"Judge Model: {config.judge_model}")
    print(f"Engine LLM: {config.engine_llm_model} (extraction)")
    print(f"Embedding: {config.embed_model} ({config.embed_dims}d)")
    print(f"Top-K: {config.top_k}")
    print(f"Retrieval: vector + BM25 + entity spreading activation + RRF (engine)")
    print(f"RRF weights: vector={config.rrf_vector_weight} fts={config.rrf_fts_weight} entity={config.rrf_entity_weight} k={config.rrf_k}")
    print(f"Forgetting curve: {config.enable_forgetting_curve}")
    print(f"Concurrency: LLM={config.max_llm_concurrent}")
    print(f"Output: {results_file}")
    print()

    all_results = []

    async with AsyncAPIClient(config) as client:
        for conv_idx, conv in enumerate(conversations):
            sample_id = conv.get("sample_id", f"conv_{conv_idx}")
            print(f"\n--- Conversation {conv_idx + 1}/{len(conversations)}: {sample_id} ---")

            # Phase 1: Ingest (append_events → compact → meditate)
            print("  Ingesting (append → compact → meditate)...", end="", flush=True)
            t0 = time.time()
            result = ingest_conversation_engine(config, conv)
            if result is None:
                print(" SKIPPED")
                continue
            store, speaker_a, speaker_b, uid_a, uid_b = result
            ingest_time = time.time() - t0
            print(f" done in {ingest_time:.0f}s")

            # Phase 2: Answer questions (async for speed)
            qa_pairs = list(conv.get("qa", []))
            if config.categories is not None:
                qa_pairs = [qa for qa in qa_pairs if qa["category"] in config.categories]

            # Phase 2.5: Pre-search all questions (sync — engine search is sync)
            print(f"  Searching {len(qa_pairs)} questions (engine four-channel)...", end="", flush=True)
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
            search_time = time.time() - t_search
            print(f" done in {search_time:.0f}s", flush=True)

            # Phase 3: Answer + Judge (async for speed)
            print(f"  Answering {len(qa_pairs)} questions (async)...", flush=True)
            t1 = time.time()

            batch_size = config.max_llm_concurrent
            for batch_start in range(0, len(qa_pairs), batch_size):
                batch = qa_pairs[batch_start:batch_start + batch_size]
                tasks = [
                    answer_and_judge_question(
                        config, client, qa, sample_id,
                        speaker_a, speaker_b, uid_a, uid_b,
                        pre_searched,
                    )
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
                    print(f"    [{done}/{len(qa_pairs)}] F1={last.f1:.3f} J={last.judge_mean:.1f} | {last.prediction[:50]}")

            answer_time = time.time() - t1
            print(f"  Questions done in {answer_time:.0f}s")

            # Print intermediate scores after each conversation
            interim = generate_summary(all_results)
            if interim.get("overall"):
                o = interim["overall"]
                cats = interim.get("by_category", {})
                parts = []
                for cat in ['single-hop', 'multi-hop', 'temporal', 'open-domain']:
                    if cat in cats:
                        parts.append(f"{cat}={cats[cat]['judge_mean']:.1f}")
                print(f"  >>> Cumulative ({len(all_results)} Q): Judge={o['judge_mean']:.1f}% | {' | '.join(parts)}", flush=True)

        print(f"\n  API stats: {client.stats}")

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
        "protocol": "engine_mode",
        "overall": {},
        "by_category": {},
        "baselines": {
            "mem0":  {"single-hop": 67.13, "multi-hop": 51.15, "temporal": 55.51, "open-domain": 72.93},
            "zep":   {"single-hop": 61.70, "multi-hop": 41.35, "temporal": 49.31, "open-domain": 76.60},
            "mem0g": {"single-hop": 65.71, "multi-hop": 47.19, "temporal": 58.13, "open-domain": 75.71},
        },
        "v2_scores": {
            "single-hop": 15.62, "multi-hop": 70.27, "temporal": 15.38, "open-domain": 50.00,
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
    print("  MemMe LOCOMO Benchmark — Engine Mode")
    print("  Retrieval: vector + BM25 + entity spreading activation + RRF")
    print("=" * 90)

    if summary.get("overall"):
        o = summary["overall"]
        print(f"\n  Overall ({summary['total_questions']} questions):")
        print(f"    F1:    {o['f1_mean']:.2f}")
        print(f"    BLEU1: {o['bleu1_mean']:.2f}")
        print(f"    Judge: {o['judge_mean']:.2f}")

    baselines = summary.get("baselines", {})
    v2 = summary.get("v2_scores", {})
    print(f"\n  {'Category':<15} {'N':>5} {'F1':>8} {'B1':>8} {'Judge':>8} | {'v2':>8} {'Mem0':>8} {'Zep':>8} {'Mem0g':>8}")
    print("  " + "-" * 85)

    for cat in ['single-hop', 'multi-hop', 'temporal', 'open-domain']:
        scores = summary.get("by_category", {}).get(cat)
        if not scores: continue
        m0 = baselines.get("mem0", {}).get(cat, 0)
        zp = baselines.get("zep", {}).get(cat, 0)
        mg = baselines.get("mem0g", {}).get(cat, 0)
        v2s = v2.get(cat, 0)
        j = scores["judge_mean"]
        print(f"  {cat:<15} {scores['count']:>5} {scores['f1_mean']:>8.2f} {scores['bleu1_mean']:>8.2f} {j:>8.2f} | {v2s:>8.2f} {m0:>8.2f} {zp:>8.2f} {mg:>8.2f}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="MemMe LOCOMO Benchmark — Engine Mode")
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""), help="DashScope API key (embedding)")
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="LLM API key (extraction + answer + judge)")
    parser.add_argument("--base-url", default=os.environ.get("EMBED_BASE_URL", ""), help="Embed base URL")
    parser.add_argument("--chat-base-url", default=os.environ.get("LLM_BASE_URL", ""), help="Chat base URL for LLM calls")
    parser.add_argument("--llm-base-url", default=None, help="LLM base URL for Rust engine (without /v1)")
    parser.add_argument("--chat-model", default="gpt-4o-mini", help="Model for answer")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Model for judge (mem0 uses gpt-4o-mini)")
    parser.add_argument("--engine-llm-model", default="gpt-4o-mini", help="Model for engine LLM extraction")
    parser.add_argument("--embed-model", default="text-embedding-v3")
    parser.add_argument("--embed-dims", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--judge-runs", type=int, default=1)
    parser.add_argument("--conversations", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--data-path", default="locomo10.json")
    parser.add_argument("--output-dir", default="results_engine")
    parser.add_argument("--max-llm-concurrent", type=int, default=5)
    parser.add_argument("--results-only", type=str, default=None)
    parser.add_argument("--reuse-cache", action="store_true", help="Skip ingestion, reuse cached .duckdb files")
    parser.add_argument("--enable-forgetting-curve", action="store_true", default=False)
    parser.add_argument("--rrf-vector-weight", type=float, default=0.5)
    parser.add_argument("--rrf-fts-weight", type=float, default=0.3)
    parser.add_argument("--rrf-entity-weight", type=float, default=0.2)
    parser.add_argument("--rrf-k", type=int, default=30)
    parser.add_argument("--rrf-temporal-weight", type=float, default=0.15)
    parser.add_argument("--rerank-api-key", type=str, default="", help="API key for rerank service (Jina/Cohere)")
    parser.add_argument("--rerank-base-url", type=str, default="", help="Base URL for rerank API (e.g. https://api.jina.ai)")
    parser.add_argument("--rerank-model", type=str, default="", help="Rerank model name")

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

    llm_base = args.llm_base_url or args.chat_base_url

    config = BenchConfig(
        api_key=args.api_key,
        llm_api_key=args.llm_api_key,
        chat_base_url=args.chat_base_url,
        llm_base_url=llm_base,
        embed_base_url=args.base_url,
        chat_model=args.chat_model,
        judge_model=args.judge_model,
        engine_llm_model=args.engine_llm_model,
        embed_model=args.embed_model,
        embed_dims=args.embed_dims,
        top_k=args.top_k,
        judge_runs=args.judge_runs,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_llm_concurrent=args.max_llm_concurrent,
        conversations=[int(x) for x in args.conversations.split(",")] if args.conversations else None,
        categories=[int(x) for x in args.categories.split(",")] if args.categories else None,
        enable_forgetting_curve=args.enable_forgetting_curve,
        rrf_vector_weight=args.rrf_vector_weight,
        rrf_fts_weight=args.rrf_fts_weight,
        rrf_entity_weight=args.rrf_entity_weight,
        rrf_k=args.rrf_k,
        rrf_temporal_weight=args.rrf_temporal_weight,
        rerank_api_key=args.rerank_api_key,
        rerank_base_url=args.rerank_base_url,
        rerank_model=args.rerank_model,
    )
    config.reuse_cache = args.reuse_cache
    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()
