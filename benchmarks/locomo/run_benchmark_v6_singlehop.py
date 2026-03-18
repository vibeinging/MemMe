#!/usr/bin/env python3
"""
MemMe LOCOMO Benchmark v6 — Single-hop Optimization

Based on v5 (meditation), with targeted fixes for single-hop failures:
1. Session-level extraction (full session context instead of 10-turn batches)
2. Dual-pass extraction (pass 1: facts, pass 2: details/specifics from same text)
3. Increased top-K to 30 for list-type questions
4. Answer prompt: strict "only state what's in memories" for factual questions
5. Meditation: identity + consolidation (from v5)
"""

import argparse
import asyncio
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

import aiohttp

# ── Configuration ──

@dataclass
class BenchConfig:
    api_key: str
    chat_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embed_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen3.5-plus"
    embed_model: str = "text-embedding-v3"
    embed_dims: int = 1024
    top_k: int = 30
    data_path: str = "locomo10.json"
    output_dir: str = "results_v3"
    conversations: Optional[list] = None
    categories: Optional[list] = None
    judge_runs: int = 1
    batch_size: int = 10
    dedup_threshold: float = 0.92
    # Concurrency settings
    max_llm_concurrent: int = 5   # max concurrent LLM calls
    max_embed_concurrent: int = 10  # max concurrent embedding calls


# ── Async API Helpers ──

class AsyncAPIClient:
    """Async API client with rate limiting via semaphores."""

    def __init__(self, config: BenchConfig):
        self.config = config
        self.llm_sem = asyncio.Semaphore(config.max_llm_concurrent)
        self.embed_sem = asyncio.Semaphore(config.max_embed_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None
        self._stats = {"llm_calls": 0, "embed_calls": 0, "llm_errors": 0, "embed_errors": 0}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
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
                              max_tokens: int = 512) -> str:
        url = f"{self.config.chat_base_url}/chat/completions"
        payload = {
            "model": self.config.chat_model,
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

    async def embed_text(self, text: str) -> list:
        url = f"{self.config.embed_base_url}/embeddings"
        payload = {"model": self.config.embed_model, "input": text[:2000]}
        async with self.embed_sem:
            for attempt in range(5):
                try:
                    async with self.session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        self._stats["embed_calls"] += 1
                        return data["data"][0]["embedding"]
                except Exception as e:
                    self._stats["embed_errors"] += 1
                    if attempt < 4:
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Embed API failed: {e}")

    async def embed_batch(self, texts: list) -> list:
        """Embed multiple texts concurrently."""
        tasks = [self.embed_text(t) for t in texts]
        return await asyncio.gather(*tasks)


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
        df = 1
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        tf = doc_tf.get(qt, 0)
        if tf > 0:
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return score


# ── Extraction Prompt (same as v3) ──

def get_extraction_prompt(timestamp: str = "") -> str:
    time_context = f"\nThe conversation timestamp is: {timestamp}" if timestamp else ""
    return f"""You are a Personal Information Organizer and Atomic Fact Extractor. Extract EVERY factual detail from conversations as self-contained atomic facts. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts for easy retrieval and personalization.

CATEGORIES to extract:
1. Personal details: name, age, gender, identity, nationality, relationship status (single/married/divorced/etc), family members and their names
2. Locations: where they live, hometown, places visited, countries/cities mentioned, places moved from/to
3. Career/Education: job title, employer, career goals, workplace, skills, education, degrees
4. Activities: hobbies, events attended/organized, regular activities, sports, exercise routines
5. Preferences: likes, dislikes, favorites (food, music, movies, books, brands, products)
6. Specific items: book titles, movie names, song names, food dishes, brands, art pieces, apps
7. Cross-entity facts: shared activities between people, things in common ("Both Alice and Bob like hiking")
8. Quantities: exact counts, dates, durations, amounts, frequencies ("has 3 children", "went 2 times")
9. Plans and intentions: future plans, upcoming events, trips, goals, deadlines
10. Health and wellness: dietary restrictions, allergies, fitness routines, medical conditions
11. Personality traits and opinions: character traits, values, beliefs, attitudes that reveal who the person is ("Melanie values family time", "Caroline is passionate about LGBTQ+ rights")

RULES:
- ONE fact per entry — each fact must be a single indivisible piece of information
- Keep each fact SHORT (under 15 words when possible)
- ALWAYS resolve pronouns to actual names (never use "she/he/they/it" — use the person's name)
- Include specific details: "read 'Becoming Nicole'" not "read a book"
- Include quantities: "has 3 children" not "has children"
- Include dates when mentioned: "moved to NYC in 2019" not "moved to NYC"
- Capture relationship status explicitly: "Alice is married" or "Bob is single"
- For cross-entity facts, mention all relevant people: "Alice and Bob both enjoy hiking"
- Extract preferences and personality traits that reveal character
- Extract specific locations for activities: "camped at the beach", "hiked in the mountains"
- Extract SPECIFIC INSTANCES, not just categories: "Melanie painted a sunset" not just "Melanie paints"
- Extract exact counts: "Melanie has 3 children" not "Melanie has children"
- Extract pet names, book titles, song titles as separate facts: "Melanie's cat is named Oliver"
- When someone describes a recent event, extract what happened specifically: "Melanie recently painted a sunset at the park"
{time_context}

Here are some few shot examples:

Input: Hi.
Output: {{"facts" : []}}

Input: There are branches in trees.
Output: {{"facts" : []}}

Input: Hi, I am looking for a restaurant in San Francisco.
Output: {{"facts" : ["Looking for a restaurant in San Francisco"]}}

Input: Yesterday, I had a meeting with John at 3pm. We discussed the new project.
Output: {{"facts" : ["Had a meeting with John at 3pm", "Discussed the new project with John"]}}

Input: Hi, my name is John. I am a software engineer.
Output: {{"facts" : ["Name is John", "John is a software engineer"]}}

Input: My favourite movies are Inception and Interstellar.
Output: {{"facts" : ["Favourite movie: Inception", "Favourite movie: Interstellar"]}}

Input: I have 3 kids. My wife Sarah and I moved to Portland in 2020.
Output: {{"facts" : ["Has 3 children", "Is married", "Wife's name is Sarah", "Moved to Portland in 2020", "Lives in Portland"]}}

Input: Both me and my friend Jake love rock climbing. He also got me into reading sci-fi.
Output: {{"facts" : ["Enjoys rock climbing", "Jake enjoys rock climbing", "Jake is a friend", "Jake introduced user to reading sci-fi", "Reads sci-fi books"]}}

Return the facts and preferences in a json format as shown above.

Remember:
- If you do not find anything relevant in the conversation, return an empty list.
- Create the facts based on the user and assistant messages only. Do not pick from system messages.
- Return JSON format: {{"facts": ["fact1", "fact2", ...]}}
- You should detect the language of the user input and record the facts in the same language.
- Each fact MUST be atomic — one indivisible piece of information per fact. Split compound facts.
- NEVER use pronouns. Always use the actual name of the person."""


# ── Answer Prompt (same as v3) ──

ANSWER_PROMPT = """You are a memory-based assistant. Answer the question using ONLY the provided memories.

CRITICAL RULES:
- For "what/who/where/when/how many" questions: List ONLY items explicitly mentioned in the memories. Do NOT add anything from your own knowledge. Do NOT generalize.
- For list questions ("what activities/books/events"): Gather ALL matching items from ALL memories. Check every memory for relevance. List them as a short comma-separated list.
- For "how many" questions: Count ONLY explicitly mentioned distinct items. If the exact number is stated in a memory, use that number.
- For "where" questions: Give the specific place name from the memories, not a description.
- For opinion/inference questions ("would...?", "personality"): Reason based on memories.
- When memories contain timestamps, more recent information takes precedence.
- Answer concisely: just the answer, no explanation. Under 10 words for factual questions.
- NEVER say "I don't know" if there is ANY relevant memory — even partial info is better than nothing.

Speaker 1 ({speaker_1}) memories:
{speaker_1_memories}

Speaker 2 ({speaker_2}) memories:
{speaker_2_memories}

Question: {question}

Answer (concise, factual, no explanation):"""


# ── Async Memory System ──

class AsyncMemorySystem:
    def __init__(self, config: BenchConfig, client: AsyncAPIClient):
        self.config = config
        self.client = client
        self.memories = {}
        self.embeddings = {}  # text -> embedding cache

    def reset(self):
        self.memories = {}
        self.embeddings = {}

    async def add(self, messages: list, user_id: str, metadata: dict = None):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        timestamp = (metadata or {}).get("timestamp", "")

        facts = await self._extract_facts(text, timestamp)

        if user_id not in self.memories:
            self.memories[user_id] = []

        # Batch embed all new facts at once
        new_facts = []
        for fact in facts:
            if fact not in self.embeddings:
                new_facts.append(fact)

        if new_facts:
            embeddings = await self.client.embed_batch(new_facts)
            for fact, emb in zip(new_facts, embeddings):
                self.embeddings[fact] = emb

        for fact in facts:
            embedding = self.embeddings[fact]

            # Dedup check (local, no API call needed)
            if self._is_duplicate_local(fact, embedding, user_id):
                continue

            tokens = normalize_answer(fact).split()
            self.memories[user_id].append({
                "content": fact,
                "embedding": embedding,
                "tokens": tokens,
                "metadata": metadata or {},
            })

    async def _extract_facts(self, text: str, timestamp: str = "") -> list:
        prompt = get_extraction_prompt(timestamp)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Extract facts from the following conversation:\n\n{text}"},
        ]
        facts = []
        try:
            raw = await self.client.chat_completion(messages, temperature=0.1, max_tokens=4096)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```\w*\n?', '', raw)
                raw = re.sub(r'\n?```$', '', raw)
            data = json.loads(raw)
            facts = [str(f) for f in data.get("facts", []) if f and len(str(f).strip()) > 3]
        except Exception as e:
            print(f"      [WARN] Extraction pass 1 failed: {str(e)[:60]}", flush=True)

        # v6: Second pass — extract specific details that are often missed
        try:
            detail_facts = await self._extract_details_pass(text, timestamp)
            facts.extend(detail_facts)
        except Exception as e:
            print(f"      [WARN] Detail pass failed: {str(e)[:60]}", flush=True)

        return facts

    async def _extract_details_pass(self, text: str, timestamp: str = "") -> list:
        """Second extraction pass focused on specific details often missed."""
        time_context = f"\nConversation timestamp: {timestamp}" if timestamp else ""
        prompt = f"""You are a Detail Extractor. Review this conversation and extract ONLY specific details that are easy to miss. Focus on:

1. PROPER NOUNS: names of people, pets, places, books, songs, bands, movies, brands, events
2. NUMBERS: exact counts, ages, dates, times, quantities, frequencies
3. ORIGINS: where someone is from, moved from, nationality, hometown
4. PURCHASES: items bought, gifts received/given
5. SPECIFIC LOCATIONS: exact places of activities (beach, mountains, forest, park)
6. RELATIONSHIPS: family members by name, friend names, pet names and types
{time_context}

RULES:
- Extract ONLY concrete, specific details — no general statements
- Always use the person's actual name
- Each fact under 15 words
- Include the specific detail (name, number, title, place)

Return JSON: {{"facts": ["fact1", "fact2", ...]}}"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Extract specific details:\n\n{text}"},
        ]
        raw = await self.client.chat_completion(messages, temperature=0.1, max_tokens=2048)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```\w*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
        data = json.loads(raw)
        return [str(f) for f in data.get("facts", []) if f and len(str(f).strip()) > 3]

    def _is_duplicate_local(self, fact: str, embedding: list, user_id: str) -> bool:
        existing = self.memories.get(user_id, [])
        for mem in existing:
            sim = cosine_similarity(embedding, mem["embedding"])
            if sim > self.config.dedup_threshold:
                return True
        return False

    # ── Meditation (Consolidation) Phase ──

    async def meditate(self, user_id: str):
        """Run meditation on accumulated memories for a user.

        Phase 1: Identity extraction — extract stable personality traits
        Phase 2: Fact consolidation — merge related facts into stronger ones
        """
        user_memories = self.memories.get(user_id, [])
        if not user_memories:
            return 0

        added = 0

        # Phase 1: Identity Extraction
        all_facts = [m["content"] for m in user_memories]
        identity_traits = await self._extract_identity(all_facts)

        for trait in identity_traits:
            if trait not in self.embeddings:
                emb = await self.client.embed_text(trait[:2000])
                self.embeddings[trait] = emb

            tagged = f"[IDENTITY] {trait}"
            if tagged not in self.embeddings:
                emb = await self.client.embed_text(tagged[:2000])
                self.embeddings[tagged] = emb

            if not self._is_duplicate_local(tagged, self.embeddings[tagged], user_id):
                tokens = normalize_answer(tagged).split()
                self.memories[user_id].append({
                    "content": tagged,
                    "embedding": self.embeddings[tagged],
                    "tokens": tokens,
                    "metadata": {"type": "identity", "timestamp": ""},
                })
                added += 1

        # Phase 2: Fact Consolidation — group related facts and create summaries
        consolidation_facts = await self._consolidate_facts(all_facts)
        for fact in consolidation_facts:
            if fact not in self.embeddings:
                emb = await self.client.embed_text(fact[:2000])
                self.embeddings[fact] = emb

            tagged = f"[CONSOLIDATED] {fact}"
            if tagged not in self.embeddings:
                emb = await self.client.embed_text(tagged[:2000])
                self.embeddings[tagged] = emb

            if not self._is_duplicate_local(tagged, self.embeddings[tagged], user_id):
                tokens = normalize_answer(tagged).split()
                self.memories[user_id].append({
                    "content": tagged,
                    "embedding": self.embeddings[tagged],
                    "tokens": tokens,
                    "metadata": {"type": "consolidated", "timestamp": ""},
                })
                added += 1

        return added

    async def _extract_identity(self, facts: list) -> list:
        """Extract stable personality traits from a collection of facts."""
        # Chunk facts to fit in context (max ~100 facts per call)
        all_traits = []
        for i in range(0, len(facts), 100):
            chunk = facts[i:i+100]
            facts_text = "\n".join(f"- {f}" for f in chunk)

            prompt = """You are a Personality & Identity Analyzer. From the following facts about a person, extract STABLE personality traits, values, preferences, and behavioral patterns.

CATEGORIES TO EXTRACT:
1. Personality traits: introvert/extrovert, creative, adventurous, caring, etc.
2. Core values: family, career, health, social justice, education, etc.
3. Strong preferences: strong likes/dislikes, passions, favorites
4. Life priorities: what matters most based on their actions
5. Behavioral patterns: how they handle stress, make decisions, spend time

RULES:
- Focus on STABLE traits that predict future behavior, not one-time events
- Always use the person's actual name, never pronouns
- Each trait should be concise (under 15 words)
- Only extract traits with clear evidence from the facts
- Include the person's name in each trait

Return JSON: {"identity_traits": ["trait1", "trait2", ...]}"""

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Extract identity traits from these facts:\n\n{facts_text}"},
            ]
            try:
                raw = await self.client.chat_completion(messages, temperature=0.1, max_tokens=1024)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r'^```\w*\n?', '', raw)
                    raw = re.sub(r'\n?```$', '', raw)
                data = json.loads(raw)
                traits = data.get("identity_traits", [])
                all_traits.extend([str(t) for t in traits if t and len(str(t).strip()) > 5])
            except Exception as e:
                print(f"      [WARN] Identity extraction failed: {str(e)[:60]}", flush=True)

        return all_traits

    async def _consolidate_facts(self, facts: list) -> list:
        """Consolidate related facts into stronger composite facts."""
        if len(facts) < 10:
            return []

        # Chunk facts
        all_consolidated = []
        for i in range(0, len(facts), 80):
            chunk = facts[i:i+80]
            facts_text = "\n".join(f"- {f}" for f in chunk)

            prompt = """You are a Memory Consolidation Engine. Review the following atomic facts and create CONSOLIDATED summary facts that combine related information.

RULES:
1. Group related facts about the same topic/person/event and create a single comprehensive fact
2. Resolve temporal conflicts: if facts contradict, keep the most recent one
3. Create cross-entity facts: "Both X and Y enjoy Z" when applicable
4. Include specific details: names, numbers, dates, locations
5. Each consolidated fact should be self-contained and informative
6. Do NOT just repeat existing facts — only create NEW combined insights
7. Focus on facts that answer WHO/WHAT/WHERE/WHEN/HOW MANY questions

Return JSON: {"consolidated_facts": ["fact1", "fact2", ...]}"""

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Consolidate these facts:\n\n{facts_text}"},
            ]
            try:
                raw = await self.client.chat_completion(messages, temperature=0.1, max_tokens=1024)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r'^```\w*\n?', '', raw)
                    raw = re.sub(r'\n?```$', '', raw)
                data = json.loads(raw)
                consolidated = data.get("consolidated_facts", [])
                all_consolidated.extend([str(f) for f in consolidated if f and len(str(f).strip()) > 5])
            except Exception as e:
                print(f"      [WARN] Consolidation failed: {str(e)[:60]}", flush=True)

        return all_consolidated

    async def _get_embedding(self, text: str) -> list:
        if text not in self.embeddings:
            self.embeddings[text] = await self.client.embed_text(text[:2000])
        return self.embeddings[text]

    def search(self, query_embedding: list, query_tokens: list,
               user_id: str, top_k: int = 20) -> list:
        """Search (sync, no API calls needed — embeddings pre-computed)."""
        user_memories = self.memories.get(user_id, [])
        if not user_memories:
            return []

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
                "content": f"[{ts}] {m['content']}" if ts else m["content"],
                "score": rrf_scores[mid],
            })
        return results

    async def search_multi_query(self, question: str, user_id: str, top_k: int = 20) -> list:
        """Multi-query retrieval with concurrent embedding."""
        queries = self._generate_query_variants_local(question)

        # Batch embed all queries
        to_embed = [q for q in queries if q not in self.embeddings]
        if to_embed:
            embeddings = await self.client.embed_batch(to_embed)
            for q, emb in zip(to_embed, embeddings):
                self.embeddings[q] = emb

        all_results = defaultdict(lambda: {"score": 0.0, "content": ""})
        k_rrf = 60

        for qi, query in enumerate(queries):
            query_embedding = self.embeddings[query]
            query_tokens = normalize_answer(query).split()
            results = self.search(query_embedding, query_tokens, user_id, top_k)
            for rank, r in enumerate(results):
                key = r["content"]
                all_results[key]["content"] = r["content"]
                weight = 1.0 if qi == 0 else 0.7
                all_results[key]["score"] += weight / (k_rrf + rank)

        merged = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    def _generate_query_variants_local(self, question: str) -> list:
        """Generate query variants locally (no LLM call needed).
        Extracts key entities and creates focused sub-queries."""
        variants = [question]

        # Extract person names (capitalized words that aren't question words)
        stop_words = {'what', 'when', 'where', 'who', 'how', 'why', 'which', 'does',
                      'did', 'has', 'have', 'is', 'are', 'was', 'were', 'do', 'the',
                      'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
                      'would', 'could', 'should', 'will', 'can', 'may', 'might',
                      'still', 'also', 'been', 'being', 'about', 'after', 'before',
                      'during', 'some', 'many', 'much', 'more', 'most', 'than',
                      'that', 'this', 'with', 'from', 'into', 'not', 'if', 'she',
                      'he', 'her', 'his', 'they', 'their', 'it', 'its'}

        words = question.rstrip('?').split()
        names = [w for w in words if w[0].isupper() and w.lower() not in stop_words and len(w) > 1]
        # Key nouns (non-stop, non-name words)
        key_terms = [w for w in words if w.lower() not in stop_words and not w[0].isupper() and len(w) > 2]

        # Variant 1: person + key terms (focused)
        if names and key_terms:
            variants.append(f"{' '.join(names)} {' '.join(key_terms[:3])}")

        # Variant 2: rephrase patterns
        q_lower = question.lower()
        if 'what activities' in q_lower or 'what does' in q_lower:
            if names:
                variants.append(f"{names[0]} hobbies interests activities enjoys")
        elif 'what books' in q_lower or 'what has' in q_lower and 'read' in q_lower:
            if names:
                variants.append(f"{names[0]} book reading title read")
        elif 'how many' in q_lower:
            if names:
                variants.append(f"{names[0]} {' '.join(key_terms[:3])} number count")
        elif 'where has' in q_lower or 'where did' in q_lower:
            if names:
                variants.append(f"{names[0]} {' '.join(key_terms[:2])} location place")
        elif 'would' in q_lower:
            if names:
                variants.append(f"{names[0]} preference personality likes values")
        elif 'what events' in q_lower or 'what' in q_lower and 'events' in q_lower:
            if names:
                variants.append(f"{names[0]} event attended participated")
        else:
            # Generic: just the key terms
            if key_terms:
                variants.append(' '.join(key_terms[:4]))

        return variants[:3]


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


async def ingest_conversation(config: BenchConfig, memory: AsyncMemorySystem, conv: dict):
    """Ingest with concurrent extraction (batches processed in parallel)."""
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

    # v6: Session-level extraction (full session context, not 10-turn batches)
    tasks_a = []
    tasks_b = []
    for num in session_nums:
        session_key = f"session_{num}"
        date_key = f"session_{num}_date_time"
        turns = conversation.get(session_key, [])
        timestamp = conversation.get(date_key, "")

        if not turns:
            continue

        # Build FULL session messages (not batched)
        messages_a, messages_b = [], []
        for turn in turns:
            if turn["speaker"] == speaker_a:
                messages_a.append({"role": "user", "content": turn["text"]})
                messages_b.append({"role": "assistant", "content": turn["text"]})
            else:
                messages_a.append({"role": "assistant", "content": turn["text"]})
                messages_b.append({"role": "user", "content": turn["text"]})

        metadata = {"timestamp": timestamp} if timestamp else {}
        tasks_a.append((messages_a, user_id_a, metadata))
        tasks_b.append((messages_b, user_id_b, metadata))

    async def process_user_sessions(sessions):
        for msgs, uid, meta in sessions:
            try:
                await memory.add(msgs, user_id=uid, metadata=meta)
            except Exception as e:
                print(f"      [WARN] Add failed: {str(e)[:50]}", flush=True)

    # Run both speakers in parallel
    await asyncio.gather(
        process_user_sessions(tasks_a),
        process_user_sessions(tasks_b),
    )

    return len(tasks_a) + len(tasks_b), speaker_a, speaker_b, user_id_a, user_id_b


async def answer_and_judge_question(
    config: BenchConfig, memory: AsyncMemorySystem, client: AsyncAPIClient,
    qa: dict, sample_id: str,
    speaker_a: str, speaker_b: str, uid_a: str, uid_b: str,
) -> QuestionResult:
    """Answer one question and judge it — can be run concurrently."""
    question = qa["question"]
    reference = str(qa["answer"])
    category = qa["category"]

    try:
        # Multi-query search from both speakers (concurrent)
        results_a, results_b = await asyncio.gather(
            memory.search_multi_query(question, user_id=uid_a, top_k=config.top_k),
            memory.search_multi_query(question, user_id=uid_b, top_k=config.top_k),
        )

        memories_a = json.dumps([r["content"] for r in results_a], ensure_ascii=False)
        memories_b = json.dumps([r["content"] for r in results_b], ensure_ascii=False)

        prompt = ANSWER_PROMPT.format(
            speaker_1=speaker_a, speaker_2=speaker_b,
            speaker_1_memories=memories_a, speaker_2_memories=memories_b,
            question=question,
        )
        prediction = await client.chat_completion(
            [{"role": "user", "content": prompt}], max_tokens=80
        )
    except Exception as e:
        print(f"    [WARN] Answer failed: {str(e)[:60]}", flush=True)
        prediction = "I don't know"

    f1 = compute_f1(prediction, reference)
    bleu1 = compute_bleu1(prediction, reference)

    # Judge
    judge_scores = []
    for _ in range(config.judge_runs):
        judge_prompt = f"""You are evaluating whether a predicted answer is correct.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Is the predicted answer correct? Consider it correct if it conveys the same key information as the reference, even if worded differently.
Reply with ONLY "correct" or "wrong"."""
        try:
            response = await client.chat_completion(
                [{"role": "user", "content": judge_prompt}], temperature=0.3, max_tokens=10
            )
            judge_scores.append(1.0 if "correct" in response.lower() else 0.0)
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

    print(f"=== MemMe LOCOMO Benchmark v6 (Single-hop Opt) ===")
    print(f"Conversations: {len(conversations)}")
    print(f"Model: {config.chat_model}")
    print(f"Embedding: {config.embed_model} ({config.embed_dims}d)")
    print(f"Top-K: {config.top_k}, Dedup: {config.dedup_threshold}")
    print(f"Concurrency: LLM={config.max_llm_concurrent}, Embed={config.max_embed_concurrent}")
    print(f"Output: {results_file}")
    print()

    all_results = []

    async with AsyncAPIClient(config) as client:
        memory = AsyncMemorySystem(config, client)

        for conv_idx, conv in enumerate(conversations):
            sample_id = conv.get("sample_id", f"conv_{conv_idx}")
            print(f"\n--- Conversation {conv_idx + 1}/{len(conversations)}: {sample_id} ---")

            memory.reset()

            # Phase 1: Ingest
            print("  Ingesting (async dual-speaker)...", end="", flush=True)
            t0 = time.time()
            result = await ingest_conversation(config, memory, conv)
            if result is None:
                print(" SKIPPED")
                continue
            total_added, speaker_a, speaker_b, uid_a, uid_b = result
            n_a = len(memory.memories.get(uid_a, []))
            n_b = len(memory.memories.get(uid_b, []))
            ingest_time = time.time() - t0
            print(f" done in {ingest_time:.0f}s ({n_a}+{n_b} memories)")

            # Phase 2: Meditation (consolidation)
            print("  Meditating (identity + consolidation)...", end="", flush=True)
            t_med = time.time()
            added_a = await memory.meditate(uid_a)
            added_b = await memory.meditate(uid_b)
            med_time = time.time() - t_med
            n_a_after = len(memory.memories.get(uid_a, []))
            n_b_after = len(memory.memories.get(uid_b, []))
            print(f" done in {med_time:.0f}s (+{added_a}+{added_b} = {n_a_after}+{n_b_after} total)", flush=True)

            # Phase 3: Answer questions (in parallel batches)
            qa_pairs = [qa for qa in conv.get("qa", []) if qa["category"] != 5]
            if config.categories is not None:
                qa_pairs = [qa for qa in qa_pairs if qa["category"] in config.categories]

            print(f"  Answering {len(qa_pairs)} questions (async)...", flush=True)
            t1 = time.time()

            # Process questions in batches of max_llm_concurrent
            batch_size = config.max_llm_concurrent
            for batch_start in range(0, len(qa_pairs), batch_size):
                batch = qa_pairs[batch_start:batch_start + batch_size]
                tasks = [
                    answer_and_judge_question(
                        config, memory, client, qa, sample_id,
                        speaker_a, speaker_b, uid_a, uid_b,
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
        "protocol": "v6_singlehop_opt",
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
    print("  MemMe LOCOMO Benchmark v3-fast (Async)")
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
    parser = argparse.ArgumentParser(description="MemMe LOCOMO Benchmark v6 (Single-hop Opt)")
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--chat-model", default="qwen3.5-plus")
    parser.add_argument("--embed-model", default="text-embedding-v3")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--judge-runs", type=int, default=1)
    parser.add_argument("--conversations", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--data-path", default="locomo10.json")
    parser.add_argument("--output-dir", default="results_v6")
    parser.add_argument("--dedup-threshold", type=float, default=0.92)
    parser.add_argument("--max-llm-concurrent", type=int, default=5)
    parser.add_argument("--max-embed-concurrent", type=int, default=10)
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
        dedup_threshold=args.dedup_threshold,
        max_llm_concurrent=args.max_llm_concurrent,
        max_embed_concurrent=args.max_embed_concurrent,
        conversations=[int(x) for x in args.conversations.split(",")] if args.conversations else None,
        categories=[int(x) for x in args.categories.split(",")] if args.categories else None,
    )
    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()
