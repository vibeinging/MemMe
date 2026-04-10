#!/usr/bin/env python3
"""MemMe LoCoMo-Plus Benchmark — Cognitive Memory (Synchronous)

Tests cue-trigger semantic disconnect: can MemMe recall implicit memories
when the trigger has LOW semantic similarity to the original cue?

Pipeline: build_context → append_events → search → answer → judge

Usage:
    python3 run_benchmark.py --api-key $KEY --chat-base-url $URL --base-url $EMBED_URL
"""

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

import requests as http

try:
    import memme
except ImportError:
    print("ERROR: memme not found.")
    sys.exit(1)


# ── Context building (from data/build_conv.py) ──

_NUM_WORD = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
             "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}

def parse_locomo_time(s):
    return datetime.strptime(s, "%I:%M %p on %d %B, %Y")

def parse_time_gap(tg):
    s = tg.lower().strip()
    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)\b\s*(week|weeks|month|months|year|years)\b", s)
    if not m: return 0
    num, unit = m.groups()
    count = int(num) if num.isdigit() else (1 if num in ("a","an") else _NUM_WORD.get(num, 0))
    if unit.startswith("week"): return count * 7
    if unit.startswith("month"): return count * 30
    if unit.startswith("year"): return count * 365
    return 0

def parse_ab(text):
    turns = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("A:"): turns.append({"speaker":"A","text":line[2:].strip()})
        elif line.startswith("B:"): turns.append({"speaker":"B","text":line[2:].strip()})
    return turns

def build_context(plus_item, locomo_item):
    conv = locomo_item["conversation"]
    speaker_a, speaker_b = conv["speaker_a"], conv["speaker_b"]
    sessions, times, date_strs = [], [], []
    idx = 1
    while f"session_{idx}" in conv:
        sessions.append((f"session_{idx}", conv[f"session_{idx}"]))
        times.append(parse_locomo_time(conv[f"session_{idx}_date_time"]))
        date_strs.append(conv[f"session_{idx}_date_time"])
        idx += 1

    query_time = times[-1] + timedelta(days=7)
    cue_time = query_time - timedelta(days=parse_time_gap(plus_item["time_gap"]))

    cue_turns = parse_ab(plus_item["cue_dialogue"])
    for t in cue_turns: t["speaker"] = speaker_a if t["speaker"]=="A" else speaker_b
    trigger_turns = parse_ab(plus_item["trigger_query"])
    for t in trigger_turns: t["speaker"] = speaker_a if t["speaker"]=="A" else speaker_b

    events = []
    for (sk, turns), t, ds in zip(sessions, times, date_strs):
        events.append({"time":t, "date_str":ds, "turns":turns, "type":"original"})
    events.append({"time":cue_time, "date_str":cue_time.strftime("%I:%M %p on %d %B, %Y"), "turns":cue_turns, "type":"cue"})
    events.append({"time":query_time, "date_str":query_time.strftime("%I:%M %p on %d %B, %Y"), "turns":trigger_turns, "type":"trigger"})
    events.sort(key=lambda x: x["time"])

    return {"speaker_a":speaker_a, "speaker_b":speaker_b, "sessions":events,
            "sample_id":locomo_item.get("sample_id","unknown")}


# ── LLM call ──

def llm_chat(messages, api_key, base_url, model="gpt-4o-mini", temperature=0.0, max_tokens=256):
    for attempt in range(3):
        try:
            r = http.post(base_url,
                headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                json={"model":model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens},
                timeout=60)
            data = r.json()
            if "error" in data:
                if attempt < 2: time.sleep(2**attempt); continue
                return ""
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt < 2: time.sleep(2**attempt); continue
            return ""


# ── Judge ──

JUDGE_PROMPT = """You are a Memory Awareness Judge.
Judge whether the Model Prediction considers or is linked to the Evidence.

Labels:
- "correct": The prediction explicitly or implicitly reflects the evidence. Give 1 point.
- "wrong": No link to the evidence. No point.

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

Return JSON: {{"label": "correct"|"wrong", "reason": "..."}}"""


# ── Metrics ──

@dataclass
class SampleMetrics:
    sample_idx: int = 0
    relation_type: str = ""
    cue_dialogue: str = ""
    trigger_query: str = ""
    prediction: str = ""
    judge_result: str = ""
    correct: bool = False
    ingest_time: float = 0.0
    search_time: float = 0.0
    answer_time: float = 0.0
    judge_time: float = 0.0
    total_time: float = 0.0
    n_sessions: int = 0
    n_search_results: int = 0
    error: str = ""


# ── Process one sample ──

def process_one(si, plus_item, context, total, config, run_path, done_idxs):
    if si in done_idxs:
        return None

    m = SampleMetrics(sample_idx=si, relation_type=plus_item["relation_type"],
                      cue_dialogue=plus_item["cue_dialogue"], trigger_query=plus_item["trigger_query"])
    t_total = time.time()

    try:
        db_path = os.path.join(config["cache_dir"], f"lcp_{si}.db")
        skip = config["reuse_cache"] and os.path.exists(db_path)
        user_id = f"{context['speaker_a']}_{context['speaker_b']}"

        store = memme.MemoryStore(
            db_path=db_path, embedder="openai",
            api_key=config["api_key"],
            base_url=config.get("embed_base_url") or None,
            embed_model=config.get("embed_model", "text-embedding-3-small"),
            dims=config.get("embed_dims", 1536),
            enable_forgetting_curve=False,
            rerank_api_key=config.get("rerank_api_key") or None,
            rerank_base_url=config.get("rerank_base_url") or None,
            rerank_model=config.get("rerank_model") or None,
        )

        if not skip:
            t_ingest = time.time()
            for idx, sess in enumerate(context["sessions"]):
                messages = []
                for turn in sess["turns"]:
                    speaker = turn.get("speaker", "unknown")
                    text = turn.get("text", "")
                    if text:
                        messages.append(("user", f"[{sess['date_str']}] {speaker}: {text}"))
                if messages:
                    store.append_events(messages=messages, session_id=f"s{idx}_{sess['type']}", user_id=user_id)
            m.ingest_time = time.time() - t_ingest
            m.n_sessions = len(context["sessions"])
            try:
                store.rebuild_fts_index()
            except Exception:
                pass

        # Trigger text
        trigger_text = plus_item["trigger_query"]
        for line in trigger_text.split("\n"):
            if line.strip().startswith("A:"):
                trigger_text = line.strip()[2:].strip()
                break

        # Search
        t_search = time.time()
        results = store.search(trigger_text, user_id=user_id, limit=config.get("top_k", 30))
        m.search_time = time.time() - t_search
        m.n_search_results = len(results)

        memories = "\n".join(
            f"[Memory {i+1}] {(r['content'] if isinstance(r,dict) else r.content)}"
            for i, r in enumerate(results[:30])
        )

        prompt = f"""You are {context['speaker_b']}, talking with {context['speaker_a']}.
Based on memories from past conversations, respond to {context['speaker_a']}'s message.
Show awareness of relevant past context.

## Memories
{memories}

## {context['speaker_a']} says:
{trigger_text}

## Your response as {context['speaker_b']}:"""

        # Answer
        t_ans = time.time()
        provider = config["chat_providers"][si % len(config["chat_providers"])]
        m.prediction = llm_chat(
            [{"role":"user","content":prompt}],
            api_key=provider["api_key"], base_url=provider["base_url"],
            model=config.get("chat_model","gpt-4o-mini"), max_tokens=256)
        m.answer_time = time.time() - t_ans

        # Judge
        t_judge = time.time()
        judge_text = JUDGE_PROMPT.format(evidence=plus_item["cue_dialogue"], pred=m.prediction)
        resp = llm_chat(
            [{"role":"user","content":judge_text}],
            api_key=provider["api_key"], base_url=provider["base_url"],
            model=config.get("judge_model","gpt-4o-mini"), temperature=0.0, max_tokens=128)
        m.judge_result = resp
        m.correct = '"correct"' in resp.lower() if resp else False
        m.judge_time = time.time() - t_judge

    except Exception as e:
        m.error = str(e)[:200]

    m.total_time = time.time() - t_total
    sym = "+" if m.correct else "-"
    print(f"  [{si+1}/{total}] {m.relation_type:8s} {sym} {m.total_time:.0f}s (I={m.ingest_time:.0f} S={m.search_time:.1f})", flush=True)

    with open(run_path, "a") as f:
        f.write(json.dumps(asdict(m)) + "\n")
    return m


# ── Main ──

def main():
    p = argparse.ArgumentParser(description="MemMe LoCoMo-Plus Benchmark")
    p.add_argument("--api-key", required=True)
    p.add_argument("--chat-base-url", required=True)
    p.add_argument("--base-url", default="")
    p.add_argument("--chat-model", default="gpt-4o-mini")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument("--embed-dims", type=int, default=1536)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--data-path", default="data/locomo_plus.json")
    p.add_argument("--locomo-path", default="data/locomo10.json")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument("--relation-types", type=str, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--rerank-api-key", default="")
    p.add_argument("--rerank-base-url", default="")
    p.add_argument("--rerank-model", default="")
    p.add_argument("--extra-providers", type=str, default=None)
    p.add_argument("--results-only", type=str, default=None)
    args = p.parse_args()

    if args.results_only:
        ms = []
        with open(args.results_only) as f:
            for line in f:
                d = json.loads(line)
                ms.append(SampleMetrics(**{k:v for k,v in d.items() if k in SampleMetrics.__dataclass_fields__}))
        total = len(ms)
        c = sum(1 for m in ms if m.correct)
        print(f"Overall: {c}/{total} = {c/total*100:.1f}%")
        by_type = defaultdict(lambda:[0,0])
        for m in ms:
            by_type[m.relation_type][1] += 1
            by_type[m.relation_type][0] += int(m.correct)
        for t,(cc,n) in sorted(by_type.items()):
            print(f"  {t:10s} {cc}/{n} = {cc/n*100:.1f}%")
        return

    with open(args.data_path) as f:
        plus_data = json.load(f)
    with open(args.locomo_path) as f:
        locomo_data = json.load(f)
    print(f"LoCoMo-Plus: {len(plus_data)} | LoCoMo: {len(locomo_data)} convs")

    if args.relation_types:
        plus_data = [d for d in plus_data if d["relation_type"] in args.relation_types.split(",")]
    if args.max_questions > 0:
        plus_data = plus_data[:args.max_questions]
    print(f"Testing: {len(plus_data)} samples")

    random.seed(42)
    contexts = [build_context(item, random.choice(locomo_data)) for item in plus_data]

    chat_providers = [{"api_key":args.api_key, "base_url":args.chat_base_url}]
    if args.extra_providers:
        chat_providers += json.loads(args.extra_providers)

    config = dict(
        api_key=args.api_key, embed_base_url=args.base_url,
        embed_model=args.embed_model, embed_dims=args.embed_dims,
        chat_model=args.chat_model, judge_model=args.judge_model,
        top_k=args.top_k, cache_dir=args.cache_dir, reuse_cache=args.reuse_cache,
        rerank_api_key=args.rerank_api_key, rerank_base_url=args.rerank_base_url,
        rerank_model=args.rerank_model, chat_providers=chat_providers,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(args.output_dir, f"run_{run_id}.jsonl")

    done_idxs = set()
    existing = []
    for fn in os.listdir(args.output_dir):
        if fn.startswith("run_") and fn.endswith(".jsonl") and fn != os.path.basename(run_path):
            with open(os.path.join(args.output_dir, fn)) as f:
                for line in f:
                    d = json.loads(line)
                    done_idxs.add(d["sample_idx"])
                    existing.append(SampleMetrics(**{k:v for k,v in d.items() if k in SampleMetrics.__dataclass_fields__}))

    print(f"Workers: {args.workers} | Providers: {len(chat_providers)} | Resume: {len(done_idxs)}")
    print(f"Output: {run_path}\n")

    all_metrics = list(existing)
    remaining = [(si, plus_data[si], contexts[si]) for si in range(len(plus_data)) if si not in done_idxs]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_one, si, item, ctx, len(plus_data), config, run_path, done_idxs): si
                for si, item, ctx in remaining}
        for fut in as_completed(futs):
            r = fut.result()
            if r: all_metrics.append(r)
            done = len([m for m in all_metrics if m.relation_type])
            correct = sum(1 for m in all_metrics if m.correct)
            if done % 20 == 0 and done > 0:
                print(f"  >>> {correct}/{done} = {correct/done*100:.1f}%", flush=True)

    total = len(all_metrics)
    c = sum(1 for m in all_metrics if m.correct)
    print(f"\n{'='*60}")
    print(f"  LoCoMo-Plus: {c}/{total} = {c/total*100:.1f}%")
    by_type = defaultdict(lambda:[0,0])
    for m in all_metrics:
        by_type[m.relation_type][1] += 1
        by_type[m.relation_type][0] += int(m.correct)
    for t,(cc,n) in sorted(by_type.items()):
        print(f"  {t:10s} {cc}/{n} = {cc/n*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
