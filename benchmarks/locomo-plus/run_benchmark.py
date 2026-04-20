#!/usr/bin/env python3
"""MemMe LoCoMo-Plus Benchmark — based on run_benchmark_base.py"""

import json
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_benchmark_base import BenchmarkRunner, Sample, SampleResult
from base_benchmark import BenchmarkAPIConfig, llm_chat


# ── Context building ──

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
    elif unit.startswith("month"): return count * 30
    elif unit.startswith("year"): return count * 365
    return 0

def parse_ab(text):
    turns = []
    for line in text.strip().split("\n"):
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

    return {"speaker_a":speaker_a, "speaker_b":speaker_b, "sessions":events}


# ── Judge prompt ──

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


class LoCoMoPlusBenchmark(BenchmarkRunner):
    name = "LoCoMo-Plus"
    description = "Cognitive memory: cue-trigger semantic disconnect"

    def add_extra_args(self, parser):
        parser.add_argument("--locomo-path", required=True, help="Path to locomo10.json (base conversations)")
        parser.add_argument("--relation-types", default=None, help="Filter by type: causal,state,value,goal")

    def load_samples(self, args) -> list[Sample]:
        with open(args.data_path) as f:
            plus_data = json.load(f)
        with open(args.locomo_path) as f:
            locomo_data = json.load(f)

        # Filter by relation type
        if args.relation_types:
            types = set(args.relation_types.split(","))
            plus_data = [p for p in plus_data if p["relation_type"] in types]

        random.seed(42)
        samples = []
        for i, item in enumerate(plus_data):
            context = build_context(item, random.choice(locomo_data))
            session_ids = [f"s{idx}_{sess['type']}" for idx, sess in enumerate(context["sessions"])]
            user_id = f"{context['speaker_a']}_{context['speaker_b']}"
            samples.append(Sample(
                id=str(i),
                data={
                    "plus_item": item,
                    "context": context,
                    "_session_ids": session_ids,
                    "_user_id": user_id,
                },
                category=item["relation_type"],
                db_name=f"lcp_{i}.db",
            ))
        return samples

    def ingest_sample(self, store, sample: Sample):
        context = sample.data["context"]
        user_id = sample.data["_user_id"]
        for idx, sess in enumerate(context["sessions"]):
            messages = []
            for turn in sess["turns"]:
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                if text:
                    messages.append(("user", f"[{sess['date_str']}] {speaker}: {text}"))
            if messages:
                store.append_events(messages=messages, session_id=f"s{idx}_{sess['type']}", user_id=user_id)

    def evaluate_sample(self, config: BenchmarkAPIConfig, store, sample: Sample) -> SampleResult:
        import time
        plus_item = sample.data["plus_item"]
        context = sample.data["context"]
        user_id = sample.data["_user_id"]

        # Extract trigger text (first A: line)
        trigger_text = plus_item["trigger_query"]
        for line in trigger_text.split("\n"):
            if line.strip().startswith("A:"):
                trigger_text = line.strip()[2:].strip()
                break

        # Search
        t0 = time.time()
        results = store.search(trigger_text, user_id=user_id, limit=config.top_k)
        search_time = time.time() - t0

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
        prediction = llm_chat(config, "answer", [{"role": "user", "content": prompt}], max_tokens=256)

        # Judge
        judge_text = JUDGE_PROMPT.format(evidence=plus_item["cue_dialogue"], pred=prediction)
        judge_resp = llm_chat(config, "judge", [{"role": "user", "content": judge_text}], max_tokens=128)
        correct = '"correct"' in judge_resp.lower() if judge_resp else False

        return SampleResult(
            sample_id=sample.id,
            category=plus_item["relation_type"],
            correct=correct,
            prediction=prediction,
            reference=plus_item["cue_dialogue"][:200],
            judge_result=judge_resp,
            search_time=search_time,
            extra={
                "trigger_query": plus_item["trigger_query"],
                "n_search_results": len(results),
            },
        )


if __name__ == "__main__":
    LoCoMoPlusBenchmark().run()
