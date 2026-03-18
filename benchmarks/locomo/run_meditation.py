#!/usr/bin/env python3
"""Run meditation on cached db: identity extraction + fact consolidation."""
import memme
import json
import os
import time
import sys
import aiohttp
import asyncio

API_KEY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LLM_API_KEY", "")
BASE_URL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("LLM_BASE_URL", "")
MODEL = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o-mini"
DB_PATH = sys.argv[4] if len(sys.argv) > 4 else "cache/conv-26.duckdb"

store = memme.MemoryStore(
    db_path=DB_PATH,
    embedder="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    embed_model="text-embedding-3-small",
    dims=1536,
    llm_api_key=API_KEY,
    llm_model=MODEL,
    llm_base_url=BASE_URL.rstrip("/").removesuffix("/v1"),
)

async def llm_call(session, prompt, temperature=0.1, max_tokens=2048):
    for attempt in range(3):
        try:
            async with session.post(f"{BASE_URL}/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }) as resp:
                data = await resp.json()
                import re
                content = data["choices"][0]["message"]["content"].strip()
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2)
            else:
                print(f"  LLM error: {e}")
                return None

async def meditate():
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as session:
        for user_id in ["conv-26_Melanie", "conv-26_Caroline"]:
            stats = store.user_stats(user_id)
            name = user_id.split("_")[1]
            print(f"\n=== Meditating on {name} ({stats['total_memories']} memories) ===")

            # Get all memories
            memories = store.list(user_id=user_id, limit=1000)
            all_facts = [m["content"] for m in memories]

            # Phase 1: Identity extraction
            print(f"  Phase 1: Identity extraction...", end="", flush=True)
            for i in range(0, len(all_facts), 100):
                chunk = all_facts[i:i+100]
                facts_text = "\n".join(f"- {f}" for f in chunk)
                prompt = f"""From these facts about {name}, extract stable IDENTITY traits.

Facts:
{facts_text}

Extract:
1. Core identity (gender, orientation, nationality, relationship status)
2. Personality traits
3. Core values and passions
4. Key relationships

Return JSON: {{"identity": ["trait1", "trait2", ...]}}"""

                raw = await llm_call(session, prompt)
                if raw:
                    try:
                        import re
                        raw = re.sub(r'^```\w*\n?', '', raw.strip())
                        raw = re.sub(r'\n?```$', '', raw)
                        data = json.loads(raw)
                        traits = data.get("identity", [])
                        for trait in traits:
                            if trait and len(str(trait)) > 5:
                                store.add(f"[IDENTITY] {trait}", user_id=user_id)
                        print(f" +{len(traits)} traits", end="", flush=True)
                    except:
                        pass
            print(" done")

            # Phase 2: Fact consolidation — group related facts
            print(f"  Phase 2: Fact consolidation...", end="", flush=True)
            for i in range(0, len(all_facts), 80):
                chunk = all_facts[i:i+80]
                facts_text = "\n".join(f"- {f}" for f in chunk)
                prompt = f"""Review these facts about {name} and create CONSOLIDATED summary facts.

Facts:
{facts_text}

Create consolidated facts that:
1. Group related items: "painted: horse, sunset, sunrise" instead of 3 separate facts
2. Include specific details: names, titles, places, dates
3. Create cross-reference facts: "Both X and Y enjoy Z"
4. Each consolidated fact should answer a potential question completely

Return JSON: {{"consolidated": ["fact1", "fact2", ...]}}"""

                raw = await llm_call(session, prompt)
                if raw:
                    try:
                        import re
                        raw = re.sub(r'^```\w*\n?', '', raw.strip())
                        raw = re.sub(r'\n?```$', '', raw)
                        data = json.loads(raw)
                        facts = data.get("consolidated", [])
                        for fact in facts:
                            if fact and len(str(fact)) > 10:
                                store.add(f"[CONSOLIDATED] {fact}", user_id=user_id)
                        print(f" +{len(facts)}", end="", flush=True)
                    except:
                        pass
            print(" done")

            # Rebuild FTS after adding new memories
            store.rebuild_fts_index()

            final_stats = store.user_stats(user_id)
            print(f"  {name}: {stats['total_memories']} → {final_stats['total_memories']} memories")

asyncio.run(meditate())
print("\nMeditation complete. Run benchmark with --reuse-cache to test.")
