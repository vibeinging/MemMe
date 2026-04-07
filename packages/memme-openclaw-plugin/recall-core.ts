/**
 * Shared recall logic used by both the recall hook and the recall tool.
 * Fetches memories, episodes, and identity traits in parallel.
 */

import type { MemoryStore } from "memme";
import type { MemmePluginConfig } from "./types";

export interface RecallResults {
  memories: Array<{ content: string; importance: number | null }>;
  episodes: Array<{ startedAt: string; title: string; summary: string }>;
  traits: Array<{ traitType: string; content: string; confidence: number }>;
}

/**
 * Fetch all recall layers in parallel. Errors in any layer are logged
 * and that layer returns empty — never throws.
 */
export async function fetchRecall(
  store: MemoryStore,
  query: string,
  userId: string,
  config: MemmePluginConfig,
  log?: (level: string, msg: string) => void
): Promise<RecallResults> {
  const warn = (msg: string) => log?.("warn", msg);

  const [memories, episodes, traits] = await Promise.all([
    store
      .hybridSearch(query, userId, undefined, undefined, config.recall_limit)
      .catch((err: unknown) => {
        warn(`[memme-memory] hybridSearch failed: ${err}`);
        return [] as Awaited<ReturnType<typeof store.hybridSearch>>;
      }),

    store
      .searchEpisodes(query, userId, config.recall_episode_limit)
      .catch((err: unknown) => {
        warn(`[memme-memory] searchEpisodes failed: ${err}`);
        return [] as Awaited<ReturnType<typeof store.searchEpisodes>>;
      }),

    config.include_identity
      ? store.listIdentityTraits(userId).catch((err: unknown) => {
          warn(`[memme-memory] listIdentityTraits failed: ${err}`);
          return [] as Awaited<ReturnType<typeof store.listIdentityTraits>>;
        })
      : Promise.resolve(
          [] as Awaited<ReturnType<typeof store.listIdentityTraits>>
        ),
  ]);

  return {
    memories: memories.map((m) => ({
      content: m.content,
      importance: m.importance ?? null,
    })),
    episodes: episodes.map((ep) => ({
      startedAt: ep.startedAt,
      title: ep.title,
      summary: ep.summary,
    })),
    traits: traits.map((t) => ({
      traitType: t.traitType,
      content: t.content,
      confidence: t.confidence,
    })),
  };
}
