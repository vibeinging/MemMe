/**
 * before_agent_start hook — inject relevant memories into the agent context.
 *
 * Fetches all three recall layers in parallel via fetchRecall(),
 * then formats them as XML tags for injection into the system prompt.
 */

import type { MemoryStore } from "memme";
import type {
  BeforeAgentStartHandler,
  BeforeAgentStartPayload,
  BeforeAgentStartResult,
  MemmePluginConfig,
} from "../types";
import { fetchRecall } from "../recall-core";

export function createRecallHook(
  store: MemoryStore,
  getUserId: () => string,
  config: MemmePluginConfig,
  log?: (level: string, msg: string) => void
): BeforeAgentStartHandler {
  return async (
    payload: BeforeAgentStartPayload
  ): Promise<BeforeAgentStartResult> => {
    const query = payload.prompt;
    if (!query || query.trim().length === 0) {
      return {};
    }

    const userId = getUserId();
    const { memories, episodes, traits } = await fetchRecall(
      store, query, userId, config, log
    );

    const parts: string[] = [];

    if (memories.length > 0) {
      const lines = memories.map(
        (m) => `- ${m.content} (confidence: ${(m.importance ?? 0.5).toFixed(2)})`
      );
      parts.push(`<memories>\n${lines.join("\n")}\n</memories>`);
    }

    if (episodes.length > 0) {
      const lines = episodes.map(
        (ep) => `- [${ep.startedAt}] ${ep.title}: ${ep.summary}`
      );
      parts.push(`<episodes>\n${lines.join("\n")}\n</episodes>`);
    }

    if (traits.length > 0) {
      const lines = traits.map(
        (t) => `- [${t.traitType}] ${t.content} (confidence: ${t.confidence.toFixed(2)})`
      );
      parts.push(`<identity>\n${lines.join("\n")}\n</identity>`);
    }

    if (parts.length === 0) {
      return {};
    }

    return {
      prependContext: `<relevant-memories>\n${parts.join("\n\n")}\n</relevant-memories>`,
    };
  };
}
