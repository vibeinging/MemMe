/**
 * memory_recall tool — agent-facing semantic search across memories and episodes.
 */

import type { MemoryStore } from "memme";
import type { AgentTool, MemmePluginConfig } from "../types";
import { fetchRecall } from "../recall-core";

export function createRecallTool(
  store: MemoryStore,
  getUserId: () => string,
  config: MemmePluginConfig
): AgentTool {
  return {
    name: "memory_recall",
    description:
      "Search your memory for relevant facts, past conversations, and episodes. " +
      "Use this when you need to recall something about the user or past interactions.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "What to search for in memory",
        },
        limit: {
          type: "number",
          description: "Maximum number of results",
          default: 10,
        },
      },
      required: ["query"],
    },
    async execute(params) {
      const userId = getUserId();
      const query = params.query as string;
      const limit = (params.limit as number) ?? config.recall_limit;
      const overrides = { ...config, recall_limit: limit };

      const { memories, episodes } = await fetchRecall(
        store, query, userId, overrides
      );

      const lines: string[] = [];

      if (memories.length > 0) {
        lines.push("## Memories");
        for (const m of memories) {
          lines.push(`- ${m.content}`);
        }
      }

      if (episodes.length > 0) {
        lines.push("\n## Episodes");
        for (const ep of episodes) {
          lines.push(`- [${ep.startedAt}] **${ep.title}**: ${ep.summary}`);
        }
      }

      if (lines.length === 0) {
        return {
          content: [{ type: "text", text: "No relevant memories found." }],
        };
      }

      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: {
          memory_count: memories.length,
          episode_count: episodes.length,
        },
      };
    },
  };
}
