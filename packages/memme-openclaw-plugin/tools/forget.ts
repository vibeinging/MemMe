/**
 * memory_forget tool — agent-facing memory deletion.
 */

import type { MemoryStore } from "memme";
import type { AgentTool } from "../types";

export function createForgetTool(
  store: MemoryStore,
  getUserId: () => string
): AgentTool {
  return {
    name: "memory_forget",
    description:
      "Remove a specific memory by ID, or search and remove memories matching a query. " +
      "Use this when the user asks you to forget something.",
    parameters: {
      type: "object",
      properties: {
        id: {
          type: "string",
          description: "Specific memory ID to delete",
        },
        query: {
          type: "string",
          description: "Search query — will delete the top matching memory",
        },
      },
    },
    async execute(params) {
      const userId = getUserId();

      // Direct deletion by ID — verify ownership first
      if (params.id) {
        const mem = await store.get(params.id as string);
        if (!mem || mem.userId !== userId) {
          return {
            content: [
              { type: "text", text: "Memory not found or not owned by you." },
            ],
          };
        }
        await store.delete(params.id as string);
        return {
          content: [
            { type: "text", text: `Deleted memory ${params.id}` },
          ],
        };
      }

      // Search-based deletion
      if (params.query) {
        const results = await store.search(
          params.query as string,
          userId,
          undefined,
          undefined,
          1
        );
        if (results.length === 0) {
          return {
            content: [
              { type: "text", text: "No matching memory found to delete." },
            ],
          };
        }
        const target = results[0];
        await store.delete(target.id);
        return {
          content: [
            {
              type: "text",
              text: `Deleted memory: "${target.content}" (id: ${target.id})`,
            },
          ],
          details: { deleted_id: target.id },
        };
      }

      return {
        content: [
          { type: "text", text: "Please provide either an id or query to forget." },
        ],
      };
    },
  };
}
