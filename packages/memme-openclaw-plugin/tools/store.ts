/**
 * memory_store tool — agent-facing manual memory storage.
 */

import type { MemoryStore } from "memme";
import type { AgentTool } from "../types";

export function createStoreTool(
  store: MemoryStore,
  getUserId: () => string
): AgentTool {
  return {
    name: "memory_store",
    description:
      "Explicitly store an important fact or preference about the user. " +
      "Use this when the user asks you to remember something specific.",
    parameters: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "The fact or information to remember",
        },
        metadata: {
          type: "string",
          description: "Optional JSON metadata",
        },
      },
      required: ["text"],
    },
    async execute(params) {
      const userId = getUserId();
      const text = params.text as string;
      const metadata = params.metadata as string | undefined;

      const result = await store.add(text, userId, undefined, undefined, metadata);

      return {
        content: [
          {
            type: "text",
            text: `Stored memory: "${text}" (id: ${result.id})`,
          },
        ],
        details: { memory_id: result.id },
      };
    },
  };
}
