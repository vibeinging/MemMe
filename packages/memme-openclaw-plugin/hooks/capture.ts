/**
 * agent_end hook — capture conversation into MemMe session.
 *
 * Flow:
 * 1. Convert OpenClaw messages to ChatMessage format
 * 2. Append events to a session (keyed by date or conversation ID)
 * 3. If compact_needed, trigger compact to extract memories + create episode
 */

import type { MemoryStore, ChatMessage } from "memme";
import type {
  AgentEndHandler,
  AgentEndPayload,
  ConversationMessage,
  ContentBlock,
  MemmePluginConfig,
} from "../types";

/** Extract text content from a message, handling both string and block formats. */
function extractContent(msg: ConversationMessage): string {
  if (typeof msg.content === "string") {
    return msg.content;
  }
  return (msg.content as ContentBlock[])
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
}

/** Generate a stable session ID for today's conversation. */
function todaySessionId(userId: string): string {
  const date = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  return `openclaw-${userId}-${date}`;
}

export function createCaptureHook(
  store: MemoryStore,
  getUserId: () => string,
  config: MemmePluginConfig
): AgentEndHandler {
  return async (payload: AgentEndPayload): Promise<void> => {
    if (!payload.success || payload.messages.length === 0) {
      return;
    }

    const userId = getUserId();
    const sessionId = todaySessionId(userId);

    // Convert messages to ChatMessage format
    const chatMessages: ChatMessage[] = payload.messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({
        role: m.role,
        content: extractContent(m),
      }))
      .filter((m) => m.content.trim().length > 0);

    if (chatMessages.length === 0) {
      return;
    }

    // Append events to session
    const result = await store.appendEvents(
      sessionId,
      chatMessages,
      userId
    );

    // Auto-compact if threshold reached
    if (
      result.compactNeeded ||
      result.totalUnprocessed >= config.auto_compact_threshold
    ) {
      try {
        await store.compact(sessionId);
      } catch (err) {
        // Compact failure is non-fatal — will be retried next time
        // or picked up by meditation
      }
    }
  };
}
