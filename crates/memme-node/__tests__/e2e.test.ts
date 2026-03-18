/**
 * E2E Integration Tests — Full pipeline with real store + mock LLM
 *
 * Tests the complete flow that OpenClaw plugin would exercise:
 *   appendEvents → compact → search → identity → session context
 *
 * Uses ScriptedMockLlm so no external services are needed.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { MemoryStore } from "../";

// ---------------------------------------------------------------------------
// Mock LLM responses
//
// compact() makes 2 LLM calls:
//   1. purify_events — coreference resolution
//   2. generate_episode_summary — title/summary/significance
// ---------------------------------------------------------------------------

const PURIFY_RESPONSE_2 = JSON.stringify({
  purified: [
    {
      content: "Alice's name is Alice and Alice loves painting watercolors.",
      event_time: null,
      location: null,
    },
    {
      content:
        "The assistant greeted Alice and said painting is a wonderful hobby.",
      event_time: null,
      location: null,
    },
  ],
});

const SUMMARY_RESPONSE_1 = JSON.stringify({
  title: "Meeting Alice the Painter",
  summary:
    "Alice introduced herself and shared her passion for painting watercolors. A friendly first conversation.",
  significance: 0.7,
});

const PURIFY_RESPONSE_3 = JSON.stringify({
  purified: [
    {
      content: "Alice asked about the best watercolor paper brands.",
      event_time: null,
      location: null,
    },
    {
      content:
        "The assistant recommended Arches and Fabriano for watercolor painting.",
      event_time: null,
      location: null,
    },
    {
      content: "Alice said she would try Arches 140lb cold press paper.",
      event_time: null,
      location: null,
    },
  ],
});

const SUMMARY_RESPONSE_2 = JSON.stringify({
  title: "Watercolor Paper Recommendations",
  summary:
    "Alice asked about watercolor paper brands. Arches and Fabriano were recommended. Alice decided to try Arches 140lb cold press.",
  significance: 0.5,
});

describe("E2E: Full Pipeline", () => {
  let store: InstanceType<typeof MemoryStore>;

  beforeEach(() => {
    // Provide enough mock LLM responses for 2 compact cycles
    store = MemoryStore.newMockWithLlm([
      PURIFY_RESPONSE_2,
      SUMMARY_RESPONSE_1,
      PURIFY_RESPONSE_3,
      SUMMARY_RESPONSE_2,
    ]);
  });

  it("appendEvents → compact → search finds the episode narrative", async () => {
    // Step 1: Append events
    const appendResult = await store.appendEvents(
      "session-alice",
      [
        { role: "user", content: "My name is Alice and I love painting watercolors." },
        { role: "assistant", content: "Nice to meet you, Alice! Painting is wonderful." },
      ],
      "user-1"
    );

    expect(appendResult.sessionId).toBe("session-alice");
    expect(appendResult.eventsAppended).toBe(2);

    // Step 2: Compact — extracts episode narrative
    const compactResult = await store.compact("session-alice");

    expect(compactResult.sessionId).toBe("session-alice");
    expect(compactResult.episodeId).toBeTruthy();
    expect(compactResult.eventsProcessed).toBe(2);

    // Step 3: Search should find the narrative trace
    const searchResults = await store.search("Alice painting", "user-1");
    expect(searchResults.length).toBeGreaterThan(0);
    // The narrative trace contains "Meeting Alice the Painter"
    const narrativeFound = searchResults.some(
      (r) => r.content.includes("Meeting Alice") || r.content.includes("painting")
    );
    expect(narrativeFound).toBe(true);
  });

  it("compact marks events as processed", async () => {
    await store.appendEvents(
      "session-processed",
      [
        { role: "user", content: "Test message one" },
        { role: "assistant", content: "Test response one" },
      ],
      "user-1"
    );

    await store.compact("session-processed");

    // After compact, events should be marked as processed
    const events = await store.getSessionEvents("session-processed");
    expect(events.length).toBe(2);
    for (const ev of events) {
      expect(ev.processed).toBe(true);
      // purifiedContent should be set after compact
      expect(ev.purifiedContent).toBeTruthy();
    }
  });

  it("getSessionContext returns events within token budget", async () => {
    await store.appendEvents(
      "session-ctx",
      [
        { role: "user", content: "Short message" },
        { role: "assistant", content: "Short reply" },
      ],
      "user-1"
    );

    await store.compact("session-ctx");

    const ctx = await store.getSessionContext("session-ctx", 2000, true);

    expect(ctx.sessionId).toBe("session-ctx");
    expect(ctx.events.length).toBe(2);
    expect(ctx.tokensUsed).toBeGreaterThan(0);
    expect(ctx.tokensUsed).toBeLessThanOrEqual(ctx.tokenBudget);
    expect(ctx.purifiedCount).toBe(2); // both purified by compact
  });

  it("getSessionContext respects small token budget", async () => {
    await store.appendEvents(
      "session-budget",
      [
        { role: "user", content: "A".repeat(200) },
        { role: "assistant", content: "B".repeat(200) },
      ],
      "user-1"
    );

    await store.compact("session-budget");

    // Very small budget — should only fit partial events
    const ctx = await store.getSessionContext("session-budget", 20, false);
    expect(ctx.events.length).toBeLessThan(2);
  });

  it("multi-session: two compacts then cross-session search", async () => {
    // Session 1: about painting
    await store.appendEvents(
      "session-paint",
      [
        { role: "user", content: "My name is Alice and I love painting watercolors." },
        { role: "assistant", content: "Nice to meet you, Alice! Painting is wonderful." },
      ],
      "user-1"
    );
    await store.compact("session-paint");

    // Session 2: about paper
    await store.appendEvents(
      "session-paper",
      [
        { role: "user", content: "What watercolor paper do you recommend?" },
        { role: "assistant", content: "I recommend Arches and Fabriano." },
        { role: "user", content: "I'll try Arches 140lb cold press." },
      ],
      "user-1"
    );
    await store.compact("session-paper");

    // Search across all sessions
    const results = await store.search("watercolor", "user-1");
    expect(results.length).toBeGreaterThanOrEqual(2);

    // List sessions
    const sessions = await store.listSessions("user-1");
    expect(sessions.length).toBe(2);

    // List episodes
    // Episodes are narrative traces stored as memories, not separate episode table entries
    // So we check memories instead
    const memories = await store.list("user-1");
    expect(memories.length).toBe(2); // 2 narrative traces
  });

  it("identity traits persist and are retrievable", async () => {
    // Add identity traits
    await store.addIdentityTrait("Style", "Prefers watercolor painting", "user-1", 0.9);
    await store.addIdentityTrait("Goal", "Wants to learn oil painting next", "user-1", 0.7);

    const traits = await store.listIdentityTraits("user-1");
    expect(traits.length).toBe(2);

    const styles = traits.filter((t) => t.traitType === "style");
    expect(styles.length).toBe(1);
    expect(styles[0].content).toBe("Prefers watercolor painting");

    const goals = traits.filter((t) => t.traitType === "goal");
    expect(goals.length).toBe(1);
    expect(goals[0].confidence).toBeCloseTo(0.7, 1);
  });

  it("delete session removes session but keeps memories", async () => {
    await store.appendEvents(
      "session-temp",
      [
        { role: "user", content: "Temporary conversation" },
        { role: "assistant", content: "Noted." },
      ],
      "user-1"
    );
    await store.compact("session-temp");

    // Verify memory exists
    const beforeDelete = await store.list("user-1");
    expect(beforeDelete.length).toBeGreaterThan(0);

    // Delete session
    await store.deleteSession("session-temp");
    const session = await store.getSession("session-temp");
    expect(session).toBeNull();

    // Memories should still exist (sessions are raw data, memories are derived)
    const afterDelete = await store.list("user-1");
    expect(afterDelete.length).toBe(beforeDelete.length);
  });

  it("LLM configuration is available after newMockWithLlm", () => {
    expect(store.hasLlm()).toBe(true);
  });
});
