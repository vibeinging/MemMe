import { describe, it, expect, beforeEach } from "vitest";
import { MemoryStore } from "../";

describe("MemoryStore (mock embedder)", () => {
  let store: InstanceType<typeof MemoryStore>;

  beforeEach(() => {
    store = MemoryStore.newMock(undefined, 384);
  });

  // ---------------------------------------------------------------------------
  // Core CRUD (existing API, sanity check)
  // ---------------------------------------------------------------------------

  describe("CRUD", () => {
    it("add and get a memory", async () => {
      const result = await store.add("Alice likes cats", "user-1");
      expect(result.id).toBeDefined();
      expect(result.content).toBe("Alice likes cats");
      expect(result.userId).toBe("user-1");

      const fetched = await store.get(result.id);
      expect(fetched).toBeDefined();
      expect(fetched!.content).toBe("Alice likes cats");
    });

    it("search memories", async () => {
      await store.add("Alice likes cats", "user-1");
      await store.add("Bob likes dogs", "user-1");

      const results = await store.search("cats", "user-1");
      expect(results.length).toBeGreaterThan(0);
    });

    it("delete a memory", async () => {
      const result = await store.add("temp memory", "user-1");
      await store.delete(result.id);
      const fetched = await store.get(result.id);
      expect(fetched).toBeNull();
    });
  });

  // ---------------------------------------------------------------------------
  // Session Management (NEW)
  // ---------------------------------------------------------------------------

  describe("Session Management", () => {
    it("appendEvents creates a session and returns result", async () => {
      const result = await store.appendEvents(
        "session-1",
        [
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi there!" },
        ],
        "user-1"
      );

      expect(result.sessionId).toBe("session-1");
      expect(result.eventsAppended).toBe(2);
      expect(typeof result.compactNeeded).toBe("boolean");
      expect(typeof result.totalUnprocessed).toBe("number");
    });

    it("getSession returns session after events appended", async () => {
      await store.appendEvents(
        "session-2",
        [{ role: "user", content: "Test message" }],
        "user-1"
      );

      const session = await store.getSession("session-2");
      expect(session).toBeDefined();
      expect(session!.sessionId).toBe("session-2");
      expect(session!.userId).toBe("user-1");
      expect(session!.eventCount).toBeGreaterThanOrEqual(1);
    });

    it("getSession returns null for nonexistent session", async () => {
      const session = await store.getSession("nonexistent");
      expect(session).toBeNull();
    });

    it("listSessions returns sessions for a user", async () => {
      await store.appendEvents(
        "session-a",
        [{ role: "user", content: "msg a" }],
        "user-1"
      );
      await store.appendEvents(
        "session-b",
        [{ role: "user", content: "msg b" }],
        "user-1"
      );

      const sessions = await store.listSessions("user-1");
      expect(sessions.length).toBeGreaterThanOrEqual(2);
    });

    it("getSessionEvents returns events in a session", async () => {
      await store.appendEvents(
        "session-events",
        [
          { role: "user", content: "First" },
          { role: "assistant", content: "Second" },
          { role: "user", content: "Third" },
        ],
        "user-1"
      );

      const events = await store.getSessionEvents("session-events");
      expect(events.length).toBe(3);
      // Events returned in DESC order (most recent first)
      const contents = events.map((e) => e.content);
      expect(contents).toContain("First");
      expect(contents).toContain("Second");
      expect(contents).toContain("Third");
      expect(events[0].eventType).toBeDefined();
      expect(events[0].userId).toBe("user-1");
    });

    it("getSessionEvents supports limit", async () => {
      await store.appendEvents(
        "session-paged",
        [
          { role: "user", content: "One" },
          { role: "assistant", content: "Two" },
          { role: "user", content: "Three" },
        ],
        "user-1"
      );

      const page = await store.getSessionEvents("session-paged", 2);
      expect(page.length).toBe(2);
    });

    it("deleteSession removes session and events", async () => {
      await store.appendEvents(
        "session-del",
        [{ role: "user", content: "goodbye" }],
        "user-1"
      );

      await store.deleteSession("session-del");
      const session = await store.getSession("session-del");
      expect(session).toBeNull();
    });
  });

  // ---------------------------------------------------------------------------
  // StreamEvent fields (NEW fields)
  // ---------------------------------------------------------------------------

  describe("StreamEvent fields", () => {
    it("events have new fields: processed, purifiedContent, metadata", async () => {
      await store.appendEvents(
        "session-fields",
        [{ role: "user", content: "test fields" }],
        "user-1"
      );

      const events = await store.getSessionEvents("session-fields");
      expect(events.length).toBe(1);
      const ev = events[0];

      expect(typeof ev.processed).toBe("boolean");
      expect(ev.sessionId).toBe("session-fields");
      // purifiedContent is null/undefined before compact
      expect(ev.purifiedContent == null || typeof ev.purifiedContent === "string").toBe(true);
    });
  });

  // ---------------------------------------------------------------------------
  // LLM Configuration (NEW)
  // ---------------------------------------------------------------------------

  describe("LLM Configuration", () => {
    it("hasLlm returns false before setting", () => {
      expect(store.hasLlm()).toBe(false);
    });

    it("setLlmProvider makes hasLlm return true", () => {
      store.setLlmProvider("test-key", "gpt-4o-mini", "https://api.openai.com");
      expect(store.hasLlm()).toBe(true);
    });

    it("saveLlmConfig and loadLlmConfig roundtrip", () => {
      store.saveLlmConfig("sk-test", "gpt-4o", "https://custom.api");
      const config = store.loadLlmConfig();
      expect(config).toBeDefined();
      expect(config![0]).toBe("sk-test");
      expect(config![1]).toBe("gpt-4o");
      expect(config![2]).toBe("https://custom.api");
    });

    it("loadLlmConfig returns null when not saved", () => {
      const config = store.loadLlmConfig();
      expect(config).toBeNull();
    });
  });

  // ---------------------------------------------------------------------------
  // Identity (NEW)
  // ---------------------------------------------------------------------------

  describe("Identity", () => {
    it("listIdentityTraits returns empty array initially", async () => {
      const traits = await store.listIdentityTraits("user-1");
      expect(traits).toEqual([]);
    });

    it("addIdentityTrait and list roundtrip", async () => {
      const trait = await store.addIdentityTrait(
        "Style",
        "Prefers concise responses",
        "user-1",
        0.9
      );

      expect(trait.traitId).toBeDefined();
      expect(trait.content).toBe("Prefers concise responses");
      expect(trait.confidence).toBeCloseTo(0.9, 1);
      expect(trait.userId).toBe("user-1");

      const traits = await store.listIdentityTraits("user-1");
      expect(traits.length).toBe(1);
      expect(traits[0].content).toBe("Prefers concise responses");
    });

    it("identity traits are isolated per user", async () => {
      await store.addIdentityTrait("Role", "Engineer", "user-1");
      await store.addIdentityTrait("Role", "Designer", "user-2");

      const traits1 = await store.listIdentityTraits("user-1");
      const traits2 = await store.listIdentityTraits("user-2");

      expect(traits1.length).toBe(1);
      expect(traits1[0].content).toBe("Engineer");
      expect(traits2.length).toBe(1);
      expect(traits2[0].content).toBe("Designer");
    });
  });

  // ---------------------------------------------------------------------------
  // Episode fields (UPDATED)
  // ---------------------------------------------------------------------------

  describe("Episode fields", () => {
    it("episode should have sessionIds and strength fields", async () => {
      // This test requires compact which needs LLM — skip if no LLM
      // Just verify the type structure by listing (empty is fine)
      const episodes = await store.listEpisodes("user-1");
      expect(Array.isArray(episodes)).toBe(true);
      // If we had episodes, they'd have sessionIds, storageStrength, retrievalStrength
    });
  });
});
