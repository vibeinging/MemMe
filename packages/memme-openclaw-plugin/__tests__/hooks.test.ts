import { describe, it, expect, vi, beforeEach } from "vitest";
import { createRecallHook } from "../hooks/recall";
import { createCaptureHook } from "../hooks/capture";
import type { MemmePluginConfig } from "../types";

// ---------------------------------------------------------------------------
// Mock MemoryStore
// ---------------------------------------------------------------------------

function createMockStore() {
  return {
    hybridSearch: vi.fn().mockResolvedValue([
      {
        id: "mem-1",
        content: "User likes TypeScript",
        importance: 0.85,
        userId: "user-1",
      },
      {
        id: "mem-2",
        content: "User works at Acme Corp",
        importance: 0.72,
        userId: "user-1",
      },
    ]),
    searchEpisodes: vi.fn().mockResolvedValue([
      {
        episodeId: "ep-1",
        title: "Project discussion",
        summary: "Discussed migrating to Rust",
        startedAt: "2026-03-30T10:00:00Z",
        userId: "user-1",
      },
    ]),
    listIdentityTraits: vi.fn().mockResolvedValue([
      {
        traitId: "trait-1",
        traitType: "Style",
        content: "Prefers concise answers",
        confidence: 0.9,
        userId: "user-1",
      },
    ]),
    appendEvents: vi.fn().mockResolvedValue({
      sessionId: "openclaw-user-1-2026-03-30",
      eventsAppended: 2,
      totalUnprocessed: 5,
      compactNeeded: false,
    }),
    compact: vi.fn().mockResolvedValue({
      sessionId: "openclaw-user-1-2026-03-30",
      episodeId: "ep-new",
      memories: [],
      eventsProcessed: 5,
    }),
  };
}

const DEFAULT_CONFIG: MemmePluginConfig = {
  db_path: ":memory:",
  embedding_provider: "openai",
  embedding_api_key: "test",
  embedding_model: "text-embedding-3-small",
  embedding_dims: 1536,
  llm_api_key: "test",
  llm_model: "gpt-4o-mini",
  llm_base_url: "https://api.openai.com",
  auto_compact_threshold: 20,
  recall_limit: 10,
  recall_episode_limit: 5,
  include_identity: true,
};

// ---------------------------------------------------------------------------
// Recall Hook Tests
// ---------------------------------------------------------------------------

describe("recall hook (before_agent_start)", () => {
  let store: ReturnType<typeof createMockStore>;
  let hook: ReturnType<typeof createRecallHook>;

  beforeEach(() => {
    store = createMockStore();
    hook = createRecallHook(store as any, () => "user-1", DEFAULT_CONFIG);
  });

  it("returns relevant-memories XML with all three layers", async () => {
    const result = await hook({
      prompt: "What do I like?",
      messages: [],
    });

    expect(result.prependContext).toBeDefined();
    expect(result.prependContext).toContain("<relevant-memories>");
    expect(result.prependContext).toContain("<memories>");
    expect(result.prependContext).toContain("<episodes>");
    expect(result.prependContext).toContain("<identity>");
    expect(result.prependContext).toContain("User likes TypeScript");
    expect(result.prependContext).toContain("Project discussion");
    expect(result.prependContext).toContain("Prefers concise answers");
  });

  it("calls hybridSearch with correct params", async () => {
    await hook({ prompt: "test query", messages: [] });

    expect(store.hybridSearch).toHaveBeenCalledWith(
      "test query",
      "user-1",
      undefined,
      undefined,
      10
    );
  });

  it("returns empty object for empty prompt", async () => {
    const result = await hook({ prompt: "", messages: [] });
    expect(result).toEqual({});
    expect(store.hybridSearch).not.toHaveBeenCalled();
  });

  it("skips identity when disabled in config", async () => {
    const noIdentityConfig = { ...DEFAULT_CONFIG, include_identity: false };
    const hookNoId = createRecallHook(
      store as any,
      () => "user-1",
      noIdentityConfig
    );

    const result = await hookNoId({
      prompt: "test",
      messages: [],
    });

    expect(result.prependContext).not.toContain("<identity>");
    expect(store.listIdentityTraits).not.toHaveBeenCalled();
  });

  it("gracefully handles search errors", async () => {
    store.hybridSearch.mockRejectedValue(new Error("DB error"));
    store.searchEpisodes.mockRejectedValue(new Error("DB error"));
    store.listIdentityTraits.mockRejectedValue(new Error("DB error"));

    const result = await hook({ prompt: "test", messages: [] });
    // Should not throw, returns empty
    expect(result).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// Capture Hook Tests
// ---------------------------------------------------------------------------

describe("capture hook (agent_end)", () => {
  let store: ReturnType<typeof createMockStore>;
  let hook: ReturnType<typeof createCaptureHook>;

  beforeEach(() => {
    store = createMockStore();
    hook = createCaptureHook(store as any, () => "user-1", DEFAULT_CONFIG);
  });

  it("appends user and assistant messages as events", async () => {
    await hook({
      success: true,
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there!" },
      ],
    });

    expect(store.appendEvents).toHaveBeenCalledTimes(1);
    const [sessionId, messages, userId] = store.appendEvents.mock.calls[0];
    expect(sessionId).toContain("openclaw-user-1-");
    expect(messages).toHaveLength(2);
    expect(messages[0].role).toBe("user");
    expect(messages[1].role).toBe("assistant");
    expect(userId).toBe("user-1");
  });

  it("filters out empty messages", async () => {
    await hook({
      success: true,
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "   " },
        { role: "user", content: "Real question" },
      ],
    });

    const [, messages] = store.appendEvents.mock.calls[0];
    expect(messages).toHaveLength(2);
  });

  it("skips on failure", async () => {
    await hook({
      success: false,
      messages: [{ role: "user", content: "Hello" }],
    });

    expect(store.appendEvents).not.toHaveBeenCalled();
  });

  it("skips on empty messages", async () => {
    await hook({ success: true, messages: [] });
    expect(store.appendEvents).not.toHaveBeenCalled();
  });

  it("triggers compact when threshold reached", async () => {
    store.appendEvents.mockResolvedValue({
      sessionId: "s1",
      eventsAppended: 2,
      totalUnprocessed: 25,
      compactNeeded: false,
    });

    await hook({
      success: true,
      messages: [
        { role: "user", content: "msg" },
        { role: "assistant", content: "reply" },
      ],
    });

    expect(store.compact).toHaveBeenCalledTimes(1);
  });

  it("triggers compact when compactNeeded is true", async () => {
    store.appendEvents.mockResolvedValue({
      sessionId: "s1",
      eventsAppended: 2,
      totalUnprocessed: 5,
      compactNeeded: true,
    });

    await hook({
      success: true,
      messages: [
        { role: "user", content: "msg" },
        { role: "assistant", content: "reply" },
      ],
    });

    expect(store.compact).toHaveBeenCalledTimes(1);
  });

  it("does not compact when below threshold", async () => {
    store.appendEvents.mockResolvedValue({
      sessionId: "s1",
      eventsAppended: 2,
      totalUnprocessed: 5,
      compactNeeded: false,
    });

    await hook({
      success: true,
      messages: [
        { role: "user", content: "msg" },
        { role: "assistant", content: "reply" },
      ],
    });

    expect(store.compact).not.toHaveBeenCalled();
  });

  it("handles content block format", async () => {
    await hook({
      success: true,
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "Block format" }],
        },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Part 1" },
            { type: "text", text: "Part 2" },
          ],
        },
      ],
    });

    const [, messages] = store.appendEvents.mock.calls[0];
    expect(messages[0].content).toBe("Block format");
    expect(messages[1].content).toBe("Part 1\nPart 2");
  });
});
