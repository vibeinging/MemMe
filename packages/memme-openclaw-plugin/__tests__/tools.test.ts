import { describe, it, expect, vi, beforeEach } from "vitest";
import { createRecallTool } from "../tools/recall";
import { createStoreTool } from "../tools/store";
import { createForgetTool } from "../tools/forget";
import type { MemmePluginConfig } from "../types";

function createMockStore() {
  return {
    hybridSearch: vi.fn().mockResolvedValue([
      { id: "mem-1", content: "User likes cats", userId: "user-1" },
    ]),
    searchEpisodes: vi.fn().mockResolvedValue([
      {
        episodeId: "ep-1",
        title: "Chat about pets",
        summary: "Discussed cats and dogs",
        startedAt: "2026-03-30T10:00:00Z",
      },
    ]),
    search: vi.fn().mockResolvedValue([
      { id: "mem-1", content: "User likes cats", userId: "user-1" },
    ]),
    listIdentityTraits: vi.fn().mockResolvedValue([]),
    get: vi.fn().mockResolvedValue({
      id: "mem-1",
      content: "User likes cats",
      userId: "user-1",
    }),
    add: vi.fn().mockResolvedValue({
      id: "mem-new",
      content: "test fact",
      userId: "user-1",
    }),
    delete: vi.fn().mockResolvedValue(undefined),
  };
}

const config: MemmePluginConfig = {
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
// memory_recall
// ---------------------------------------------------------------------------

describe("memory_recall tool", () => {
  let store: ReturnType<typeof createMockStore>;

  beforeEach(() => {
    store = createMockStore();
  });

  it("returns memories and episodes", async () => {
    const tool = createRecallTool(store as any, () => "user-1", config);
    const result = await tool.execute({ query: "cats" });

    expect(result.content[0].text).toContain("User likes cats");
    expect(result.content[0].text).toContain("Chat about pets");
    expect(result.details?.memory_count).toBe(1);
    expect(result.details?.episode_count).toBe(1);
  });

  it("returns no results message when empty", async () => {
    store.hybridSearch.mockResolvedValue([]);
    store.searchEpisodes.mockResolvedValue([]);

    const tool = createRecallTool(store as any, () => "user-1", config);
    const result = await tool.execute({ query: "unknown" });

    expect(result.content[0].text).toContain("No relevant memories");
  });

  it("passes limit to search", async () => {
    const tool = createRecallTool(store as any, () => "user-1", config);
    await tool.execute({ query: "test", limit: 5 });

    expect(store.hybridSearch).toHaveBeenCalledWith(
      "test",
      "user-1",
      undefined,
      undefined,
      5
    );
  });
});

// ---------------------------------------------------------------------------
// memory_store
// ---------------------------------------------------------------------------

describe("memory_store tool", () => {
  let store: ReturnType<typeof createMockStore>;

  beforeEach(() => {
    store = createMockStore();
  });

  it("stores a fact and returns confirmation", async () => {
    const tool = createStoreTool(store as any, () => "user-1");
    const result = await tool.execute({ text: "I love pizza" });

    expect(store.add).toHaveBeenCalledWith(
      "I love pizza",
      "user-1",
      undefined,
      undefined,
      undefined
    );
    expect(result.content[0].text).toContain("Stored memory");
    expect(result.details?.memory_id).toBe("mem-new");
  });

  it("passes metadata when provided", async () => {
    const tool = createStoreTool(store as any, () => "user-1");
    await tool.execute({
      text: "Important fact",
      metadata: '{"source":"manual"}',
    });

    expect(store.add).toHaveBeenCalledWith(
      "Important fact",
      "user-1",
      undefined,
      undefined,
      '{"source":"manual"}'
    );
  });
});

// ---------------------------------------------------------------------------
// memory_forget
// ---------------------------------------------------------------------------

describe("memory_forget tool", () => {
  let store: ReturnType<typeof createMockStore>;

  beforeEach(() => {
    store = createMockStore();
  });

  it("deletes by ID", async () => {
    const tool = createForgetTool(store as any, () => "user-1");
    const result = await tool.execute({ id: "mem-1" });

    expect(store.delete).toHaveBeenCalledWith("mem-1");
    expect(result.content[0].text).toContain("Deleted memory mem-1");
  });

  it("deletes by query (top match)", async () => {
    const tool = createForgetTool(store as any, () => "user-1");
    const result = await tool.execute({ query: "cats" });

    expect(store.search).toHaveBeenCalledWith("cats", "user-1", undefined, undefined, 1);
    expect(store.delete).toHaveBeenCalledWith("mem-1");
    expect(result.content[0].text).toContain("User likes cats");
  });

  it("handles no match on query", async () => {
    store.search.mockResolvedValue([]);
    const tool = createForgetTool(store as any, () => "user-1");
    const result = await tool.execute({ query: "nonexistent" });

    expect(result.content[0].text).toContain("No matching memory");
    expect(store.delete).not.toHaveBeenCalled();
  });

  it("prompts when no id or query given", async () => {
    const tool = createForgetTool(store as any, () => "user-1");
    const result = await tool.execute({});

    expect(result.content[0].text).toContain("provide either");
  });
});
