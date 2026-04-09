/**
 * MemMe Memory Plugin for OpenClaw
 *
 * Replaces OpenClaw's built-in Markdown + sqlite-vec memory with MemMe's
 * structured Session/Episode architecture, identity traits, and meditation.
 *
 * Installation:
 *   1. Copy this directory to ~/.openclaw/extensions/memme-memory/
 *   2. Set plugins.slots.memory = "memme-memory" in OpenClaw config
 *   3. Configure embedding_api_key and llm_api_key
 */

import { MemoryStore } from "memme";
import type { PluginManifest, OpenClawPluginApi, MemmePluginConfig } from "./types";
import { createRecallHook } from "./hooks/recall";
import { createCaptureHook } from "./hooks/capture";
import { createRecallTool } from "./tools/recall";
import { createStoreTool } from "./tools/store";
import { createForgetTool } from "./tools/forget";

const DEFAULT_CONFIG: MemmePluginConfig = {
  db_path: "~/.openclaw/memme.db",
  embedding_provider: "openai",
  embedding_api_key: "",
  embedding_model: "text-embedding-3-small",
  embedding_dims: 1536,
  llm_api_key: "",
  llm_model: "gpt-4o-mini",
  llm_base_url: "https://api.openai.com",
  auto_compact_threshold: 20,
  recall_limit: 10,
  recall_episode_limit: 5,
  include_identity: true,
};

/** Resolve ~ to home directory. */
function expandPath(p: string): string {
  if (p.startsWith("~/")) {
    const home = process.env.HOME || process.env.USERPROFILE || "";
    return p.replace("~", home);
  }
  return p;
}

const plugin: PluginManifest = {
  id: "memme-memory",
  name: "MemMe Memory",
  kind: "memory",

  register(api: OpenClawPluginApi) {
    const userConfig = api.getConfig<Partial<MemmePluginConfig>>();
    const config: MemmePluginConfig = { ...DEFAULT_CONFIG, ...userConfig };

    // Validate required keys
    if (!config.embedding_api_key) {
      throw new Error(
        "[memme-memory] embedding_api_key is required. Set OPENAI_API_KEY or configure explicitly."
      );
    }
    if (!config.llm_api_key) {
      throw new Error(
        "[memme-memory] llm_api_key is required. Set OPENAI_API_KEY or configure explicitly."
      );
    }

    // Initialize MemMe store
    const dbPath = expandPath(config.db_path);
    const store = MemoryStore.newOpenai(
      config.embedding_api_key,
      dbPath,
      config.embedding_base_url,
      config.embedding_model,
      config.embedding_dims
    );

    // Configure LLM provider (one-time setup)
    store.setLlmProvider(
      config.llm_api_key,
      config.llm_model,
      config.llm_base_url
    );

    api.log("info", `[memme-memory] Initialized with db: ${dbPath}`);

    const getUserId = () => api.getUserId();

    // Register lifecycle hooks
    const log = (level: string, msg: string) => api.log(level as "debug" | "info" | "warn" | "error", msg);
    api.on("before_agent_start", createRecallHook(store, getUserId, config, log));
    api.on("agent_end", createCaptureHook(store, getUserId, config));

    // Register agent tools
    api.registerTool(createRecallTool(store, getUserId, config));
    api.registerTool(createStoreTool(store, getUserId));
    api.registerTool(createForgetTool(store, getUserId));

    api.log("info", "[memme-memory] Plugin registered successfully");
  },
};

export default plugin;
