/**
 * OpenClaw Plugin SDK type stubs.
 *
 * OpenClaw does not ship public TypeScript types, so we define the subset
 * we actually use. These are derived from openclaw/openclaw source and
 * community plugin conventions.
 */

// ---------------------------------------------------------------------------
// Plugin entry
// ---------------------------------------------------------------------------

export interface PluginManifest {
  id: string;
  name: string;
  kind: "memory" | "tool" | "adapter";
  configSchema?: { parse(value: unknown): unknown };
  register(api: OpenClawPluginApi): void | Promise<void>;
}

// ---------------------------------------------------------------------------
// Plugin API
// ---------------------------------------------------------------------------

export interface OpenClawPluginApi {
  /** Register an event handler. */
  on(event: "before_agent_start", handler: BeforeAgentStartHandler): void;
  on(event: "agent_end", handler: AgentEndHandler): void;

  /** Register an agent-facing tool. */
  registerTool(tool: AgentTool): void;

  /** Get current user context. */
  getUserId(): string;

  /** Get plugin configuration. */
  getConfig<T = unknown>(): T;

  /** Log a message. */
  log(level: "debug" | "info" | "warn" | "error", message: string): void;
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

export interface BeforeAgentStartPayload {
  prompt: string;
  messages: ConversationMessage[];
}

export interface BeforeAgentStartResult {
  prependContext?: string;
}

export type BeforeAgentStartHandler = (
  payload: BeforeAgentStartPayload
) => Promise<BeforeAgentStartResult | void>;

export interface AgentEndPayload {
  success: boolean;
  messages: ConversationMessage[];
}

export type AgentEndHandler = (
  payload: AgentEndPayload
) => Promise<void>;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

export interface ConversationMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | ContentBlock[];
}

export interface ContentBlock {
  type: "text";
  text: string;
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export interface AgentTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  execute(
    params: Record<string, unknown>
  ): Promise<ToolResult>;
}

export interface ToolResult {
  content: Array<{ type: "text"; text: string }>;
  details?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// MemMe Plugin Config
// ---------------------------------------------------------------------------

export interface MemmePluginConfig {
  db_path: string;
  embedding_provider: "openai";
  embedding_api_key: string;
  embedding_base_url?: string;
  embedding_model: string;
  embedding_dims: number;
  llm_api_key: string;
  llm_model: string;
  llm_base_url: string;
  auto_compact_threshold: number;
  recall_limit: number;
  recall_episode_limit: number;
  include_identity: boolean;
}
