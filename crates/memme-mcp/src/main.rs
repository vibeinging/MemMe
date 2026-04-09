//! MemMe MCP Server
//!
//! Implements the Model Context Protocol (MCP) over stdio for integration
//! with Claude Desktop, Cursor, Windsurf, and other MCP-compatible clients.
//!
//! Usage:
//!   OPENAI_API_KEY=<your-api-key> memme-mcp [--db-path memory.db]

use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tracing::debug;

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::Embedder;

mod protocol;
use protocol::*;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("memme_mcp=debug,memme_core=info")
        .with_writer(io::stderr)
        .init();

    let db_path = std::env::args()
        .skip_while(|a| a != "--db-path")
        .nth(1)
        .unwrap_or_else(|| "memory.db".into());

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: OPENAI_API_KEY env var required");
        std::process::exit(1);
    });

    let embed_url = std::env::var("EMBEDDING_URL").expect(
        "EMBEDDING_URL env var required (full endpoint URL, e.g. https://api.openai.com/v1/embeddings)",
    );
    let llm_url = std::env::var("LLM_URL").expect(
        "LLM_URL env var required (full endpoint URL, e.g. https://api.openai.com/v1/chat/completions)",
    );

    // Build store inside tokio runtime for OpenAI provider
    let rt = tokio::runtime::Runtime::new()?;
    let _guard = rt.enter();

    let config = MemoryConfig {
        db_path,
        embedding_dims: 1536,
        ..Default::default()
    };

    let embedder: Arc<dyn Embedder> = Arc::new(memme_embeddings::openai::OpenAiEmbedder::new(
        &api_key, &embed_url,
    ));

    let llm_config = memme_llm::openai::OpenAIConfig {
        api_key: api_key.clone(),
        base_url: llm_url,
        model: "gpt-4.1-nano".to_string(),
    };
    let llm = Arc::new(memme_llm::openai::OpenAIProvider::new(llm_config));

    let store = MemoryStore::new(config, embedder)?.with_llm(llm);

    let store = Arc::new(Mutex::new(store));

    debug!("MemMe MCP server starting on stdio");

    // Stdio JSON-RPC loop
    let stdin = io::stdin();
    let stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    -32700,
                    &format!("Parse error: {e}"),
                );
                writeln!(&stdout, "{}", serde_json::to_string(&resp)?)?;
                continue;
            }
        };

        let response = handle_request(&request, &store);
        let out = serde_json::to_string(&response)?;
        writeln!(&stdout, "{}", out)?;
        stdout.lock().flush()?;
    }

    Ok(())
}

fn handle_request(req: &JsonRpcRequest, store: &Arc<Mutex<MemoryStore>>) -> JsonRpcResponse {
    match req.method.as_str() {
        "initialize" => {
            let result = serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "memme",
                    "version": env!("CARGO_PKG_VERSION")
                }
            });
            JsonRpcResponse::ok(req.id.clone(), result)
        }

        "notifications/initialized" | "initialized" => {
            // No response needed for notifications, but send ok if id present
            JsonRpcResponse::ok(req.id.clone(), serde_json::json!({}))
        }

        "tools/list" => {
            let tools = serde_json::json!({
                "tools": [
                    {
                        "name": "add_memory",
                        "description": "Add a new memory. Extracts facts and stores with vector embedding.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string", "description": "The text to memorize" },
                                "user_id": { "type": "string", "description": "User identifier" },
                                "agent_id": { "type": "string", "description": "Agent identifier (optional)" },
                                "metadata": { "type": "object", "description": "Optional metadata" }
                            },
                            "required": ["content", "user_id"]
                        }
                    },
                    {
                        "name": "search_memory",
                        "description": "Search memories by semantic similarity to a query.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string", "description": "Search query" },
                                "user_id": { "type": "string", "description": "User identifier" },
                                "limit": { "type": "integer", "description": "Max results (default 5)" }
                            },
                            "required": ["query", "user_id"]
                        }
                    },
                    {
                        "name": "get_memory",
                        "description": "Get a specific memory by its ID.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "Memory ID" }
                            },
                            "required": ["id"]
                        }
                    },
                    {
                        "name": "update_memory",
                        "description": "Update the content of an existing memory.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "Memory ID" },
                                "content": { "type": "string", "description": "New content" }
                            },
                            "required": ["id", "content"]
                        }
                    },
                    {
                        "name": "delete_memory",
                        "description": "Delete a memory by ID.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "Memory ID" }
                            },
                            "required": ["id"]
                        }
                    },
                    {
                        "name": "list_memories",
                        "description": "List all memories for a user.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": { "type": "string", "description": "User identifier" },
                                "limit": { "type": "integer", "description": "Max results (default 10)" }
                            },
                            "required": ["user_id"]
                        }
                    },
                    {
                        "name": "delete_all_memories",
                        "description": "Delete all memories for a user.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": { "type": "string", "description": "User identifier" }
                            },
                            "required": ["user_id"]
                        }
                    }
                ]
            });
            JsonRpcResponse::ok(req.id.clone(), tools)
        }

        "tools/call" => {
            let params = req.params.as_ref();
            let tool_name = params
                .and_then(|p| p.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let args = params
                .and_then(|p| p.get("arguments"))
                .cloned()
                .unwrap_or(serde_json::json!({}));

            let store = store.lock().unwrap_or_else(|e| {
                tracing::warn!("store mutex was poisoned, recovering");
                e.into_inner()
            });
            let result = call_tool(tool_name, &args, &store);

            match result {
                Ok(text) => JsonRpcResponse::ok(
                    req.id.clone(),
                    serde_json::json!({
                        "content": [{
                            "type": "text",
                            "text": text
                        }]
                    }),
                ),
                Err(e) => JsonRpcResponse::ok(
                    req.id.clone(),
                    serde_json::json!({
                        "content": [{
                            "type": "text",
                            "text": format!("Error: {e}")
                        }],
                        "isError": true
                    }),
                ),
            }
        }

        _ => JsonRpcResponse::error(
            req.id.clone(),
            -32601,
            &format!("Method not found: {}", req.method),
        ),
    }
}

fn call_tool(name: &str, args: &serde_json::Value, store: &MemoryStore) -> Result<String, String> {
    match name {
        "add_memory" => {
            let content = args["content"].as_str().ok_or("content required")?;
            let user_id = args["user_id"].as_str().ok_or("user_id required")?;
            let mut opts = AddOptions::new(user_id);
            if let Some(aid) = args.get("agent_id").and_then(|v| v.as_str()) {
                opts = opts.agent_id(aid);
            }
            if let Some(m) = args.get("metadata") {
                if !m.is_null() {
                    opts = opts.metadata(m.clone());
                }
            }
            let result = store.add(content, opts).map_err(|e| e.to_string())?;
            serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
        }

        "search_memory" => {
            let query = args["query"].as_str().ok_or("query required")?;
            let user_id = args["user_id"].as_str().ok_or("user_id required")?;
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
            let opts = SearchOptions::new(user_id).limit(limit);
            let results = store.search(query, opts).map_err(|e| e.to_string())?;
            serde_json::to_string_pretty(&results).map_err(|e| e.to_string())
        }

        "get_memory" => {
            let id = args["id"].as_str().ok_or("id required")?;
            let result = store.get_trace(id).map_err(|e| e.to_string())?;
            match result {
                Some(m) => serde_json::to_string_pretty(&m).map_err(|e| e.to_string()),
                None => Err(format!("Memory '{id}' not found")),
            }
        }

        "update_memory" => {
            let id = args["id"].as_str().ok_or("id required")?;
            let content = args["content"].as_str().ok_or("content required")?;
            let result = store
                .update_trace(id, content, None)
                .map_err(|e| e.to_string())?;
            serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
        }

        "delete_memory" => {
            let id = args["id"].as_str().ok_or("id required")?;
            store.delete_trace(id).map_err(|e| e.to_string())?;
            Ok(format!("Memory '{id}' deleted"))
        }

        "list_memories" => {
            let user_id = args["user_id"].as_str().ok_or("user_id required")?;
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            let opts = ListOptions::new(user_id).limit(limit);
            let results = store.list_traces(opts).map_err(|e| e.to_string())?;
            serde_json::to_string_pretty(&results).map_err(|e| e.to_string())
        }

        "delete_all_memories" => {
            let user_id = args["user_id"].as_str().ok_or("user_id required")?;
            let count = store
                .delete_all_traces(user_id, None, None, None)
                .map_err(|e| e.to_string())?;
            Ok(format!("Deleted {count} memories"))
        }

        _ => Err(format!("Unknown tool: {name}")),
    }
}
