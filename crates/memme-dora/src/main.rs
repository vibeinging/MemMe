//! MemMe Dora Node — Long-term memory for dora-rs robots
//!
//! See lib.rs for handler logic and tests.

use std::sync::Arc;

use dora_node_api::{dora_core::config::DataId, DoraNode, Event};
use eyre::{Context, Result};
use tracing::{error, info, warn};

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_embeddings::Embedder;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "memme_dora=info,memme_core=warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    info!("memme-dora node starting");

    let (store, _rt) = init_store().wrap_err("failed to initialize MemMe store")?;
    info!("MemMe store initialized");

    let (mut node, mut events) = DoraNode::init_from_env()?;
    info!("dora node initialized, entering event loop");

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => {
                let input_id = id.as_str();

                // dora raw bytes arrive as a single-buffer byte array (UInt8Array).
                // For StringArray inputs (from Python nodes), the last buffer contains the data.
                let arrow_data = data.to_data();
                let buffers = arrow_data.buffers();
                let raw_bytes = match buffers.last() {
                    Some(buf) => buf.as_slice(),
                    None => {
                        warn!("empty Arrow data on '{input_id}'");
                        continue;
                    }
                };
                let text = match std::str::from_utf8(raw_bytes) {
                    Ok(s) => s.trim(),
                    Err(_) => {
                        warn!("non-UTF8 input on '{input_id}'");
                        continue;
                    }
                };

                if text.is_empty() {
                    continue;
                }

                let (output_id, payload) = match memme_dora::dispatch(input_id, text, &store) {
                    Ok((out_id, json)) => (out_id, json),
                    Err(e) => {
                        error!("error handling '{input_id}': {e}");
                        let err_json = serde_json::json!({
                            "input": input_id,
                            "error": e.to_string()
                        });
                        ("error", serde_json::to_string(&err_json).unwrap())
                    }
                };

                let bytes = payload.as_bytes();
                node.send_output_bytes(
                    DataId::from(output_id.to_string()),
                    metadata.parameters.clone(),
                    bytes.len(),
                    bytes,
                )?;
            }
            Event::Stop(_) => {
                info!("stop event received, shutting down");
                break;
            }
            _ => {}
        }
    }

    info!("memme-dora node stopped");
    Ok(())
}

/// Returns (store, runtime). The runtime must be kept alive for the process lifetime
/// because the OpenAI embedder uses reqwest which requires a tokio runtime.
fn init_store() -> Result<(MemoryStore, tokio::runtime::Runtime)> {
    let db_path = std::env::var("MEMME_DB_PATH").unwrap_or_else(|_| "robot_memory.duckdb".into());
    let collection = std::env::var("MEMME_COLLECTION").unwrap_or_else(|_| "default".into());
    let dims: usize = std::env::var("MEMME_EMBEDDING_DIMS")
        .unwrap_or_else(|_| "1536".into())
        .parse()
        .unwrap_or(1536);

    let api_key = std::env::var("OPENAI_API_KEY").wrap_err("OPENAI_API_KEY env var required")?;
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());

    let rt = tokio::runtime::Runtime::new()?;
    let _guard = rt.enter();

    let config = MemoryConfig {
        db_path,
        collection_name: collection,
        embedding_dims: dims,
        ..Default::default()
    };

    let embedder: Arc<dyn Embedder> =
        Arc::new(memme_embeddings::openai::OpenAiEmbedder::new(&api_key).with_base_url(&base_url));

    let llm_model = std::env::var("MEMME_LLM_MODEL").unwrap_or_else(|_| "gpt-4.1-nano".into());
    let llm_config = memme_llm::openai::OpenAIConfig {
        api_key,
        base_url: base_url.trim_end_matches("/v1").to_string(),
        model: llm_model,
        retry: Default::default(),
    };
    let llm = Arc::new(memme_llm::openai::OpenAIProvider::new(llm_config));

    let store = MemoryStore::new(config, embedder)?.with_llm(llm);
    Ok((store, rt))
}
