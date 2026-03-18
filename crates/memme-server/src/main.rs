use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Router,
};
use clap::Parser;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{info, warn};

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_embeddings::Embedder;

mod handlers;

/// MemMe REST API Server
#[derive(Parser)]
#[command(name = "memme-server", about = "Self-hosted MemMe REST API server")]
struct Cli {
    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value = "8080")]
    port: u16,

    /// DuckDB database path
    #[arg(long, default_value = "memory.duckdb")]
    db_path: String,

    /// Embedding dimensions
    #[arg(long, default_value = "1536")]
    embedding_dims: usize,

    /// Embedding model name (default: text-embedding-3-small)
    #[arg(long, default_value = "text-embedding-3-small")]
    embedding_model: String,

    /// OpenAI API key for embeddings and LLM
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,

    /// OpenAI base URL
    #[arg(long, env = "OPENAI_BASE_URL")]
    openai_base_url: Option<String>,

    /// LLM model name
    #[arg(long, default_value = "gpt-4.1-nano")]
    llm_model: String,

    /// Use mock embedder (no external API needed, for UI testing)
    #[arg(long, default_value = "false")]
    mock: bool,

    /// API key for bearer token authentication (optional, if not set no auth required)
    #[arg(long, env = "MEMME_API_KEY")]
    api_key: Option<String>,

    /// Allowed CORS origin (can be specified multiple times). If not set, only same-origin is allowed.
    #[arg(long)]
    cors_origin: Vec<String>,
}

pub struct AppState {
    pub store: MemoryStore,
    pub api_key: Option<String>,
}

/// Middleware that checks for bearer token authentication.
async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(ref expected_key) = state.api_key {
        let auth_header = request
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        match auth_header.as_deref() {
            Some(header) if header.starts_with("Bearer ") => {
                let token = &header[7..];
                if token != expected_key.as_str() {
                    return (
                        StatusCode::UNAUTHORIZED,
                        axum::Json(serde_json::json!({
                            "success": false,
                            "error": "Invalid API key"
                        })),
                    )
                        .into_response();
                }
            }
            _ => {
                return (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(serde_json::json!({
                        "success": false,
                        "error": "Missing or invalid Authorization header. Expected: Bearer <api-key>"
                    })),
                )
                    .into_response();
            }
        }
    }

    next.run(request).await
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "memme_server=info,memme_core=info".into()),
        )
        .init();

    let cli = Cli::parse();

    // Warn if binding to 0.0.0.0
    if cli.host == "0.0.0.0" {
        warn!("Binding to 0.0.0.0 exposes the server to all network interfaces. Consider using 127.0.0.1 for local-only access.");
    }

    let config = MemoryConfig {
        db_path: cli.db_path.clone(),
        embedding_dims: cli.embedding_dims,
        ..Default::default()
    };

    let store = if cli.mock {
        // Mock mode: no external API needed, for UI testing
        info!("Running in MOCK mode — no external API calls");
        let embedder: Arc<dyn Embedder> = Arc::new(memme_embeddings::mock::MockEmbedder::new(
            cli.embedding_dims,
        ));
        MemoryStore::new(config, embedder)?
    } else {
        let api_key = cli.openai_api_key.as_deref().unwrap_or_else(|| {
            eprintln!("Error: OPENAI_API_KEY required. Set via --openai-api-key or env var, or use --mock for testing.");
            std::process::exit(1);
        });
        let base_url = cli
            .openai_base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");

        let embed_model = if cli.embedding_model == "text-embedding-3-small"
            && cli.embedding_dims == 1536
        {
            memme_embeddings::openai::OpenAiModel::TextEmbedding3Small
        } else if cli.embedding_model == "text-embedding-3-large" && cli.embedding_dims == 3072 {
            memme_embeddings::openai::OpenAiModel::TextEmbedding3Large
        } else {
            memme_embeddings::openai::OpenAiModel::Custom {
                name: cli.embedding_model.clone(),
                dims: cli.embedding_dims,
            }
        };

        let embedder: Arc<dyn Embedder> = Arc::new(
            memme_embeddings::openai::OpenAiEmbedder::new(api_key)
                .with_base_url(base_url)
                .with_model(embed_model),
        );

        let llm_config = memme_llm::openai::OpenAIConfig {
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches("/v1").to_string(),
            model: cli.llm_model.clone(),
        };
        let llm = Arc::new(memme_llm::openai::OpenAIProvider::new(llm_config));

        MemoryStore::new(config, embedder)?.with_llm(llm)
    };

    let state = Arc::new(AppState {
        store,
        api_key: cli.api_key.clone(),
    });

    // Build CORS layer
    let cors = if cli.cors_origin.is_empty() {
        // No origins specified — restrictive default (no cross-origin allowed)
        CorsLayer::new()
    } else {
        let origins: Vec<_> = cli
            .cors_origin
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::PUT,
                axum::http::Method::DELETE,
            ])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
            ])
    };

    if cli.api_key.is_some() {
        info!("API key authentication enabled");
    } else {
        warn!("No API key set — authentication disabled (suitable for local dev only)");
    }

    let app = Router::new()
        // Basic CRUD
        .route("/v1/memories", post(handlers::add_memory))
        .route("/v1/memories/search", post(handlers::search_memories))
        .route("/v1/memories/hybrid-search", post(handlers::hybrid_search))
        .route("/v1/memories/list", post(handlers::list_memories))
        .route("/v1/memories/export", post(handlers::export_memories))
        .route("/v1/memories/import", post(handlers::import_memories))
        .route("/v1/memories/{id}", get(handlers::get_memory))
        .route("/v1/memories/{id}", put(handlers::update_memory))
        .route("/v1/memories/{id}", delete(handlers::delete_memory))
        .route("/v1/memories/{id}/history", get(handlers::memory_history))
        .route("/v1/memories", delete(handlers::delete_all_memories))
        // Smart (LLM-powered)
        .route("/v1/memories/smart", post(handlers::smart_add))
        .route(
            "/v1/memories/smart/messages",
            post(handlers::smart_add_messages),
        )
        // Episodes: removed (merged into traces; use search() with resolution filter)
        // Knowledge graph
        .route("/v1/graph", post(handlers::graph_add))
        .route("/v1/graph/search", post(handlers::graph_search))
        // Recall (multi-layer unified retrieval)
        .route("/v1/recall", post(handlers::recall))
        .route(
            "/v1/recall/{recall_id}/feedback",
            post(handlers::recall_feedback),
        )
        // Config
        .route("/v1/config", get(handlers::get_config))
        .route("/v1/config/llm", post(handlers::set_llm_config))
        // Analytics
        .route("/v1/users/{user_id}/stats", get(handlers::user_stats))
        .route(
            "/v1/users/{user_id}/frequency",
            get(handlers::memory_frequency),
        )
        .route(
            "/v1/users/{user_id}/top-entities",
            get(handlers::top_entities),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .route("/health", get(handlers::health))
        .layer(cors)
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    info!("MemMe server starting on http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
