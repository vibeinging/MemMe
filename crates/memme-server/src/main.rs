use std::collections::HashSet;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Router,
};
use clap::{Parser, ValueEnum};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{info, warn};

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::MemoryError;
use memme_embeddings::Embedder;

mod handlers;

const FULL_IMPORT_BODY_LIMIT_BYTES: usize = 32 * 1024 * 1024;

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

    /// SQLite database path
    #[arg(long, default_value = "memory.db")]
    db_path: String,

    /// Embedding provider. ONNX runs locally and needs no API key.
    #[arg(long, env = "EMBEDDING_PROVIDER", value_enum, default_value = "onnx")]
    embedding_provider: EmbeddingProvider,

    /// Embedding dimensions. Defaults to the selected model's dimensions.
    #[arg(long)]
    embedding_dims: Option<usize>,

    /// OpenAI-compatible embedding model name
    #[arg(long, default_value = "text-embedding-3-small")]
    embedding_model: String,

    /// Local ONNX model. The default is the compact Chinese model.
    #[arg(
        long,
        env = "ONNX_EMBEDDING_MODEL",
        value_enum,
        default_value = "bge-small-zh-v15"
    )]
    onnx_embedding_model: OnnxEmbeddingModel,

    /// API key used only by the embedding provider
    #[arg(long, env = "EMBEDDING_API_KEY", hide_env_values = true)]
    embedding_api_key: Option<String>,

    /// Legacy shared OpenAI key. Prefer EMBEDDING_API_KEY and LLM_API_KEY.
    #[arg(long, env = "OPENAI_API_KEY", hide_env_values = true)]
    openai_api_key: Option<String>,

    /// Embedding endpoint URL (e.g. https://api.openai.com/v1/embeddings)
    #[arg(long, env = "EMBEDDING_URL")]
    embedding_url: Option<String>,

    /// API key used only by the optional LLM provider
    #[arg(long, env = "LLM_API_KEY", hide_env_values = true)]
    llm_api_key: Option<String>,

    /// Optional LLM endpoint URL (e.g. https://api.openai.com/v1/chat/completions)
    #[arg(long, env = "LLM_URL")]
    llm_url: Option<String>,

    /// LLM model name
    #[arg(long, default_value = "gpt-4.1-nano")]
    llm_model: String,

    /// Hostnames allowed to use private addresses or plain HTTP for an LLM.
    /// Use only for trusted, operator-managed services such as local Ollama.
    #[arg(
        long,
        env = "MEMME_LLM_ALLOWED_HOSTS",
        value_delimiter = ',',
        hide_env_values = true
    )]
    llm_allowed_hosts: Vec<String>,

    /// Use mock embedder (no external API needed, for UI testing)
    #[arg(long, default_value = "false")]
    mock: bool,

    /// API key for bearer token authentication (optional, if not set no auth required)
    #[arg(long, env = "MEMME_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Allowed CORS origin (can be specified multiple times). If not set, only same-origin is allowed.
    #[arg(long)]
    cors_origin: Vec<String>,

    /// Directory used by POST /v1/backups
    #[arg(long, env = "MEMME_BACKUP_DIR", default_value = "backups")]
    backup_dir: PathBuf,
}

#[derive(Clone, Debug, ValueEnum)]
enum EmbeddingProvider {
    Onnx,
    Openai,
    Mock,
}

#[derive(Clone, Debug, ValueEnum)]
enum OnnxEmbeddingModel {
    MultilingualE5Small,
    MultilingualE5Base,
    MultilingualE5Large,
    BgeSmallZhV15,
    AllMiniLmL6V2,
}

impl OnnxEmbeddingModel {
    fn as_core_model(&self) -> memme_embeddings::onnx::OnnxModel {
        use memme_embeddings::onnx::OnnxModel;
        match self {
            Self::MultilingualE5Small => OnnxModel::MultilingualE5Small,
            Self::MultilingualE5Base => OnnxModel::MultilingualE5Base,
            Self::MultilingualE5Large => OnnxModel::MultilingualE5Large,
            Self::BgeSmallZhV15 => OnnxModel::BgeSmallZhV15,
            Self::AllMiniLmL6V2 => OnnxModel::AllMiniLmL6V2,
        }
    }
}

#[derive(Clone)]
pub struct RestoreControl {
    pending: Arc<tokio::sync::Mutex<Option<PathBuf>>>,
    notify: Arc<tokio::sync::Notify>,
}

impl RestoreControl {
    fn new() -> Self {
        Self {
            pending: Arc::new(tokio::sync::Mutex::new(None)),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub async fn request(&self, path: PathBuf) -> std::result::Result<(), String> {
        let mut pending = self.pending.lock().await;
        if pending.is_some() {
            return Err("a backup restore is already pending".to_string());
        }
        *pending = Some(path);
        drop(pending);
        self.notify.notify_one();
        Ok(())
    }

    async fn wait(&self) {
        self.notify.notified().await;
    }

    async fn take(&self) -> Option<PathBuf> {
        self.pending.lock().await.take()
    }
}

type SharedLlmConfig = Arc<RwLock<Option<memme_llm::openai::OpenAIConfig>>>;

pub struct AppState {
    pub store: MemoryStore,
    pub api_key: Option<String>,
    pub backup_dir: PathBuf,
    pub restore_control: RestoreControl,
    pub llm_config: SharedLlmConfig,
    pub llm_allowed_hosts: Arc<HashSet<String>>,
    pub full_import_semaphore: Arc<tokio::sync::Semaphore>,
}

#[derive(Clone)]
pub struct FullImportPermit(pub(crate) Arc<tokio::sync::OwnedSemaphorePermit>);

fn build_store(config: &MemoryConfig, embedder: Arc<dyn Embedder>) -> Result<MemoryStore> {
    Ok(MemoryStore::new(config.clone(), embedder)?)
}

fn forbidden_llm_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let octets = ip.octets();
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.is_broadcast()
                || ip.is_multicast()
                || octets[0] == 0
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
                || (octets[0] == 198 && (octets[1] == 18 || octets[1] == 19))
                || (octets[0] >= 224)
        }
        IpAddr::V6(ip) => {
            if let Some(ipv4) = ip.to_ipv4() {
                return forbidden_llm_ip(IpAddr::V4(ipv4));
            }
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || (ip.segments()[0] & 0xfe00) == 0xfc00
                || (ip.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

async fn resolve_llm_endpoint(
    raw: &str,
    allowed_hosts: &HashSet<String>,
) -> std::result::Result<Vec<std::net::SocketAddr>, String> {
    let parsed = url::Url::parse(raw).map_err(|error| format!("invalid LLM URL: {error}"))?;
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err("LLM URL must not contain credentials".to_string());
    }
    if parsed.fragment().is_some() {
        return Err("LLM URL must not contain a fragment".to_string());
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| "LLM URL must include a hostname".to_string())?
        .trim_end_matches('.')
        .to_ascii_lowercase();
    let explicitly_allowed = allowed_hosts.contains(&host);
    if explicitly_allowed {
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err("LLM URL protocol must be http or https".to_string());
        }
    } else if parsed.scheme() != "https" {
        return Err(
            "LLM URL must use https unless its hostname is in MEMME_LLM_ALLOWED_HOSTS".to_string(),
        );
    }
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| "LLM URL has no usable port".to_string())?;
    let resolved: Vec<_> = tokio::net::lookup_host((host.as_str(), port))
        .await
        .map_err(|error| format!("cannot resolve LLM hostname: {error}"))?
        .collect();
    if resolved.is_empty() {
        return Err("LLM hostname resolved to no addresses".to_string());
    }
    if !explicitly_allowed
        && resolved
            .iter()
            .any(|address| forbidden_llm_ip(address.ip()))
    {
        return Err(
            "LLM hostname resolves to a loopback, private, link-local, or reserved address"
                .to_string(),
        );
    }
    Ok(resolved)
}

pub(crate) async fn validate_llm_endpoint(
    raw: &str,
    allowed_hosts: &HashSet<String>,
) -> std::result::Result<(), String> {
    resolve_llm_endpoint(raw, allowed_hosts).await.map(|_| ())
}

async fn apply_persisted_llm_config(
    store: &MemoryStore,
    shared: &SharedLlmConfig,
    allowed_hosts: &HashSet<String>,
) -> Result<()> {
    let configured = shared
        .read()
        .map_err(|_| anyhow::anyhow!("LLM configuration lock is poisoned"))?
        .clone();
    let Some(mut config) = configured else {
        if store.load_persisted_llm_config()?.is_some() {
            warn!(
                "Persisted LLM endpoint is present but no LLM_API_KEY is available; LLM remains disabled"
            );
        }
        return Ok(());
    };
    if let Some((model, base_url)) = store.load_persisted_llm_config()? {
        config.model = model;
        config.base_url = base_url;
    }
    let resolved = resolve_llm_endpoint(&config.base_url, allowed_hosts)
        .await
        .map_err(anyhow::Error::msg)?;
    let provider_config = config.clone();
    let provider = tokio::task::spawn_blocking(move || {
        Arc::new(
            memme_llm::openai::OpenAIProvider::new_with_resolved_addresses(
                provider_config,
                &resolved,
            ),
        )
    })
    .await
    .map_err(|error| anyhow::anyhow!("cannot create LLM client: {error}"))?;
    store.set_llm_provider(provider);
    *shared
        .write()
        .map_err(|_| anyhow::anyhow!("LLM configuration lock is poisoned"))? = Some(config);
    Ok(())
}

async fn build_openai_provider(
    config: memme_llm::openai::OpenAIConfig,
    resolved: Vec<std::net::SocketAddr>,
) -> std::result::Result<Arc<memme_llm::openai::OpenAIProvider>, String> {
    tokio::task::spawn_blocking(move || {
        Arc::new(memme_llm::openai::OpenAIProvider::new_with_resolved_addresses(config, &resolved))
    })
    .await
    .map_err(|error| format!("cannot create LLM client: {error}"))
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let max_len = left.len().max(right.len());
    let mut difference = left.len() ^ right.len();
    for index in 0..max_len {
        difference |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
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
                if !constant_time_eq(token.as_bytes(), expected_key.as_bytes()) {
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

/// Admit only one full import before Axum reads and deserializes its JSON body.
/// This keeps concurrent 32 MiB requests from being buffered at the same time.
async fn full_import_admission(
    State(state): State<Arc<AppState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let permit = match state.full_import_semaphore.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            let mut response = (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({
                    "success": false,
                    "error": "another full import is already in progress"
                })),
            )
                .into_response();
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
            return response;
        }
    };
    request
        .extensions_mut()
        .insert(FullImportPermit(Arc::new(permit)));
    next.run(request).await
}

async fn normalize_request_error(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    let status = response.status();
    let should_normalize = matches!(
        status,
        StatusCode::BAD_REQUEST
            | StatusCode::PAYLOAD_TOO_LARGE
            | StatusCode::UNSUPPORTED_MEDIA_TYPE
            | StatusCode::UNPROCESSABLE_ENTITY
    ) && response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map_or(true, |value| !value.starts_with("application/json"));
    if !should_normalize {
        return response;
    }
    let message = match status {
        StatusCode::PAYLOAD_TOO_LARGE => "request body is too large",
        StatusCode::UNSUPPORTED_MEDIA_TYPE => "content-type must be application/json",
        StatusCode::UNPROCESSABLE_ENTITY => "request body does not match the endpoint schema",
        _ => "invalid request body",
    };
    (
        status,
        axum::Json(serde_json::json!({
            "success": false,
            "error": message,
        })),
    )
        .into_response()
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

    let provider = if cli.mock {
        EmbeddingProvider::Mock
    } else {
        cli.embedding_provider.clone()
    };
    let embedder: Arc<dyn Embedder> = match provider {
        EmbeddingProvider::Onnx => {
            let embedder = memme_embeddings::onnx::OnnxEmbedder::with_model(
                cli.onnx_embedding_model.as_core_model(),
            )?;
            if let Some(dims) = cli.embedding_dims {
                anyhow::ensure!(
                    dims == embedder.dimensions(),
                    "--embedding-dims {dims} does not match ONNX model dimensions {}",
                    embedder.dimensions()
                );
            }
            info!(model = embedder.model_name(), "Using local ONNX embeddings");
            Arc::new(embedder)
        }
        EmbeddingProvider::Mock => {
            let dims = cli.embedding_dims.unwrap_or(384);
            info!(dims, "Running with mock embeddings");
            Arc::new(memme_embeddings::mock::MockEmbedder::new(dims))
        }
        EmbeddingProvider::Openai => {
            let api_key = cli
                .embedding_api_key
                .as_deref()
                .or(cli.openai_api_key.as_deref())
                .ok_or_else(|| {
                    anyhow::anyhow!("EMBEDDING_API_KEY is required for --embedding-provider openai")
                })?;
            let embed_url = cli
                .embedding_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1/embeddings");
            let dims = cli.embedding_dims.unwrap_or_else(|| {
                if cli.embedding_model == "text-embedding-3-large" {
                    3072
                } else {
                    1536
                }
            });
            let model = if cli.embedding_model == "text-embedding-3-small" && dims == 1536 {
                memme_embeddings::openai::OpenAiModel::TextEmbedding3Small
            } else if cli.embedding_model == "text-embedding-3-large" && dims == 3072 {
                memme_embeddings::openai::OpenAiModel::TextEmbedding3Large
            } else {
                memme_embeddings::openai::OpenAiModel::Custom {
                    name: cli.embedding_model.clone(),
                    dims,
                    send_dims: true,
                }
            };
            info!(model = %cli.embedding_model, dims, "Using remote embeddings");
            Arc::new(
                memme_embeddings::openai::OpenAiEmbedder::new(api_key, embed_url).with_model(model),
            )
        }
    };

    let config = MemoryConfig {
        db_path: cli.db_path.clone(),
        embedding_dims: embedder.dimensions(),
        ..Default::default()
    };

    let llm_allowed_hosts: Arc<HashSet<String>> = Arc::new(
        cli.llm_allowed_hosts
            .iter()
            .map(|host| host.trim().trim_end_matches('.').to_ascii_lowercase())
            .filter(|host| !host.is_empty())
            .collect(),
    );

    let llm_key = cli
        .llm_api_key
        .as_deref()
        .or_else(|| cli.llm_url.as_ref().and(cli.openai_api_key.as_deref()));
    let llm_config = if let Some(api_key) = llm_key {
        let llm_url = cli
            .llm_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1/chat/completions");
        let llm_config = memme_llm::openai::OpenAIConfig {
            api_key: api_key.to_string(),
            base_url: llm_url.to_string(),
            model: cli.llm_model.clone(),
        };
        validate_llm_endpoint(&llm_config.base_url, &llm_allowed_hosts)
            .await
            .map_err(anyhow::Error::msg)?;
        info!(model = %cli.llm_model, "LLM enabled");
        Some(llm_config)
    } else {
        info!("LLM disabled; add/search/event ingestion remain available");
        None
    };
    let llm_config = Arc::new(RwLock::new(llm_config));

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

    let addr = format!("{}:{}", cli.host, cli.port);
    loop {
        let store = build_store(&config, embedder.clone())?;
        apply_persisted_llm_config(&store, &llm_config, &llm_allowed_hosts).await?;
        let restore_control = RestoreControl::new();
        let state = Arc::new(AppState {
            store,
            api_key: cli.api_key.clone(),
            backup_dir: cli.backup_dir.clone(),
            restore_control: restore_control.clone(),
            llm_config: llm_config.clone(),
            llm_allowed_hosts: llm_allowed_hosts.clone(),
            full_import_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        });
        let app = build_router(state, cors.clone());

        info!("MemMe server starting on http://{addr}");
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        let shutdown = restore_control.clone();
        axum::serve(listener, app)
            .with_graceful_shutdown(async move { shutdown.wait().await })
            .await?;

        let Some(backup_path) = restore_control.take().await else {
            return Ok(());
        };
        let backup_path = backup_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("backup path is not valid UTF-8"))?;
        match MemoryStore::restore_from_backup(backup_path, &config) {
            Ok(()) => info!(
                backup = backup_path,
                "Backup restored; REST server restarting"
            ),
            Err(error @ MemoryError::RollbackFailed { .. }) => {
                return Err(anyhow::anyhow!(
                    "backup restore rollback failed; server stopped to avoid creating an empty database: {error}"
                ));
            }
            Err(error) => {
                MemoryStore::validate_backup(&config.db_path).map_err(|verification_error| {
                    anyhow::anyhow!(
                        "backup restore failed ({error}) and the previous primary is not valid; server stopped: {verification_error}"
                    )
                })?;
                let verified = build_store(&config, embedder.clone()).map_err(
                    |verification_error| {
                        anyhow::anyhow!(
                            "backup restore failed ({error}) and the previous primary cannot be reopened; server stopped: {verification_error}"
                        )
                    },
                )?;
                drop(verified);
                warn!(
                    backup = backup_path,
                    %error,
                    "Backup restore failed; verified previous database kept; REST server restarting"
                );
            }
        }
    }
}

fn build_router(state: Arc<AppState>, cors: CorsLayer) -> Router {
    Router::new()
        // Basic CRUD
        .route("/v1/memories", post(handlers::add_memory))
        .route("/v1/memories/search", post(handlers::search_memories))
        .route("/v1/memories/hybrid-search", post(handlers::hybrid_search))
        .route("/v1/memories/list", post(handlers::list_memories))
        .route("/v1/memories/export", post(handlers::export_memories))
        .route("/v1/memories/import", post(handlers::import_memories))
        .route("/v1/data/export", post(handlers::full_export))
        .route(
            "/v1/data/import",
            post(handlers::full_import)
                .layer(DefaultBodyLimit::max(FULL_IMPORT_BODY_LIMIT_BYTES))
                .layer(middleware::from_fn_with_state(
                    state.clone(),
                    full_import_admission,
                )),
        )
        .route("/v1/memories/{id}", get(handlers::get_memory))
        .route("/v1/memories/{id}", put(handlers::update_memory))
        .route("/v1/memories/{id}", delete(handlers::delete_memory))
        .route("/v1/memories/{id}/history", get(handlers::memory_history))
        .route("/v1/memories", delete(handlers::delete_all_memories))
        // Episodes: removed (merged into traces; use search() with resolution filter)
        // Knowledge graph
        .route("/v1/graph", post(handlers::graph_add))
        .route("/v1/graph/search", post(handlers::graph_search))
        // Recall (multi-layer unified retrieval)
        .route("/v1/recall", post(handlers::recall))
        // Durable conversation ingestion and memory lifecycle
        .route("/v1/events", post(handlers::append_events))
        .route(
            "/v1/sessions/{session_id}/compact",
            post(handlers::compact_session),
        )
        .route("/v1/meditations", post(handlers::meditate))
        // Data safety
        .route("/v1/backups", post(handlers::create_backup))
        .route("/v1/backups/restore", post(handlers::restore_backup))
        .route("/v1/replica/status", get(handlers::replica_status))
        .route("/v1/replica/sync", post(handlers::sync_replica))
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
        .route("/v1/users/{user_id}", delete(handlers::delete_user_data))
        .route("/diagnose", get(handlers::diagnose))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .route("/health", get(handlers::health))
        .layer(middleware::from_fn(normalize_request_error))
        .layer(cors)
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request, StatusCode};
    use clap::CommandFactory;
    use memme_embeddings::mock::MockEmbedder;
    use tower::ServiceExt;

    fn test_app() -> Router {
        test_app_with_api_key(None)
    }

    fn test_app_with_api_key(api_key: Option<&str>) -> Router {
        let config = MemoryConfig::new(":memory:", 32);
        let store = MemoryStore::new(config, Arc::new(MockEmbedder::new(32))).unwrap();
        let state = Arc::new(AppState {
            store,
            api_key: api_key.map(str::to_string),
            backup_dir: PathBuf::from("target/test-backups"),
            restore_control: RestoreControl::new(),
            llm_config: Arc::new(RwLock::new(None)),
            llm_allowed_hosts: Arc::new(HashSet::new()),
            full_import_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        });
        build_router(state, CorsLayer::new())
    }

    #[test]
    fn cli_hides_all_secret_environment_values() {
        let command = Cli::command();
        for id in [
            "embedding_api_key",
            "openai_api_key",
            "llm_api_key",
            "api_key",
        ] {
            let argument = command
                .get_arguments()
                .find(|argument| argument.get_id() == id)
                .unwrap_or_else(|| panic!("missing CLI argument {id}"));
            assert!(
                argument.is_hide_env_values_set(),
                "{id} must hide its environment value in --help"
            );
        }
    }

    #[test]
    fn bearer_tokens_use_length_safe_equality() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secrex"));
        assert!(!constant_time_eq(b"secret", b"secret-longer"));
        assert!(!constant_time_eq(b"secret-longer", b"secret"));
    }

    #[tokio::test]
    async fn llm_endpoint_rejects_private_and_unsafe_destinations() {
        let allowed = HashSet::new();
        for endpoint in [
            "http://api.example.com/v1/chat/completions",
            "file:///etc/passwd",
            "https://127.0.0.1/v1/chat/completions",
            "https://[::1]/v1/chat/completions",
            "https://10.1.2.3/v1/chat/completions",
            "https://169.254.169.254/latest/meta-data",
        ] {
            assert!(
                validate_llm_endpoint(endpoint, &allowed).await.is_err(),
                "{endpoint} must be rejected"
            );
        }

        let explicitly_allowed = HashSet::from(["localhost".to_string()]);
        assert!(validate_llm_endpoint(
            "http://localhost:11434/v1/chat/completions",
            &explicitly_allowed
        )
        .await
        .is_ok());
    }

    #[tokio::test]
    async fn persisted_llm_model_and_url_replace_startup_defaults() {
        let store = MemoryStore::new(
            MemoryConfig::new(":memory:", 32),
            Arc::new(MockEmbedder::new(32)),
        )
        .unwrap();
        store
            .save_llm_config(
                "not-persisted",
                "deepseek-chat",
                "http://localhost:11434/v1/chat/completions",
            )
            .unwrap();
        let shared = Arc::new(RwLock::new(Some(memme_llm::openai::OpenAIConfig {
            api_key: "runtime-key".into(),
            base_url: "https://api.openai.com/v1/chat/completions".into(),
            model: "startup-default".into(),
        })));
        let allowed = HashSet::from(["localhost".to_string()]);

        apply_persisted_llm_config(&store, &shared, &allowed)
            .await
            .unwrap();

        assert!(store.has_llm());
        let configured = shared.read().unwrap();
        let configured = configured.as_ref().unwrap();
        assert_eq!(configured.model, "deepseek-chat");
        assert_eq!(
            configured.base_url,
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(configured.api_key, "runtime-key");
    }

    #[tokio::test]
    async fn health_is_public_but_diagnose_requires_bearer_auth() {
        let app = test_app_with_api_key(Some("diagnostic-secret"));
        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/diagnose")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let authorized = app
            .oneshot(
                Request::builder()
                    .uri("/diagnose")
                    .header("authorization", "Bearer diagnostic-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn full_import_has_a_route_specific_body_limit() {
        let app = test_app();
        let oversized = format!(
            "{{\"padding\":\"{}\"}}",
            "x".repeat(FULL_IMPORT_BODY_LIMIT_BYTES + 1)
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/data/import")
                    .header("content-type", "application/json")
                    .body(Body::from(oversized))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn json_extractor_errors_use_the_api_error_envelope() {
        let response = test_app()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/events")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"session_id": 7}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["success"], false);
        assert!(json["error"].is_string());
    }

    #[tokio::test]
    async fn full_import_admission_runs_before_json_deserialization() {
        let config = MemoryConfig::new(":memory:", 32);
        let store = MemoryStore::new(config, Arc::new(MockEmbedder::new(32))).unwrap();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
        let held = semaphore.clone().acquire_owned().await.unwrap();
        let state = Arc::new(AppState {
            store,
            api_key: None,
            backup_dir: PathBuf::from("target/test-backups"),
            restore_control: RestoreControl::new(),
            llm_config: Arc::new(RwLock::new(None)),
            llm_allowed_hosts: Arc::new(HashSet::new()),
            full_import_semaphore: semaphore,
        });
        let app = build_router(state, CorsLayer::new());

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/data/import")
                    .header("content-type", "application/json")
                    .body(Body::from("not-json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("1")
        );
        drop(held);
    }

    async fn request_json(
        app: &Router,
        method: Method,
        path: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    async fn post_json(app: &Router, path: &str, body: serde_json::Value) -> serde_json::Value {
        let (status, body) = request_json(app, Method::POST, path, body).await;
        assert_eq!(status, StatusCode::OK);
        body
    }

    #[tokio::test]
    async fn event_retries_are_idempotent_and_recall_is_pet_scoped() {
        let app = test_app();
        let pet_a = serde_json::json!({
            "session_id": "session-a",
            "user_id": "owner-1",
            "agent_id": "pet-a",
            "app_id": "xiaozhi",
            "run_id": "run-a",
            "messages": [{
                "event_id": "session-a-user-1",
                "role": "user",
                "content": "我给小蓝买了蓝色鲸鱼玩具",
                "image_url": null,
                "image_type": null,
                "timestamp": "2026-09-01T10:00:00Z"
            }]
        });
        let first = post_json(&app, "/v1/events", pet_a.clone()).await;
        assert_eq!(first["data"]["events_appended"], 1);
        assert_eq!(first["data"]["events_replayed"], 0);

        let replay = post_json(&app, "/v1/events", pet_a).await;
        assert_eq!(replay["data"]["events_appended"], 0);
        assert_eq!(replay["data"]["events_replayed"], 1);

        post_json(
            &app,
            "/v1/events",
            serde_json::json!({
                "session_id": "session-b",
                "user_id": "owner-1",
                "agent_id": "pet-b",
                "app_id": "xiaozhi",
                "run_id": "run-b",
                "messages": [{
                    "event_id": "session-b-user-1",
                    "role": "user",
                    "content": "我给小红买了红色火箭玩具",
                    "image_url": null,
                    "image_type": null,
                    "timestamp": "2026-09-01T10:01:00Z"
                }]
            }),
        )
        .await;

        let recalled = post_json(
            &app,
            "/v1/recall",
            serde_json::json!({
                "query": "玩具",
                "user_id": "owner-1",
                "agent_id": "pet-a",
                "app_id": "xiaozhi",
                "limit": 10
            }),
        )
        .await;
        let memories = recalled["data"]["memories"].as_array().unwrap();
        assert!(memories
            .iter()
            .any(|memory| memory["content"].as_str().unwrap().contains("蓝色鲸鱼")));
        assert!(!memories
            .iter()
            .any(|memory| memory["content"].as_str().unwrap().contains("红色火箭")));
    }

    #[tokio::test]
    async fn session_owner_and_pet_scope_conflicts_are_rejected() {
        let app = test_app();
        let event = |event_id: &str, user_id: &str, agent_id: &str| {
            serde_json::json!({
                "session_id": "shared-session",
                "user_id": user_id,
                "agent_id": agent_id,
                "app_id": "xiaozhi",
                "run_id": "run-1",
                "messages": [{
                    "event_id": event_id,
                    "role": "user",
                    "content": "session boundary test",
                    "image_url": null,
                    "image_type": null,
                    "timestamp": "2026-09-01T10:00:00Z"
                }]
            })
        };

        post_json(
            &app,
            "/v1/events",
            event("session-owner-event-1", "owner-1", "pet-a"),
        )
        .await;

        let (status, body) = request_json(
            &app,
            Method::POST,
            "/v1/events",
            event("session-owner-event-2", "owner-2", "pet-a"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("another user"));

        let (status, body) = request_json(
            &app,
            Method::POST,
            "/v1/events",
            event("session-owner-event-3", "owner-1", "pet-b"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("agent_id"));

        let exported = post_json(
            &app,
            "/v1/data/export",
            serde_json::json!({"user_id": "owner-1"}),
        )
        .await;
        assert_eq!(exported["data"]["events"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn full_export_import_and_confirmed_user_delete_cover_all_layers() {
        let app = test_app();
        post_json(
            &app,
            "/v1/events",
            serde_json::json!({
                "session_id": "portable-session",
                "user_id": "owner-portable",
                "agent_id": "pet-a",
                "app_id": "xiaozhi",
                "run_id": "run-1",
                "messages": [{
                    "event_id": "portable-event-1",
                    "role": "user",
                    "content": "portable event",
                    "image_url": null,
                    "image_type": null,
                    "timestamp": "2026-09-01T10:00:00Z"
                }]
            }),
        )
        .await;

        let exported = post_json(
            &app,
            "/v1/data/export",
            serde_json::json!({"user_id": "owner-portable"}),
        )
        .await;
        let full_export = exported["data"].clone();
        assert_eq!(full_export["sessions"].as_array().unwrap().len(), 1);
        assert_eq!(full_export["events"].as_array().unwrap().len(), 1);

        let (status, _) = request_json(
            &app,
            Method::DELETE,
            "/v1/users/owner-portable",
            serde_json::json!({"confirm_user_id": "wrong-user"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, deleted) = request_json(
            &app,
            Method::DELETE,
            "/v1/users/owner-portable",
            serde_json::json!({"confirm_user_id": "owner-portable"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(deleted["data"]["deleted"], true);

        let after_delete = post_json(
            &app,
            "/v1/data/export",
            serde_json::json!({"user_id": "owner-portable"}),
        )
        .await;
        assert!(after_delete["data"]["sessions"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(after_delete["data"]["events"]
            .as_array()
            .unwrap()
            .is_empty());

        let (status, _) = request_json(
            &app,
            Method::POST,
            "/v1/data/import",
            serde_json::json!({
                "confirm": "wrong-confirmation",
                "export": full_export.clone(),
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let imported = post_json(
            &app,
            "/v1/data/import",
            serde_json::json!({
                "confirm": "import-full-export",
                "export": full_export,
            }),
        )
        .await;
        assert_eq!(imported["data"]["sessions"], 1);
        assert_eq!(imported["data"]["events"], 1);

        let after_import = post_json(
            &app,
            "/v1/data/export",
            serde_json::json!({"user_id": "owner-portable"}),
        )
        .await;
        assert_eq!(after_import["data"]["events"].as_array().unwrap().len(), 1);

        let recalled = post_json(
            &app,
            "/v1/recall",
            serde_json::json!({
                "query": "portable event",
                "user_id": "owner-portable",
                "agent_id": "pet-a",
                "app_id": "xiaozhi",
                "limit": 10
            }),
        )
        .await;
        assert!(recalled["data"]["memories"]
            .as_array()
            .unwrap()
            .iter()
            .any(|memory| memory["content"] == "portable event"));
    }

    #[tokio::test]
    async fn backup_restore_requires_exact_confirmation_and_valid_memme_database() {
        let test_dir =
            PathBuf::from("target").join(format!("restore-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&test_dir).unwrap();
        let db_path = test_dir.join("primary.db");
        let backup_path = test_dir.join("known-good.db");
        let config = MemoryConfig::new(db_path.to_str().unwrap(), 32);
        let store = MemoryStore::new(config, Arc::new(MockEmbedder::new(32))).unwrap();
        store
            .add(
                "backup marker",
                memme_core::types::AddOptions::new("owner-1"),
            )
            .unwrap();
        store.backup_to_path(backup_path.to_str().unwrap()).unwrap();

        let restore_control = RestoreControl::new();
        let state = Arc::new(AppState {
            store,
            api_key: None,
            backup_dir: test_dir.clone(),
            restore_control: restore_control.clone(),
            llm_config: Arc::new(RwLock::new(None)),
            llm_allowed_hosts: Arc::new(HashSet::new()),
            full_import_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        });
        let app = build_router(state, CorsLayer::new());

        let (backup_status, _) = request_json(
            &app,
            Method::POST,
            "/v1/backups",
            serde_json::json!({"filename": "private-created.db"}),
        )
        .await;
        assert_eq!(backup_status, StatusCode::OK);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&test_dir).unwrap().permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                std::fs::metadata(test_dir.join("private-created.db"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }

        let (status, _) = request_json(
            &app,
            Method::POST,
            "/v1/backups/restore",
            serde_json::json!({
                "filename": "known-good.db",
                "confirm": "wrong",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let directory_backup = test_dir.join("directory.db");
        std::fs::create_dir(&directory_backup).unwrap();
        let (status, _) = request_json(
            &app,
            Method::POST,
            "/v1/backups/restore",
            serde_json::json!({
                "filename": "directory.db",
                "confirm": "restore:directory.db",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        #[cfg(unix)]
        {
            let link_path = test_dir.join("linked.db");
            std::os::unix::fs::symlink(&backup_path, &link_path).unwrap();
            let (status, _) = request_json(
                &app,
                Method::POST,
                "/v1/backups/restore",
                serde_json::json!({
                    "filename": "linked.db",
                    "confirm": "restore:linked.db",
                }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
        }

        let (status, body) = request_json(
            &app,
            Method::POST,
            "/v1/backups/restore",
            serde_json::json!({
                "filename": "known-good.db",
                "confirm": "restore:known-good.db",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(body["data"]["server_restarting"], true);
        assert_eq!(
            restore_control.take().await.as_deref(),
            Some(backup_path.as_path())
        );

        drop(app);
        std::fs::remove_dir_all(test_dir).unwrap();
    }

    #[test]
    fn openapi_documents_registered_routes_without_removed_smart_routes() {
        let document: serde_yaml::Value =
            serde_yaml::from_str(include_str!("../../../docs/openapi.yaml")).unwrap();
        let paths = document["paths"].as_mapping().unwrap();
        for path in [
            "/health",
            "/diagnose",
            "/v1/config",
            "/v1/config/llm",
            "/v1/memories",
            "/v1/memories/search",
            "/v1/memories/hybrid-search",
            "/v1/memories/list",
            "/v1/memories/export",
            "/v1/memories/import",
            "/v1/data/export",
            "/v1/data/import",
            "/v1/memories/{id}",
            "/v1/memories/{id}/history",
            "/v1/graph",
            "/v1/graph/search",
            "/v1/recall",
            "/v1/events",
            "/v1/sessions/{session_id}/compact",
            "/v1/meditations",
            "/v1/backups",
            "/v1/backups/restore",
            "/v1/replica/status",
            "/v1/replica/sync",
            "/v1/users/{user_id}/stats",
            "/v1/users/{user_id}/frequency",
            "/v1/users/{user_id}/top-entities",
            "/v1/users/{user_id}",
        ] {
            assert!(
                paths.contains_key(serde_yaml::Value::String(path.to_string())),
                "OpenAPI is missing {path}"
            );
        }
        for removed in [
            "/v1/memories/smart",
            "/v1/memories/smart/messages",
            "/v1/recall/{recall_id}/feedback",
        ] {
            assert!(
                !paths.contains_key(serde_yaml::Value::String(removed.to_string())),
                "OpenAPI still documents removed route {removed}"
            );
        }
    }
}
