use std::sync::Arc;

use axum::{
    extract::{Extension, Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};

use memme_core::memory::MemoryStore;
use memme_core::types::*;

use crate::{AppState, FullImportPermit};

// ── Request / Response Types ──

#[derive(Deserialize)]
pub struct AddRequest {
    pub content: String,
    pub user_id: String,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub importance: Option<f32>,
    pub immutable: Option<bool>,
    pub expiration_date: Option<String>,
    pub categories: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub user_id: String,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
    pub limit: Option<usize>,
    pub threshold: Option<f32>,
    pub keyword_search: Option<bool>,
    pub fields: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct UpdateRequest {
    pub content: String,
    pub metadata: Option<serde_json::Value>,
    pub timestamp: Option<String>,
}

#[derive(Deserialize)]
pub struct ListRequest {
    pub user_id: String,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct DeleteAllQuery {
    pub user_id: String,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
}

#[derive(Deserialize)]
pub struct ExportRequest {
    pub user_id: Option<String>,
}

#[derive(Deserialize)]
pub struct FullExportRequest {
    pub user_id: String,
}

#[derive(Deserialize)]
pub struct FullImportRequest {
    pub confirm: String,
    pub export: FullExport,
}

#[derive(Deserialize)]
pub struct DeleteUserRequest {
    pub confirm_user_id: String,
}

#[derive(Serialize)]
struct ApiOk<T: Serialize> {
    success: bool,
    data: T,
}

#[derive(Serialize)]
struct ApiErr {
    success: bool,
    error: String,
}

fn ok_json<T: Serialize>(data: T) -> impl IntoResponse {
    Json(ApiOk {
        success: true,
        data,
    })
}

fn err_json(status: StatusCode, msg: impl Into<String>) -> impl IntoResponse {
    (
        status,
        Json(ApiErr {
            success: false,
            error: msg.into(),
        }),
    )
}

// ── Handlers ──

pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "service": "memme" }))
}

pub async fn diagnose(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || state.store.diagnose()).await {
        Ok(report) => Json(report).into_response(),
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("diagnostic task failed: {error}"),
        )
        .into_response(),
    }
}

fn scoped_metadata(
    metadata: Option<serde_json::Value>,
    agent_id: Option<&str>,
    app_id: Option<&str>,
    run_id: Option<&str>,
) -> Result<Option<serde_json::Value>, String> {
    let mut object = match metadata {
        Some(serde_json::Value::Object(object)) => object,
        Some(_) => return Err("metadata must be a JSON object".to_string()),
        None => serde_json::Map::new(),
    };

    for key in ["agent_id", "app_id", "run_id"] {
        if let Some(value) = object.get(key) {
            let valid = value
                .as_str()
                .map(|value| !value.trim().is_empty())
                .unwrap_or(false);
            if !valid {
                return Err(format!("metadata.{key} must be a non-empty string"));
            }
        }
    }

    for (key, value) in [
        ("agent_id", agent_id),
        ("app_id", app_id),
        ("run_id", run_id),
    ] {
        let Some(value) = value else {
            continue;
        };
        if value.trim().is_empty() {
            return Err(format!("{key} must not be empty"));
        }
        if let Some(existing) = object.get(key) {
            if existing.as_str() != Some(value) {
                return Err(format!("metadata.{key} conflicts with {key}"));
            }
        }
        object.insert(
            key.to_string(),
            serde_json::Value::String(value.to_string()),
        );
    }

    if object.is_empty() {
        Ok(None)
    } else {
        Ok(Some(serde_json::Value::Object(object)))
    }
}

// ── Config Handlers ──

#[derive(Deserialize)]
pub struct SetLlmRequest {
    pub api_key: String,
    pub model: Option<String>,
    pub base_url: Option<String>,
}

pub async fn get_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let has_llm = state.store.has_llm();
    ok_json(serde_json::json!({
        "has_llm": has_llm,
        "embedding_dims": state.store.embedding_dims(),
    }))
}

pub async fn set_llm_config(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetLlmRequest>,
) -> impl IntoResponse {
    let model = req.model.unwrap_or_else(|| "gpt-4.1-nano".to_string());
    let base_url = match req.base_url {
        Some(url) => url,
        None => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({"error": "base_url required (full endpoint URL)"})),
            )
                .into_response();
        }
    };

    let resolved = match crate::resolve_llm_endpoint(&base_url, &state.llm_allowed_hosts).await {
        Ok(resolved) => resolved,
        Err(error) => return err_json(StatusCode::BAD_REQUEST, error).into_response(),
    };

    let llm_config = memme_llm::openai::OpenAIConfig {
        api_key: req.api_key.clone(),
        base_url: base_url.clone(),
        model: model.clone(),
    };
    let llm = match crate::build_openai_provider(llm_config.clone(), resolved).await {
        Ok(llm) => llm,
        Err(error) => {
            return err_json(StatusCode::INTERNAL_SERVER_ERROR, error).into_response();
        }
    };
    let mut configured = match state.llm_config.write() {
        Ok(configured) => configured,
        Err(_) => {
            return err_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                "LLM configuration lock is poisoned",
            )
            .into_response()
        }
    };
    if let Err(error) = state.store.save_llm_config(&req.api_key, &model, &base_url) {
        return err_json(StatusCode::BAD_REQUEST, error.to_string()).into_response();
    }
    state.store.set_llm_provider(llm);
    *configured = Some(llm_config);
    ok_json(serde_json::json!({
        "configured": true,
        "model": model,
        "base_url": base_url,
    }))
    .into_response()
}

pub async fn add_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddRequest>,
) -> impl IntoResponse {
    let mut opts = AddOptions::new(&req.user_id);
    if let Some(aid) = req.agent_id {
        opts = opts.agent_id(aid);
    }
    if let Some(aid) = req.app_id {
        opts = opts.app_id(aid);
    }
    if let Some(rid) = req.run_id {
        opts = opts.run_id(rid);
    }
    if let Some(m) = req.metadata {
        opts = opts.metadata(m);
    }
    if let Some(i) = req.importance {
        opts = opts.importance(i);
    }
    if let Some(imm) = req.immutable {
        opts = opts.immutable(imm);
    }
    if let Some(exp) = req.expiration_date {
        opts = opts.expiration_date(exp);
    }
    if let Some(cats) = req.categories {
        opts = opts.categories(cats);
    }

    match state.store.add(&req.content, opts) {
        Ok(result) => ok_json(result).into_response(),
        Err(e) => err_json(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

pub async fn search_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let mut opts = SearchOptions::new(&req.user_id);
    if let Some(aid) = req.agent_id {
        opts = opts.agent_id(aid);
    }
    if let Some(aid) = req.app_id {
        opts = opts.app_id(aid);
    }
    if let Some(rid) = req.run_id {
        opts = opts.run_id(rid);
    }
    if let Some(k) = req.limit {
        opts = opts.limit(k);
    }
    if let Some(t) = req.threshold {
        opts = opts.threshold(t);
    }
    if req.keyword_search.unwrap_or(false) {
        opts = opts.keyword_search(true);
    }
    if let Some(f) = req.fields {
        opts = opts.fields(f);
    }

    match state.store.search(&req.query, opts) {
        Ok(results) => ok_json(results).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn get_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.get_trace(&id) {
        Ok(Some(result)) => ok_json(result).into_response(),
        Ok(None) => {
            err_json(StatusCode::NOT_FOUND, format!("Memory '{id}' not found")).into_response()
        }
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn update_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateRequest>,
) -> impl IntoResponse {
    let opts = if req.metadata.is_some() || req.timestamp.is_some() {
        let mut o = UpdateOptions::new();
        if let Some(m) = req.metadata {
            o = o.metadata(m);
        }
        if let Some(t) = req.timestamp {
            o = o.timestamp(t);
        }
        Some(o)
    } else {
        None
    };

    match state.store.update_trace(&id, &req.content, opts) {
        Ok(result) => ok_json(result).into_response(),
        Err(e) => {
            let status = if e.to_string().contains("not found") {
                StatusCode::NOT_FOUND
            } else if e.to_string().contains("immutable") || e.to_string().contains("Immutable") {
                StatusCode::FORBIDDEN
            } else {
                StatusCode::BAD_REQUEST
            };
            err_json(status, e.to_string()).into_response()
        }
    }
}

pub async fn delete_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.delete_trace(&id) {
        Ok(()) => ok_json(serde_json::json!({"deleted": true})).into_response(),
        Err(e) => {
            let status = if e.to_string().contains("not found") {
                StatusCode::NOT_FOUND
            } else if e.to_string().contains("immutable") || e.to_string().contains("Immutable") {
                StatusCode::FORBIDDEN
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            err_json(status, e.to_string()).into_response()
        }
    }
}

pub async fn memory_history(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.trace_history(&id) {
        Ok(records) => ok_json(records).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn delete_all_memories(
    State(state): State<Arc<AppState>>,
    Query(q): Query<DeleteAllQuery>,
) -> impl IntoResponse {
    match state.store.delete_all_traces(
        &q.user_id,
        q.agent_id.as_deref(),
        q.run_id.as_deref(),
        q.app_id.as_deref(),
    ) {
        Ok(count) => ok_json(serde_json::json!({"deleted_count": count})).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn list_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ListRequest>,
) -> impl IntoResponse {
    let mut opts = ListOptions::new(&req.user_id);
    if let Some(aid) = req.agent_id {
        opts = opts.agent_id(aid);
    }
    if let Some(aid) = req.app_id {
        opts = opts.app_id(aid);
    }
    if let Some(rid) = req.run_id {
        opts = opts.run_id(rid);
    }
    if let Some(l) = req.limit {
        opts = opts.limit(l);
    }

    match state.store.list_traces(opts) {
        Ok(results) => ok_json(results).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn export_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExportRequest>,
) -> impl IntoResponse {
    match state.store.export(req.user_id.as_deref()) {
        Ok(data) => ok_json(data).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn import_memories(
    State(state): State<Arc<AppState>>,
    Json(memories): Json<Vec<MemoryExport>>,
) -> impl IntoResponse {
    match state.store.import_memories(&memories) {
        Ok(count) => ok_json(serde_json::json!({"imported_count": count})).into_response(),
        Err(e) => err_json(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

pub async fn full_export(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FullExportRequest>,
) -> impl IntoResponse {
    let user_id = req.user_id.trim().to_string();
    if user_id.is_empty() {
        return err_json(StatusCode::BAD_REQUEST, "user_id must not be empty").into_response();
    }
    match tokio::task::spawn_blocking(move || state.store.full_export(Some(&user_id))).await {
        Ok(Ok(data)) => ok_json(data).into_response(),
        Ok(Err(error)) => {
            err_json(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response()
        }
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("full export task failed: {error}"),
        )
        .into_response(),
    }
}

pub async fn full_import(
    State(state): State<Arc<AppState>>,
    Extension(permit): Extension<FullImportPermit>,
    Json(req): Json<FullImportRequest>,
) -> impl IntoResponse {
    if req.confirm != "import-full-export" {
        return err_json(
            StatusCode::BAD_REQUEST,
            "confirm must equal 'import-full-export'",
        )
        .into_response();
    }
    match tokio::task::spawn_blocking(move || {
        let _permit = permit.0;
        state.store.full_import(&req.export)
    })
    .await
    {
        Ok(Ok(data)) => ok_json(data).into_response(),
        Ok(Err(error)) => err_json(StatusCode::BAD_REQUEST, error.to_string()).into_response(),
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("full import task failed: {error}"),
        )
        .into_response(),
    }
}

pub async fn delete_user_data(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
    Json(req): Json<DeleteUserRequest>,
) -> impl IntoResponse {
    if user_id.trim().is_empty() {
        return err_json(StatusCode::BAD_REQUEST, "user_id must not be empty").into_response();
    }
    if req.confirm_user_id != user_id {
        return err_json(
            StatusCode::BAD_REQUEST,
            "confirm_user_id must exactly match the user_id in the path",
        )
        .into_response();
    }
    let response_user_id = user_id.clone();
    match tokio::task::spawn_blocking(move || state.store.delete_user_data(&user_id)).await {
        Ok(Ok(())) => ok_json(serde_json::json!({
            "deleted": true,
            "user_id": response_user_id,
        }))
        .into_response(),
        Ok(Err(error)) => err_json(StatusCode::BAD_REQUEST, error.to_string()).into_response(),
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("delete user task failed: {error}"),
        )
        .into_response(),
    }
}

// ── Durable conversation ingestion ──

#[derive(Deserialize)]
pub struct AppendEventsRequest {
    pub session_id: String,
    pub user_id: String,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub messages: Vec<IdentifiedChatMessage>,
}

pub async fn append_events(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AppendEventsRequest>,
) -> impl IntoResponse {
    let metadata = match scoped_metadata(
        req.metadata,
        req.agent_id.as_deref(),
        req.app_id.as_deref(),
        req.run_id.as_deref(),
    ) {
        Ok(metadata) => metadata,
        Err(error) => return err_json(StatusCode::BAD_REQUEST, error).into_response(),
    };
    let session_id = req.session_id;
    let user_id = req.user_id;
    let messages = req.messages;
    match tokio::task::spawn_blocking(move || {
        state
            .store
            .append_events_idempotent(&session_id, &messages, &user_id, metadata)
    })
    .await
    {
        Ok(Ok(result)) => ok_json(result).into_response(),
        Ok(Err(error)) => err_json(StatusCode::BAD_REQUEST, error.to_string()).into_response(),
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("event task failed: {error}"),
        )
        .into_response(),
    }
}

pub async fn compact_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || state.store.compact(&session_id)).await {
        Ok(Ok(result)) => ok_json(result).into_response(),
        Ok(Err(error)) => err_json(StatusCode::BAD_REQUEST, error.to_string()).into_response(),
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("compact task failed: {error}"),
        )
        .into_response(),
    }
}

#[derive(Deserialize)]
pub struct MeditateRequest {
    pub user_id: String,
    pub triggered_by: Option<String>,
    pub since: Option<String>,
}

pub async fn meditate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MeditateRequest>,
) -> impl IntoResponse {
    let mut options = MeditateOptions::new(
        req.user_id,
        req.triggered_by.unwrap_or_else(|| "rest_api".to_string()),
    );
    if let Some(since) = req.since {
        options = options.since(since);
    }
    match tokio::task::spawn_blocking(move || state.store.meditate(options)).await {
        Ok(Ok(result)) => ok_json(result).into_response(),
        Ok(Err(error)) => err_json(StatusCode::BAD_REQUEST, error.to_string()).into_response(),
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("meditation task failed: {error}"),
        )
        .into_response(),
    }
}

// ── Data safety ──

#[derive(Deserialize)]
pub struct BackupRequest {
    pub filename: Option<String>,
}

#[derive(Deserialize)]
pub struct RestoreBackupRequest {
    pub filename: String,
    pub confirm: String,
}

fn is_safe_backup_filename(filename: &str) -> bool {
    std::path::Path::new(filename)
        .file_name()
        .and_then(|name| name.to_str())
        == Some(filename)
        && filename.ends_with(".db")
}

#[cfg(unix)]
fn set_private_directory_permissions(path: &std::path::Path) -> std::result::Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .map_err(|error| format!("cannot set private backup directory permissions: {error}"))
}

#[cfg(not(unix))]
fn set_private_directory_permissions(_path: &std::path::Path) -> std::result::Result<(), String> {
    Ok(())
}

pub async fn create_backup(
    State(state): State<Arc<AppState>>,
    request: Option<Json<BackupRequest>>,
) -> impl IntoResponse {
    let filename = request
        .and_then(|Json(request)| request.filename)
        .unwrap_or_else(|| format!("memme-{}.db", uuid::Uuid::new_v4()));
    if !is_safe_backup_filename(&filename) {
        return err_json(
            StatusCode::BAD_REQUEST,
            "filename must be a plain name ending in .db",
        )
        .into_response();
    }

    match tokio::task::spawn_blocking(move || {
        std::fs::create_dir_all(&state.backup_dir)
            .map_err(|error| format!("cannot create backup directory: {error}"))?;
        set_private_directory_permissions(&state.backup_dir)?;
        let path = state.backup_dir.join(filename);
        if path.exists() {
            return Err("backup file already exists; choose a new filename".to_string());
        }
        let path = path
            .to_str()
            .ok_or_else(|| "backup path is not valid UTF-8".to_string())?;
        state
            .store
            .backup_to_path(path)
            .map_err(|error| error.to_string())
    })
    .await
    {
        Ok(Ok(result)) => ok_json(result).into_response(),
        Ok(Err(error)) => err_json(StatusCode::BAD_REQUEST, error).into_response(),
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("backup task failed: {error}"),
        )
        .into_response(),
    }
}

pub async fn restore_backup(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RestoreBackupRequest>,
) -> impl IntoResponse {
    if !is_safe_backup_filename(&req.filename) {
        return err_json(
            StatusCode::BAD_REQUEST,
            "filename must be a plain name ending in .db",
        )
        .into_response();
    }
    if req.confirm != format!("restore:{}", req.filename) {
        return err_json(
            StatusCode::BAD_REQUEST,
            "confirm must equal 'restore:<filename>'",
        )
        .into_response();
    }

    let path = state.backup_dir.join(&req.filename);
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) => {
            return err_json(
                StatusCode::BAD_REQUEST,
                format!("cannot access backup file: {error}"),
            )
            .into_response();
        }
    };
    if !metadata.file_type().is_file() {
        return err_json(
            StatusCode::BAD_REQUEST,
            "backup must be a regular file, not a directory or symbolic link",
        )
        .into_response();
    }
    if path.to_str().is_none() {
        return err_json(StatusCode::BAD_REQUEST, "backup path is not valid UTF-8").into_response();
    }
    let validation_path = path.clone();
    match tokio::task::spawn_blocking(move || {
        MemoryStore::validate_backup(
            validation_path
                .to_str()
                .ok_or_else(|| "backup path is not valid UTF-8".to_string())?,
        )
        .map_err(|error| error.to_string())
    })
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            return err_json(StatusCode::BAD_REQUEST, error).into_response();
        }
        Err(error) => {
            return err_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("backup validation task failed: {error}"),
            )
            .into_response();
        }
    }

    match state.restore_control.request(path).await {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(ApiOk {
                success: true,
                data: serde_json::json!({
                    "accepted": true,
                    "server_restarting": true,
                }),
            }),
        )
            .into_response(),
        Err(error) => err_json(StatusCode::CONFLICT, error).into_response(),
    }
}

pub async fn replica_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || state.store.replica_status()).await {
        Ok(Ok(result)) => ok_json(result).into_response(),
        Ok(Err(error)) => {
            err_json(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response()
        }
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("replica status task failed: {error}"),
        )
        .into_response(),
    }
}

pub async fn sync_replica(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || state.store.sync_replica()).await {
        Ok(Ok(result)) => ok_json(result).into_response(),
        Ok(Err(error)) => {
            err_json(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response()
        }
        Err(error) => err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("replica sync task failed: {error}"),
        )
        .into_response(),
    }
}

#[derive(Deserialize)]
pub struct GraphAddRequest {
    pub text: String,
    pub user_id: String,
}

#[derive(Deserialize)]
pub struct GraphSearchRequest {
    pub query: String,
    pub user_id: String,
    pub depth: Option<usize>,
}

#[derive(Deserialize)]
pub struct HybridSearchRequest {
    pub query: String,
    pub user_id: String,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct FrequencyQuery {
    pub granularity: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct TopEntitiesQuery {
    pub limit: Option<usize>,
}

pub async fn graph_add(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GraphAddRequest>,
) -> impl IntoResponse {
    match state.store.add_graph_auto(&req.text, &req.user_id) {
        Ok(result) => ok_json(result).into_response(),
        Err(e) => err_json(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

pub async fn graph_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GraphSearchRequest>,
) -> impl IntoResponse {
    match state
        .store
        .search_graph(&req.query, &req.user_id, req.depth.unwrap_or(2))
    {
        Ok(result) => ok_json(result).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn hybrid_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<HybridSearchRequest>,
) -> impl IntoResponse {
    let mut opts = SearchOptions::new(&req.user_id).keyword_search(true);
    if let Some(aid) = req.agent_id {
        opts = opts.agent_id(aid);
    }
    if let Some(aid) = req.app_id {
        opts = opts.app_id(aid);
    }
    if let Some(rid) = req.run_id {
        opts = opts.run_id(rid);
    }
    if let Some(k) = req.limit {
        opts = opts.limit(k);
    }

    match state.store.search(&req.query, opts) {
        Ok(results) => ok_json(results).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

// ── Analytics Handlers ──

pub async fn user_stats(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    match state.store.user_stats(&user_id) {
        Ok(stats) => ok_json(stats).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn memory_frequency(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
    Query(q): Query<FrequencyQuery>,
) -> impl IntoResponse {
    let granularity = q.granularity.as_deref().unwrap_or("day");
    let limit = q.limit.unwrap_or(30);
    match state.store.memory_frequency(&user_id, granularity, limit) {
        Ok(buckets) => ok_json(buckets).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn top_entities(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
    Query(q): Query<TopEntitiesQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(10);
    match state.store.top_entities(&user_id, limit) {
        Ok(entities) => ok_json(entities).into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

// ── Recall (redirects to search) ──

#[derive(Deserialize)]
pub struct RecallRequest {
    pub query: String,
    pub user_id: String,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
    pub limit: Option<usize>,
    pub threshold: Option<f32>,
    pub fields: Option<Vec<String>>,
}

pub async fn recall(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RecallRequest>,
) -> impl IntoResponse {
    // Recall is now implemented as search() with keyword_search enabled
    let mut opts = SearchOptions::new(&req.user_id).keyword_search(true);
    if let Some(agent_id) = req.agent_id {
        opts = opts.agent_id(agent_id);
    }
    if let Some(app_id) = req.app_id {
        opts = opts.app_id(app_id);
    }
    if let Some(run_id) = req.run_id {
        opts = opts.run_id(run_id);
    }
    if let Some(l) = req.limit {
        opts = opts.limit(l);
    }
    if let Some(threshold) = req.threshold {
        opts = opts.threshold(threshold);
    }
    if let Some(fields) = req.fields {
        opts = opts.fields(fields);
    }

    match state.store.search(&req.query, opts) {
        Ok(results) => ok_json(serde_json::json!({
            "memories": results,
            "recall_id": uuid::Uuid::new_v4().to_string(),
        }))
        .into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}
