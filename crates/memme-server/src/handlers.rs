use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};

use memme_core::types::*;

use crate::AppState;

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
    let report = tokio::task::spawn_blocking(move || state.store.diagnose())
        .await
        .unwrap();
    Json(report)
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
    let model = req.model.as_deref().unwrap_or("gpt-4.1-nano");
    let base_url = match req.base_url.as_deref() {
        Some(url) => url,
        None => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({"error": "base_url required (full endpoint URL)"})),
            )
                .into_response();
        }
    };

    // Create OpenAI provider and configure the store
    let llm_config = memme_llm::openai::OpenAIConfig {
        api_key: req.api_key.clone(),
        base_url: base_url.to_string(),
        model: model.to_string(),
    };
    let llm = std::sync::Arc::new(memme_llm::openai::OpenAIProvider::new(llm_config));
    state.store.set_llm_provider(llm);
    match state.store.save_llm_config(&req.api_key, model, base_url) {
        Ok(()) => ok_json(serde_json::json!({
            "configured": true,
            "model": model,
            "base_url": base_url,
        }))
        .into_response(),
        Err(e) => err_json(StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
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
    pub vector_weight: Option<f64>,
    pub fts_weight: Option<f64>,
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

// ── Episode Handlers (removed: episodes merged into traces) ──
// Episode operations are now internal (pub(crate)) and handled by compact().
// Use search() with resolution filter to find narrative traces instead.

#[derive(Deserialize)]
pub struct PaginationQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
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
    pub limit: Option<usize>,
    pub include_episodes: Option<bool>,
    pub include_identity: Option<bool>,
    pub include_graph: Option<bool>,
}

pub async fn recall(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RecallRequest>,
) -> impl IntoResponse {
    // Recall is now implemented as search() with keyword_search enabled
    let mut opts = SearchOptions::new(&req.user_id).keyword_search(true);
    if let Some(l) = req.limit {
        opts = opts.limit(l);
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

#[derive(Deserialize)]
pub struct RecallFeedbackRequest {
    pub feedback: String,
}

pub async fn recall_feedback(
    State(_state): State<Arc<AppState>>,
    Path(_recall_id): Path<String>,
    Json(_req): Json<RecallFeedbackRequest>,
) -> impl IntoResponse {
    // Recall feedback is no longer supported; return a no-op success
    ok_json(serde_json::json!({"updated": true, "deprecated": true}))
}
