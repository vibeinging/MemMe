//! # memme-core
//!
//! Core library for MemMe — an edge-first AI memory engine backed by SQLite.
//!
//! This crate provides the main [`MemoryStore`] API for storing, searching, and
//! managing memories with vector similarity, knowledge graphs, and an Ebbinghaus
//! forgetting curve. It is designed for local-first / offline use on any platform
//! (mobile, desktop, server).
//!
//! ## Architecture
//!
//! Data flows through a layered pipeline:
//!
//! 1. **Stream** — raw events are ingested via [`MemoryStore::ingest_event`] or
//!    [`MemoryStore::append_events`].
//! 2. **Session / Episode** — events are grouped into sessions and compacted into
//!    episodes via [`MemoryStore::compact`].
//! 3. **Memory (Trace)** — atomic facts extracted from episodes, stored with
//!    embeddings for vector search.
//! 4. **Identity** — high-level personality traits distilled from memories.
//! 5. **Graph** — entity-relation knowledge graph extracted alongside memories.
//!
//! ## Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use memme_core::{MemoryConfig, MemoryStore};
//! use memme_embeddings::mock::MockEmbedder;
//!
//! let config = MemoryConfig::new(":memory:", 384);
//! let embedder = Arc::new(MockEmbedder::new(384));
//! let store = MemoryStore::new(config, embedder).unwrap();
//! ```

pub mod analytics;
pub mod config;
pub(crate) mod dedup;
pub(crate) mod entity_index;
pub mod error;
pub(crate) mod graph;
pub mod import;
pub mod memory;
pub mod procedural;
pub mod rerank;
pub(crate) mod search;
pub(crate) mod storage;
pub mod sync;
pub(crate) mod text_utils;
pub mod types;
#[cfg(feature = "webhooks")]
pub mod webhook;

pub use config::{MemoryConfig, PowerConfig};
pub use error::{MemoryError, Result};
pub use import::{ImportConversationsResult, ImportedConversation};
pub use memme_llm::prompts::FeedbackItem;
pub use memory::{
    CheckResult, DiagnoseReport, LearnFromFeedbackOptions, LearnFromFeedbackResult, MemoryStore,
    ReflectOptions, ReflectResult,
};
pub use procedural::{Procedure, ProcedureStep};
pub use types::{
    // types/identity.rs
    AddIdentityTraitOptions,
    // types/mod.rs
    AddOptions,
    AppendEventsResult,
    BackupInfo,
    ChatMessage,
    CompactResult,
    ConsolidateResult,
    // types/episode.rs
    CreateEpisodeOptions,
    Entity,
    Episode,
    EpisodeMessagesOptions,
    // types/stream.rs
    Event,
    EventType,
    // types/filter.rs
    FilterExpression,
    FilterOp,
    FullExport,
    FullImportResult,
    // types/session.rs
    GetSessionContextOptions,
    GraphRelation,
    GraphSearchResult,
    HistoryEvent,
    HistoryRecord,
    IdentityTrait,
    IngestEventOptions,
    ListEpisodesOptions,
    ListEventsOptions,
    ListOptions,
    ListSessionsOptions,
    // types/meditation.rs
    MeditateOptions,
    MeditationRecord,
    MeditationStatus,
    MemoryExport,
    MemoryResult,
    Privacy,
    PruningStrategy,
    ReplicaStatus,
    ReplicaSyncResult,
    Resolution,
    SearchEpisodesOptions,
    SearchOptions,
    Session,
    SessionContext,
    Source,
    TraitType,
    UpdateOptions,
};
