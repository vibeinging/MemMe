//! # memme-core
//!
//! Core library for MemMe — an edge-first AI memory engine backed by DuckDB.
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
pub mod memory;
pub mod procedural;
pub mod rerank;
pub(crate) mod search;
pub(crate) mod smart;
pub(crate) mod storage;
pub mod sync;
pub mod types;
#[cfg(feature = "webhooks")]
pub mod webhook;

pub use config::{MemoryConfig, PowerConfig};
pub use error::{MemoryError, Result};
pub use memory::MemoryStore;
pub use procedural::{Procedure, ProcedureStep};
pub use types::{
    // types/mod.rs
    AddOptions, AppendEventsResult, ChatMessage, CompactResult, ConsolidateResult, Entity,
    FullExport, GraphRelation, GraphSearchResult, HistoryEvent, HistoryRecord, ListOptions,
    MemoryExport, MemoryResult, Privacy, PruningStrategy, Resolution, SearchOptions,
    SmartAddResult, UpdateOptions,
    // types/filter.rs
    FilterExpression, FilterOp,
    // types/stream.rs
    Event, EventType, IngestEventOptions, ListEventsOptions, Source,
    // types/session.rs
    GetSessionContextOptions, ListSessionsOptions, Session, SessionContext,
    // types/episode.rs
    CreateEpisodeOptions, Episode, EpisodeMessagesOptions, ListEpisodesOptions,
    SearchEpisodesOptions,
    // types/identity.rs
    AddIdentityTraitOptions, IdentityTrait, TraitType,
    // types/meditation.rs
    MeditateOptions, MeditationRecord, MeditationStatus,
};
