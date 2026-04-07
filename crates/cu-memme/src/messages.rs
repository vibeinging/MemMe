//! Message types for Copper inter-task communication.
//!
//! These types implement the traits required by cu29's CuMsgPayload:
//! Default + Debug + Clone + Encode + Decode + Serialize + DeserializeOwned + Reflect
//!
//! NOTE: cu29 re-exports its own bincode as `cu29::bincode` (cu-bincode ^2.0).
//! The Encode/Decode derives must come from cu29's re-export, not the standalone
//! bincode crate. If compilation fails, ensure cu29 version matches.

use cu29::bincode::{Decode, Encode};
use cu29::prelude::Reflect;
use serde::{Deserialize, Serialize};

/// Request sent to the MemMe task.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode, Reflect)]
pub enum MemMeRequest {
    Store {
        content: String,
        user_id: String,
        metadata: Option<String>,
    },
    Search {
        query: String,
        user_id: String,
        limit: u32,
    },
    IngestEvent {
        content: String,
        user_id: String,
        session_id: Option<String>,
        event_type: Option<String>,
    },
    Compact {
        session_id: String,
    },
    #[default]
    Noop,
}

/// Response from the MemMe task.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode, Reflect)]
pub enum MemMeResponse {
    Stored { id: String, content: String },
    SearchResults { results: Vec<MemoryHit> },
    EventAck { event_id: String },
    Compacted {
        episode_id: String,
        memory_count: u32,
    },
    #[default]
    Empty,
    Error { message: String },
}

/// A single memory search hit.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode, Reflect)]
pub struct MemoryHit {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub created_at: String,
}
