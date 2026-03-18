//! cu-memme — Copper-rs memory task powered by MemMe.
//!
//! Gives any Copper robot persistent, searchable long-term memory.
//!
//! # RON Configuration
//!
//! ```ron
//! (
//!     tasks: [
//!         (id: "memme", type: "cu_memme::MemMeTask", config: {
//!             "db_path": "robot_memory.duckdb",
//!             "user_id": "robot-001",
//!             "embedding_dims": "384",
//!         }),
//!     ],
//!     cnx: [
//!         (src: "perception", dst: "memme", msg: "cu_memme::MemMeRequest"),
//!         (src: "memme", dst: "planner", msg: "cu_memme::MemMeResponse"),
//!     ],
//! )
//! ```

mod messages;
mod task;

pub use messages::{MemMeRequest, MemMeResponse, MemoryHit};
pub use task::MemMeTask;
