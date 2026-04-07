// Prompt design adapted from mem0 (Apache 2.0 License)
// https://github.com/mem0ai/mem0
//
// These prompts have been adapted for MemMe's edge-first architecture.
// Original prompt structures by the mem0 team, modified for Rust usage.

mod helpers;

mod agent_memory;
mod entity_extraction;
mod fact_extraction;
mod reflect;
mod update_memory;

pub use agent_memory::*;
pub use entity_extraction::*;
pub use fact_extraction::*;
pub use reflect::*;
pub use update_memory::*;
