//! Session/Episode workflow example: demonstrates event ingestion, session
//! management, and the event -> session -> episode data flow.
//!
//! This example shows the recommended ingestion pattern using append_events()
//! which is the primary way to add data to MemMe. Events are grouped into
//! sessions, and sessions can be compacted into episodes with memory extraction.
//!
//! Run with:
//!   MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)" \
//!     cargo run -p memme-core --example session

use std::sync::Arc;

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::mock::MockEmbedder;

fn main() {
    println!("=== MemMe Session/Episode Workflow Example ===\n");

    // ── 1. Create an in-memory store ────────────────────────────────
    let config = {
        let mut c = MemoryConfig::new(":memory:", 384);
        c.collection_name = "session_demo".into();
        c
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    let store = MemoryStore::new(config, embedder).expect("Failed to create MemoryStore");
    println!("[1] MemoryStore created.\n");

    // ── 2. Append events to a session (recommended API) ─────────────
    // append_events() is the primary ingestion method. It accepts
    // ChatMessage structs and handles session creation automatically.
    println!("[2] Appending messages to session 'session-001'...\n");

    let messages_1 = vec![
        ChatMessage {
            role: "user".into(),
            content: "Hi, I need help setting up a Rust project.".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T09:00:00Z".into()),
        },
        ChatMessage {
            role: "assistant".into(),
            content: "Sure! You can start with `cargo init` to create a new project.".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T09:00:05Z".into()),
        },
        ChatMessage {
            role: "user".into(),
            content: "How do I add dependencies?".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T09:00:30Z".into()),
        },
        ChatMessage {
            role: "assistant".into(),
            content: "Add them to Cargo.toml under [dependencies], then run `cargo build`.".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T09:00:35Z".into()),
        },
    ];

    let result1 = store
        .append_events("session-001", &messages_1, "user_alice", None)
        .expect("append_events failed");

    println!("  Session: {}", result1.session_id);
    println!("  Events appended: {}", result1.events_appended);
    println!("  Total unprocessed: {}", result1.total_unprocessed);
    println!("  Compact needed: {}", result1.compact_needed);
    println!();

    // ── 3. Append events to a second session ────────────────────────
    println!("[3] Appending messages to session 'session-002'...\n");

    let messages_2 = vec![
        ChatMessage {
            role: "user".into(),
            content: "What is SQLite and why would I use it?".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T10:00:00Z".into()),
        },
        ChatMessage {
            role: "assistant".into(),
            content: "SQLite is an in-process relational database. It is great for embedded apps, local-first storage, and edge computing because it runs as a single file with no server needed.".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T10:00:05Z".into()),
        },
        ChatMessage {
            role: "user".into(),
            content: "Can I use it from Rust?".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T10:00:30Z".into()),
        },
        ChatMessage {
            role: "assistant".into(),
            content: "Yes, the rusqlite crate on crates.io provides excellent Rust bindings for SQLite.".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T10:00:35Z".into()),
        },
    ];

    let result2 = store
        .append_events("session-002", &messages_2, "user_alice", None)
        .expect("append_events failed");

    println!("  Session: {}", result2.session_id);
    println!("  Events appended: {}", result2.events_appended);
    println!("  Total unprocessed: {}", result2.total_unprocessed);
    println!();

    // ── 4. Append follow-up messages to the same session ────────────
    println!("[4] Appending follow-up messages to session 'session-002'...\n");

    let followup = vec![
        ChatMessage {
            role: "user".into(),
            content: "What about vector search? Can SQLite do that?".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T10:01:00Z".into()),
        },
        ChatMessage {
            role: "assistant".into(),
            content: "SQLite supports vector similarity search through extensions like VexDB-Lite. MemMe uses VexDB-Lite for embedding-based memory retrieval.".into(),
            image_url: None,
            image_type: None,
            timestamp: Some("2026-04-01T10:01:05Z".into()),
        },
    ];

    let result3 = store
        .append_events("session-002", &followup, "user_alice", None)
        .expect("append_events failed");

    println!("  Events appended: {}", result3.events_appended);
    println!("  Total unprocessed: {}", result3.total_unprocessed);
    println!();

    // ── 5. List sessions ────────────────────────────────────────────
    println!("[5] Listing all sessions for user_alice...\n");

    let sessions = store
        .list_sessions(ListSessionsOptions::new("user_alice"))
        .expect("list_sessions failed");

    for s in &sessions {
        println!(
            "  Session: {} (events: {}, started: {})",
            s.session_id, s.event_count, s.started_at
        );
    }
    println!();

    // ── 6. Get events from a session ────────────────────────────────
    println!("[6] Getting events from session 'session-001'...\n");

    let events = store
        .get_session_events("session-001", None, None)
        .expect("get_session_events failed");

    for e in &events {
        let content_preview: String = e.content.chars().take(60).collect();
        println!(
            "  [{}] {} (processed: {})",
            e.event_type.as_str(),
            content_preview,
            e.processed
        );
    }
    println!();

    // ── 7. List events with filters ─────────────────────────────────
    println!("[7] Listing unprocessed events for user_alice...\n");

    let unprocessed = store
        .list_events(
            ListEventsOptions::new("user_alice")
                .unprocessed_only()
                .limit(20),
        )
        .expect("list_events failed");

    println!("  Found {} unprocessed events:", unprocessed.len());
    for e in unprocessed.iter().take(5) {
        let content_preview: String = e.content.chars().take(50).collect();
        println!(
            "    [{}] {} (session: {})",
            e.event_type.as_str(),
            content_preview,
            e.session_id.as_deref().unwrap_or("none")
        );
    }
    if unprocessed.len() > 5 {
        println!("    ... and {} more", unprocessed.len() - 5);
    }
    println!();

    // ── 8. Get session context ──────────────────────────────────────
    // Session context returns events within a token budget, useful for
    // providing conversation history during retrieval.
    println!("[8] Getting session context for 'session-002' (token budget: 500)...\n");

    let context = store
        .get_session_context(
            "session-002",
            GetSessionContextOptions::new().token_budget(500),
        )
        .expect("get_session_context failed");

    println!("  Session: {}", context.session_id);
    println!("  Events included: {}", context.events.len());
    println!(
        "  Tokens used: {} / {}",
        context.tokens_used, context.token_budget
    );
    println!("  Purified events: {}", context.purified_count);
    println!("  Raw events: {}", context.raw_count);
    if let Some(ref summary) = context.episode_summary {
        println!("  Episode summary: {}", summary);
    }
    println!("\n  Context events:");
    for e in &context.events {
        let content_preview: String = e.content.chars().take(60).collect();
        println!("    [{}] {}", e.event_type.as_str(), content_preview);
    }
    println!();

    // ── 9. Ingest individual events (low-level API) ─────────────────
    // For finer control, ingest_event() lets you specify event type,
    // metadata, and parent relationships directly.
    println!("[9] Low-level: ingesting individual events...\n");

    let opts = IngestEventOptions::new("user_alice")
        .session_id("session-001")
        .event_type("user_message");
    let event = store
        .ingest_event("Thanks, that was really helpful!", opts)
        .expect("Ingest failed");
    println!(
        "  Ingested event: \"{}\" (id: {})",
        event.content,
        &event.event_id[..8]
    );
    println!();

    // ── 10. Create an episode manually ──────────────────────────────
    // Episodes can also be created manually (e.g., for pre-processed data).
    // Normally, compact() creates episodes automatically when an LLM is configured.
    println!("[10] Creating an episode manually...\n");

    let episode_opts = CreateEpisodeOptions::new(
        "Rust Project Setup Discussion",
        "User asked about setting up a Rust project, adding dependencies, and using cargo.",
        "user_alice",
        "2026-04-01T09:00:00Z",
    )
    .ended_at("2026-04-01T09:15:00Z")
    .significance(0.7)
    .outcome("success")
    .session_ids(vec!["session-001".into()]);

    let episode = store
        .create_episode(episode_opts)
        .expect("create_episode failed");

    println!("  Episode created:");
    println!("    ID: {}", &episode.episode_id[..8]);
    println!("    Title: {}", episode.title);
    println!("    Summary: {}", episode.summary);
    println!("    Significance: {:.1}", episode.significance);
    println!(
        "    Outcome: {}",
        episode.outcome.as_deref().unwrap_or("none")
    );
    println!();

    // ── 11. List episodes ───────────────────────────────────────────
    println!("[11] Listing episodes for user_alice...\n");

    let episodes = store
        .list_episodes(ListEpisodesOptions::new("user_alice"))
        .expect("list_episodes failed");

    for ep in &episodes {
        println!(
            "  Episode: \"{}\" (significance: {:.1}, recalls: {})",
            ep.title, ep.significance, ep.recall_count
        );
    }

    println!("\n=== Done! ===");
}
