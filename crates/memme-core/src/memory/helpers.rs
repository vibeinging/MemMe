use std::sync::{Mutex, MutexGuard};

use crate::procedural::Procedure;
use crate::types::{MemoryResult, Resolution};

/// Acquire a mutex lock, recovering from poison if a previous thread panicked.
/// This prevents cascading panics when one thread fails while holding a lock.
pub(crate) fn recover_lock<'a, T>(mutex: &'a Mutex<T>, label: &str) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|e| {
        tracing::warn!("{} mutex was poisoned, recovering", label);
        e.into_inner()
    })
}

pub(crate) fn procedure_row_to_result(row: crate::storage::ProcedureRow) -> Procedure {
    let steps: Vec<crate::procedural::ProcedureStep> = row
        .steps
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    Procedure {
        id: row.id,
        name: row.name,
        description: row.description,
        steps,
        user_id: row.user_id,
        trigger: row.trigger_pattern,
        confidence: row.confidence,
        usage_count: row.usage_count,
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}

pub(crate) fn row_to_result(row: crate::storage::MemoryRow) -> MemoryResult {
    let metadata = row.metadata.and_then(|s| serde_json::from_str(&s).ok());
    let categories = crate::storage::Storage::parse_categories(row.categories);

    MemoryResult {
        id: row.id,
        content: row.content,
        user_id: row.user_id,
        agent_id: row.agent_id,
        app_id: row.app_id,
        run_id: row.run_id,
        score: row.score,
        created_at: row.created_at,
        updated_at: row.updated_at,
        metadata,
        importance: row.importance,
        access_count: row.access_count,
        immutable: row.immutable,
        expiration_date: row.expiration_date,
        categories,
        memory_type: row.memory_type,
        retention: None,
        stability: row.stability,
        privacy: row.privacy.unwrap_or_else(|| "syncable".to_string()),
        event_time: row.event_time,
        episode_id: row.episode_id,
        session_id: row.session_id,
        resolution: Resolution::parse(row.resolution.as_deref().unwrap_or("granular")),
    }
}

/// Filter a MemoryResult to only include specified fields.
/// Unspecified fields are set to their default/None values.
pub(crate) fn filter_fields(r: MemoryResult, fields: &[String]) -> MemoryResult {
    MemoryResult {
        id: r.id, // always included
        content: if fields.iter().any(|f| f == "content" || f == "memory") {
            r.content
        } else {
            String::new()
        },
        user_id: r.user_id, // always included
        agent_id: if fields.iter().any(|f| f == "agent_id") {
            r.agent_id
        } else {
            None
        },
        app_id: if fields.iter().any(|f| f == "app_id") {
            r.app_id
        } else {
            None
        },
        run_id: if fields.iter().any(|f| f == "run_id") {
            r.run_id
        } else {
            None
        },
        score: if fields.iter().any(|f| f == "score") {
            r.score
        } else {
            None
        },
        created_at: if fields.iter().any(|f| f == "created_at") {
            r.created_at
        } else {
            String::new()
        },
        updated_at: if fields.iter().any(|f| f == "updated_at") {
            r.updated_at
        } else {
            String::new()
        },
        metadata: if fields.iter().any(|f| f == "metadata") {
            r.metadata
        } else {
            None
        },
        importance: if fields.iter().any(|f| f == "importance") {
            r.importance
        } else {
            None
        },
        access_count: if fields.iter().any(|f| f == "access_count") {
            r.access_count
        } else {
            None
        },
        immutable: if fields.iter().any(|f| f == "immutable") {
            r.immutable
        } else {
            false
        },
        expiration_date: if fields.iter().any(|f| f == "expiration_date") {
            r.expiration_date
        } else {
            None
        },
        categories: if fields.iter().any(|f| f == "categories") {
            r.categories
        } else {
            None
        },
        memory_type: if fields.iter().any(|f| f == "memory_type") {
            r.memory_type
        } else {
            None
        },
        retention: if fields.iter().any(|f| f == "retention") {
            r.retention
        } else {
            None
        },
        stability: if fields.iter().any(|f| f == "stability") {
            r.stability
        } else {
            None
        },
        privacy: if fields.iter().any(|f| f == "privacy") {
            r.privacy
        } else {
            "syncable".to_string()
        },
        event_time: if fields.iter().any(|f| f == "event_time") {
            r.event_time
        } else {
            None
        },
        episode_id: if fields.iter().any(|f| f == "episode_id") {
            r.episode_id
        } else {
            None
        },
        session_id: if fields.iter().any(|f| f == "session_id") {
            r.session_id
        } else {
            None
        },
        resolution: r.resolution, // always include
    }
}

/// Compute retention using FSRS power-law forgetting curve.
/// R(t, S) = (1 + t / (c * S))^(-p), where c = 5.0, p = 0.5
pub(crate) fn compute_retention(updated_at: &str, stability: f32) -> f32 {
    let days_elapsed = parse_days_since(updated_at);
    let s = stability.max(0.01);
    let retention = (1.0 + days_elapsed / (5.0 * s)).powf(-0.5);
    retention.clamp(0.0, 1.0)
}

/// Parse a timestamp string and return days elapsed since then.
///
/// Handles formats: `YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DDTHH:MM:SS`,
/// with optional fractional seconds and timezone suffix (`Z`, `+08:00`).
fn parse_days_since(timestamp: &str) -> f32 {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let ts = timestamp.replace('T', " ");
    let ts = ts.trim();
    let parts: Vec<&str> = ts.splitn(2, ' ').collect();
    if parts.len() < 2 {
        return 0.0;
    }
    let date_parts: Vec<u32> = parts[0].split('-').filter_map(|s| s.parse().ok()).collect();
    // Strip fractional seconds and timezone suffix (+HH:MM or Z)
    let time_raw = parts[1].split('.').next().unwrap_or("00:00:00");
    let time_str = time_raw.split('+').next().unwrap_or(time_raw);
    let time_str = time_str.strip_suffix('Z').unwrap_or(time_str);
    let time_parts: Vec<u32> = time_str.split(':').filter_map(|s| s.parse().ok()).collect();
    if date_parts.len() != 3 || time_parts.len() < 2 {
        return 0.0;
    }
    let (year, month, day) = (date_parts[0], date_parts[1], date_parts[2]);
    let (hour, minute, second) = (
        time_parts[0],
        time_parts[1],
        if time_parts.len() > 2 {
            time_parts[2]
        } else {
            0
        },
    );
    // Days from epoch using formula (avoids O(n) year loop)
    let y = year as i64;
    let m = month as i64;
    let d = day as i64;
    // Adjust for Jan/Feb (treat as months 13/14 of previous year)
    let (y_adj, m_adj) = if m <= 2 { (y - 1, m + 12) } else { (y, m) };
    let days_from_epoch =
        365 * y_adj + y_adj / 4 - y_adj / 100 + y_adj / 400 + (153 * (m_adj - 3) + 2) / 5 + d
            - 719469;
    let ts_epoch =
        days_from_epoch * 86400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64;
    let elapsed_seconds = now - ts_epoch as f64;
    (elapsed_seconds / 86400.0).max(0.0) as f32
}

fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

/// Compute initial stability based on memory importance tier.
pub(crate) fn initial_stability_for_tier(importance: f32, access_count: u32) -> f32 {
    if importance >= 0.8 || access_count >= 10 {
        3.0 // long-term memory
    } else if importance >= 0.5 || access_count >= 3 {
        1.5 // short-term memory
    } else {
        1.0 // working memory
    }
}

/// Simple deterministic hash of content for fast exact-match dedup optimization.
/// Uses FNV-1a algorithm for a stable, deterministic hash.
pub(crate) fn content_hash(content: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in content.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    format!("{:016x}", hash)
}

/// Simple timestamp string using std::time (no chrono dependency).
pub(crate) fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", secs)
}
