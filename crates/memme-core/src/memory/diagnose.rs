use std::fmt;
use std::time::Instant;

use serde::Serialize;

use super::helpers::recover_lock;
use memme_llm::{GenerateOptions, Message, MessageRole};

/// Result of a single diagnostic check.
#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub name: &'static str,
    pub ok: bool,
    pub latency_ms: u64,
    pub detail: String,
}

/// Aggregated diagnostic report for a MemoryStore.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnoseReport {
    pub all_ok: bool,
    pub checks: Vec<CheckResult>,
}

impl fmt::Display for DiagnoseReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in &self.checks {
            let status = if c.ok { "OK" } else { "FAIL" };
            writeln!(f, "[{status}] {} ({}ms): {}", c.name, c.latency_ms, c.detail)?;
        }
        if self.all_ok {
            writeln!(f, "All checks passed.")
        } else {
            let failed: Vec<&str> = self.checks.iter().filter(|c| !c.ok).map(|c| c.name).collect();
            writeln!(f, "Failed: {}", failed.join(", "))
        }
    }
}

impl super::MemoryStore {
    /// Run diagnostic checks on all configured components.
    ///
    /// Tests: storage (DuckDB), embedder (embed a short string), LLM (if configured).
    /// Returns a report with per-check latency and error details.
    pub fn diagnose(&self) -> DiagnoseReport {
        let mut checks = Vec::new();

        checks.push(check_storage(&self.storage));
        checks.push(check_embedder(self.embedder.as_ref(), self.config.embedding_dims));

        if let Some(llm) = recover_lock(&self.llm, "llm").clone() {
            checks.push(check_llm(llm.as_ref()));
        }

        let all_ok = checks.iter().all(|c| c.ok);
        DiagnoseReport { all_ok, checks }
    }
}

fn check_storage(storage: &crate::storage::Storage) -> CheckResult {
    let start = Instant::now();
    let result = storage.get_config("__diagnose_ping");
    let latency_ms = start.elapsed().as_millis() as u64;
    match result {
        Ok(_) => CheckResult {
            name: "storage",
            ok: true,
            latency_ms,
            detail: "DuckDB responsive".to_string(),
        },
        Err(e) => CheckResult {
            name: "storage",
            ok: false,
            latency_ms,
            detail: format!("{e}"),
        },
    }
}

fn check_embedder(embedder: &dyn memme_embeddings::Embedder, expected_dims: usize) -> CheckResult {
    let start = Instant::now();
    let result = embedder.embed("hello");
    let latency_ms = start.elapsed().as_millis() as u64;
    match result {
        Ok(vec) if vec.len() == expected_dims => CheckResult {
            name: "embedder",
            ok: true,
            latency_ms,
            detail: format!("{}d, model={}", vec.len(), embedder.model_name()),
        },
        Ok(vec) => CheckResult {
            name: "embedder",
            ok: false,
            latency_ms,
            detail: format!("dimension mismatch: got {} expected {}", vec.len(), expected_dims),
        },
        Err(e) => CheckResult {
            name: "embedder",
            ok: false,
            latency_ms,
            detail: format!("{e}"),
        },
    }
}

fn check_llm(llm: &dyn memme_llm::LlmProvider) -> CheckResult {
    let start = Instant::now();
    let messages = vec![Message {
        role: MessageRole::User,
        content: "Reply with exactly: ok".to_string(),
    }];
    let options = GenerateOptions {
        max_tokens: Some(3),
        temperature: Some(0.0),
        ..Default::default()
    };
    let result = llm.generate(&messages, &options);
    let latency_ms = start.elapsed().as_millis() as u64;
    match result {
        Ok(resp) => {
            let preview: String = resp.chars().take(30).collect();
            CheckResult {
                name: "llm",
                ok: true,
                latency_ms,
                detail: format!("provider={}, response={:?}", llm.name(), preview),
            }
        }
        Err(e) => CheckResult {
            name: "llm",
            ok: false,
            latency_ms,
            detail: format!("{e}"),
        },
    }
}
