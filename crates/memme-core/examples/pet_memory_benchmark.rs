//! Offline baseline benchmark for the AI-pet memory use case.
//!
//! This example intentionally uses a deterministic lexical embedder. It measures
//! MemMe/VexDB-Lite engine behavior, not production embedding model quality.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use memme_core::{
    AddOptions, ChatMessage, MemoryConfig, MemoryError, MemoryResult, MemoryStore, SearchOptions,
    UpdateOptions, VEXDB_LITE_EXTENSION_ENV,
};
use memme_embeddings::{EmbedError, Embedder};
use serde::{Deserialize, Serialize};

const EMBEDDING_DIMS: usize = 256;

#[derive(Debug)]
struct Args {
    dataset: PathBuf,
    output: PathBuf,
    extension: PathBuf,
    memory_count: usize,
    query_count: usize,
}

#[derive(Debug, Deserialize)]
struct Dataset {
    version: u32,
    name: String,
    description: String,
    records: Vec<RecordSpec>,
    queries: Vec<QuerySpec>,
}

#[derive(Debug, Deserialize)]
struct RecordSpec {
    id: String,
    kind: String,
    user_id: String,
    content: String,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    expiration_date: Option<String>,
    #[serde(default)]
    event_time: Option<String>,
    #[serde(default)]
    memory_type: Option<String>,
    #[serde(default)]
    immutable: bool,
}

#[derive(Debug, Deserialize)]
struct QuerySpec {
    id: String,
    category: String,
    query: String,
    user_id: String,
    #[serde(default)]
    agent_id: Option<String>,
    limit: usize,
    #[serde(default = "default_true")]
    keyword_search: bool,
    #[serde(default)]
    expected_all: Vec<String>,
    #[serde(default)]
    forbidden_any: Vec<String>,
    required: bool,
    note: String,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct EmbeddingCounts {
    single_calls: u64,
    batch_calls: u64,
    texts_embedded: u64,
}

impl EmbeddingCounts {
    fn delta(self, earlier: Self) -> Self {
        Self {
            single_calls: self.single_calls.saturating_sub(earlier.single_calls),
            batch_calls: self.batch_calls.saturating_sub(earlier.batch_calls),
            texts_embedded: self.texts_embedded.saturating_sub(earlier.texts_embedded),
        }
    }
}

#[derive(Debug, Default)]
struct CountingLexicalEmbedder {
    single_calls: AtomicU64,
    batch_calls: AtomicU64,
    texts_embedded: AtomicU64,
}

impl CountingLexicalEmbedder {
    fn counts(&self) -> EmbeddingCounts {
        EmbeddingCounts {
            single_calls: self.single_calls.load(Ordering::Relaxed),
            batch_calls: self.batch_calls.load(Ordering::Relaxed),
            texts_embedded: self.texts_embedded.load(Ordering::Relaxed),
        }
    }

    fn vectorize(text: &str) -> Vec<f32> {
        let normalized = text.to_lowercase();
        let mut vector = vec![0.0_f32; EMBEDDING_DIMS];

        let mut token = String::new();
        for ch in normalized.chars().chain(std::iter::once(' ')) {
            if ch.is_alphanumeric() {
                token.push(ch);
            } else if !token.is_empty() {
                add_feature(&mut vector, &format!("word:{token}"), 2.0);
                token.clear();
            }
        }

        let chars: Vec<char> = normalized
            .chars()
            .filter(|ch| ch.is_alphanumeric())
            .collect();
        for width in [1_usize, 2, 3] {
            if chars.len() < width {
                continue;
            }
            let weight = match width {
                1 => 0.25,
                2 => 0.75,
                _ => 1.0,
            };
            for window in chars.windows(width) {
                let feature: String = window.iter().collect();
                add_feature(&mut vector, &format!("char{width}:{feature}"), weight);
            }
        }

        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        vector
    }
}

impl Embedder for CountingLexicalEmbedder {
    fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, EmbedError> {
        self.single_calls.fetch_add(1, Ordering::Relaxed);
        self.texts_embedded.fetch_add(1, Ordering::Relaxed);
        Ok(Self::vectorize(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        self.texts_embedded
            .fetch_add(texts.len() as u64, Ordering::Relaxed);
        Ok(texts.iter().map(|text| Self::vectorize(text)).collect())
    }

    fn dimensions(&self) -> usize {
        EMBEDDING_DIMS
    }

    fn model_name(&self) -> &str {
        "petmem-lexical-hash-v1"
    }
}

fn add_feature(vector: &mut [f32], feature: &str, weight: f32) {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in feature.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let index = (hash as usize) % vector.len();
    vector[index] += weight;
}

#[derive(Debug, Serialize)]
struct QueryResult {
    id: String,
    category: String,
    required: bool,
    passed: bool,
    latency_ms: f64,
    missing: Vec<String>,
    forbidden_found: Vec<String>,
    returned: Vec<ReturnedMemory>,
    note: String,
}

#[derive(Debug, Serialize)]
struct ReturnedMemory {
    id: String,
    content: String,
    score: Option<f32>,
    agent_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct LifecycleResult {
    id: String,
    required: bool,
    passed: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct FunctionalSummary {
    required_passed: usize,
    required_total: usize,
    target_passed: usize,
    target_total: usize,
    queries: Vec<QueryResult>,
    lifecycle_checks: Vec<LifecycleResult>,
    embedding_counts: EmbeddingCounts,
}

#[derive(Debug, Serialize)]
struct PerformanceSummary {
    requested_memories: usize,
    unique_memories: usize,
    insert_ms: f64,
    inserts_per_second: f64,
    query_count: usize,
    recall_at_k: f64,
    missed_probes: Vec<MissedProbe>,
    query_avg_ms: f64,
    query_p50_ms: f64,
    query_p95_ms: f64,
    query_p99_ms: f64,
    database_bytes: u64,
    embedding_counts: EmbeddingCounts,
}

#[derive(Debug, Serialize)]
struct MissedProbe {
    index: usize,
    expected: String,
    returned: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    generated_at: String,
    dataset_version: u32,
    dataset_name: String,
    dataset_description: String,
    backend: String,
    embedder: String,
    build_profile: String,
    git_commit: Option<String>,
    git_dirty: Option<bool>,
    required_gate_passed: bool,
    functional: FunctionalSummary,
    performance: PerformanceSummary,
}

struct TempDatabase {
    path: PathBuf,
}

impl TempDatabase {
    fn new(label: &str) -> Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before UNIX epoch")?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "memme-petmem-{label}-{}-{nonce}.sqlite",
            std::process::id()
        ));
        Ok(Self { path })
    }
}

impl Drop for TempDatabase {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        for suffix in ["-wal", "-shm"] {
            let sidecar = PathBuf::from(format!("{}{suffix}", self.path.display()));
            let _ = fs::remove_file(sidecar);
        }
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let dataset_text = fs::read_to_string(&args.dataset)
        .with_context(|| format!("failed to read {}", args.dataset.display()))?;
    let dataset: Dataset = serde_json::from_str(&dataset_text)
        .with_context(|| format!("invalid dataset {}", args.dataset.display()))?;

    let functional = run_functional(&dataset, &args.extension)?;
    let performance = run_performance(&args.extension, args.memory_count, args.query_count)?;
    let required_gate_passed = functional.required_passed == functional.required_total;

    let report = BenchmarkReport {
        schema_version: 1,
        generated_at: chrono::Utc::now().to_rfc3339(),
        dataset_version: dataset.version,
        dataset_name: dataset.name,
        dataset_description: dataset.description,
        backend: "VexDB-Lite SQLite extension".to_string(),
        embedder: "petmem-lexical-hash-v1 (engine baseline only)".to_string(),
        build_profile: if cfg!(debug_assertions) {
            "debug".to_string()
        } else {
            "release".to_string()
        },
        git_commit: git_output(&["rev-parse", "HEAD"]),
        git_dirty: git_dirty(),
        required_gate_passed,
        functional,
        performance,
    };

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&args.output, format!("{json}\n"))
        .with_context(|| format!("failed to write {}", args.output.display()))?;

    println!("PetMemBench result: {}", args.output.display());
    println!(
        "Required scenarios: {}/{}; target scenarios: {}/{}",
        report.functional.required_passed,
        report.functional.required_total,
        report.functional.target_passed,
        report.functional.target_total
    );
    println!(
        "Performance: {} memories, p50 {:.3} ms, p95 {:.3} ms, p99 {:.3} ms, Recall@K {:.1}%",
        report.performance.unique_memories,
        report.performance.query_p50_ms,
        report.performance.query_p95_ms,
        report.performance.query_p99_ms,
        report.performance.recall_at_k * 100.0
    );

    if required_gate_passed {
        Ok(())
    } else {
        std::process::exit(2);
    }
}

fn parse_args() -> Result<Args> {
    let mut dataset = PathBuf::from("benchmarks/petmem/scenarios.json");
    let mut output = PathBuf::from("benchmarks/petmem/results/latest.json");
    let mut extension = std::env::var_os(VEXDB_LITE_EXTENSION_ENV).map(PathBuf::from);
    let mut memory_count = 2_000_usize;
    let mut query_count = 100_usize;

    let mut raw = std::env::args().skip(1);
    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--dataset" => dataset = PathBuf::from(next_value(&mut raw, "--dataset")?),
            "--output" => output = PathBuf::from(next_value(&mut raw, "--output")?),
            "--extension" => extension = Some(PathBuf::from(next_value(&mut raw, "--extension")?)),
            "--memory-count" => {
                memory_count = next_value(&mut raw, "--memory-count")?
                    .parse()
                    .context("--memory-count must be a positive integer")?
            }
            "--query-count" => {
                query_count = next_value(&mut raw, "--query-count")?
                    .parse()
                    .context("--query-count must be a positive integer")?
            }
            "--help" | "-h" => {
                println!(
                    "Usage: pet_memory_benchmark [--dataset PATH] [--output PATH] \
                     [--extension PATH] [--memory-count N] [--query-count N]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    if memory_count == 0 || query_count == 0 {
        bail!("--memory-count and --query-count must be greater than zero");
    }
    let extension = extension.with_context(|| {
        format!("set {VEXDB_LITE_EXTENSION_ENV} or pass --extension with a trusted library")
    })?;
    if !extension.is_file() {
        bail!(
            "VexDB-Lite extension does not exist: {}",
            extension.display()
        );
    }

    Ok(Args {
        dataset,
        output,
        extension,
        memory_count,
        query_count,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("missing value for {flag}"))
}

fn base_config(path: &Path, collection: &str) -> MemoryConfig {
    let mut config = MemoryConfig::new(path.to_string_lossy(), EMBEDDING_DIMS);
    config.collection_name = collection.to_string();
    config.enable_graph = false;
    config.tuning.enable_forgetting_curve = false;
    config.tuning.enable_rerank = false;
    config.tuning.graph_augmentation_limit = 0;
    config.tuning.dedup_threshold = 0.000_001;
    config.tuning.rrf_candidate_multiplier = 3;
    config
}

fn run_functional(dataset: &Dataset, extension: &Path) -> Result<FunctionalSummary> {
    let database = TempDatabase::new("functional")?;
    let embedder = Arc::new(CountingLexicalEmbedder::default());
    let counts_before = embedder.counts();
    let store = MemoryStore::new_with_vexdb_lite(
        base_config(&database.path, "petmem_functional"),
        embedder.clone(),
        extension,
    )?;

    let mut aliases = HashMap::new();
    for record in &dataset.records {
        match record.kind.as_str() {
            "memory" => {
                let mut options = AddOptions::new(&record.user_id).immutable(record.immutable);
                if let Some(agent_id) = &record.agent_id {
                    options = options.agent_id(agent_id);
                }
                if let Some(expiration_date) = &record.expiration_date {
                    options = options.expiration_date(expiration_date);
                }
                if let Some(event_time) = &record.event_time {
                    options = options.event_time(event_time);
                }
                if let Some(memory_type) = &record.memory_type {
                    options = options.memory_type(memory_type);
                }
                if let Some(session_id) = &record.session_id {
                    options = options.session_id(session_id);
                }
                let result = store.add(&record.content, options)?;
                aliases.insert(record.id.clone(), result.id);
            }
            "event" => {
                let session_id = record
                    .session_id
                    .as_deref()
                    .with_context(|| format!("event {} needs session_id", record.id))?;
                let message = ChatMessage {
                    role: record.role.clone().unwrap_or_else(|| "user".to_string()),
                    content: record.content.clone(),
                    image_url: None,
                    image_type: None,
                    timestamp: record.timestamp.clone(),
                };
                let metadata = record
                    .agent_id
                    .as_ref()
                    .map(|agent_id| serde_json::json!({"agent_id": agent_id}));
                let appended =
                    store.append_events(session_id, &[message], &record.user_id, metadata)?;
                if appended.events_appended != 1 {
                    bail!("event {} was not appended", record.id);
                }
            }
            other => bail!("unsupported record kind {other} for {}", record.id),
        }
    }

    let mut queries = Vec::new();
    for query in &dataset.queries {
        queries.push(run_query(&store, query)?);
    }
    let lifecycle_checks = run_lifecycle_checks(&store)?;

    let required_query_passed = queries
        .iter()
        .filter(|result| result.required && result.passed)
        .count();
    let required_query_total = queries.iter().filter(|result| result.required).count();
    let target_query_passed = queries
        .iter()
        .filter(|result| !result.required && result.passed)
        .count();
    let target_query_total = queries.iter().filter(|result| !result.required).count();
    let required_lifecycle_passed = lifecycle_checks
        .iter()
        .filter(|result| result.required && result.passed)
        .count();
    let required_lifecycle_total = lifecycle_checks
        .iter()
        .filter(|result| result.required)
        .count();
    let target_lifecycle_passed = lifecycle_checks
        .iter()
        .filter(|result| !result.required && result.passed)
        .count();
    let target_lifecycle_total = lifecycle_checks
        .iter()
        .filter(|result| !result.required)
        .count();

    drop(aliases);
    drop(store);
    Ok(FunctionalSummary {
        required_passed: required_query_passed + required_lifecycle_passed,
        required_total: required_query_total + required_lifecycle_total,
        target_passed: target_query_passed + target_lifecycle_passed,
        target_total: target_query_total + target_lifecycle_total,
        queries,
        lifecycle_checks,
        embedding_counts: embedder.counts().delta(counts_before),
    })
}

fn run_query(store: &MemoryStore, query: &QuerySpec) -> Result<QueryResult> {
    let mut options = SearchOptions::new(&query.user_id)
        .limit(query.limit)
        .keyword_search(query.keyword_search);
    if let Some(agent_id) = &query.agent_id {
        options = options.agent_id(agent_id);
    }

    let started = Instant::now();
    let memories = store.search(&query.query, options)?;
    let latency_ms = duration_ms(started.elapsed());
    let combined = memories
        .iter()
        .map(|memory| memory.content.to_lowercase())
        .collect::<Vec<_>>()
        .join("\n");
    let missing = query
        .expected_all
        .iter()
        .filter(|expected| !combined.contains(&expected.to_lowercase()))
        .cloned()
        .collect::<Vec<_>>();
    let forbidden_found = query
        .forbidden_any
        .iter()
        .filter(|forbidden| combined.contains(&forbidden.to_lowercase()))
        .cloned()
        .collect::<Vec<_>>();

    Ok(QueryResult {
        id: query.id.clone(),
        category: query.category.clone(),
        required: query.required,
        passed: missing.is_empty() && forbidden_found.is_empty(),
        latency_ms,
        missing,
        forbidden_found,
        returned: memories.into_iter().map(returned_memory).collect(),
        note: query.note.clone(),
    })
}

fn returned_memory(memory: MemoryResult) -> ReturnedMemory {
    ReturnedMemory {
        id: memory.id,
        content: memory.content,
        score: memory.score,
        agent_id: memory.agent_id,
    }
}

fn run_lifecycle_checks(store: &MemoryStore) -> Result<Vec<LifecycleResult>> {
    let mut checks = Vec::new();

    let immutable = store.add(
        "PET-IMMUTABLE 主人的花生过敏属于不可修改的安全事实。",
        AddOptions::new("lifecycle-owner").immutable(true),
    )?;
    let update_rejected = matches!(
        store.update_trace(
            &immutable.id,
            "PET-IMMUTABLE 主人可以吃花生。",
            Some(UpdateOptions::new()),
        ),
        Err(MemoryError::ImmutableMemory(_))
    );
    checks.push(LifecycleResult {
        id: "immutable_update_rejected".to_string(),
        required: true,
        passed: update_rejected,
        detail: "不可变安全事实不能被更新。".to_string(),
    });

    let delete_rejected = matches!(
        store.delete_trace(&immutable.id),
        Err(MemoryError::ImmutableMemory(_))
    );
    checks.push(LifecycleResult {
        id: "immutable_delete_rejected".to_string(),
        required: true,
        passed: delete_rejected,
        detail: "不可变安全事实不能被删除。".to_string(),
    });

    let mutable = store.add(
        "PET-DOSE 当前药物剂量是10毫克。",
        AddOptions::new("lifecycle-owner"),
    )?;
    store.update_trace(
        &mutable.id,
        "PET-DOSE 当前药物剂量是20毫克。",
        Some(UpdateOptions::new()),
    )?;
    let updated = store.search(
        "PET-DOSE 当前药物剂量",
        SearchOptions::new("lifecycle-owner")
            .limit(5)
            .keyword_search(true),
    )?;
    let updated_text = joined_content(&updated);
    checks.push(LifecycleResult {
        id: "update_replaces_active_content".to_string(),
        required: true,
        passed: updated_text.contains("20毫克") && !updated_text.contains("10毫克"),
        detail: "同一条记忆更新后，旧内容不能继续出现在搜索结果中。".to_string(),
    });

    let deleted = store.add(
        "PET-DELETE-UNIQUE 这条记忆将在测试中删除。",
        AddOptions::new("lifecycle-owner"),
    )?;
    store.delete_trace(&deleted.id)?;
    let after_delete = store.search(
        "PET-DELETE-UNIQUE",
        SearchOptions::new("lifecycle-owner")
            .limit(5)
            .keyword_search(true),
    )?;
    checks.push(LifecycleResult {
        id: "delete_removes_search_result".to_string(),
        required: true,
        passed: !joined_content(&after_delete).contains("PET-DELETE-UNIQUE"),
        detail: "删除后不能再被搜索到。".to_string(),
    });

    let exact_first = store.add(
        "PET-DEDUP-EXACT 主人喜欢纸飞机。",
        AddOptions::new("lifecycle-owner"),
    )?;
    let exact_second = store.add(
        "PET-DEDUP-EXACT 主人喜欢纸飞机。",
        AddOptions::new("lifecycle-owner"),
    )?;
    checks.push(LifecycleResult {
        id: "exact_dedup_keeps_identity".to_string(),
        required: true,
        passed: exact_first.id == exact_second.id,
        detail: "完全相同的内容不应创建第二条记忆。".to_string(),
    });

    store.add(
        "PET-CALENDAR 明天遛狗时间是上午九点。",
        AddOptions::new("lifecycle-owner"),
    )?;
    store.add(
        "PET-CALENDAR 明天遛狗时间改成上午十点。",
        AddOptions::new("lifecycle-owner"),
    )?;
    let conflict = store.search(
        "PET-CALENDAR 明天几点遛狗",
        SearchOptions::new("lifecycle-owner")
            .limit(8)
            .keyword_search(true),
    )?;
    let conflict_text = joined_content(&conflict);
    checks.push(LifecycleResult {
        id: "new_fact_supersedes_old_fact".to_string(),
        required: false,
        passed: conflict_text.contains("上午十点") && !conflict_text.contains("上午九点"),
        detail: "调用方直接写入冲突事实时，系统应自动让旧事实失效。".to_string(),
    });

    Ok(checks)
}

fn joined_content(memories: &[MemoryResult]) -> String {
    memories
        .iter()
        .map(|memory| memory.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn run_performance(
    extension: &Path,
    memory_count: usize,
    query_count: usize,
) -> Result<PerformanceSummary> {
    let database = TempDatabase::new("performance")?;
    let embedder = Arc::new(CountingLexicalEmbedder::default());
    let counts_before = embedder.counts();
    let mut config = base_config(&database.path, "petmem_performance");
    config.tuning.default_limit = 10;
    let store = MemoryStore::new_with_vexdb_lite(config, embedder.clone(), extension)?;

    let items = (0..memory_count)
        .map(|index| {
            let token = format!("PETFACT-{index:06}");
            let detail = pseudo_words(index as u64);
            let agent = format!("pet-{}", index % 4);
            (
                format!("{token} 宠物日记：第{index}条共同经历，线索 {detail}，归属 {agent}。"),
                AddOptions::new("performance-owner")
                    .agent_id(agent)
                    .session_id(format!("session-{}", index % 50)),
            )
        })
        .collect::<Vec<_>>();

    let insert_started = Instant::now();
    let mut ids = HashSet::new();
    for chunk in items.chunks(100) {
        for result in store.add_batch(chunk)? {
            ids.insert(result.id);
        }
    }
    let insert_elapsed = insert_started.elapsed();

    let warmups = query_count.min(10);
    for step in 0..warmups {
        let index = probe_index(step, memory_count);
        let _ = performance_query(&store, index)?;
    }

    let mut latencies = Vec::with_capacity(query_count);
    let mut recalled = 0_usize;
    let mut missed_probes = Vec::new();
    for step in 0..query_count {
        let index = probe_index(step, memory_count);
        let expected = format!("PETFACT-{index:06}");
        let started = Instant::now();
        let results = performance_query(&store, index)?;
        latencies.push(started.elapsed());
        if joined_content(&results).contains(&expected) {
            recalled += 1;
        } else {
            missed_probes.push(MissedProbe {
                index,
                expected,
                returned: results.into_iter().map(|result| result.content).collect(),
            });
        }
    }

    let database_bytes = fs::metadata(&database.path)
        .map(|meta| meta.len())
        .unwrap_or(0);
    let insert_seconds = insert_elapsed.as_secs_f64();
    let avg_ms = latencies.iter().map(|d| duration_ms(*d)).sum::<f64>() / latencies.len() as f64;
    let p50 = percentile_ms(&latencies, 0.50);
    let p95 = percentile_ms(&latencies, 0.95);
    let p99 = percentile_ms(&latencies, 0.99);

    drop(store);
    Ok(PerformanceSummary {
        requested_memories: memory_count,
        unique_memories: ids.len(),
        insert_ms: duration_ms(insert_elapsed),
        inserts_per_second: if insert_seconds > 0.0 {
            memory_count as f64 / insert_seconds
        } else {
            0.0
        },
        query_count,
        recall_at_k: recalled as f64 / query_count as f64,
        missed_probes,
        query_avg_ms: avg_ms,
        query_p50_ms: p50,
        query_p95_ms: p95,
        query_p99_ms: p99,
        database_bytes,
        embedding_counts: embedder.counts().delta(counts_before),
    })
}

fn performance_query(store: &MemoryStore, index: usize) -> Result<Vec<MemoryResult>> {
    let agent = format!("pet-{}", index % 4);
    store
        .search(
            &format!("PETFACT-{index:06} 共同经历"),
            SearchOptions::new("performance-owner")
                .agent_id(agent)
                .limit(10)
                .keyword_search(true),
        )
        .map_err(Into::into)
}

fn probe_index(step: usize, memory_count: usize) -> usize {
    step.wrapping_mul(7919).wrapping_add(17) % memory_count
}

fn pseudo_words(mut seed: u64) -> String {
    let syllables = [
        "amber", "birch", "coral", "delta", "ember", "frost", "grove", "harbor", "indigo", "jade",
        "kite", "lunar", "maple", "nova", "opal", "pearl",
    ];
    let mut words = Vec::with_capacity(6);
    for _ in 0..6 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        words.push(syllables[(seed as usize) % syllables.len()]);
    }
    words.join("-")
}

fn percentile_ms(values: &[Duration], percentile: f64) -> f64 {
    let mut nanos = values.iter().map(Duration::as_nanos).collect::<Vec<_>>();
    nanos.sort_unstable();
    let rank = ((nanos.len() - 1) as f64 * percentile).round() as usize;
    nanos[rank] as f64 / 1_000_000.0
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn git_dirty() -> Option<bool> {
    let output = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()?;
    output.status.success().then_some(!output.stdout.is_empty())
}
