use crate::entity_index::EntityIndex;
use crate::error::Result;
use crate::time_parser::{self, TimeRange};
use crate::types::*;

use super::helpers::{
    compute_retention, filter_fields, interference_discount, recover_lock, row_to_result,
};

impl super::MemoryStore {
    /// **Internal** — Used by entity-centric search channel.
    ///
    /// Build the entity index for a user from the entities table.
    pub(crate) fn get_entity_index(&self, user_id: &str) -> EntityIndex {
        match self.storage.all_entity_names(user_id) {
            Ok(names) => EntityIndex::build(&names),
            Err(_) => EntityIndex::build(&[]),
        }
    }

    /// Entity-centric retrieval (used by graph augmentation).
    /// 1. Extract entity names from query (Aho-Corasick, <1ms)
    /// 2. Spread activation: find related entities (1-hop graph traversal)
    /// 3. Collect memories linked to these entities, ranked by cosine distance to query
    #[allow(dead_code)] // available for direct use; V4 pipeline uses graph_augment_results instead
    fn entity_channel_search(
        &self,
        query: &str,
        query_embedding: &[f32],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<MemoryResult>> {
        // Step 1: Extract entities from query
        let entity_index = self.get_entity_index(user_id);
        let matched_entities = entity_index.extract(query);

        if matched_entities.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: Spreading activation — find connected entities (configurable depth)
        let seed_refs: Vec<&str> = matched_entities.iter().map(|s| s.as_str()).collect();
        let depth = self.config.tuning.graph_spreading_depth;
        let expanded_entities = self
            .storage
            .spread_entity_names(&seed_refs, user_id, depth)?;

        // Step 3: Collect memories linked to these entities, ranked by semantic relevance.
        // Cosine distance ranking naturally prioritizes memories relevant to the query,
        // even when hub nodes expand to many entities.
        let entity_refs: Vec<&str> = expanded_entities.iter().map(|s| s.as_str()).collect();
        let rows = self.storage.entity_associated_memories(
            &entity_refs,
            user_id,
            Some(query_embedding),
            limit,
        )?;

        Ok(rows.into_iter().map(row_to_result).collect())
    }

    /// Detect whether a query has temporal intent (asks about time or contains date references).
    /// Also returns parsed time ranges when detected, so the caller can skip re-parsing.
    fn detect_temporal_intent(query: &str) -> (bool, Vec<TimeRange>) {
        let q = query.to_lowercase();

        // First, try the time parser — if it finds concrete ranges, that's definitive.
        let ranges = time_parser::parse_time_references(query);
        if !ranges.is_empty() {
            return (true, ranges);
        }

        // Fallback heuristics for queries that ask *about* time without a concrete range.
        // "When did...", "When is...", "When was..."
        if q.starts_with("when ") {
            return (true, Vec::new());
        }

        // "How long ago...", "How many years/months/weeks..."
        if q.starts_with("how long ")
            || q.starts_with("how many year")
            || q.starts_with("how many month")
            || q.starts_with("how many week")
        {
            return (true, Vec::new());
        }

        // Contains month names (English)
        let months = [
            "january",
            "february",
            "march",
            "april",
            "may ",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ];
        if months.iter().any(|m| q.contains(m)) {
            return (true, Vec::new());
        }

        // Contains year patterns (4 digits starting with 19 or 20)
        let bytes = q.as_bytes();
        for i in 0..bytes.len().saturating_sub(3) {
            if (bytes[i] == b'1' && bytes[i + 1] == b'9'
                || bytes[i] == b'2' && bytes[i + 1] == b'0')
                && bytes[i + 2].is_ascii_digit()
                && bytes[i + 3].is_ascii_digit()
            {
                return (true, Vec::new());
            }
        }

        // Relative time references (English)
        let time_refs = [
            "yesterday",
            "last week",
            "last month",
            "last year",
            "ago",
            "before",
            "after",
            "during",
            "recently",
        ];
        if time_refs.iter().any(|t| q.contains(t)) {
            return (true, Vec::new());
        }

        // Chinese temporal keywords
        let cn_refs = [
            "昨天",
            "今天",
            "上周",
            "上个星期",
            "上个月",
            "上月",
            "去年",
            "本周",
            "这周",
            "本月",
            "这个月",
            "天前",
            "周前",
            "月前",
            "年前",
            "最近",
        ];
        if cn_refs.iter().any(|t| q.contains(t)) {
            return (true, Vec::new());
        }

        (false, Vec::new())
    }

    /// # Core API — Primary search method
    ///
    /// V4 pipeline: graph-powered query expansion → dual-channel recall (vector + BM25)
    /// → fusion (RRF or CombMAX) + temporal filter → graph augmentation
    /// → cross-encoder rerank → post-processing (superseded filter, session diversity).
    ///
    /// Zero LLM calls at retrieval time.
    pub fn search(&self, query: &str, options: SearchOptions) -> Result<Vec<MemoryResult>> {
        let embedding = self
            .embedder
            .embed(query)
            .map_err(crate::error::MemoryError::Embedding)?;

        let limit = options.limit.unwrap_or(self.config.tuning.default_limit);
        let threshold = options.threshold;

        let has_reranker = self.reranker.is_some() && self.config.tuning.enable_rerank;
        let rerank_mult = if has_reranker {
            self.config.tuning.rerank_candidate_multiplier.max(1)
        } else {
            1
        };

        // ── Step 1: Query analysis (rule-based, zero LLM) ──
        let (has_temporal_intent, temporal_ranges) = Self::detect_temporal_intent(query);
        let entity_index = if self.config.enable_graph {
            Some(self.get_entity_index(&options.user_id))
        } else {
            None
        };
        let query_entities: Vec<String> = entity_index
            .as_ref()
            .map(|idx| idx.extract(query))
            .unwrap_or_default();

        // ── Step 2: Graph-powered query expansion (pre-search) ──
        // Expand query entities via 1-hop graph traversal, then feed expanded
        // entities into BM25 as additional OR terms.
        let expanded_entities = if !query_entities.is_empty() && self.config.enable_graph {
            let seed_refs: Vec<&str> = query_entities.iter().map(|s| s.as_str()).collect();
            self.storage
                .spread_entity_names(&seed_refs, &options.user_id, self.config.tuning.graph_spreading_depth)
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Build expanded BM25 query: original query + expanded entity names
        let expanded_fts_query = if expanded_entities.is_empty() {
            query.to_string()
        } else {
            // Add graph-expanded entities as OR terms for broader BM25 recall
            let entity_terms: Vec<&str> = expanded_entities.iter()
                .filter(|e| !query.to_lowercase().contains(&e.to_lowercase()))
                .map(|s| s.as_str())
                .take(10) // cap to avoid overly broad queries
                .collect();
            if entity_terms.is_empty() {
                query.to_string()
            } else {
                format!("{} OR {}", query, entity_terms.join(" OR "))
            }
        };

        let mut fused = false; // tracks whether fusion was applied
        let results = if options.keyword_search || self.config.enable_graph {
            // ── Step 3: Three-channel recall (vector + BM25 + entity, parallel) ──
            let candidate_limit =
                limit * self.config.tuning.rrf_candidate_multiplier.max(1) * rerank_mult;

            let do_fts = options.keyword_search;
            let do_graph = self.config.enable_graph;

            let (vector_results, fts_results, entity_results) = std::thread::scope(|s| {
                let h_vector = s.spawn(|| {
                    match self.storage.vector_search(
                        &embedding,
                        &options.user_id,
                        options.agent_id.as_deref(),
                        options.run_id.as_deref(),
                        options.app_id.as_deref(),
                        options.filter.as_ref(),
                        candidate_limit,
                    ) {
                        Ok(r) => r.into_iter().map(row_to_result).collect::<Vec<_>>(),
                        Err(e) => {
                            tracing::warn!("Vector search failed: {e}");
                            Vec::new()
                        }
                    }
                });

                let h_fts = s.spawn(|| {
                    if !do_fts {
                        return Vec::new();
                    }
                    // Use graph-expanded query for broader BM25 recall
                    match self.storage.fts_search(
                        &expanded_fts_query,
                        &options.user_id,
                        options.agent_id.as_deref(),
                        options.run_id.as_deref(),
                        options.app_id.as_deref(),
                        candidate_limit,
                    ) {
                        Ok(rows) => rows.into_iter().map(row_to_result).collect(),
                        Err(e) => {
                            tracing::warn!("FTS search failed: {e}");
                            Vec::new()
                        }
                    }
                });

                // Entity channel: spreading activation via graph (kept as recall channel)
                let h_entity = s.spawn(|| {
                    if !do_graph {
                        return Vec::new();
                    }
                    match self.entity_channel_search(
                        query,
                        &embedding,
                        &options.user_id,
                        candidate_limit,
                    ) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::warn!("Entity channel search failed: {e}");
                            Vec::new()
                        }
                    }
                });

                (
                    h_vector.join().unwrap_or_default(),
                    h_fts.join().unwrap_or_default(),
                    h_entity.join().unwrap_or_default(),
                )
            });

            // Word overlap: re-score all candidates from other channels.
            // Low weight — acts as a tiebreaker for surface-level keyword matches
            // that vector and FTS both missed (e.g., stemming differences).
            let word_overlap_results = {
                let mut pool = vector_results.clone();
                for r in &fts_results {
                    if !pool.iter().any(|p| p.id == r.id) {
                        pool.push(r.clone());
                    }
                }
                for r in &entity_results {
                    if !pool.iter().any(|p| p.id == r.id) {
                        pool.push(r.clone());
                    }
                }
                crate::search::word_overlap_rank_with(query, &pool, self.tokenizer.as_ref())
            };

            // ── Step 4: Fusion + temporal filter ──
            let ranked_lists: Vec<(&[MemoryResult], f64)> = [
                (vector_results.as_slice(), self.config.tuning.rrf_vector_weight),
                (fts_results.as_slice(), self.config.tuning.rrf_fts_weight),
                (entity_results.as_slice(), self.config.tuning.rrf_entity_weight),
                (word_overlap_results.as_slice(), self.config.tuning.rrf_word_overlap_weight),
            ]
            .into_iter()
            .filter(|(results, _)| !results.is_empty())
            .collect();

            if ranked_lists.is_empty() {
                Vec::new()
            } else if ranked_lists.len() == 1 {
                ranked_lists[0].0.to_vec()
            } else {
                fused = true;
                let fuse_limit = limit * rerank_mult;
                // Default: RRF fusion (multiple confirmations = stronger signal)
                crate::search::rrf_fuse(&ranked_lists, self.config.tuning.rrf_k, fuse_limit)
            }
        } else {
            // Pure vector search (no FTS configured)
            let vector_limit = limit * rerank_mult;
            let rows = self.storage.vector_search(
                &embedding,
                &options.user_id,
                options.agent_id.as_deref(),
                options.run_id.as_deref(),
                options.app_id.as_deref(),
                options.filter.as_ref(),
                vector_limit,
            )?;
            rows.into_iter()
                .filter(|r| match (threshold, r.score) {
                    (Some(t), Some(d)) => d <= t,
                    _ => true,
                })
                .map(row_to_result)
                .collect()
        };

        // ── Temporal filter: apply time range constraint on fused results ──
        let results = if has_temporal_intent && !temporal_ranges.is_empty() {
            self.apply_temporal_filter(results, &temporal_ranges)
        } else {
            results
        };

        // ── Step 5: Graph augmentation (post-recall) ──
        // From top results, follow entity edges to discover related memories
        // that recall channels might have missed. Conservative: only graph-linked,
        // not broad entity discovery (which introduces noise without reranker).
        let results = if self.config.enable_graph && self.config.tuning.graph_augmentation_limit > 0 {
            self.graph_augment_results(results, &embedding, &options.user_id)
        } else {
            results
        };

        // ── Step 6: Cross-encoder rerank ──
        let mut was_reranked = false;
        let results = if let Some(ref reranker) = self.reranker {
            if self.config.tuning.enable_rerank {
                let fallback = results;
                match reranker.rerank(query, fallback.clone(), limit) {
                    Ok(reranked) => {
                        was_reranked = true;
                        tracing::debug!(
                            before = fallback.len(),
                            after = reranked.len(),
                            "Cross-encoder reranked"
                        );
                        reranked
                    }
                    Err(e) => {
                        tracing::warn!("Reranking failed, using fusion results: {e}");
                        fallback.into_iter().take(limit).collect()
                    }
                }
            } else {
                results
            }
        } else {
            results
        };

        // ── Forgetting curve scoring ──
        let mut results = if self.config.tuning.enable_forgetting_curve {
            let max_score = if fused || was_reranked {
                results.iter().filter_map(|r| r.score).fold(0.0f32, f32::max).max(0.001)
            } else {
                1.0
            };
            let mem_count = self.storage.count_user_memories(&options.user_id).unwrap_or(0);
            let i_discount = interference_discount(mem_count);

            results
                .into_iter()
                .map(|mut r| {
                    let stability = r.stability.unwrap_or(1.0);
                    let retention = compute_retention(&r.updated_at, stability) * i_discount;
                    r.retention = Some(retention);
                    if let Some(raw_score) = r.score {
                        let similarity = if was_reranked || fused {
                            (raw_score / max_score).clamp(0.0, 1.0)
                        } else {
                            1.0 - (raw_score / 2.0)
                        };
                        let imp = r.importance.unwrap_or(0.5);
                        let w = self.config.tuning.retention_weight;
                        r.score = Some(similarity * (w * retention + (1.0 - w) * imp));
                    }
                    r
                })
                .collect::<Vec<_>>()
        } else {
            results
        };

        if self.config.tuning.enable_forgetting_curve {
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Resolution-based score adjustment
        let mut results: Vec<MemoryResult> = results
            .into_iter()
            .map(|mut r| {
                if let Some(score) = r.score {
                    let res_mult = match r.resolution {
                        Resolution::Granular => self.config.tuning.resolution_weight_granular,
                        Resolution::Narrative => self.config.tuning.resolution_weight_narrative,
                        Resolution::Identity => self.config.tuning.resolution_weight_identity,
                    };
                    r.score = Some(score * res_mult);
                }
                r
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // ── Step 7: Post-processing ──

        // 7a. Filter superseded memories (knowledge-update correctness)
        // TODO: filter superseded memories once superseded_by is exposed in MemoryResult

        // 7b. Session diversity: cap results per session to ensure cross-session coverage
        let max_per_session = self.config.tuning.max_per_session;
        if max_per_session > 0 {
            let mut session_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            results.retain(|r| {
                if let Some(ref sid) = r.session_id {
                    let count = session_counts.entry(sid.clone()).or_insert(0);
                    *count += 1;
                    *count <= max_per_session
                } else {
                    true // no session_id → always keep
                }
            });
        }

        // Defer access tracking writes
        {
            const DEFERRED_QUEUE_CAP: usize = 500;
            let needs_time_flush = self.should_time_flush();
            {
                let queue = recover_lock(&self.deferred_writes, "deferred_writes");
                if queue.len() >= DEFERRED_QUEUE_CAP || needs_time_flush {
                    drop(queue);
                    self.flush_deferred_writes();
                }
            }
            let mut queue = recover_lock(&self.deferred_writes, "deferred_writes");
            let mut seen_sessions = std::collections::HashSet::new();
            for r in &results {
                if self.config.tuning.enable_forgetting_curve {
                    queue.push(super::DeferredWrite::ReinforceStability(r.id.clone(), self.config.tuning.stability_growth_factor));
                } else {
                    queue.push(super::DeferredWrite::IncrementAccess(r.id.clone()));
                }
                if let Some(ref sid) = r.session_id {
                    if seen_sessions.insert(sid.clone()) {
                        queue.push(super::DeferredWrite::MarkSessionQueried(sid.clone()));
                    }
                }
            }
        }

        // Enforce limit
        let results: Vec<MemoryResult> = results.into_iter().take(limit).collect();

        // Apply threshold filter
        let results = if let Some(t) = threshold {
            if fused || was_reranked || self.config.tuning.enable_forgetting_curve {
                results.into_iter().filter(|r| r.score.unwrap_or(0.0) >= t).collect()
            } else {
                results.into_iter().filter(|r| r.score.map_or(true, |s| s <= t)).collect()
            }
        } else {
            results
        };

        // Process one background task opportunistically
        if self.has_llm() {
            let _ = self.process_background();
        }

        // Apply field filtering if specified
        if let Some(ref fields) = options.fields {
            Ok(results.into_iter().map(|r| filter_fields(r, fields)).collect())
        } else {
            Ok(results)
        }
    }

    /// Apply temporal filter: keep results that overlap with detected time ranges,
    /// or boost temporally relevant results. Non-destructive — results without
    /// timestamps are always kept.
    fn apply_temporal_filter(
        &self,
        mut results: Vec<MemoryResult>,
        _ranges: &[TimeRange],
    ) -> Vec<MemoryResult> {
        // For now, boost results that have event_time within the ranges.
        // Full interval overlap matching (Hindsight-style) is a future enhancement.
        // Non-destructive: we don't remove results, just re-sort with temporal boost.
        for r in &mut results {
            if let Some(ref et) = r.event_time {
                for range in _ranges {
                    if (range.start.is_empty() || et.as_str() >= range.start.as_str())
                        && (range.end.is_empty() || et.as_str() <= range.end.as_str())
                    {
                        // Boost score for temporally matching results
                        if let Some(ref mut score) = r.score {
                            *score *= 1.5;
                        }
                        break;
                    }
                }
            }
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Graph augmentation: from top results, follow entity edges to discover
    /// related memories that the recall channels missed.
    fn graph_augment_results(
        &self,
        mut results: Vec<MemoryResult>,
        query_embedding: &[f32],
        user_id: &str,
    ) -> Vec<MemoryResult> {
        let aug_limit = self.config.tuning.graph_augmentation_limit;
        if aug_limit == 0 || results.is_empty() {
            return results;
        }

        let entity_index = self.get_entity_index(user_id);
        let mut result_entities = Vec::new();
        for r in results.iter().take(10) {
            result_entities.extend(entity_index.extract(&r.content));
        }
        result_entities.sort();
        result_entities.dedup();

        if result_entities.is_empty() {
            return results;
        }

        let existing_ids: std::collections::HashSet<String> =
            results.iter().map(|r| r.id.clone()).collect();
        let entity_refs: Vec<&str> = result_entities.iter().map(|s| s.as_str()).collect();

        if let Ok(rows) = self.storage.entity_associated_memories(
            &entity_refs,
            user_id,
            Some(query_embedding),
            aug_limit * 2,
        ) {
            let mut added = 0;
            for row in rows {
                if added >= aug_limit { break; }
                if !existing_ids.contains(&row.id) {
                    results.push(row_to_result(row));
                    added += 1;
                }
            }
        }

        results
    }

    /// **Advanced** — Most users should use `search()` instead.
    ///
    /// Search with reranking.
    /// First performs vector search for candidates, then reranks with the provided reranker.
    #[allow(dead_code)]
    pub(crate) fn search_reranked(
        &self,
        query: &str,
        options: SearchOptions,
        reranker: &dyn crate::rerank::Reranker,
        candidate_multiplier: usize,
    ) -> Result<Vec<MemoryResult>> {
        let limit = options.limit.unwrap_or(self.config.tuning.default_limit);
        let candidate_limit = limit * candidate_multiplier.max(1);

        let mut expanded_opts = SearchOptions::new(&options.user_id).limit(candidate_limit);
        if let Some(t) = options.threshold {
            expanded_opts = expanded_opts.threshold(t);
        }
        if let Some(ref aid) = options.agent_id {
            expanded_opts = expanded_opts.agent_id(aid);
        }
        if let Some(ref rid) = options.run_id {
            expanded_opts = expanded_opts.run_id(rid);
        }
        if let Some(ref aid) = options.app_id {
            expanded_opts = expanded_opts.app_id(aid);
        }
        if let Some(ref f) = options.filter {
            expanded_opts = expanded_opts.filter(f.clone());
        }

        let candidates = self.search(query, expanded_opts)?;
        reranker.rerank(query, candidates, limit)
    }

    /// **Advanced** — Most users should use `search()` instead.
    ///
    /// Rebuild the FTS index. Call this after batch insertions.
    ///
    /// The FTS index is not automatically updated when memories are
    /// added/updated/deleted. Call this method to rebuild the index
    /// so that `hybrid_search` returns up-to-date FTS results.
    pub fn rebuild_fts_index(&self) -> Result<()> {
        self.storage.create_fts_index()?;
        self.storage.create_fts_index_events()
    }

    /// **Advanced** — Utility method for inspecting forgetting curve state.
    ///
    /// Get the current retention value for a specific memory.
    /// Returns None if memory not found or forgetting curve is disabled.
    #[allow(dead_code)] // planned API: forgetting curve inspection
    pub(crate) fn get_retention(&self, id: &str) -> Result<Option<f32>> {
        if !self.config.tuning.enable_forgetting_curve {
            return Ok(None);
        }
        let row = self.storage.get_memory(id)?;
        Ok(row.map(|r| {
            let stability = r.stability.unwrap_or(1.0);
            compute_retention(&r.updated_at, stability)
        }))
    }
}
