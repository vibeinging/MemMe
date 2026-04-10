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

    /// Entity-centric retrieval channel:
    /// 1. Extract entity names from query (Aho-Corasick, <1ms)
    /// 2. Spread activation: find related entities (1-hop graph traversal)
    /// 3. Collect memories linked to these entities, ranked by cosine distance to query
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
    /// Search across all memory layers (facts, episode summaries, identity traits)
    /// using multi-channel retrieval (vector + BM25 + entity spreading + temporal) with RRF fusion.
    ///
    /// Results include traces of all resolutions (granular facts, narrative summaries,
    /// identity traits) ranked by relevance.
    ///
    /// This is the recommended way to query MemMe. For most use cases,
    /// you only need three methods: `append_events()`, `search()`, and `compact()`.
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

        let mut rrf_fused = false; // tracks whether RRF fusion was actually applied
        let results = if options.keyword_search || self.config.enable_graph {
            // Four-channel retrieval — execute channels in parallel using thread::scope.
            // Each channel acquires its own read connection from the pool (Multi mode
            // has 4 read connections by default), so they can run concurrently.
            let candidate_limit =
                limit * self.config.tuning.rrf_candidate_multiplier.max(1) * rerank_mult;

            let do_fts = options.keyword_search;
            let do_graph = self.config.enable_graph;
            let (do_temporal, temporal_ranges) = Self::detect_temporal_intent(query);

            let (vector_results, fts_results, entity_results, temporal_results) =
                std::thread::scope(|s| {
                    // Channel 1: Vector search
                    let h_vector = s.spawn(|| {
                        let rows = self.storage.vector_search(
                            &embedding,
                            &options.user_id,
                            options.agent_id.as_deref(),
                            options.run_id.as_deref(),
                            options.app_id.as_deref(),
                            options.filter.as_ref(),
                            candidate_limit,
                        );
                        match rows {
                            Ok(r) => r.into_iter().map(row_to_result).collect::<Vec<_>>(),
                            Err(e) => {
                                tracing::warn!("Vector search failed: {e}");
                                Vec::new()
                            }
                        }
                    });

                    // Channel 2: BM25/FTS search
                    let h_fts = s.spawn(|| {
                        if !do_fts {
                            return Vec::new();
                        }
                        match self.storage.fts_search(
                            query,
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

                    // Channel 3: Entity-centric retrieval
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

                    // Channel 4: Temporal retrieval (range-based when time ranges detected)
                    let h_temporal = s.spawn(|| {
                        if !do_temporal {
                            return Vec::new();
                        }
                        let result = if temporal_ranges.is_empty() {
                            // No concrete ranges — fall back to recency ordering
                            self.storage
                                .temporal_search(&options.user_id, candidate_limit)
                        } else {
                            // Concrete date ranges extracted from query
                            self.storage.temporal_range_search(
                                &options.user_id,
                                &temporal_ranges,
                                candidate_limit,
                            )
                        };
                        match result {
                            Ok(rows) => rows.into_iter().map(row_to_result).collect(),
                            Err(e) => {
                                tracing::warn!("Temporal search failed: {e}");
                                Vec::new()
                            }
                        }
                    });

                    (
                        h_vector.join().unwrap_or_default(),
                        h_fts.join().unwrap_or_default(),
                        h_entity.join().unwrap_or_default(),
                        h_temporal.join().unwrap_or_default(),
                    )
                });

            // Channel 5: Word overlap re-scoring on vector candidates
            // Pure string matching — no embedding or FTS needed.
            // Uses all vector candidates as the pool for overlap scoring.
            let word_overlap_results = {
                // Merge all available candidates as pool for word overlap
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
                for r in &temporal_results {
                    if !pool.iter().any(|p| p.id == r.id) {
                        pool.push(r.clone());
                    }
                }
                crate::search::word_overlap_rank_with(query, &pool, self.tokenizer.as_ref())
            };

            // Build ranked lists with weights (optionally adaptive)
            let alpha = self.config.tuning.adaptive_rrf_alpha as f64;
            let adapt = |base_weight: f64, confidence: f64| -> f64 {
                base_weight * (1.0 - alpha + alpha * confidence)
            };
            let confidence_k = 5; // top-k results used for confidence estimation

            // (results, base_weight, is_distance_score)
            let channels: [(&[MemoryResult], f64, bool); 5] = [
                (&vector_results, self.config.tuning.rrf_vector_weight, true),
                (&fts_results, self.config.tuning.rrf_fts_weight, false),
                (&entity_results, self.config.tuning.rrf_entity_weight, true),
                (
                    &temporal_results,
                    self.config.tuning.rrf_temporal_weight,
                    false,
                ),
                (
                    &word_overlap_results,
                    self.config.tuning.rrf_word_overlap_weight,
                    false,
                ),
            ];

            let ranked_lists: Vec<(&[MemoryResult], f64)> = channels
                .into_iter()
                .filter(|(results, _, _)| !results.is_empty())
                .map(|(results, base_weight, is_distance)| {
                    let w = if alpha > 0.0 {
                        let c = crate::search::compute_channel_confidence(
                            results,
                            is_distance,
                            confidence_k,
                        );
                        adapt(base_weight, c)
                    } else {
                        base_weight
                    };
                    (results, w)
                })
                .collect();

            if ranked_lists.is_empty() {
                Vec::new()
            } else if ranked_lists.len() == 1 {
                // Only one channel has results, use it directly
                ranked_lists[0].0.to_vec()
            } else {
                rrf_fused = true;
                let fuse_limit = limit * rerank_mult;
                crate::search::rrf_fuse(&ranked_lists, self.config.tuning.rrf_k, fuse_limit)
            }
        } else {
            // Pure vector search (no FTS or graph configured)
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

        // Apply cross-encoder reranking if configured
        let mut was_reranked = false;
        let results = if let Some(ref reranker) = self.reranker {
            if self.config.tuning.enable_rerank {
                let fallback = results; // keep original
                match reranker.rerank(query, fallback.clone(), limit) {
                    Ok(reranked) => {
                        was_reranked = true;
                        tracing::debug!(
                            before = limit * rerank_mult,
                            after = reranked.len(),
                            "Reranked results"
                        );
                        reranked
                    }
                    Err(e) => {
                        tracing::warn!("Reranking failed, using original results: {e}");
                        fallback.into_iter().take(limit).collect()
                    }
                }
            } else {
                results
            }
        } else {
            results
        };

        // Apply forgetting curve scoring if enabled
        let mut results = if self.config.tuning.enable_forgetting_curve {
            // Compute dynamic normalization from actual max score (for RRF and reranker paths)
            let max_score = if rrf_fused || was_reranked {
                results
                    .iter()
                    .filter_map(|r| r.score)
                    .fold(0.0f32, f32::max)
                    .max(0.001)
            } else {
                1.0 // not used in vector-only path
            };

            // Interference discount: query memory count once for the whole batch.
            // When a user has many memories, similar ones compete for retrieval,
            // mildly reducing each memory's effective retention (cognitive crowding).
            let mem_count = self
                .storage
                .count_user_memories(&options.user_id)
                .unwrap_or(0);
            let i_discount = interference_discount(mem_count);

            results
                .into_iter()
                .map(|mut r| {
                    let stability = r.stability.unwrap_or(1.0);
                    let retention = compute_retention(&r.updated_at, stability) * i_discount;
                    r.retention = Some(retention);

                    if let Some(raw_score) = r.score {
                        // Convert raw_score to a 0-1 similarity based on score source:
                        // - Reranker: already a relevance score, normalize to 0-1
                        // - RRF fusion: divide by max to normalize
                        // - Vector-only: cosine distance, convert to similarity
                        let similarity = if was_reranked || rrf_fused {
                            // Relevance scores (higher = better): normalize by batch max
                            (raw_score / max_score).clamp(0.0, 1.0)
                        } else {
                            // Cosine distance: convert to similarity
                            1.0 - (raw_score / 2.0)
                        };
                        let imp = r.importance.unwrap_or(0.5);
                        let w = self.config.tuning.retention_weight;
                        let weighted = similarity * (w * retention + (1.0 - w) * imp);
                        r.score = Some(weighted);
                    }
                    r
                })
                .collect::<Vec<_>>()
        } else {
            results
        };

        if self.config.tuning.enable_forgetting_curve {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Apply resolution-based score adjustment independently of forgetting curve.
        // Penalizes coarser-grain memories (narrative summaries, identity traits) to
        // bias toward precise atomic facts.
        let results = results
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
            .collect::<Vec<_>>();

        // Memory-type weighting removed: keyword-based boosting of preference/decision
        // memories caused regressions in LongMemEval (80.6% → 61.1%) by promoting
        // false positives containing common words like "like", "love", "best".
        // TODO: revisit with a more precise classification (e.g., LLM-tagged memory types).

        let mut results = results;
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Defer access tracking writes to avoid blocking reads with write locks.
        // Auto-flush when queue exceeds cap or flush interval has elapsed.
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
            // Collect unique session_ids to mark as queried (feedback-driven consolidation).
            let mut seen_sessions = std::collections::HashSet::new();
            for r in &results {
                if self.config.tuning.enable_forgetting_curve {
                    queue.push(super::DeferredWrite::ReinforceStability(
                        r.id.clone(),
                        self.config.tuning.stability_growth_factor,
                    ));
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

        // Feedback-driven compact: if a session has been queried >= 3 times
        // and hasn't been compacted yet, enqueue a background CompactSession task.
        // This ensures only frequently-accessed sessions consume LLM resources.
        {
            let mut seen_sessions = std::collections::HashSet::new();
            for r in &results {
                if let Some(ref sid) = r.session_id {
                    if !seen_sessions.insert(sid.clone()) {
                        continue;
                    }
                    // The mark_session_queried write hasn't been flushed yet,
                    // so read the stored count (will be +1 after next flush).
                    // Threshold of 3 means the session has been searched at least
                    // 3 times, indicating it's worth investing LLM resources.
                    const COMPACT_QUERY_THRESHOLD: u32 = 3;
                    match self.storage.get_session_queried_count(sid) {
                        Ok(count) if count >= COMPACT_QUERY_THRESHOLD => {
                            match self.storage.session_has_episode(sid) {
                                Ok(false) => {
                                    self.enqueue_background(
                                        super::background::BackgroundTask::CompactSession(
                                            sid.clone(),
                                        ),
                                    );
                                }
                                Ok(true) => {} // already compacted
                                Err(e) => {
                                    tracing::debug!(
                                        session_id = sid.as_str(),
                                        error = %e,
                                        "Failed to check session episode status"
                                    );
                                }
                            }
                        }
                        Ok(_) => {} // not enough queries yet
                        Err(e) => {
                            tracing::debug!(
                                session_id = sid.as_str(),
                                error = %e,
                                "Failed to read session queried_count"
                            );
                        }
                    }
                }
            }
        }

        // Enforce limit — earlier stages may return more candidates than requested
        let results: Vec<MemoryResult> = results.into_iter().take(limit).collect();

        // Apply threshold filter (works in all paths: vector distance, RRF, or weighted score)
        let results = if let Some(t) = threshold {
            if rrf_fused || was_reranked || self.config.tuning.enable_forgetting_curve {
                // Score is a relevance/weighted score (higher = better): keep if >= threshold
                results
                    .into_iter()
                    .filter(|r| r.score.unwrap_or(0.0) >= t)
                    .collect()
            } else {
                // Pure vector: score is cosine distance (lower = better): keep if <= threshold
                results
                    .into_iter()
                    .filter(|r| r.score.map_or(true, |s| s <= t))
                    .collect()
            }
        } else {
            results
        };

        // Process one background task opportunistically (non-blocking).
        if self.has_llm() {
            let _ = self.process_background();
        }

        // Apply field filtering if specified
        if let Some(ref fields) = options.fields {
            Ok(results
                .into_iter()
                .map(|r| filter_fields(r, fields))
                .collect())
        } else {
            Ok(results)
        }
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
        self.storage.create_fts_index()
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
