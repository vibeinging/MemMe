use crate::entity_index::EntityIndex;
use crate::error::Result;
use crate::types::*;

use super::helpers::{compute_retention, filter_fields, recover_lock, row_to_result};

impl super::MemoryStore {
    /// Batch-embed facts and search for similar memories.
    /// Uses embed_batch to minimize API calls (respects config.embed_batch_size).
    pub(crate) fn batch_search_facts(
        &self,
        facts: &[String],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<Vec<MemoryResult>>> {
        let batch_size = self.config.embed_batch_size.max(1);
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(facts.len());

        for chunk in facts.chunks(batch_size) {
            let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
            let batch = self
                .embedder
                .embed_batch(&refs)
                .map_err(crate::error::MemoryError::Embedding)?;
            all_embeddings.extend(batch);
        }

        let mut results = Vec::with_capacity(facts.len());
        for embedding in &all_embeddings {
            let rows = self
                .storage
                .vector_search(embedding, user_id, None, None, None, None, limit)?;
            results.push(
                rows.into_iter()
                    .map(super::helpers::row_to_result)
                    .collect(),
            );
        }

        Ok(results)
    }

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
    /// 3. Collect ALL memories linked to these entities
    fn entity_channel_search(
        &self,
        query: &str,
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
        let depth = self.config.graph_spreading_depth;
        let expanded_entities = self
            .storage
            .spread_entity_names(&seed_refs, user_id, depth)?;

        // Step 3: Collect all memories linked to these entities
        let entity_refs: Vec<&str> = expanded_entities.iter().map(|s| s.as_str()).collect();
        let rows = self
            .storage
            .entity_associated_memories(&entity_refs, user_id, limit)?;

        Ok(rows.into_iter().map(row_to_result).collect())
    }

    /// Detect whether a query has temporal intent (asks about time or contains date references).
    fn has_temporal_intent(query: &str) -> bool {
        let q = query.to_lowercase();

        // "When did...", "When is...", "When was..."
        if q.starts_with("when ") {
            return true;
        }

        // "How long ago...", "How many years/months/weeks..."
        if q.starts_with("how long ")
            || q.starts_with("how many year")
            || q.starts_with("how many month")
            || q.starts_with("how many week")
        {
            return true;
        }

        // Contains month names
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
            return true;
        }

        // Contains year patterns (4 digits starting with 19 or 20)
        let bytes = q.as_bytes();
        for i in 0..bytes.len().saturating_sub(3) {
            if (bytes[i] == b'1' && bytes[i + 1] == b'9'
                || bytes[i] == b'2' && bytes[i + 1] == b'0')
                && bytes[i + 2].is_ascii_digit()
                && bytes[i + 3].is_ascii_digit()
            {
                return true;
            }
        }

        // Relative time references
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
        time_refs.iter().any(|t| q.contains(t))
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

        let limit = options.limit.unwrap_or(self.config.default_limit);
        let threshold = options.threshold;

        let has_reranker = self.reranker.is_some() && self.config.enable_rerank;
        let rerank_mult = if has_reranker {
            self.config.rerank_candidate_multiplier.max(1)
        } else {
            1
        };

        let mut rrf_fused = false; // tracks whether RRF fusion was actually applied
        let results = if options.keyword_search || self.config.enable_graph {
            // Four-channel retrieval
            let candidate_limit = limit * self.config.rrf_candidate_multiplier.max(1) * rerank_mult;

            // Channel 1: Vector search
            let vector_rows = self.storage.vector_search(
                &embedding,
                &options.user_id,
                options.agent_id.as_deref(),
                options.run_id.as_deref(),
                options.app_id.as_deref(),
                options.filter.as_ref(),
                candidate_limit,
            )?;
            let vector_results: Vec<MemoryResult> =
                vector_rows.into_iter().map(row_to_result).collect();

            // Channel 2: BM25/FTS search
            let fts_results: Vec<MemoryResult> = if options.keyword_search {
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
            } else {
                Vec::new()
            };

            // Channel 3: Entity-centric retrieval (via Aho-Corasick + graph spreading)
            let entity_results: Vec<MemoryResult> = if self.config.enable_graph {
                match self.entity_channel_search(query, &options.user_id, candidate_limit) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("Entity channel search failed: {e}");
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            // Channel 4: Temporal retrieval (when query has temporal intent)
            let temporal_results: Vec<MemoryResult> = if Self::has_temporal_intent(query) {
                match self
                    .storage
                    .temporal_search(&options.user_id, candidate_limit)
                {
                    Ok(rows) => rows.into_iter().map(row_to_result).collect(),
                    Err(e) => {
                        tracing::warn!("Temporal search failed: {e}");
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            // Build ranked lists with weights
            let mut ranked_lists: Vec<(&[MemoryResult], f64)> = Vec::new();

            if !vector_results.is_empty() {
                ranked_lists.push((&vector_results, self.config.rrf_vector_weight));
            }
            if !fts_results.is_empty() {
                ranked_lists.push((&fts_results, self.config.rrf_fts_weight));
            }
            if !entity_results.is_empty() {
                ranked_lists.push((&entity_results, self.config.rrf_entity_weight));
            }
            if !temporal_results.is_empty() {
                ranked_lists.push((&temporal_results, self.config.rrf_temporal_weight));
            }

            if ranked_lists.is_empty() {
                Vec::new()
            } else if ranked_lists.len() == 1 {
                // Only one channel has results, use it directly
                ranked_lists[0].0.to_vec()
            } else {
                rrf_fused = true;
                let fuse_limit = limit * rerank_mult;
                crate::search::rrf_fuse(&ranked_lists, self.config.rrf_k, fuse_limit)
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
            if self.config.enable_rerank {
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
        let mut results = if self.config.enable_forgetting_curve {
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

            results
                .into_iter()
                .map(|mut r| {
                    let stability = r.stability.unwrap_or(1.0);
                    let retention = compute_retention(&r.updated_at, stability);
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
                        let w = self.config.retention_weight;
                        let weighted = similarity * (w * retention + (1.0 - w) * imp);
                        r.score = Some(weighted);
                    }
                    r
                })
                .collect::<Vec<_>>()
        } else {
            results
        };

        if self.config.enable_forgetting_curve {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

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
            for r in &results {
                if self.config.enable_forgetting_curve {
                    queue.push(super::DeferredWrite::ReinforceStability(
                        r.id.clone(),
                        self.config.stability_growth_factor,
                    ));
                } else {
                    queue.push(super::DeferredWrite::IncrementAccess(r.id.clone()));
                }
            }
        }

        // Enforce limit — earlier stages may return more candidates than requested
        let results: Vec<MemoryResult> = results.into_iter().take(limit).collect();

        // Apply threshold filter (works in all paths: vector distance, RRF, or weighted score)
        let results = if let Some(t) = threshold {
            if rrf_fused || was_reranked || self.config.enable_forgetting_curve {
                // Score is a relevance/weighted score (higher = better): keep if >= threshold
                results
                    .into_iter()
                    .filter(|r| r.score.unwrap_or(0.0) >= t)
                    .collect()
            } else {
                // Pure vector: score is cosine distance (lower = better): keep if <= threshold
                results
                    .into_iter()
                    .filter(|r| r.score.is_none_or(|s| s <= t))
                    .collect()
            }
        } else {
            results
        };

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
    pub fn search_reranked(
        &self,
        query: &str,
        options: SearchOptions,
        reranker: &dyn crate::rerank::Reranker,
        candidate_multiplier: usize,
    ) -> Result<Vec<MemoryResult>> {
        let limit = options.limit.unwrap_or(self.config.default_limit);
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
        if !self.config.enable_forgetting_curve {
            return Ok(None);
        }
        let row = self.storage.get_memory(id)?;
        Ok(row.map(|r| {
            let stability = r.stability.unwrap_or(1.0);
            compute_retention(&r.updated_at, stability)
        }))
    }
}
