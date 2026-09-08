use uuid::Uuid;

use crate::error::{MemoryError, Result};
use crate::types::*;

impl super::MemoryStore {
    /// **Internal** — Most users should use `compact()` which creates episodes automatically.
    ///
    /// Create an episode manually.
    pub fn create_episode(&self, options: CreateEpisodeOptions) -> Result<Episode> {
        let episode_id = Uuid::new_v4().to_string();
        let embedding = self
            .embedder
            .embed(&options.summary)
            .map_err(MemoryError::Embedding)?;
        self.storage.insert_episode(
            &episode_id,
            &options.title,
            &options.summary,
            &embedding,
            &options,
        )?;
        self.storage
            .get_episode(&episode_id)?
            .ok_or_else(|| MemoryError::NotFound(episode_id))
    }

    /// **Internal** — Most users should use `search()` which returns episodes via the Trace model.
    ///
    /// Search episodes using multi-channel retrieval (vector + FTS + entity) with RRF fusion.
    pub fn search_episodes(
        &self,
        query: &str,
        options: SearchEpisodesOptions,
    ) -> Result<Vec<Episode>> {
        let limit = options.limit.unwrap_or(10);
        let candidate_limit = limit * 3;

        let use_entity = self.config.enable_graph;

        // Channel 1: Vector search
        let embedding = self.embedder.embed(query).map_err(MemoryError::Embedding)?;
        let vector_results = self.storage.search_episodes_by_vector(
            &embedding,
            &options.user_id,
            candidate_limit,
        )?;

        // Channel 2: FTS search
        let fts_results: Vec<Episode> =
            match self
                .storage
                .fts_search_episodes(query, &options.user_id, candidate_limit)
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Episode FTS search failed (index may not exist): {e}");
                    Vec::new()
                }
            };

        // Channel 3: Entity-centric retrieval
        let entity_results: Vec<Episode> = if use_entity {
            let entity_index = self.get_entity_index(&options.user_id);
            let matched = entity_index.extract(query);
            if matched.is_empty() {
                Vec::new()
            } else {
                let seed_refs: Vec<&str> = matched.iter().map(|s: &String| s.as_str()).collect();
                let depth = self.config.tuning.graph_spreading_depth;
                let expanded =
                    self.storage
                        .spread_entity_names(&seed_refs, &options.user_id, depth)?;
                let entity_refs: Vec<&str> = expanded.iter().map(|s| s.as_str()).collect();
                match self.storage.entity_associated_episodes(
                    &entity_refs,
                    &options.user_id,
                    candidate_limit,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("Entity episode search failed: {e}");
                        Vec::new()
                    }
                }
            }
        } else {
            Vec::new()
        };

        // RRF fusion
        let mut ranked_lists: Vec<(&[Episode], f64)> = Vec::new();
        if !vector_results.is_empty() {
            ranked_lists.push((&vector_results, 0.4));
        }
        if !fts_results.is_empty() {
            ranked_lists.push((&fts_results, 0.3));
        }
        if !entity_results.is_empty() {
            ranked_lists.push((&entity_results, 0.3));
        }

        if ranked_lists.is_empty() {
            Ok(Vec::new())
        } else if ranked_lists.len() == 1 {
            let mut results = ranked_lists[0].0.to_vec();
            results.truncate(limit);
            Ok(results)
        } else {
            Ok(crate::search::rrf_fuse_episodes(&ranked_lists, 60, limit))
        }
    }

    /// Get a single episode, reinforcing its recall strength.
    pub fn get_episode(&self, episode_id: &str) -> Result<Option<Episode>> {
        let ep = self.storage.get_episode(episode_id)?;
        if ep.is_some() {
            let _ = self.storage.reinforce_episode(episode_id);
        }
        Ok(ep)
    }

    /// List episodes for a user with pagination (limit + offset).
    pub fn list_episodes(&self, options: ListEpisodesOptions) -> Result<Vec<Episode>> {
        self.storage.list_episodes_paged(&options)
    }

    /// Get the messages (events) associated with an episode.
    /// Supports pagination via `options`.
    pub fn get_episode_messages(
        &self,
        episode_id: &str,
        options: EpisodeMessagesOptions,
    ) -> Result<Vec<Event>> {
        let ep = self
            .storage
            .get_episode(episode_id)?
            .ok_or_else(|| MemoryError::NotFound(episode_id.to_string()))?;
        if ep.event_ids.is_empty() {
            return Ok(Vec::new());
        }

        let limit = options.limit.unwrap_or(50);
        let offset = options.offset.unwrap_or(0);

        // Apply offset/limit to event_ids before querying
        let paged_ids: Vec<String> = ep.event_ids.into_iter().skip(offset).take(limit).collect();

        if paged_ids.is_empty() {
            return Ok(Vec::new());
        }
        self.storage
            .get_events_by_ids_for_user(&paged_ids, &ep.user_id)
    }

    /// Search episode messages by semantic similarity.
    pub fn search_episode_messages(
        &self,
        episode_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Event>> {
        let ep = self
            .storage
            .get_episode(episode_id)?
            .ok_or_else(|| MemoryError::NotFound(episode_id.to_string()))?;
        if ep.event_ids.is_empty() {
            return Ok(Vec::new());
        }
        // Get all events, batch-embed them, then rank by similarity to query
        let all_events = self
            .storage
            .get_events_by_ids_for_user(&ep.event_ids, &ep.user_id)?;
        if all_events.is_empty() {
            return Ok(Vec::new());
        }
        let query_embedding = self.embedder.embed(query).map_err(MemoryError::Embedding)?;
        let texts: Vec<&str> = all_events.iter().map(|ev| ev.content.as_str()).collect();
        let embeddings = self
            .embedder
            .embed_batch(&texts)
            .map_err(MemoryError::Embedding)?;

        let mut scored: Vec<(f32, Event)> = all_events
            .into_iter()
            .zip(embeddings)
            .map(|(ev, emb)| {
                let score = cosine_similarity(&query_embedding, &emb);
                (score, ev)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(_, ev)| ev).collect())
    }

    /// Delete an episode by ID.
    pub fn delete_episode(&self, episode_id: &str) -> Result<()> {
        self.storage.delete_episode(episode_id)
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}
