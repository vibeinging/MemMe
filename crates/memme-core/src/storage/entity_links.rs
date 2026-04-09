use crate::error::Result;
use crate::types::SqlParam;

use super::{MemoryRow, Storage};

impl Storage {
    /// Link a memory to an entity.
    pub(crate) fn link_memory_entity(
        &self,
        memory_id: &str,
        entity_id: &str,
        entity_name: &str,
        user_id: &str,
    ) -> Result<()> {
        let sql = "INSERT INTO memory_entities (memory_id, entity_id, entity_name, user_id) \
                   VALUES ($1, $2, $3, $4) \
                   ON CONFLICT DO NOTHING";
        self.backend.execute(
            sql,
            &[
                SqlParam::Text(memory_id.to_string()),
                SqlParam::Text(entity_id.to_string()),
                SqlParam::Text(entity_name.to_string()),
                SqlParam::Text(user_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Find all memories linked to ANY of the given entity names (case-insensitive).
    /// This is the core of entity-centric retrieval.
    ///
    /// When `query_embedding` is provided, results are ranked by cosine distance
    /// (semantic relevance) instead of importance. This enables entity channel results
    /// to participate meaningfully in RRF fusion.
    pub(crate) fn entity_associated_memories(
        &self,
        entity_names: &[&str],
        user_id: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
    ) -> Result<Vec<MemoryRow>> {
        if entity_names.is_empty() {
            return Ok(Vec::new());
        }

        let escaped_names: Vec<String> = entity_names
            .iter()
            .map(|n| format!("'{}'", n.to_lowercase().replace('\'', "''")))
            .collect();
        let in_clause = escaped_names.join(", ");

        // When query embedding is available, compute cosine distance for semantic ranking
        let (score_expr, order_clause) = if let Some(emb) = query_embedding {
            let lit = self.format_embedding(emb, self.config.embedding_dims)?;
            let dist = self.dialect().cosine_distance_expr("m.embedding", &lit);
            (Some(dist), "ORDER BY score ASC".to_string())
        } else {
            (None, "ORDER BY m.importance DESC NULLS LAST".to_string())
        };

        let cols = super::query::memory_select_cols(score_expr.as_deref(), "m.");
        let sql = format!(
            "SELECT DISTINCT {cols} FROM memories m \
             JOIN memory_entities me ON m.id = me.memory_id \
             WHERE me.user_id = $1 AND LOWER(me.entity_name) IN ({in_clause}) \
             {order_clause} LIMIT {limit}"
        );

        self.backend.query_read(
            &sql,
            &[SqlParam::Text(user_id.to_string())],
            super::query::map_memory_row,
        )
    }

    /// Find episodes whose title or summary mention any of the given entity names.
    /// Uses a single SQL query with OR conditions instead of N+1 queries.
    pub(crate) fn entity_associated_episodes(
        &self,
        entity_names: &[&str],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<crate::types::Episode>> {
        if entity_names.is_empty() {
            return Ok(Vec::new());
        }

        // Build OR conditions for LIKE matching, with parameterized patterns
        let mut like_conditions = Vec::new();
        let mut dynamic_params: Vec<SqlParam> =
            vec![SqlParam::Text(user_id.to_string())];
        for (i, name) in entity_names.iter().enumerate() {
            let idx = i + 2; // $1 is user_id
                             // Escape LIKE wildcards in entity name
            let escaped = name.to_lowercase().replace('%', "\\%").replace('_', "\\_");
            like_conditions.push(format!(
                "(LOWER(title) LIKE ${idx} OR LOWER(summary) LIKE ${idx})"
            ));
            dynamic_params.push(SqlParam::Text(format!("%{escaped}%")));
        }

        let where_like = like_conditions.join(" OR ");
        let sql = format!(
            r#"SELECT DISTINCT episode_id, title, summary,
                      started_at, ended_at,
                      significance, outcome, source_id, event_ids, user_id,
                      created_at, last_recalled,
                      recall_count, storage_strength, retrieval_strength,
                      session_ids, last_meditated_at
               FROM episodes
               WHERE user_id = $1 AND ({where_like})
               ORDER BY significance DESC
               LIMIT {limit}"#
        );

        self.backend.query_read(&sql, &dynamic_params, |row| {
            let event_ids_raw: Option<String> = row.get_opt_string(8)?;
            let event_ids: Vec<String> = event_ids_raw
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
            let session_ids_raw: Option<String> = row.get_opt_string(15)?;
            let session_ids: Vec<String> = session_ids_raw
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
            Ok(crate::types::Episode {
                episode_id: row.get_string(0)?,
                title: row.get_string(1)?,
                summary: row.get_string(2)?,
                started_at: row.get_string(3)?,
                ended_at: row.get_opt_string(4)?,
                significance: row.get_opt_f64(5)?.unwrap_or(0.5) as f32,
                outcome: row.get_opt_string(6)?,
                source_id: row.get_opt_string(7)?,
                event_ids,
                session_ids,
                user_id: row.get_string(9)?,
                created_at: row.get_string(10)?,
                last_recalled: row.get_opt_string(11)?,
                recall_count: row.get_opt_i64(12)?.unwrap_or(0) as u32,
                storage_strength: row.get_opt_f64(13)?.unwrap_or(1.0) as f32,
                retrieval_strength: row.get_opt_f64(14)?.unwrap_or(1.0) as f32,
                last_meditated_at: row.get_opt_string(16)?,
                score: None,
            })
        })
    }

    /// Get all entity names for a user (for building Aho-Corasick dictionary).
    pub(crate) fn all_entity_names(&self, user_id: &str) -> Result<Vec<String>> {
        let collection = &self.config.collection_name;
        let sql =
            format!("SELECT DISTINCT LOWER(name) FROM entities_{collection} WHERE user_id = $1");
        self.backend.query_read(
            &sql,
            &[SqlParam::Text(user_id.to_string())],
            |row| row.get_string(0),
        )
    }

    /// Spreading activation: expand seed entity names by traversing the graph.
    ///
    /// Each hop discovers entities connected to the current frontier via relationships.
    /// `depth` controls how many hops to traverse (1 = direct neighbors, 2 = friends-of-friends).
    /// Results are capped at 100 expanded entities to prevent explosion in dense graphs.
    pub(crate) fn spread_entity_names(
        &self,
        seed_names: &[&str],
        user_id: &str,
        depth: usize,
    ) -> Result<Vec<String>> {
        Ok(self
            .spread_entity_names_with_context(seed_names, user_id, depth)?
            .into_iter()
            .map(|(name, _)| name)
            .collect())
    }

    /// Like `spread_entity_names`, but also returns the relationship description
    /// that connects each expanded entity to its seed. Seeds have an empty description.
    /// Returns `Vec<(entity_name, relationship_description)>`.
    pub(crate) fn spread_entity_names_with_context(
        &self,
        seed_names: &[&str],
        user_id: &str,
        depth: usize,
    ) -> Result<Vec<(String, String)>> {
        let seeds: Vec<(String, String)> = seed_names
            .iter()
            .map(|s| (s.to_lowercase(), String::new()))
            .collect();
        if seed_names.is_empty() || depth == 0 {
            return Ok(seeds);
        }

        const MAX_SPREAD: usize = 100;
        let collection = &self.config.collection_name;

        let mut seen: std::collections::HashSet<String> =
            seed_names.iter().map(|s| s.to_lowercase()).collect();
        let mut all: Vec<(String, String)> = seeds;
        let mut frontier: Vec<String> = seen.iter().cloned().collect();

        for _hop in 0..depth {
            if frontier.is_empty() || all.len() >= MAX_SPREAD {
                break;
            }

            let escaped: Vec<String> = frontier
                .iter()
                .map(|n| format!("'{}'", n.replace('\'', "''")))
                .collect();
            let in_clause = escaped.join(", ");

            let sql = format!(
                r#"SELECT DISTINCT LOWER(e2.name), COALESCE(r.description, r.relation_type)
                   FROM entities_{collection} e1
                   JOIN relationships_{collection} r ON (e1.id = r.source_id OR e1.id = r.target_id)
                   JOIN entities_{collection} e2 ON (e2.id = r.source_id OR e2.id = r.target_id)
                   WHERE LOWER(e1.name) IN ({in_clause})
                     AND e1.user_id = $1
                     AND e2.id != e1.id
                   LIMIT {limit}"#,
                limit = MAX_SPREAD
            );

            let rows: Vec<(String, String)> = self.backend.query_read(
                &sql,
                &[SqlParam::Text(user_id.to_string())],
                |row| Ok((row.get_string(0)?, row.get_string(1)?)),
            )?;

            let mut next_frontier = Vec::new();
            for (name, desc) in rows {
                if seen.insert(name.clone()) {
                    all.push((name.clone(), desc));
                    next_frontier.push(name);
                }
            }
            frontier = next_frontier;
        }

        Ok(all)
    }
}

