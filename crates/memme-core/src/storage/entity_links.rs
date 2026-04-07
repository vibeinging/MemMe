use crate::error::Result;

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
        let conn = self.write_conn();
        conn.execute(
            sql,
            duckdb::params![memory_id, entity_id, entity_name, user_id],
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
        let (distance_col, order_clause) = if let Some(emb) = query_embedding {
            let lit = Self::format_embedding(emb, self.config.embedding_dims)?;
            (
                format!("array_cosine_distance(m.embedding, {lit}) AS distance"),
                "ORDER BY distance ASC".to_string(),
            )
        } else {
            (
                "NULL AS distance".to_string(),
                "ORDER BY m.importance DESC NULLS LAST".to_string(),
            )
        };

        let sql = format!(
            r#"SELECT DISTINCT m.id, m.content, m.user_id,
                      m.created_at::VARCHAR, m.updated_at::VARCHAR, m.metadata,
                      {distance_col},
                      m.importance, m.access_count, m.agent_id, m.app_id, m.run_id,
                      m.immutable, m.expiration_date::VARCHAR, m.categories::VARCHAR,
                      m.memory_type, m.stability, m.privacy, m.event_time::VARCHAR,
                      m.episode_id, m.session_id, m.resolution
               FROM memories m
               JOIN memory_entities me ON m.id = me.memory_id
               WHERE me.user_id = $1
                 AND LOWER(me.entity_name) IN ({in_clause})
               {order_clause}
               LIMIT {limit}"#
        );

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(duckdb::params![user_id], |row| {
            Ok(MemoryRow {
                id: row.get(0)?,
                content: row.get(1)?,
                user_id: row.get(2)?,
                created_at: row.get::<_, String>(3)?,
                updated_at: row.get::<_, String>(4)?,
                metadata: row.get::<_, Option<String>>(5)?,
                score: row.get::<_, Option<f64>>(6)?.map(|d| d as f32),
                importance: row
                    .get::<_, Option<f64>>(7)
                    .ok()
                    .flatten()
                    .map(|v| v as f32),
                access_count: row
                    .get::<_, Option<i32>>(8)
                    .ok()
                    .flatten()
                    .map(|v| v as u32),
                agent_id: row.get::<_, Option<String>>(9)?,
                app_id: row.get::<_, Option<String>>(10)?,
                run_id: row.get::<_, Option<String>>(11)?,
                immutable: row.get::<_, Option<bool>>(12)?.unwrap_or(false),
                expiration_date: row.get::<_, Option<String>>(13)?,
                categories: row.get::<_, Option<String>>(14)?,
                memory_type: row.get::<_, Option<String>>(15)?,
                stability: row
                    .get::<_, Option<f64>>(16)
                    .ok()
                    .flatten()
                    .map(|v| v as f32),
                privacy: row.get::<_, Option<String>>(17).ok().flatten(),
                event_time: row.get::<_, Option<String>>(18).ok().flatten(),
                episode_id: row.get::<_, Option<String>>(19).ok().flatten(),
                session_id: row.get::<_, Option<String>>(20).ok().flatten(),
                resolution: row.get::<_, Option<String>>(21).ok().flatten(),
            })
        })?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(crate::error::MemoryError::DuckDb)
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
        let mut dynamic_params: Vec<duckdb::types::Value> =
            vec![duckdb::types::Value::Text(user_id.to_string())];
        for (i, name) in entity_names.iter().enumerate() {
            let idx = i + 2; // $1 is user_id
                             // Escape LIKE wildcards in entity name
            let escaped = name.to_lowercase().replace('%', "\\%").replace('_', "\\_");
            like_conditions.push(format!(
                "(LOWER(title) LIKE ${idx} OR LOWER(summary) LIKE ${idx})"
            ));
            dynamic_params.push(duckdb::types::Value::Text(format!("%{escaped}%")));
        }

        let where_like = like_conditions.join(" OR ");
        let sql = format!(
            r#"SELECT DISTINCT episode_id, title, summary,
                      CAST(started_at AS VARCHAR), CAST(ended_at AS VARCHAR),
                      significance, outcome, source_id, event_ids, user_id,
                      CAST(created_at AS VARCHAR), CAST(last_recalled AS VARCHAR),
                      recall_count, storage_strength, retrieval_strength,
                      session_ids, CAST(last_meditated_at AS VARCHAR)
               FROM episodes
               WHERE user_id = $1 AND ({where_like})
               ORDER BY significance DESC
               LIMIT {limit}"#
        );

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn duckdb::ToSql> = dynamic_params
            .iter()
            .map(|p| p as &dyn duckdb::ToSql)
            .collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                let event_ids_raw: Option<String> = row.get(8)?;
                let event_ids: Vec<String> = event_ids_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                let session_ids_raw: Option<String> =
                    row.get::<_, Option<String>>(15).ok().flatten();
                let session_ids: Vec<String> = session_ids_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                Ok(crate::types::Episode {
                    episode_id: row.get(0)?,
                    title: row.get(1)?,
                    summary: row.get(2)?,
                    started_at: row.get::<_, String>(3)?,
                    ended_at: row.get::<_, Option<String>>(4)?,
                    significance: row.get::<_, Option<f64>>(5)?.unwrap_or(0.5) as f32,
                    outcome: row.get::<_, Option<String>>(6)?,
                    source_id: row.get::<_, Option<String>>(7)?,
                    event_ids,
                    session_ids,
                    user_id: row.get(9)?,
                    created_at: row.get::<_, String>(10)?,
                    last_recalled: row.get::<_, Option<String>>(11)?,
                    recall_count: row.get::<_, Option<i32>>(12)?.unwrap_or(0) as u32,
                    storage_strength: row.get::<_, Option<f64>>(13)?.unwrap_or(1.0) as f32,
                    retrieval_strength: row.get::<_, Option<f64>>(14)?.unwrap_or(1.0) as f32,
                    last_meditated_at: row.get::<_, Option<String>>(16)?,
                    score: None,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get all entity names for a user (for building Aho-Corasick dictionary).
    pub(crate) fn all_entity_names(&self, user_id: &str) -> Result<Vec<String>> {
        let collection = &self.config.collection_name;
        let sql =
            format!("SELECT DISTINCT LOWER(name) FROM entities_{collection} WHERE user_id = $1");
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(duckdb::params![user_id], |row| row.get::<_, String>(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(crate::error::MemoryError::DuckDb)
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
        let conn = self.read_conn();

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

            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(duckdb::params![user_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;

            let mut next_frontier = Vec::new();
            for pair in rows {
                match pair {
                    Ok((name, desc)) => {
                        if seen.insert(name.clone()) {
                            all.push((name.clone(), desc));
                            next_frontier.push(name);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Error reading entity name in spread: {e}");
                    }
                }
            }
            frontier = next_frontier;
        }

        Ok(all)
    }
}
