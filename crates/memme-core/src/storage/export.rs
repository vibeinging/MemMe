use duckdb::params;

use crate::error::Result;
use crate::types::{
    Entity, Episode, Event, EventType, FullExport, GraphRelation, IdentityTrait, MemoryExport,
    Session, Source, TraitType,
};

use super::Storage;

impl Storage {
    /// Export all memories, optionally filtered by user_id.
    /// By default, skips `local_only` privacy memories.
    #[allow(dead_code)]
    pub(crate) fn export_memories(&self, user_id: Option<&str>) -> Result<Vec<MemoryExport>> {
        self.export_memories_with_privacy(user_id, false)
    }

    /// Export memories with control over including local_only privacy memories.
    #[allow(dead_code)]
    pub(crate) fn export_memories_with_privacy(
        &self,
        user_id: Option<&str>,
        include_local: bool,
    ) -> Result<Vec<MemoryExport>> {
        let privacy_filter = if include_local {
            ""
        } else {
            " AND (privacy IS NULL OR privacy != 'local_only')"
        };
        let (sql, has_user) = if user_id.is_some() {
            (
                format!(
                    "SELECT id, content, user_id, agent_id, app_id, run_id,
                        CAST(metadata AS VARCHAR) AS metadata,
                        importance, immutable,
                        CAST(expiration_date AS VARCHAR) AS expiration_date,
                        CAST(categories AS VARCHAR) AS categories,
                        CAST(created_at AS VARCHAR) AS created_at,
                        CAST(updated_at AS VARCHAR) AS updated_at,
                        stability
                 FROM memories WHERE user_id = $1{privacy_filter}
                 ORDER BY created_at"
                ),
                true,
            )
        } else {
            (
                format!(
                    "SELECT id, content, user_id, agent_id, app_id, run_id,
                        CAST(metadata AS VARCHAR) AS metadata,
                        importance, immutable,
                        CAST(expiration_date AS VARCHAR) AS expiration_date,
                        CAST(categories AS VARCHAR) AS categories,
                        CAST(created_at AS VARCHAR) AS created_at,
                        CAST(updated_at AS VARCHAR) AS updated_at,
                        stability
                 FROM memories WHERE 1=1{privacy_filter}
                 ORDER BY created_at"
                ),
                false,
            )
        };

        let map_export = |row: &duckdb::Row<'_>| -> duckdb::Result<MemoryExport> {
            let metadata_str: Option<String> = row.get(6)?;
            let metadata = metadata_str.and_then(|s| serde_json::from_str(&s).ok());
            let categories_raw: Option<String> = row.get(10)?;
            Ok(MemoryExport {
                id: row.get(0)?,
                content: row.get(1)?,
                user_id: row.get(2)?,
                agent_id: row.get(3)?,
                app_id: row.get(4)?,
                run_id: row.get(5)?,
                metadata,
                importance: row.get::<_, Option<f64>>(7)?.unwrap_or(0.5) as f32,
                immutable: row.get::<_, Option<bool>>(8)?.unwrap_or(false),
                expiration_date: row.get(9)?,
                categories: Self::parse_categories(categories_raw),
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
                stability: row.get::<_, Option<f64>>(13)?.map(|v| v as f32),
            })
        };

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = if has_user {
            stmt.query_map(params![user_id.unwrap()], map_export)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map([], map_export)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(rows)
    }

    // ── Full export ──

    /// Export all data layers, optionally filtered by user_id.
    #[allow(dead_code)]
    pub(crate) fn full_export(
        &self,
        user_id: Option<&str>,
        collection: &str,
    ) -> Result<FullExport> {
        let memories = self.export_memories(user_id)?;

        let sessions = if let Some(uid) = user_id {
            let opts = crate::types::ListSessionsOptions::new(uid);
            self.list_sessions(&opts)?
        } else {
            self.list_all_sessions()?
        };

        let events = if let Some(uid) = user_id {
            let opts = crate::types::ListEventsOptions {
                user_id: uid.to_string(),
                limit: None,
                ..Default::default()
            };
            self.list_events(&opts)?
        } else {
            self.list_all_events()?
        };

        let episodes = if let Some(uid) = user_id {
            self.list_episodes_for_user(uid)?
        } else {
            self.list_all_episodes()?
        };

        let entities = self.export_entities(user_id)?;
        let relations = self.export_relations(user_id)?;

        let identity_traits = if let Some(uid) = user_id {
            self.list_identity_traits(uid)?
        } else {
            self.list_all_identity_traits()?
        };

        let sources = if let Some(uid) = user_id {
            self.list_sources(uid)?
        } else {
            self.list_all_sources()?
        };

        Ok(FullExport {
            version: "2.0".into(),
            collection: collection.to_string(),
            exported_at: chrono::Utc::now().to_rfc3339(),
            memories,
            entities,
            relations,
            sessions,
            episodes,
            events,
            identity_traits,
            sources,
        })
    }

    // ── List-all methods (no user_id filter) ──

    /// List all sessions without user_id filter.
    fn list_all_sessions(&self) -> Result<Vec<Session>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT s.session_id, s.user_id, s.source_id,
                      CAST(s.started_at AS VARCHAR), CAST(s.ended_at AS VARCHAR),
                      s.metadata, CAST(s.created_at AS VARCHAR),
                      (SELECT COUNT(*) FROM events WHERE session_id = s.session_id) AS event_count,
                      s.structured_notes
               FROM sessions s
               ORDER BY s.started_at DESC"#,
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Session {
                    session_id: row.get(0)?,
                    user_id: row.get(1)?,
                    source_id: row.get::<_, Option<String>>(2)?,
                    started_at: row.get::<_, String>(3)?,
                    ended_at: row.get::<_, Option<String>>(4)?,
                    metadata: row
                        .get::<_, Option<String>>(5)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    created_at: row.get::<_, String>(6)?,
                    event_count: row.get::<_, Option<i64>>(7)?.unwrap_or(0) as u32,
                    structured_notes: row.get::<_, Option<String>>(8)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// List all events without user_id filter.
    fn list_all_events(&self) -> Result<Vec<Event>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT event_id, source_id, session_id, CAST(timestamp AS VARCHAR),
                      event_type, content, parent_id, metadata, user_id,
                      processed, CAST(processed_at AS VARCHAR),
                      purified_content, purified, CAST(event_time AS VARCHAR), location
               FROM events ORDER BY timestamp DESC"#,
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Event {
                    event_id: row.get(0)?,
                    source_id: row.get::<_, Option<String>>(1)?,
                    session_id: row.get::<_, Option<String>>(2)?,
                    timestamp: row.get::<_, String>(3)?,
                    event_type: EventType::parse(
                        &row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                    ),
                    content: row.get(5)?,
                    parent_id: row.get::<_, Option<String>>(6)?,
                    metadata: row
                        .get::<_, Option<String>>(7)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    user_id: row.get(8)?,
                    processed: row.get::<_, Option<bool>>(9)?.unwrap_or(false),
                    processed_at: row.get::<_, Option<String>>(10)?,
                    purified_content: row.get::<_, Option<String>>(11)?,
                    purified: row.get::<_, Option<bool>>(12)?.unwrap_or(false),
                    event_time: row.get::<_, Option<String>>(13)?,
                    location: row.get::<_, Option<String>>(14)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// List all episodes without user_id filter.
    fn list_all_episodes(&self) -> Result<Vec<Episode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT episode_id, title, summary,
                      CAST(started_at AS VARCHAR), CAST(ended_at AS VARCHAR),
                      significance, outcome, source_id, event_ids, user_id,
                      CAST(created_at AS VARCHAR), CAST(last_recalled AS VARCHAR),
                      recall_count, storage_strength, retrieval_strength,
                      session_ids, CAST(last_meditated_at AS VARCHAR)
               FROM episodes ORDER BY started_at DESC"#,
        )?;
        let rows = stmt
            .query_map([], |row| {
                let event_ids_raw: Option<String> = row.get(8)?;
                let event_ids: Vec<String> = event_ids_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                let session_ids_raw: Option<String> =
                    row.get::<_, Option<String>>(15).ok().flatten();
                let session_ids: Vec<String> = session_ids_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                Ok(Episode {
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

    /// List all identity traits without user_id filter.
    fn list_all_identity_traits(&self) -> Result<Vec<IdentityTrait>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT trait_id, trait_type, content, confidence, evidence_ids, user_id,
                      CAST(created_at AS VARCHAR), CAST(updated_at AS VARCHAR)
               FROM identity ORDER BY confidence DESC"#,
        )?;
        let rows = stmt
            .query_map([], |row| {
                let evidence_raw: Option<String> = row.get(4)?;
                let evidence_ids: Vec<String> = evidence_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                Ok(IdentityTrait {
                    trait_id: row.get(0)?,
                    trait_type: TraitType::parse(&row.get::<_, String>(1).unwrap_or_default()),
                    content: row.get(2)?,
                    confidence: row.get::<_, Option<f64>>(3)?.unwrap_or(0.5) as f32,
                    evidence_ids,
                    user_id: row.get(5)?,
                    created_at: row.get::<_, String>(6)?,
                    updated_at: row.get::<_, Option<String>>(7)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// List all sources without user_id filter.
    fn list_all_sources(&self) -> Result<Vec<Source>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT source_id, source_type, name, CAST(registered_at AS VARCHAR), metadata FROM sources ORDER BY registered_at DESC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Source {
                    source_id: row.get(0)?,
                    source_type: row.get(1)?,
                    name: row.get::<_, Option<String>>(2)?,
                    registered_at: row.get::<_, String>(3)?,
                    metadata: row
                        .get::<_, Option<String>>(4)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ── Export entities and relations as typed structs ──

    /// Export entities as Entity structs, optionally filtered by user_id.
    fn export_entities(&self, user_id: Option<&str>) -> Result<Vec<Entity>> {
        let collection = &self.config.collection_name;
        let (sql, has_user) = if user_id.is_some() {
            (
                format!(
                    "SELECT id, name, entity_type, user_id FROM entities_{collection} WHERE user_id = $1 ORDER BY created_at"
                ),
                true,
            )
        } else {
            (
                format!(
                    "SELECT id, name, entity_type, user_id FROM entities_{collection} ORDER BY created_at"
                ),
                false,
            )
        };

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let map_entity = |row: &duckdb::Row<'_>| -> duckdb::Result<Entity> {
            Ok(Entity {
                id: row.get(0)?,
                name: row.get(1)?,
                entity_type: row.get::<_, Option<String>>(2)?,
                user_id: row.get(3)?,
            })
        };
        let rows = if has_user {
            stmt.query_map(params![user_id.unwrap()], map_entity)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map([], map_entity)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        Ok(rows)
    }

    /// Export relationships as GraphRelation structs, optionally filtered by user_id.
    fn export_relations(&self, user_id: Option<&str>) -> Result<Vec<GraphRelation>> {
        let collection = &self.config.collection_name;
        let (sql, has_user) = if user_id.is_some() {
            (
                format!(
                    r#"SELECT r.id, r.source_id, r.target_id, r.relation_type, r.user_id,
                              s.name AS source_name, t.name AS target_name
                       FROM relationships_{collection} r
                       JOIN entities_{collection} s ON r.source_id = s.id
                       JOIN entities_{collection} t ON r.target_id = t.id
                       WHERE r.user_id = $1"#
                ),
                true,
            )
        } else {
            (
                format!(
                    r#"SELECT r.id, r.source_id, r.target_id, r.relation_type, r.user_id,
                              s.name AS source_name, t.name AS target_name
                       FROM relationships_{collection} r
                       JOIN entities_{collection} s ON r.source_id = s.id
                       JOIN entities_{collection} t ON r.target_id = t.id"#
                ),
                false,
            )
        };

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let map_relation = |row: &duckdb::Row<'_>| -> duckdb::Result<GraphRelation> {
            Ok(GraphRelation {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: row.get(3)?,
                user_id: row.get(4)?,
                source: row.get(5)?,
                target: row.get(6)?,
            })
        };
        let rows = if has_user {
            stmt.query_map(params![user_id.unwrap()], map_relation)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map([], map_relation)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        Ok(rows)
    }

    // ── Bulk import methods ──

    /// Import sources from export records. Returns the number of successfully imported sources.
    #[allow(dead_code)]
    pub(crate) fn import_sources(&self, sources: &[Source]) -> Result<u64> {
        let mut count: u64 = 0;
        let conn = self.write_conn();
        for src in sources {
            let name_val: duckdb::types::Value = match &src.name {
                Some(n) => duckdb::types::Value::Text(n.clone()),
                None => duckdb::types::Value::Null,
            };
            let meta_val: duckdb::types::Value = match &src.metadata {
                Some(m) => {
                    duckdb::types::Value::Text(serde_json::to_string(m).unwrap_or_default())
                }
                None => duckdb::types::Value::Null,
            };
            let affected = conn.execute(
                r#"INSERT INTO sources (source_id, source_type, name, registered_at, metadata)
                   VALUES ($1, $2, $3, CAST($4 AS TIMESTAMP), $5)
                   ON CONFLICT (source_id) DO NOTHING"#,
                params![src.source_id, src.source_type, name_val, src.registered_at, meta_val],
            )?;
            count += affected as u64;
        }
        Ok(count)
    }

    /// Import sessions from export records. Returns the number of successfully imported sessions.
    #[allow(dead_code)]
    pub(crate) fn import_sessions(&self, sessions: &[Session]) -> Result<u64> {
        let mut count: u64 = 0;
        let conn = self.write_conn();
        for sess in sessions {
            let source_val: duckdb::types::Value = match &sess.source_id {
                Some(s) => duckdb::types::Value::Text(s.clone()),
                None => duckdb::types::Value::Null,
            };
            let ended_val: duckdb::types::Value = match &sess.ended_at {
                Some(e) => duckdb::types::Value::Text(e.clone()),
                None => duckdb::types::Value::Null,
            };
            let meta_val: duckdb::types::Value = match &sess.metadata {
                Some(m) => {
                    duckdb::types::Value::Text(serde_json::to_string(m).unwrap_or_default())
                }
                None => duckdb::types::Value::Null,
            };
            let notes_val: duckdb::types::Value = match &sess.structured_notes {
                Some(n) => duckdb::types::Value::Text(n.clone()),
                None => duckdb::types::Value::Null,
            };
            let affected = conn.execute(
                r#"INSERT INTO sessions (session_id, user_id, source_id, started_at, ended_at, metadata, created_at, structured_notes)
                   VALUES ($1, $2, $3, CAST($4 AS TIMESTAMP),
                           CASE WHEN $5 IS NULL THEN NULL ELSE CAST($5 AS TIMESTAMP) END,
                           $6, CAST($7 AS TIMESTAMP), $8)
                   ON CONFLICT DO NOTHING"#,
                params![
                    sess.session_id,
                    sess.user_id,
                    source_val,
                    sess.started_at,
                    ended_val,
                    meta_val,
                    sess.created_at,
                    notes_val
                ],
            )?;
            count += affected as u64;
        }
        Ok(count)
    }

    /// Import events from export records. Returns the number of successfully imported events.
    /// Embeddings (content_vec) are set to NULL since they are not exported.
    #[allow(dead_code)]
    pub(crate) fn import_events(&self, events: &[Event]) -> Result<u64> {
        let mut count: u64 = 0;
        let conn = self.write_conn();
        for evt in events {
            let source_val: duckdb::types::Value = match &evt.source_id {
                Some(s) => duckdb::types::Value::Text(s.clone()),
                None => duckdb::types::Value::Null,
            };
            let session_val: duckdb::types::Value = match &evt.session_id {
                Some(s) => duckdb::types::Value::Text(s.clone()),
                None => duckdb::types::Value::Null,
            };
            let parent_val: duckdb::types::Value = match &evt.parent_id {
                Some(p) => duckdb::types::Value::Text(p.clone()),
                None => duckdb::types::Value::Null,
            };
            let meta_val: duckdb::types::Value = match &evt.metadata {
                Some(m) => {
                    duckdb::types::Value::Text(serde_json::to_string(m).unwrap_or_default())
                }
                None => duckdb::types::Value::Null,
            };
            let purified_val: duckdb::types::Value = match &evt.purified_content {
                Some(p) => duckdb::types::Value::Text(p.clone()),
                None => duckdb::types::Value::Null,
            };
            let event_time_val: duckdb::types::Value = match &evt.event_time {
                Some(t) => duckdb::types::Value::Text(t.clone()),
                None => duckdb::types::Value::Null,
            };
            let location_val: duckdb::types::Value = match &evt.location {
                Some(l) => duckdb::types::Value::Text(l.clone()),
                None => duckdb::types::Value::Null,
            };
            let affected = conn.execute(
                r#"INSERT INTO events (event_id, source_id, session_id, timestamp, event_type,
                                       content, parent_id, metadata, user_id,
                                       processed, purified_content, purified,
                                       event_time, location)
                   VALUES ($1, $2, $3, CAST($4 AS TIMESTAMP), $5,
                           $6, $7, $8, $9,
                           $10, $11, $12,
                           CASE WHEN $13 IS NULL THEN NULL ELSE CAST($13 AS TIMESTAMP) END,
                           $14)
                   ON CONFLICT (event_id) DO NOTHING"#,
                params![
                    evt.event_id,
                    source_val,
                    session_val,
                    evt.timestamp,
                    evt.event_type.as_str(),
                    evt.content,
                    parent_val,
                    meta_val,
                    evt.user_id,
                    evt.processed,
                    purified_val,
                    evt.purified,
                    event_time_val,
                    location_val
                ],
            )?;
            count += affected as u64;
        }
        Ok(count)
    }

    /// Import episodes from export records. Returns the number of successfully imported episodes.
    /// Embeddings (summary_vec) are set to NULL since they are not exported.
    #[allow(dead_code)]
    pub(crate) fn import_episodes(&self, episodes: &[Episode]) -> Result<u64> {
        let mut count: u64 = 0;
        let conn = self.write_conn();
        for ep in episodes {
            let ended_val: duckdb::types::Value = match &ep.ended_at {
                Some(e) => duckdb::types::Value::Text(e.clone()),
                None => duckdb::types::Value::Null,
            };
            let outcome_val: duckdb::types::Value = match &ep.outcome {
                Some(o) => duckdb::types::Value::Text(o.clone()),
                None => duckdb::types::Value::Null,
            };
            let source_val: duckdb::types::Value = match &ep.source_id {
                Some(s) => duckdb::types::Value::Text(s.clone()),
                None => duckdb::types::Value::Null,
            };
            let event_ids_str = serde_json::to_string(&ep.event_ids).unwrap_or_default();
            let session_ids_str = serde_json::to_string(&ep.session_ids).unwrap_or_default();
            let last_recalled_val: duckdb::types::Value = match &ep.last_recalled {
                Some(l) => duckdb::types::Value::Text(l.clone()),
                None => duckdb::types::Value::Null,
            };
            let last_meditated_val: duckdb::types::Value = match &ep.last_meditated_at {
                Some(l) => duckdb::types::Value::Text(l.clone()),
                None => duckdb::types::Value::Null,
            };
            let affected = conn.execute(
                r#"INSERT INTO episodes (episode_id, title, summary, started_at, ended_at,
                                         significance, outcome, source_id, event_ids, user_id,
                                         created_at, last_recalled, recall_count,
                                         storage_strength, retrieval_strength,
                                         session_ids, last_meditated_at)
                   VALUES ($1, $2, $3, CAST($4 AS TIMESTAMP),
                           CASE WHEN $5 IS NULL THEN NULL ELSE CAST($5 AS TIMESTAMP) END,
                           $6, $7, $8, $9, $10,
                           CAST($11 AS TIMESTAMP),
                           CASE WHEN $12 IS NULL THEN NULL ELSE CAST($12 AS TIMESTAMP) END,
                           $13, $14, $15, $16,
                           CASE WHEN $17 IS NULL THEN NULL ELSE CAST($17 AS TIMESTAMP) END)
                   ON CONFLICT (episode_id) DO NOTHING"#,
                params![
                    ep.episode_id,
                    ep.title,
                    ep.summary,
                    ep.started_at,
                    ended_val,
                    ep.significance as f64,
                    outcome_val,
                    source_val,
                    event_ids_str,
                    ep.user_id,
                    ep.created_at,
                    last_recalled_val,
                    ep.recall_count as i32,
                    ep.storage_strength as f64,
                    ep.retrieval_strength as f64,
                    session_ids_str,
                    last_meditated_val
                ],
            )?;
            count += affected as u64;
        }
        Ok(count)
    }

    /// Import entities from export records. Returns the number of successfully imported entities.
    #[allow(dead_code)]
    pub(crate) fn import_entities(&self, entities: &[Entity]) -> Result<u64> {
        let collection = &self.config.collection_name;
        let mut count: u64 = 0;
        let conn = self.write_conn();
        for ent in entities {
            let type_val: duckdb::types::Value = match &ent.entity_type {
                Some(t) => duckdb::types::Value::Text(t.clone()),
                None => duckdb::types::Value::Null,
            };
            let sql = format!(
                r#"INSERT INTO entities_{collection} (id, name, entity_type, user_id)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (id) DO NOTHING"#
            );
            let affected = conn.execute(&sql, params![ent.id, ent.name, type_val, ent.user_id])?;
            count += affected as u64;
        }
        Ok(count)
    }

    /// Import relations from export records. Returns the number of successfully imported relations.
    #[allow(dead_code)]
    pub(crate) fn import_relations(&self, relations: &[GraphRelation]) -> Result<u64> {
        let collection = &self.config.collection_name;
        let mut count: u64 = 0;
        let conn = self.write_conn();
        for rel in relations {
            let sql = format!(
                r#"INSERT INTO relationships_{collection} (id, source_id, target_id, relation_type, user_id)
                   VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (id) DO NOTHING"#
            );
            let affected = conn.execute(
                &sql,
                params![rel.id, rel.source_id, rel.target_id, rel.relation_type, rel.user_id],
            )?;
            count += affected as u64;
        }
        Ok(count)
    }

    /// Import identity traits from export records. Returns the number of successfully imported traits.
    /// Embeddings (content_vec) are set to NULL since they are not exported.
    #[allow(dead_code)]
    pub(crate) fn import_identity_traits(&self, traits: &[IdentityTrait]) -> Result<u64> {
        let mut count: u64 = 0;
        let conn = self.write_conn();
        for t in traits {
            let evidence_str = serde_json::to_string(&t.evidence_ids).unwrap_or_default();
            let updated_val: duckdb::types::Value = match &t.updated_at {
                Some(u) => duckdb::types::Value::Text(u.clone()),
                None => duckdb::types::Value::Null,
            };
            let affected = conn.execute(
                r#"INSERT INTO identity (trait_id, trait_type, content, confidence, evidence_ids, user_id, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, CAST($7 AS TIMESTAMP),
                           CASE WHEN $8 IS NULL THEN NULL ELSE CAST($8 AS TIMESTAMP) END)
                   ON CONFLICT (trait_id) DO NOTHING"#,
                params![
                    t.trait_id,
                    t.trait_type.as_str(),
                    t.content,
                    t.confidence as f64,
                    evidence_str,
                    t.user_id,
                    t.created_at,
                    updated_val
                ],
            )?;
            count += affected as u64;
        }
        Ok(count)
    }

    /// Import memories from export records. Returns the number of successfully imported memories.
    #[allow(dead_code)]
    pub(crate) fn import_memories(&self, memories: &[MemoryExport]) -> Result<u64> {
        let mut count: u64 = 0;
        let conn = self.write_conn();
        for mem in memories {
            let meta_str = mem
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default());
            let meta_val: duckdb::types::Value = match &meta_str {
                Some(s) => duckdb::types::Value::Text(s.clone()),
                None => duckdb::types::Value::Null,
            };
            let agent_val: duckdb::types::Value = match &mem.agent_id {
                Some(a) => duckdb::types::Value::Text(a.clone()),
                None => duckdb::types::Value::Null,
            };
            let run_val: duckdb::types::Value = match &mem.run_id {
                Some(r) => duckdb::types::Value::Text(r.clone()),
                None => duckdb::types::Value::Null,
            };
            let app_val: duckdb::types::Value = match &mem.app_id {
                Some(a) => duckdb::types::Value::Text(a.clone()),
                None => duckdb::types::Value::Null,
            };
            let exp_val: duckdb::types::Value = match &mem.expiration_date {
                Some(d) => duckdb::types::Value::Text(d.clone()),
                None => duckdb::types::Value::Null,
            };
            let cats_literal = Self::format_categories(mem.categories.as_deref())?;

            // Use ON CONFLICT to skip duplicates
            let sql = format!(
                r#"INSERT INTO memories (id, content, user_id, agent_id, run_id, app_id, metadata, importance, immutable, expiration_date, categories, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
                           CASE WHEN $10 IS NULL THEN NULL ELSE CAST($10 AS TIMESTAMP) END,
                           {cats_literal},
                           CAST($11 AS TIMESTAMP), CAST($12 AS TIMESTAMP))
                   ON CONFLICT (id) DO NOTHING"#
            );
            let affected = conn.execute(
                &sql,
                params![
                    mem.id,
                    mem.content,
                    mem.user_id,
                    agent_val,
                    run_val,
                    app_val,
                    meta_val,
                    mem.importance as f64,
                    mem.immutable,
                    exp_val,
                    mem.created_at,
                    mem.updated_at
                ],
            )?;
            count += affected as u64;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::MemoryConfig;
    use crate::storage::InsertMemoryParams;

    use super::Storage;

    fn test_config(dims: usize) -> MemoryConfig {
        MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: dims,
            dedup_threshold: 0.15,
            default_limit: 10,
            ..Default::default()
        }
    }

    fn open_storage(dims: usize) -> Storage {
        Storage::open(test_config(dims)).unwrap()
    }

    fn dummy_embedding(dims: usize, seed: f32) -> Vec<f32> {
        (0..dims).map(|i| (i as f32 * 0.01 + seed).sin()).collect()
    }

    #[test]
    fn test_export_import() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);

        storage
            .insert_memory(
                "id1",
                "hello",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams {
                    agent_id: Some("agent1".into()),
                    app_id: Some("app1".into()),
                    metadata: Some(r#"{"k":"v"}"#.into()),
                    importance: Some(0.8),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "world",
                &emb,
                "user1",
                "h2",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // Export
        let exports = storage.export_memories(Some("user1")).unwrap();
        assert_eq!(exports.len(), 2);
        assert_eq!(exports[0].id, "id1");
        assert_eq!(exports[0].app_id.as_deref(), Some("app1"));

        // Import into a new storage
        let storage2 = open_storage(384);
        let imported = storage2.import_memories(&exports).unwrap();
        assert_eq!(imported, 2);

        // Verify imported data
        let row = storage2.get_memory("id1").unwrap().unwrap();
        assert_eq!(row.content, "hello");
        assert_eq!(row.app_id.as_deref(), Some("app1"));
    }

    #[test]
    fn test_cleanup_expired() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);

        storage
            .insert_memory(
                "id1",
                "expired",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams {
                    expiration_date: Some("2020-01-01T00:00:00".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "future",
                &emb,
                "user1",
                "h2",
                &InsertMemoryParams {
                    expiration_date: Some("2099-01-01T00:00:00".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id3",
                "no_exp",
                &emb,
                "user1",
                "h3",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let expired_count = storage.cleanup_expired().unwrap();
        assert_eq!(expired_count, 1);

        assert!(storage.get_memory("id1").unwrap().is_none());
        assert!(storage.get_memory("id2").unwrap().is_some());
        assert!(storage.get_memory("id3").unwrap().is_some());
    }
}
