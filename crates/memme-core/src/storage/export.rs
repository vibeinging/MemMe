use crate::error::Result;
use crate::types::{
    Entity, Episode, Event, EventType, FullExport, GraphRelation, IdentityTrait, MemoryExport,
    Session, Source, SqlParam, TraitType,
};

use super::util::opt_text;
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
        let (sql, params) = if let Some(uid) = user_id {
            (
                format!(
                    "SELECT id, content, user_id, agent_id, app_id, run_id,
                        metadata,
                        importance, immutable,
                        expiration_date,
                        categories,
                        created_at,
                        updated_at,
                        stability
                 FROM memories WHERE user_id = $1{privacy_filter}
                 ORDER BY created_at"
                ),
                vec![SqlParam::Text(uid.to_string())],
            )
        } else {
            (
                format!(
                    "SELECT id, content, user_id, agent_id, app_id, run_id,
                        metadata,
                        importance, immutable,
                        expiration_date,
                        categories,
                        created_at,
                        updated_at,
                        stability
                 FROM memories WHERE 1=1{privacy_filter}
                 ORDER BY created_at"
                ),
                vec![],
            )
        };

        self.backend.query_read(&sql, &params, |row| {
            let metadata = row
                .get_opt_string(6)?
                .and_then(|s| serde_json::from_str(&s).ok());
            let categories_raw: Option<String> = row.get_opt_string(10)?;
            Ok(MemoryExport {
                id: row.get_string(0)?,
                content: row.get_string(1)?,
                user_id: row.get_string(2)?,
                agent_id: row.get_opt_string(3)?,
                app_id: row.get_opt_string(4)?,
                run_id: row.get_opt_string(5)?,
                metadata,
                importance: row.get_opt_f64(7)?.unwrap_or(0.5) as f32,
                immutable: row.get_opt_bool(8)?.unwrap_or(false),
                expiration_date: row.get_opt_string(9)?,
                categories: Self::parse_categories(categories_raw),
                created_at: row.get_string(11)?,
                updated_at: row.get_string(12)?,
                stability: row.get_opt_f64(13)?.map(|v| v as f32),
            })
        })
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
        let sql = r#"SELECT s.session_id, s.user_id, s.source_id,
                      s.started_at, s.ended_at,
                      s.metadata, s.created_at,
                      (SELECT COUNT(*) FROM events WHERE session_id = s.session_id) AS event_count,
                      s.structured_notes
               FROM sessions s
               ORDER BY s.started_at DESC"#;
        self.backend.query_read(
            sql,
            &[],
            |row| {
                Ok(Session {
                    session_id: row.get_string(0)?,
                    user_id: row.get_string(1)?,
                    source_id: row.get_opt_string(2)?,
                    started_at: row.get_string(3)?,
                    ended_at: row.get_opt_string(4)?,
                    metadata: row
                        .get_opt_string(5)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    created_at: row.get_string(6)?,
                    event_count: row.get_opt_i64(7)?.unwrap_or(0) as u32,
                    structured_notes: row.get_opt_string(8)?,
                })
            },
        )
    }

    /// List all events without user_id filter.
    fn list_all_events(&self) -> Result<Vec<Event>> {
        let sql = r#"SELECT event_id, source_id, session_id, timestamp,
                      event_type, content, parent_id, metadata, user_id,
                      processed, processed_at,
                      purified_content, purified, event_time, location
               FROM events ORDER BY timestamp DESC"#;
        self.backend.query_read(
            sql,
            &[],
            |row| {
                Ok(Event {
                    event_id: row.get_string(0)?,
                    source_id: row.get_opt_string(1)?,
                    session_id: row.get_opt_string(2)?,
                    timestamp: row.get_string(3)?,
                    event_type: EventType::parse(
                        &row.get_opt_string(4)?.unwrap_or_default(),
                    ),
                    content: row.get_string(5)?,
                    parent_id: row.get_opt_string(6)?,
                    metadata: row
                        .get_opt_string(7)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    user_id: row.get_string(8)?,
                    processed: row.get_opt_bool(9)?.unwrap_or(false),
                    processed_at: row.get_opt_string(10)?,
                    purified_content: row.get_opt_string(11)?,
                    purified: row.get_opt_bool(12)?.unwrap_or(false),
                    event_time: row.get_opt_string(13)?,
                    location: row.get_opt_string(14)?,
                })
            },
        )
    }

    /// List all episodes without user_id filter.
    fn list_all_episodes(&self) -> Result<Vec<Episode>> {
        let sql = r#"SELECT episode_id, title, summary,
                      started_at, ended_at,
                      significance, outcome, source_id, event_ids, user_id,
                      created_at, last_recalled,
                      recall_count, storage_strength, retrieval_strength,
                      session_ids, last_meditated_at
               FROM episodes ORDER BY started_at DESC"#;
        self.backend.query_read(
            sql,
            &[],
            |row| {
                let event_ids_raw: Option<String> = row.get_opt_string(8)?;
                let event_ids: Vec<String> = event_ids_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                let session_ids_raw: Option<String> = row.get_opt_string(15)?;
                let session_ids: Vec<String> = session_ids_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                Ok(Episode {
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
            },
        )
    }

    /// List all identity traits without user_id filter.
    fn list_all_identity_traits(&self) -> Result<Vec<IdentityTrait>> {
        let sql =
            r#"SELECT trait_id, trait_type, content, confidence, evidence_ids, user_id,
                      created_at, updated_at
               FROM identity ORDER BY confidence DESC"#;
        self.backend.query_read(
            sql,
            &[],
            |row| {
                let evidence_raw: Option<String> = row.get_opt_string(4)?;
                let evidence_ids: Vec<String> = evidence_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                Ok(IdentityTrait {
                    trait_id: row.get_string(0)?,
                    trait_type: TraitType::parse(&row.get_string(1).unwrap_or_default()),
                    content: row.get_string(2)?,
                    confidence: row.get_opt_f64(3)?.unwrap_or(0.5) as f32,
                    evidence_ids,
                    user_id: row.get_string(5)?,
                    created_at: row.get_string(6)?,
                    updated_at: row.get_opt_string(7)?,
                })
            },
        )
    }

    /// List all sources without user_id filter.
    fn list_all_sources(&self) -> Result<Vec<Source>> {
        let sql =
            "SELECT source_id, source_type, name, registered_at, metadata FROM sources ORDER BY registered_at DESC";
        self.backend.query_read(
            sql,
            &[],
            |row| {
                Ok(Source {
                    source_id: row.get_string(0)?,
                    source_type: row.get_string(1)?,
                    name: row.get_opt_string(2)?,
                    registered_at: row.get_string(3)?,
                    metadata: row
                        .get_opt_string(4)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                })
            },
        )
    }

    // ── Export entities and relations as typed structs ──

    /// Export entities as Entity structs, optionally filtered by user_id.
    fn export_entities(&self, user_id: Option<&str>) -> Result<Vec<Entity>> {
        let collection = &self.config.collection_name;
        let (sql, params) = if let Some(uid) = user_id {
            (
                format!(
                    "SELECT id, name, entity_type, user_id FROM entities_{collection} WHERE user_id = $1 ORDER BY created_at"
                ),
                vec![SqlParam::Text(uid.to_string())],
            )
        } else {
            (
                format!(
                    "SELECT id, name, entity_type, user_id FROM entities_{collection} ORDER BY created_at"
                ),
                vec![],
            )
        };

        self.backend.query_read(&sql, &params, |row| {
            Ok(Entity {
                id: row.get_string(0)?,
                name: row.get_string(1)?,
                entity_type: row.get_opt_string(2)?,
                user_id: row.get_string(3)?,
            })
        })
    }

    /// Export relationships as GraphRelation structs, optionally filtered by user_id.
    fn export_relations(&self, user_id: Option<&str>) -> Result<Vec<GraphRelation>> {
        let collection = &self.config.collection_name;
        let (sql, params) = if let Some(uid) = user_id {
            (
                format!(
                    r#"SELECT r.id, r.source_id, r.target_id, r.relation_type, r.user_id,
                              s.name AS source_name, t.name AS target_name, r.description
                       FROM relationships_{collection} r
                       JOIN entities_{collection} s ON r.source_id = s.id
                       JOIN entities_{collection} t ON r.target_id = t.id
                       WHERE r.user_id = $1"#
                ),
                vec![SqlParam::Text(uid.to_string())],
            )
        } else {
            (
                format!(
                    r#"SELECT r.id, r.source_id, r.target_id, r.relation_type, r.user_id,
                              s.name AS source_name, t.name AS target_name, r.description
                       FROM relationships_{collection} r
                       JOIN entities_{collection} s ON r.source_id = s.id
                       JOIN entities_{collection} t ON r.target_id = t.id"#
                ),
                vec![],
            )
        };

        self.backend.query_read(&sql, &params, |row| {
            Ok(GraphRelation {
                id: row.get_string(0)?,
                source_id: row.get_string(1)?,
                target_id: row.get_string(2)?,
                relation_type: row.get_string(3)?,
                user_id: row.get_string(4)?,
                source: row.get_string(5)?,
                target: row.get_string(6)?,
                description: row.get_opt_string(7)?,
            })
        })
    }

    // ── Bulk import methods ──

    /// Import sources from export records. Returns the number of successfully imported sources.
    #[allow(dead_code)]
    pub(crate) fn import_sources(&self, sources: &[Source]) -> Result<u64> {
        self.backend.execute_batch("BEGIN")?;
        let mut count: u64 = 0;
        for src in sources {
            let name_val = opt_text(src.name.as_deref());
            let meta_val = opt_text(
                src.metadata
                    .as_ref()
                    .map(|m| serde_json::to_string(m).unwrap_or_default()),
            );
            let sql = r#"INSERT INTO sources (source_id, source_type, name, registered_at, metadata)
                   VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (source_id) DO NOTHING"#;
            let affected = self.backend.execute(
                sql,
                &[
                    SqlParam::Text(src.source_id.clone()),
                    SqlParam::Text(src.source_type.clone()),
                    name_val,
                    SqlParam::Text(src.registered_at.clone()),
                    meta_val,
                ],
            )?;
            count += affected as u64;
        }
        self.backend.execute_batch("COMMIT")?;
        Ok(count)
    }

    /// Import sessions from export records. Returns the number of successfully imported sessions.
    #[allow(dead_code)]
    pub(crate) fn import_sessions(&self, sessions: &[Session]) -> Result<u64> {
        self.backend.execute_batch("BEGIN")?;
        let mut count: u64 = 0;
        for sess in sessions {
            let source_val = opt_text(sess.source_id.as_deref());
            let ended_val = opt_text(sess.ended_at.as_deref());
            let meta_val = opt_text(
                sess.metadata
                    .as_ref()
                    .map(|m| serde_json::to_string(m).unwrap_or_default()),
            );
            let notes_val = opt_text(sess.structured_notes.as_deref());
            let sql = r#"INSERT INTO sessions (session_id, user_id, source_id, started_at, ended_at, metadata, created_at, structured_notes)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT DO NOTHING"#;
            let affected = self.backend.execute(
                sql,
                &[
                    SqlParam::Text(sess.session_id.clone()),
                    SqlParam::Text(sess.user_id.clone()),
                    source_val,
                    SqlParam::Text(sess.started_at.clone()),
                    ended_val,
                    meta_val,
                    SqlParam::Text(sess.created_at.clone()),
                    notes_val,
                ],
            )?;
            count += affected as u64;
        }
        self.backend.execute_batch("COMMIT")?;
        Ok(count)
    }

    /// Import events from export records. Returns the number of successfully imported events.
    /// Embeddings (content_vec) are set to NULL since they are not exported.
    #[allow(dead_code)]
    pub(crate) fn import_events(&self, events: &[Event]) -> Result<u64> {
        self.backend.execute_batch("BEGIN")?;
        let mut count: u64 = 0;
        for evt in events {
            let source_val = opt_text(evt.source_id.as_deref());
            let session_val = opt_text(evt.session_id.as_deref());
            let parent_val = opt_text(evt.parent_id.as_deref());
            let meta_val = opt_text(
                evt.metadata
                    .as_ref()
                    .map(|m| serde_json::to_string(m).unwrap_or_default()),
            );
            let purified_val = opt_text(evt.purified_content.as_deref());
            let event_time_val = opt_text(evt.event_time.as_deref());
            let location_val = opt_text(evt.location.as_deref());
            let sql = r#"INSERT INTO events (event_id, source_id, session_id, timestamp, event_type,
                                       content, parent_id, metadata, user_id,
                                       processed, purified_content, purified,
                                       event_time, location)
                   VALUES ($1, $2, $3, $4, $5,
                           $6, $7, $8, $9,
                           $10, $11, $12,
                           $13,
                           $14)
                   ON CONFLICT (event_id) DO NOTHING"#;
            let affected = self.backend.execute(
                sql,
                &[
                    SqlParam::Text(evt.event_id.clone()),
                    source_val,
                    session_val,
                    SqlParam::Text(evt.timestamp.clone()),
                    SqlParam::Text(evt.event_type.as_str().to_string()),
                    SqlParam::Text(evt.content.clone()),
                    parent_val,
                    meta_val,
                    SqlParam::Text(evt.user_id.clone()),
                    SqlParam::Bool(evt.processed),
                    purified_val,
                    SqlParam::Bool(evt.purified),
                    event_time_val,
                    location_val,
                ],
            )?;
            count += affected as u64;
        }
        self.backend.execute_batch("COMMIT")?;
        Ok(count)
    }

    /// Import episodes from export records. Returns the number of successfully imported episodes.
    /// Embeddings (summary_vec) are set to NULL since they are not exported.
    #[allow(dead_code)]
    pub(crate) fn import_episodes(&self, episodes: &[Episode]) -> Result<u64> {
        self.backend.execute_batch("BEGIN")?;
        let mut count: u64 = 0;
        for ep in episodes {
            let ended_val = opt_text(ep.ended_at.as_deref());
            let outcome_val = opt_text(ep.outcome.as_deref());
            let source_val = opt_text(ep.source_id.as_deref());
            let event_ids_str = serde_json::to_string(&ep.event_ids).unwrap_or_default();
            let session_ids_str = serde_json::to_string(&ep.session_ids).unwrap_or_default();
            let last_recalled_val = opt_text(ep.last_recalled.as_deref());
            let last_meditated_val = opt_text(ep.last_meditated_at.as_deref());
            let sql = r#"INSERT INTO episodes (episode_id, title, summary, started_at, ended_at,
                                         significance, outcome, source_id, event_ids, user_id,
                                         created_at, last_recalled, recall_count,
                                         storage_strength, retrieval_strength,
                                         session_ids, last_meditated_at)
                   VALUES ($1, $2, $3, $4, $5,
                           $6, $7, $8, $9, $10,
                           $11, $12,
                           $13, $14, $15, $16, $17)
                   ON CONFLICT (episode_id) DO NOTHING"#;
            let affected = self.backend.execute(
                sql,
                &[
                    SqlParam::Text(ep.episode_id.clone()),
                    SqlParam::Text(ep.title.clone()),
                    SqlParam::Text(ep.summary.clone()),
                    SqlParam::Text(ep.started_at.clone()),
                    ended_val,
                    SqlParam::Float(ep.significance as f64),
                    outcome_val,
                    source_val,
                    SqlParam::Text(event_ids_str),
                    SqlParam::Text(ep.user_id.clone()),
                    SqlParam::Text(ep.created_at.clone()),
                    last_recalled_val,
                    SqlParam::Int(ep.recall_count as i64),
                    SqlParam::Float(ep.storage_strength as f64),
                    SqlParam::Float(ep.retrieval_strength as f64),
                    SqlParam::Text(session_ids_str),
                    last_meditated_val,
                ],
            )?;
            count += affected as u64;
        }
        self.backend.execute_batch("COMMIT")?;
        Ok(count)
    }

    /// Import entities from export records. Returns the number of successfully imported entities.
    #[allow(dead_code)]
    pub(crate) fn import_entities(&self, entities: &[Entity]) -> Result<u64> {
        let collection = &self.config.collection_name;
        self.backend.execute_batch("BEGIN")?;
        let mut count: u64 = 0;
        for ent in entities {
            let type_val = opt_text(ent.entity_type.as_deref());
            let sql = format!(
                r#"INSERT INTO entities_{collection} (id, name, entity_type, user_id)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (id) DO NOTHING"#
            );
            let affected = self.backend.execute(
                &sql,
                &[
                    SqlParam::Text(ent.id.clone()),
                    SqlParam::Text(ent.name.clone()),
                    type_val,
                    SqlParam::Text(ent.user_id.clone()),
                ],
            )?;
            count += affected as u64;
        }
        self.backend.execute_batch("COMMIT")?;
        Ok(count)
    }

    /// Import relations from export records. Returns the number of successfully imported relations.
    #[allow(dead_code)]
    pub(crate) fn import_relations(&self, relations: &[GraphRelation]) -> Result<u64> {
        let collection = &self.config.collection_name;
        self.backend.execute_batch("BEGIN")?;
        let mut count: u64 = 0;
        for rel in relations {
            let desc_val = opt_text(rel.description.as_deref());
            let sql = format!(
                r#"INSERT INTO relationships_{collection} (id, source_id, target_id, relation_type, user_id, description)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (id) DO NOTHING"#
            );
            let affected = self.backend.execute(
                &sql,
                &[
                    SqlParam::Text(rel.id.clone()),
                    SqlParam::Text(rel.source_id.clone()),
                    SqlParam::Text(rel.target_id.clone()),
                    SqlParam::Text(rel.relation_type.clone()),
                    SqlParam::Text(rel.user_id.clone()),
                    desc_val,
                ],
            )?;
            count += affected as u64;
        }
        self.backend.execute_batch("COMMIT")?;
        Ok(count)
    }

    /// Import identity traits from export records. Returns the number of successfully imported traits.
    /// Embeddings (content_vec) are set to NULL since they are not exported.
    #[allow(dead_code)]
    pub(crate) fn import_identity_traits(&self, traits: &[IdentityTrait]) -> Result<u64> {
        self.backend.execute_batch("BEGIN")?;
        let mut count: u64 = 0;
        for t in traits {
            let evidence_str = serde_json::to_string(&t.evidence_ids).unwrap_or_default();
            let updated_val = opt_text(t.updated_at.as_deref());
            let sql = r#"INSERT INTO identity (trait_id, trait_type, content, confidence, evidence_ids, user_id, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT (trait_id) DO NOTHING"#;
            let affected = self.backend.execute(
                sql,
                &[
                    SqlParam::Text(t.trait_id.clone()),
                    SqlParam::Text(t.trait_type.as_str().to_string()),
                    SqlParam::Text(t.content.clone()),
                    SqlParam::Float(t.confidence as f64),
                    SqlParam::Text(evidence_str),
                    SqlParam::Text(t.user_id.clone()),
                    SqlParam::Text(t.created_at.clone()),
                    updated_val,
                ],
            )?;
            count += affected as u64;
        }
        self.backend.execute_batch("COMMIT")?;
        Ok(count)
    }

    /// Import memories from export records. Returns the number of successfully imported memories.
    #[allow(dead_code)]
    pub(crate) fn import_memories(&self, memories: &[MemoryExport]) -> Result<u64> {
        self.backend.execute_batch("BEGIN")?;
        let mut count: u64 = 0;
        for mem in memories {
            let meta_val = opt_text(
                mem.metadata
                    .as_ref()
                    .map(|m| serde_json::to_string(m).unwrap_or_default()),
            );
            let agent_val = opt_text(mem.agent_id.as_deref());
            let run_val = opt_text(mem.run_id.as_deref());
            let app_val = opt_text(mem.app_id.as_deref());
            let exp_val = opt_text(mem.expiration_date.as_deref());
            let cats_literal = self.format_categories(mem.categories.as_deref())?;

            // Use ON CONFLICT to skip duplicates
            let sql = format!(
                r#"INSERT INTO memories (id, content, user_id, agent_id, run_id, app_id, metadata, importance, immutable, expiration_date, categories, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
                           $10,
                           {cats_literal},
                           $11, $12)
                   ON CONFLICT (id) DO NOTHING"#
            );
            let affected = self.backend.execute(
                &sql,
                &[
                    SqlParam::Text(mem.id.clone()),
                    SqlParam::Text(mem.content.clone()),
                    SqlParam::Text(mem.user_id.clone()),
                    agent_val,
                    run_val,
                    app_val,
                    meta_val,
                    SqlParam::Float(mem.importance as f64),
                    SqlParam::Bool(mem.immutable),
                    exp_val,
                    SqlParam::Text(mem.created_at.clone()),
                    SqlParam::Text(mem.updated_at.clone()),
                ],
            )?;
            count += affected as u64;
        }
        self.backend.execute_batch("COMMIT")?;
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
