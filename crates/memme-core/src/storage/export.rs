#[cfg(test)]
use crate::error::MemoryError;
use crate::error::Result;
use crate::types::{
    AssociationExport, Entity, Episode, Event, EventType, FullExport, GraphRelation, HistoryExport,
    IdentityTrait, MeditationRecord, MeditationStatus, MemoryEntityExport, MemoryExport,
    RecallExport, Session, Source, SqlParam, TraitType,
};

use super::backend::{Backend, BackendTransaction, RowAccess};
#[cfg(test)]
use super::util::opt_text;
use super::Storage;

trait ExportReader {
    fn query_read<T, F>(&self, sql: &str, params: &[SqlParam], mapper: F) -> Result<Vec<T>>
    where
        F: FnMut(&dyn RowAccess) -> Result<T>;
}

impl ExportReader for Backend {
    fn query_read<T, F>(&self, sql: &str, params: &[SqlParam], mapper: F) -> Result<Vec<T>>
    where
        F: FnMut(&dyn RowAccess) -> Result<T>,
    {
        Backend::query_read(self, sql, params, mapper)
    }
}

impl ExportReader for BackendTransaction<'_> {
    fn query_read<T, F>(&self, sql: &str, params: &[SqlParam], mapper: F) -> Result<Vec<T>>
    where
        F: FnMut(&dyn RowAccess) -> Result<T>,
    {
        BackendTransaction::query_read(self, sql, params, mapper)
    }
}

fn user_filter(column: &str, user_id: Option<&str>) -> (String, Vec<SqlParam>) {
    match user_id {
        Some(user_id) => (
            format!("WHERE {column} = $1"),
            vec![SqlParam::Text(user_id.to_string())],
        ),
        None => (String::new(), Vec::new()),
    }
}

fn parse_optional_json<T: serde::de::DeserializeOwned>(raw: Option<String>) -> Result<Option<T>> {
    raw.map(|value| serde_json::from_str(&value))
        .transpose()
        .map_err(Into::into)
}

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
        self.export_memories_with_privacy_on(&self.backend, user_id, include_local)
    }

    fn export_memories_with_privacy_on<R: ExportReader>(
        &self,
        reader: &R,
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
                        stability,
                        privacy, actor_id, access_count, memory_type, event_time,
                        episode_id, session_id, resolution, storage_strength,
                        retrieval_strength, superseded_by, valid_from, valid_until,
                        confidence, evidence, episode_ids, ingestion_time,
                        sync_version, device_id, sync_status, pinned
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
                        stability,
                        privacy, actor_id, access_count, memory_type, event_time,
                        episode_id, session_id, resolution, storage_strength,
                        retrieval_strength, superseded_by, valid_from, valid_until,
                        confidence, evidence, episode_ids, ingestion_time,
                        sync_version, device_id, sync_status, pinned
                 FROM memories WHERE 1=1{privacy_filter}
                 ORDER BY created_at"
                ),
                vec![],
            )
        };

        reader.query_read(&sql, &params, |row| {
            let metadata = parse_optional_json(row.get_opt_string(6)?)?;
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
                privacy: row.get_opt_string(14)?,
                actor_id: row.get_opt_string(15)?,
                access_count: row.get_opt_i64(16)?.map(|value| value as u32),
                memory_type: row.get_opt_string(17)?,
                event_time: row.get_opt_string(18)?,
                episode_id: row.get_opt_string(19)?,
                session_id: row.get_opt_string(20)?,
                resolution: row.get_opt_string(21)?,
                storage_strength: row.get_opt_f64(22)?.map(|value| value as f32),
                retrieval_strength: row.get_opt_f64(23)?.map(|value| value as f32),
                superseded_by: row.get_opt_string(24)?,
                valid_from: row.get_opt_string(25)?,
                valid_until: row.get_opt_string(26)?,
                confidence: row.get_opt_f64(27)?.map(|value| value as f32),
                evidence: parse_optional_json(row.get_opt_string(28)?)?,
                episode_ids: parse_optional_json(row.get_opt_string(29)?)?,
                ingestion_time: row.get_opt_string(30)?,
                sync_version: row.get_opt_i64(31)?.map(|value| value as u64),
                device_id: row.get_opt_string(32)?,
                sync_status: row.get_opt_string(33)?,
                pinned: row.get_opt_bool(34)?,
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
        self.backend.read_transaction(|reader| {
            // A full export is a portability/backup operation and must include
            // local-only records. Every layer is read from this same snapshot.
            let memories = self.export_memories_with_privacy_on(reader, user_id, true)?;
            let sessions = self.list_all_sessions_on(reader, user_id)?;
            let events = self.list_all_events_on(reader, user_id)?;
            let episodes = self.list_all_episodes_on(reader, user_id)?;
            let entities = self.export_entities_on(reader, user_id)?;
            let relations = self.export_relations_on(reader, user_id)?;
            let identity_traits = self.list_all_identity_traits_on(reader, user_id)?;
            let sources = self.list_all_sources_on(reader, user_id)?;
            let history = self.export_history_on(reader, user_id)?;
            let procedures = self.export_procedures_on(reader, user_id)?;
            let meditations = self.export_meditations_on(reader, user_id)?;
            let recalls = self.export_recalls_on(reader, user_id)?;
            let memory_entities = self.export_memory_entities_on(reader, user_id)?;
            let associations = self.export_associations_on(reader, user_id, collection)?;

            Ok(FullExport {
                version: super::portable_import::FULL_EXPORT_VERSION.into(),
                collection: collection.to_string(),
                exported_at: chrono::Utc::now().to_rfc3339(),
                user_id: user_id.map(str::to_string),
                memories,
                entities,
                relations,
                sessions,
                episodes,
                events,
                identity_traits,
                sources,
                history,
                procedures,
                meditations,
                recalls,
                memory_entities,
                associations,
            })
        })
    }

    // ── List-all methods (no user_id filter) ──

    /// List every session without applying API pagination defaults.
    fn list_all_sessions_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<Session>> {
        let (filter, params) = user_filter("s.user_id", user_id);
        let sql = format!(
            r#"SELECT s.session_id, s.user_id, s.source_id,
                      s.started_at, s.ended_at,
                      s.metadata, s.created_at,
                      (SELECT COUNT(*) FROM events WHERE session_id = s.session_id) AS event_count,
                      s.structured_notes, s.queried_count, s.last_queried_at
               FROM sessions s
               {filter}
               ORDER BY s.started_at DESC"#
        );
        reader.query_read(&sql, &params, |row| {
            Ok(Session {
                session_id: row.get_string(0)?,
                user_id: row.get_string(1)?,
                source_id: row.get_opt_string(2)?,
                started_at: row.get_string(3)?,
                ended_at: row.get_opt_string(4)?,
                metadata: parse_optional_json(row.get_opt_string(5)?)?,
                created_at: row.get_string(6)?,
                event_count: row.get_opt_i64(7)?.unwrap_or(0) as u32,
                structured_notes: row.get_opt_string(8)?,
                queried_count: row.get_opt_i64(9)?.unwrap_or(0) as u32,
                last_queried_at: row.get_opt_string(10)?,
            })
        })
    }

    /// List every event without applying API pagination defaults.
    fn list_all_events_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<Event>> {
        let (filter, params) = user_filter("user_id", user_id);
        let sql = format!(
            r#"SELECT event_id, source_id, session_id, timestamp,
                      event_type, content, parent_id, metadata, user_id,
                      processed, processed_at,
                      purified_content, purified, event_time, location,
                      agent_id, app_id, run_id
               FROM events {filter} ORDER BY timestamp DESC"#
        );
        reader.query_read(&sql, &params, |row| {
            Ok(Event {
                event_id: row.get_string(0)?,
                source_id: row.get_opt_string(1)?,
                session_id: row.get_opt_string(2)?,
                timestamp: row.get_string(3)?,
                event_type: EventType::parse(&row.get_opt_string(4)?.unwrap_or_default()),
                content: row.get_string(5)?,
                parent_id: row.get_opt_string(6)?,
                metadata: parse_optional_json(row.get_opt_string(7)?)?,
                user_id: row.get_string(8)?,
                agent_id: row.get_opt_string(15)?,
                app_id: row.get_opt_string(16)?,
                run_id: row.get_opt_string(17)?,
                processed: row.get_opt_bool(9)?.unwrap_or(false),
                processed_at: row.get_opt_string(10)?,
                purified_content: row.get_opt_string(11)?,
                purified: row.get_opt_bool(12)?.unwrap_or(false),
                event_time: row.get_opt_string(13)?,
                location: row.get_opt_string(14)?,
            })
        })
    }

    /// List every episode without applying API pagination defaults.
    fn list_all_episodes_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<Episode>> {
        let (filter, params) = user_filter("user_id", user_id);
        let sql = format!(
            r#"SELECT episode_id, title, summary,
                      started_at, ended_at,
                      significance, outcome, source_id, event_ids, user_id,
                      created_at, last_recalled,
                      recall_count, storage_strength, retrieval_strength,
                      session_ids, last_meditated_at
               FROM episodes {filter} ORDER BY started_at DESC"#
        );
        reader.query_read(&sql, &params, |row| {
            let event_ids = parse_optional_json(row.get_opt_string(8)?)?.unwrap_or_default();
            let session_ids = parse_optional_json(row.get_opt_string(15)?)?.unwrap_or_default();
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
        })
    }

    /// List every identity trait without applying API pagination defaults.
    fn list_all_identity_traits_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<IdentityTrait>> {
        let (filter, params) = user_filter("user_id", user_id);
        let sql = format!(
            r#"SELECT trait_id, trait_type, content, confidence, evidence_ids, user_id,
                      created_at, updated_at
               FROM identity {filter} ORDER BY confidence DESC"#
        );
        reader.query_read(&sql, &params, |row| {
            let evidence_ids = parse_optional_json(row.get_opt_string(4)?)?.unwrap_or_default();
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
        })
    }

    /// List every source without applying API pagination defaults.
    fn list_all_sources_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<Source>> {
        let (filter, params) = match user_id {
            Some(user_id) => (
                "WHERE user_id = $1 OR (user_id IS NULL AND (\
                     EXISTS (SELECT 1 FROM sessions WHERE sessions.source_id = sources.source_id AND sessions.user_id = $1) OR \
                     EXISTS (SELECT 1 FROM events WHERE events.source_id = sources.source_id AND events.user_id = $1) OR \
                     EXISTS (SELECT 1 FROM episodes WHERE episodes.source_id = sources.source_id AND episodes.user_id = $1) OR \
                     EXISTS (SELECT 1 FROM recalls WHERE recalls.source_id = sources.source_id AND recalls.user_id = $1)))"
                    .to_string(),
                vec![SqlParam::Text(user_id.to_string())],
            ),
            None => (String::new(), Vec::new()),
        };
        let sql = format!(
            "SELECT source_id, source_type, name, registered_at, metadata, user_id FROM sources {filter} ORDER BY registered_at DESC"
        );
        reader.query_read(&sql, &params, |row| {
            Ok(Source {
                source_id: row.get_string(0)?,
                source_type: row.get_string(1)?,
                name: row.get_opt_string(2)?,
                registered_at: row.get_string(3)?,
                metadata: parse_optional_json(row.get_opt_string(4)?)?,
                user_id: row.get_opt_string(5)?,
            })
        })
    }

    // ── Export entities and relations as typed structs ──

    /// Export entities as Entity structs, optionally filtered by user_id.
    fn export_entities_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<Entity>> {
        let collection = &self.config.collection_name;
        let (sql, params) = if let Some(uid) = user_id {
            (
                format!(
                    "SELECT id, name, entity_type, user_id, created_at, updated_at FROM entities_{collection} WHERE user_id = $1 ORDER BY created_at"
                ),
                vec![SqlParam::Text(uid.to_string())],
            )
        } else {
            (
                format!(
                    "SELECT id, name, entity_type, user_id, created_at, updated_at FROM entities_{collection} ORDER BY created_at"
                ),
                vec![],
            )
        };

        reader.query_read(&sql, &params, |row| {
            Ok(Entity {
                id: row.get_string(0)?,
                name: row.get_string(1)?,
                entity_type: row.get_opt_string(2)?,
                user_id: row.get_string(3)?,
                created_at: row.get_opt_string(4)?,
                updated_at: row.get_opt_string(5)?,
            })
        })
    }

    /// Export relationships as GraphRelation structs, optionally filtered by user_id.
    fn export_relations_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<GraphRelation>> {
        let collection = &self.config.collection_name;
        let (sql, params) = if let Some(uid) = user_id {
            (
                format!(
                    r#"SELECT r.id, r.source_id, r.target_id, r.relation_type, r.user_id,
                              s.name AS source_name, t.name AS target_name, r.description,
                              r.created_at, r.strength, r.context, r.episode_ids
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
                              s.name AS source_name, t.name AS target_name, r.description,
                              r.created_at, r.strength, r.context, r.episode_ids
                       FROM relationships_{collection} r
                       JOIN entities_{collection} s ON r.source_id = s.id
                       JOIN entities_{collection} t ON r.target_id = t.id"#
                ),
                vec![],
            )
        };

        reader.query_read(&sql, &params, |row| {
            Ok(GraphRelation {
                id: row.get_string(0)?,
                source_id: row.get_string(1)?,
                target_id: row.get_string(2)?,
                relation_type: row.get_string(3)?,
                user_id: row.get_string(4)?,
                source: row.get_string(5)?,
                target: row.get_string(6)?,
                description: row.get_opt_string(7)?,
                created_at: row.get_opt_string(8)?,
                strength: row.get_opt_f64(9)?.map(|value| value as f32),
                context: parse_optional_json(row.get_opt_string(10)?)?,
                episode_ids: parse_optional_json(row.get_opt_string(11)?)?,
            })
        })
    }

    fn export_history_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<HistoryExport>> {
        let (filter, params) = user_filter("user_id", user_id);
        let sql = format!(
            "SELECT id, memory_id, user_id, old_memory, new_memory, event, created_at \
             FROM history {filter} ORDER BY created_at"
        );
        reader.query_read(&sql, &params, |row| {
            Ok(HistoryExport {
                id: row.get_string(0)?,
                memory_id: row.get_opt_string(1)?,
                user_id: row.get_string(2)?,
                old_memory: row.get_opt_string(3)?,
                new_memory: row.get_opt_string(4)?,
                event: row.get_opt_string(5)?,
                created_at: row.get_string(6)?,
            })
        })
    }

    fn export_procedures_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<crate::procedural::Procedure>> {
        let (filter, params) = user_filter("user_id", user_id);
        let sql = format!(
            "SELECT id, name, description, steps, user_id, trigger_pattern, confidence, \
             usage_count, created_at, updated_at FROM procedures {filter} ORDER BY created_at"
        );
        reader.query_read(&sql, &params, |row| {
            let steps = match row.get_opt_string(3)? {
                Some(value) => serde_json::from_str(&value)?,
                None => Vec::new(),
            };
            Ok(crate::procedural::Procedure {
                id: row.get_string(0)?,
                name: row.get_string(1)?,
                description: row.get_opt_string(2)?.unwrap_or_default(),
                steps,
                user_id: row.get_string(4)?,
                trigger: row.get_opt_string(5)?,
                confidence: row.get_opt_f64(6)?.unwrap_or(0.5) as f32,
                usage_count: row.get_opt_i64(7)?.unwrap_or(0) as u32,
                created_at: row.get_string(8)?,
                updated_at: row.get_string(9)?,
            })
        })
    }

    fn export_meditations_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<MeditationRecord>> {
        let (filter, params) = user_filter("user_id", user_id);
        let sql = format!(
            r#"SELECT meditation_id, triggered_by, started_at, finished_at, status, user_id,
                       events_processed, episodes_created, memories_created, memories_updated,
                       memories_decayed, entities_created, relations_created, conflicts_found,
                       journal, metadata
                FROM meditations {filter} ORDER BY started_at"#
        );
        reader.query_read(&sql, &params, |row| {
            Ok(MeditationRecord {
                meditation_id: row.get_string(0)?,
                triggered_by: row.get_string(1)?,
                started_at: row.get_string(2)?,
                finished_at: row.get_opt_string(3)?,
                status: MeditationStatus::parse(&row.get_string(4)?),
                user_id: row.get_string(5)?,
                events_processed: row.get_opt_i64(6)?.unwrap_or(0) as u32,
                episodes_created: row.get_opt_i64(7)?.unwrap_or(0) as u32,
                memories_created: row.get_opt_i64(8)?.unwrap_or(0) as u32,
                memories_updated: row.get_opt_i64(9)?.unwrap_or(0) as u32,
                memories_decayed: row.get_opt_i64(10)?.unwrap_or(0) as u32,
                entities_created: row.get_opt_i64(11)?.unwrap_or(0) as u32,
                relations_created: row.get_opt_i64(12)?.unwrap_or(0) as u32,
                conflicts_found: row.get_opt_i64(13)?.unwrap_or(0) as u32,
                journal: row.get_opt_string(14)?,
                metadata: row
                    .get_opt_string(15)?
                    .map(|value| serde_json::from_str(&value))
                    .transpose()?,
            })
        })
    }

    fn export_recalls_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<RecallExport>> {
        let (filter, params) = user_filter("user_id", user_id);
        let sql = format!(
            "SELECT recall_id, query, timestamp, source_id, user_id, results, feedback \
             FROM recalls {filter} ORDER BY timestamp"
        );
        reader.query_read(&sql, &params, |row| {
            Ok(RecallExport {
                recall_id: row.get_string(0)?,
                query: row.get_string(1)?,
                timestamp: row.get_string(2)?,
                source_id: row.get_opt_string(3)?,
                user_id: row.get_string(4)?,
                results: row
                    .get_opt_string(5)?
                    .map(|value| serde_json::from_str(&value))
                    .transpose()?,
                feedback: row.get_opt_string(6)?,
            })
        })
    }

    fn export_memory_entities_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
    ) -> Result<Vec<MemoryEntityExport>> {
        let (filter, params) = user_filter("user_id", user_id);
        let sql = format!(
            "SELECT memory_id, entity_id, entity_name, user_id \
             FROM memory_entities {filter} ORDER BY memory_id, entity_id"
        );
        reader.query_read(&sql, &params, |row| {
            Ok(MemoryEntityExport {
                memory_id: row.get_string(0)?,
                entity_id: row.get_string(1)?,
                entity_name: row.get_string(2)?,
                user_id: row.get_string(3)?,
            })
        })
    }

    fn export_associations_on<R: ExportReader>(
        &self,
        reader: &R,
        user_id: Option<&str>,
        collection: &str,
    ) -> Result<Vec<AssociationExport>> {
        let (filter, params) = match user_id {
            Some(user_id) => (
                format!(
                    r#"WHERE EXISTS (
                           SELECT 1 FROM (
                               SELECT id, 'memory' AS layer FROM memories WHERE user_id = $1
                               UNION ALL SELECT event_id, 'event' FROM events WHERE user_id = $1
                               UNION ALL SELECT episode_id, 'episode' FROM episodes WHERE user_id = $1
                               UNION ALL SELECT trait_id, 'identity' FROM identity WHERE user_id = $1
                               UNION ALL SELECT id, 'entity' FROM entities_{collection} WHERE user_id = $1
                           ) owned_from
                           WHERE owned_from.id = a.from_id AND owned_from.layer = a.from_layer
                       ) AND EXISTS (
                           SELECT 1 FROM (
                               SELECT id, 'memory' AS layer FROM memories WHERE user_id = $1
                               UNION ALL SELECT event_id, 'event' FROM events WHERE user_id = $1
                               UNION ALL SELECT episode_id, 'episode' FROM episodes WHERE user_id = $1
                               UNION ALL SELECT trait_id, 'identity' FROM identity WHERE user_id = $1
                               UNION ALL SELECT id, 'entity' FROM entities_{collection} WHERE user_id = $1
                           ) owned_to
                           WHERE owned_to.id = a.to_id AND owned_to.layer = a.to_layer
                       )"#
                ),
                vec![SqlParam::Text(user_id.to_string())],
            ),
            None => (String::new(), Vec::new()),
        };
        let sql = format!(
            "SELECT assoc_id, from_id, from_layer, to_id, to_layer, assoc_type, strength, created_at \
             FROM associations a {filter} ORDER BY created_at"
        );
        reader.query_read(&sql, &params, |row| {
            Ok(AssociationExport {
                assoc_id: row.get_string(0)?,
                from_id: row.get_string(1)?,
                from_layer: row.get_string(2)?,
                to_id: row.get_string(3)?,
                to_layer: row.get_string(4)?,
                assoc_type: row.get_string(5)?,
                strength: row.get_opt_f64(6)?.unwrap_or(0.5) as f32,
                created_at: row.get_string(7)?,
            })
        })
    }

    // ── Bulk import methods ──

    /// Import sources from export records. Returns the number of successfully imported sources.
    #[cfg(test)]
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
            let user_val = opt_text(src.user_id.as_deref());
            let sql = r#"INSERT INTO sources (source_id, source_type, name, registered_at, metadata, user_id)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (source_id) DO NOTHING"#;
            let affected = self.backend.execute(
                sql,
                &[
                    SqlParam::Text(src.source_id.clone()),
                    SqlParam::Text(src.source_type.clone()),
                    name_val,
                    SqlParam::Text(src.registered_at.clone()),
                    meta_val,
                    user_val,
                ],
            )?;
            count += affected as u64;
        }
        self.backend.execute_batch("COMMIT")?;
        Ok(count)
    }

    /// Import sessions from export records. Returns the number of successfully imported sessions.
    #[cfg(test)]
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
    #[cfg(test)]
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
            let scope = |key: &str| {
                opt_text(
                    evt.metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get(key))
                        .and_then(serde_json::Value::as_str),
                )
            };
            let agent_val = scope("agent_id");
            let app_val = scope("app_id");
            let run_val = scope("run_id");
            let processed_at_val = opt_text(evt.processed_at.as_deref());
            let purified_val = opt_text(evt.purified_content.as_deref());
            let event_time_val = opt_text(evt.event_time.as_deref());
            let location_val = opt_text(evt.location.as_deref());
            let sql = r#"INSERT INTO events (event_id, source_id, session_id,
                                       agent_id, app_id, run_id,
                                       timestamp, event_type, content, parent_id, metadata, user_id,
                                       processed, processed_at, purified_content, purified,
                                       event_time, location)
                   VALUES ($1, $2, $3,
                           $4, $5, $6,
                           $7, $8, $9, $10, $11, $12,
                           $13, $14, $15, $16,
                           $17, $18)
                   ON CONFLICT (event_id) DO NOTHING"#;
            let affected = self.backend.execute(
                sql,
                &[
                    SqlParam::Text(evt.event_id.clone()),
                    source_val,
                    session_val,
                    agent_val,
                    app_val,
                    run_val,
                    SqlParam::Text(evt.timestamp.clone()),
                    SqlParam::Text(evt.event_type.as_str().to_string()),
                    SqlParam::Text(evt.content.clone()),
                    parent_val,
                    meta_val,
                    SqlParam::Text(evt.user_id.clone()),
                    SqlParam::Bool(evt.processed),
                    processed_at_val,
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
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn import_memories(&self, memories: &[MemoryExport]) -> Result<u64> {
        let embeddings = vec![vec![0.0; self.config.embedding_dims]; memories.len()];
        self.import_memories_atomic(memories, &embeddings)
    }

    /// Rebuild the vector entry for an imported memory without changing its
    /// timestamps, privacy, immutability, or other exported fields.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn index_imported_memory(
        &self,
        memory: &MemoryExport,
        embedding: &[f32],
    ) -> Result<()> {
        if embedding.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                embedding.len()
            )));
        }
        let emb_literal = self.format_embedding(embedding, self.config.embedding_dims)?;
        let stored = self.backend.query_one(
            "SELECT content, user_id, agent_id, run_id, app_id FROM memories WHERE id = $1",
            &[SqlParam::Text(memory.id.clone())],
            |row| {
                Ok((
                    row.get_string(0)?,
                    row.get_string(1)?,
                    row.get_opt_string(2)?,
                    row.get_opt_string(3)?,
                    row.get_opt_string(4)?,
                ))
            },
        )?;
        let Some((content, user_id, agent_id, run_id, app_id)) = stored else {
            return Err(MemoryError::NotFound(memory.id.clone()));
        };
        if content != memory.content {
            // Import is insert-only. Keep an existing record with the same ID
            // exactly as it is, including its matching vector entry.
            return Ok(());
        }

        let delete_sql = self.dialect().vector_delete_sql();
        let insert_sql = self.dialect().vector_insert_sql("$1", &emb_literal);
        self.backend.transaction(|tx| {
            tx.execute(
                &format!("UPDATE memories SET embedding = {emb_literal} WHERE id = $1"),
                &[SqlParam::Text(memory.id.clone())],
            )?;
            if let Some(delete_sql) = delete_sql {
                tx.execute(delete_sql, &[SqlParam::Text(memory.id.clone())])?;
            }
            if let Some(insert_sql) = insert_sql.as_deref() {
                tx.execute(
                    insert_sql,
                    &[
                        SqlParam::Text(memory.id.clone()),
                        SqlParam::Text(user_id),
                        opt_text(agent_id.as_deref()),
                        opt_text(run_id.as_deref()),
                        opt_text(app_id.as_deref()),
                    ],
                )?;
            }
            Ok(())
        })
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn index_imported_event(&self, event: &Event, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                embedding.len()
            )));
        }
        let emb_literal = self.format_embedding(embedding, self.config.embedding_dims)?;
        let stored = self.backend.query_one(
            "SELECT content, user_id, agent_id FROM events WHERE event_id = $1",
            &[SqlParam::Text(event.event_id.clone())],
            |row| {
                Ok((
                    row.get_string(0)?,
                    row.get_string(1)?,
                    row.get_opt_string(2)?,
                ))
            },
        )?;
        let Some((content, user_id, agent_id)) = stored else {
            return Err(MemoryError::NotFound(event.event_id.clone()));
        };
        if content != event.content {
            return Ok(());
        }

        let delete_sql = self.dialect().vector_event_delete_sql();
        let insert_sql = self.dialect().vector_event_insert_sql("$1", &emb_literal);
        self.backend.transaction(|tx| {
            tx.execute(
                &format!("UPDATE events SET content_vec = {emb_literal} WHERE event_id = $1"),
                &[SqlParam::Text(event.event_id.clone())],
            )?;
            if let Some(delete_sql) = delete_sql {
                tx.execute(delete_sql, &[SqlParam::Text(event.event_id.clone())])?;
            }
            if let Some(insert_sql) = insert_sql.as_deref() {
                tx.execute(
                    insert_sql,
                    &[
                        SqlParam::Text(event.event_id.clone()),
                        SqlParam::Text(user_id),
                        opt_text(agent_id.as_deref()),
                    ],
                )?;
            }
            Ok(())
        })
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn index_imported_episode(
        &self,
        episode: &Episode,
        embedding: &[f32],
    ) -> Result<()> {
        if embedding.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                embedding.len()
            )));
        }
        let emb_literal = self.format_embedding(embedding, self.config.embedding_dims)?;
        let stored = self.backend.query_one(
            "SELECT summary FROM episodes WHERE episode_id = $1",
            &[SqlParam::Text(episode.episode_id.clone())],
            |row| row.get_string(0),
        )?;
        let Some(summary) = stored else {
            return Err(MemoryError::NotFound(episode.episode_id.clone()));
        };
        if summary != episode.summary {
            return Ok(());
        }
        self.backend.execute(
            &format!("UPDATE episodes SET summary_vec = {emb_literal} WHERE episode_id = $1"),
            &[SqlParam::Text(episode.episode_id.clone())],
        )?;
        Ok(())
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn index_imported_identity_trait(
        &self,
        identity: &IdentityTrait,
        embedding: &[f32],
    ) -> Result<()> {
        if embedding.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                embedding.len()
            )));
        }
        let emb_literal = self.format_embedding(embedding, self.config.embedding_dims)?;
        let stored = self.backend.query_one(
            "SELECT content FROM identity WHERE trait_id = $1",
            &[SqlParam::Text(identity.trait_id.clone())],
            |row| row.get_string(0),
        )?;
        let Some(content) = stored else {
            return Err(MemoryError::NotFound(identity.trait_id.clone()));
        };
        if content != identity.content {
            return Ok(());
        }
        self.backend.execute(
            &format!("UPDATE identity SET content_vec = {emb_literal} WHERE trait_id = $1"),
            &[SqlParam::Text(identity.trait_id.clone())],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::config::MemoryConfig;
    use crate::storage::InsertMemoryParams;

    use super::Storage;

    fn test_config(dims: usize) -> MemoryConfig {
        MemoryConfig::new(":memory:", dims)
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
