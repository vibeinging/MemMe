use std::collections::HashMap;

use crate::error::{MemoryError, Result};
use crate::types::{Event, FullExport, FullImportResult, MemoryExport, Session, SqlParam};

use super::backend::BackendTransaction;
use super::util::opt_text;
use super::Storage;

pub(crate) const MAX_FULL_IMPORT_RECORDS: usize = 50_000;
pub(crate) const MAX_FULL_IMPORT_EMBEDDING_BYTES: usize = 128 * 1024 * 1024;
pub(crate) const FULL_EXPORT_VERSION: &str = "3.0";

pub(crate) struct FullImportEmbeddings<'a> {
    pub memories: &'a [Vec<f32>],
    pub events: &'a [Vec<f32>],
    pub episodes: &'a [Vec<f32>],
    pub identity_traits: &'a [Vec<f32>],
}

fn json_param(value: Option<&serde_json::Value>) -> Result<SqlParam> {
    value
        .map(serde_json::to_string)
        .transpose()
        .map(|value| opt_text(value.as_deref()))
        .map_err(Into::into)
}

fn json_vec_param(value: Option<&Vec<String>>) -> Result<SqlParam> {
    value
        .map(serde_json::to_string)
        .transpose()
        .map(|value| opt_text(value.as_deref()))
        .map_err(Into::into)
}

fn session_scope(session: &Session, key: &str) -> Result<Option<String>> {
    let Some(value) = session
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get(key))
    else {
        return Ok(None);
    };
    let value = value.as_str().ok_or_else(|| {
        MemoryError::Config(format!(
            "session '{}' metadata.{key} must be a string",
            session.session_id
        ))
    })?;
    if value.trim().is_empty() {
        return Err(MemoryError::Config(format!(
            "session '{}' metadata.{key} must not be empty",
            session.session_id
        )));
    }
    Ok(Some(value.to_string()))
}

fn event_scope(event: &Event, key: &str) -> Result<Option<String>> {
    let explicit = match key {
        "agent_id" => event.agent_id.as_deref(),
        "app_id" => event.app_id.as_deref(),
        "run_id" => event.run_id.as_deref(),
        _ => None,
    };
    let metadata = event
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get(key));
    let metadata = match metadata {
        Some(value) => Some(value.as_str().ok_or_else(|| {
            MemoryError::Config(format!(
                "event '{}' metadata.{key} must be a string",
                event.event_id
            ))
        })?),
        None => None,
    };
    if let (Some(explicit), Some(metadata)) = (explicit, metadata) {
        if explicit != metadata {
            return Err(MemoryError::Config(format!(
                "event '{}' has conflicting {key} values",
                event.event_id
            )));
        }
    }
    let value = explicit.or(metadata);
    if value.is_some_and(|value| value.trim().is_empty()) {
        return Err(MemoryError::Config(format!(
            "event '{}' {key} must not be empty",
            event.event_id
        )));
    }
    Ok(value.map(str::to_string))
}

fn validate_event_session(event: &Event, session: &Session) -> Result<()> {
    if event.user_id != session.user_id {
        return Err(MemoryError::Config(format!(
            "event '{}' cannot use session '{}' owned by another user",
            event.event_id, session.session_id
        )));
    }
    if event.source_id != session.source_id {
        return Err(MemoryError::Config(format!(
            "event '{}' conflicts with session '{}' source_id",
            event.event_id, session.session_id
        )));
    }
    for key in ["agent_id", "app_id", "run_id"] {
        if event_scope(event, key)? != session_scope(session, key)? {
            return Err(MemoryError::Config(format!(
                "event '{}' conflicts with session '{}' {key}",
                event.event_id, session.session_id
            )));
        }
    }
    Ok(())
}

fn validate_embedding_counts(
    export: &FullExport,
    embeddings: &FullImportEmbeddings<'_>,
) -> Result<()> {
    for (label, expected, actual) in [
        ("memory", export.memories.len(), embeddings.memories.len()),
        ("event", export.events.len(), embeddings.events.len()),
        ("episode", export.episodes.len(), embeddings.episodes.len()),
        (
            "identity",
            export.identity_traits.len(),
            embeddings.identity_traits.len(),
        ),
    ] {
        if expected != actual {
            return Err(MemoryError::Config(format!(
                "{label} embedding count mismatch: expected {expected}, got {actual}"
            )));
        }
    }

    Ok(())
}

fn build_owner_map<'a>(
    layer: &str,
    rows: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<HashMap<&'a str, &'a str>> {
    let mut owners = HashMap::new();
    for (id, user_id) in rows {
        if owners.insert(id, user_id).is_some() {
            return Err(MemoryError::Config(format!(
                "duplicate {layer} id '{id}' in full export"
            )));
        }
    }
    Ok(owners)
}

fn require_owned_export_reference(
    owners: &HashMap<&str, &str>,
    layer: &str,
    id: &str,
    expected_user: &str,
    context: &str,
) -> Result<()> {
    match owners.get(id).copied() {
        Some(owner) if owner == expected_user => Ok(()),
        Some(_) => Err(MemoryError::Config(format!(
            "{context} references {layer} '{id}' owned by another user"
        ))),
        None => Err(MemoryError::Config(format!(
            "{context} references missing {layer} '{id}' in full export"
        ))),
    }
}

struct ExportOwners<'a> {
    memories: HashMap<&'a str, &'a str>,
    sessions: HashMap<&'a str, &'a str>,
    events: HashMap<&'a str, &'a str>,
    episodes: HashMap<&'a str, &'a str>,
    entities: HashMap<&'a str, &'a str>,
    identity: HashMap<&'a str, &'a str>,
}

impl ExportOwners<'_> {
    fn layer(&self, layer: &str) -> Option<&HashMap<&str, &str>> {
        match layer {
            "memory" => Some(&self.memories),
            "event" => Some(&self.events),
            "episode" => Some(&self.episodes),
            "entity" => Some(&self.entities),
            "identity" => Some(&self.identity),
            _ => None,
        }
    }
}

pub(crate) fn validate_full_import_envelope(
    export: &FullExport,
    embedding_dimensions: usize,
    expected_collection: &str,
) -> Result<()> {
    if export.version != FULL_EXPORT_VERSION {
        return Err(MemoryError::Config(format!(
            "unsupported full export version '{}'; expected {FULL_EXPORT_VERSION}",
            export.version
        )));
    }
    if export.collection != expected_collection {
        return Err(MemoryError::Config(format!(
            "full export collection '{}' does not match target collection '{expected_collection}'",
            export.collection
        )));
    }

    let total_records = export.memories.len()
        + export.sessions.len()
        + export.events.len()
        + export.episodes.len()
        + export.entities.len()
        + export.relations.len()
        + export.identity_traits.len()
        + export.sources.len()
        + export.history.len()
        + export.procedures.len()
        + export.meditations.len()
        + export.recalls.len()
        + export.memory_entities.len()
        + export.associations.len();
    if total_records > MAX_FULL_IMPORT_RECORDS {
        return Err(MemoryError::Config(format!(
            "full import contains {total_records} records; maximum is {MAX_FULL_IMPORT_RECORDS}. Use a validated SQLite backup for larger migrations"
        )));
    }

    let embedded_records = export.memories.len()
        + export.events.len()
        + export.episodes.len()
        + export.identity_traits.len();
    let embedding_bytes = embedded_records
        .checked_mul(embedding_dimensions)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| MemoryError::Config("full import embedding size overflow".into()))?;
    if embedding_bytes > MAX_FULL_IMPORT_EMBEDDING_BYTES {
        return Err(MemoryError::Config(format!(
            "full import would allocate about {embedding_bytes} bytes for embeddings; maximum is {MAX_FULL_IMPORT_EMBEDDING_BYTES}. Use a validated SQLite backup for larger migrations"
        )));
    }

    if let Some(expected_user) = export.user_id.as_deref() {
        let check = |layer: &str, id: &str, actual: &str| -> Result<()> {
            if actual != expected_user {
                return Err(MemoryError::Config(format!(
                    "{layer} '{id}' belongs to '{actual}', not exported user '{expected_user}'"
                )));
            }
            Ok(())
        };
        for row in &export.memories {
            check("memory", &row.id, &row.user_id)?;
        }
        for row in &export.sessions {
            check("session", &row.session_id, &row.user_id)?;
        }
        for row in &export.events {
            check("event", &row.event_id, &row.user_id)?;
        }
        for row in &export.episodes {
            check("episode", &row.episode_id, &row.user_id)?;
        }
        for row in &export.entities {
            check("entity", &row.id, &row.user_id)?;
        }
        for row in &export.relations {
            check("relation", &row.id, &row.user_id)?;
        }
        for row in &export.identity_traits {
            check("identity", &row.trait_id, &row.user_id)?;
        }
        for row in &export.sources {
            if let Some(actual) = row.user_id.as_deref() {
                check("source", &row.source_id, actual)?;
            }
        }
        for row in &export.history {
            check("history", &row.id, &row.user_id)?;
        }
        for row in &export.procedures {
            check("procedure", &row.id, &row.user_id)?;
        }
        for row in &export.meditations {
            check("meditation", &row.meditation_id, &row.user_id)?;
        }
        for row in &export.recalls {
            check("recall", &row.recall_id, &row.user_id)?;
        }
        for row in &export.memory_entities {
            check("memory_entity", &row.memory_id, &row.user_id)?;
        }
    }

    let owners = ExportOwners {
        memories: build_owner_map(
            "memory",
            export
                .memories
                .iter()
                .map(|row| (row.id.as_str(), row.user_id.as_str())),
        )?,
        sessions: build_owner_map(
            "session",
            export
                .sessions
                .iter()
                .map(|row| (row.session_id.as_str(), row.user_id.as_str())),
        )?,
        events: build_owner_map(
            "event",
            export
                .events
                .iter()
                .map(|row| (row.event_id.as_str(), row.user_id.as_str())),
        )?,
        episodes: build_owner_map(
            "episode",
            export
                .episodes
                .iter()
                .map(|row| (row.episode_id.as_str(), row.user_id.as_str())),
        )?,
        entities: build_owner_map(
            "entity",
            export
                .entities
                .iter()
                .map(|row| (row.id.as_str(), row.user_id.as_str())),
        )?,
        identity: build_owner_map(
            "identity",
            export
                .identity_traits
                .iter()
                .map(|row| (row.trait_id.as_str(), row.user_id.as_str())),
        )?,
    };
    let sessions_by_id: HashMap<&str, &Session> = export
        .sessions
        .iter()
        .map(|session| (session.session_id.as_str(), session))
        .collect();

    // Also reject duplicate primary keys in layers that are not reference targets.
    build_owner_map(
        "relation",
        export
            .relations
            .iter()
            .map(|row| (row.id.as_str(), row.user_id.as_str())),
    )?;
    build_owner_map(
        "history",
        export
            .history
            .iter()
            .map(|row| (row.id.as_str(), row.user_id.as_str())),
    )?;
    build_owner_map(
        "procedure",
        export
            .procedures
            .iter()
            .map(|row| (row.id.as_str(), row.user_id.as_str())),
    )?;
    build_owner_map(
        "meditation",
        export
            .meditations
            .iter()
            .map(|row| (row.meditation_id.as_str(), row.user_id.as_str())),
    )?;
    build_owner_map(
        "recall",
        export
            .recalls
            .iter()
            .map(|row| (row.recall_id.as_str(), row.user_id.as_str())),
    )?;
    let mut source_ids = HashMap::new();
    for row in &export.sources {
        if source_ids.insert(row.source_id.as_str(), ()).is_some() {
            return Err(MemoryError::Config(format!(
                "duplicate source id '{}' in full export",
                row.source_id
            )));
        }
    }
    let mut association_ids = HashMap::new();
    for row in &export.associations {
        if association_ids.insert(row.assoc_id.as_str(), ()).is_some() {
            return Err(MemoryError::Config(format!(
                "duplicate association id '{}' in full export",
                row.assoc_id
            )));
        }
    }
    let mut memory_entity_ids = HashMap::new();
    for row in &export.memory_entities {
        if memory_entity_ids
            .insert((row.memory_id.as_str(), row.entity_id.as_str()), ())
            .is_some()
        {
            return Err(MemoryError::Config(format!(
                "duplicate memory-entity link '{}:{}' in full export",
                row.memory_id, row.entity_id
            )));
        }
    }

    for event in &export.events {
        if let Some(session_id) = event.session_id.as_deref() {
            require_owned_export_reference(
                &owners.sessions,
                "session",
                session_id,
                &event.user_id,
                &format!("event '{}'", event.event_id),
            )?;
            validate_event_session(event, sessions_by_id[session_id])?;
        }
    }
    for episode in &export.episodes {
        let context = format!("episode '{}'", episode.episode_id);
        for event_id in &episode.event_ids {
            require_owned_export_reference(
                &owners.events,
                "event",
                event_id,
                &episode.user_id,
                &context,
            )?;
        }
        for session_id in &episode.session_ids {
            require_owned_export_reference(
                &owners.sessions,
                "session",
                session_id,
                &episode.user_id,
                &context,
            )?;
        }
    }
    for memory in &export.memories {
        let context = format!("memory '{}'", memory.id);
        if let Some(session_id) = memory.session_id.as_deref() {
            require_owned_export_reference(
                &owners.sessions,
                "session",
                session_id,
                &memory.user_id,
                &context,
            )?;
        }
        if let Some(episode_id) = memory.episode_id.as_deref() {
            require_owned_export_reference(
                &owners.episodes,
                "episode",
                episode_id,
                &memory.user_id,
                &context,
            )?;
        }
        if let Some(episode_ids) = memory.episode_ids.as_deref() {
            for episode_id in episode_ids {
                require_owned_export_reference(
                    &owners.episodes,
                    "episode",
                    episode_id,
                    &memory.user_id,
                    &context,
                )?;
            }
        }
        if let Some(superseded_by) = memory.superseded_by.as_deref() {
            require_owned_export_reference(
                &owners.memories,
                "memory",
                superseded_by,
                &memory.user_id,
                &context,
            )?;
        }
    }
    for relation in &export.relations {
        let context = format!("relation '{}'", relation.id);
        require_owned_export_reference(
            &owners.entities,
            "entity",
            &relation.source_id,
            &relation.user_id,
            &context,
        )?;
        require_owned_export_reference(
            &owners.entities,
            "entity",
            &relation.target_id,
            &relation.user_id,
            &context,
        )?;
        if let Some(episode_ids) = relation.episode_ids.as_deref() {
            for episode_id in episode_ids {
                require_owned_export_reference(
                    &owners.episodes,
                    "episode",
                    episode_id,
                    &relation.user_id,
                    &context,
                )?;
            }
        }
    }
    for identity in &export.identity_traits {
        let context = format!("identity '{}'", identity.trait_id);
        for memory_id in &identity.evidence_ids {
            require_owned_export_reference(
                &owners.memories,
                "memory",
                memory_id,
                &identity.user_id,
                &context,
            )?;
        }
    }
    for history in &export.history {
        let Some(memory_id) = history.memory_id.as_deref() else {
            continue;
        };
        match owners.memories.get(memory_id).copied() {
            Some(owner) if owner == history.user_id => {}
            Some(_) => {
                return Err(MemoryError::Config(format!(
                    "history '{}' references memory '{memory_id}' owned by another user",
                    history.id
                )))
            }
            None if history
                .event
                .as_deref()
                .is_some_and(|event| event.eq_ignore_ascii_case("delete")) => {}
            None => {
                return Err(MemoryError::Config(format!(
                    "history '{}' references missing memory '{memory_id}' in full export",
                    history.id
                )))
            }
        }
    }
    for link in &export.memory_entities {
        let context = "memory-entity link";
        require_owned_export_reference(
            &owners.memories,
            "memory",
            &link.memory_id,
            &link.user_id,
            context,
        )?;
        require_owned_export_reference(
            &owners.entities,
            "entity",
            &link.entity_id,
            &link.user_id,
            context,
        )?;
    }
    for association in &export.associations {
        let from_owners = owners.layer(&association.from_layer).ok_or_else(|| {
            MemoryError::Config(format!(
                "association '{}' uses unsupported layer '{}'",
                association.assoc_id, association.from_layer
            ))
        })?;
        let to_owners = owners.layer(&association.to_layer).ok_or_else(|| {
            MemoryError::Config(format!(
                "association '{}' uses unsupported layer '{}'",
                association.assoc_id, association.to_layer
            ))
        })?;
        let from_user = from_owners
            .get(association.from_id.as_str())
            .ok_or_else(|| {
                MemoryError::Config(format!(
                    "association '{}' references missing {} '{}' in full export",
                    association.assoc_id, association.from_layer, association.from_id
                ))
            })?;
        let to_user = to_owners.get(association.to_id.as_str()).ok_or_else(|| {
            MemoryError::Config(format!(
                "association '{}' references missing {} '{}' in full export",
                association.assoc_id, association.to_layer, association.to_id
            ))
        })?;
        if from_user != to_user {
            return Err(MemoryError::Config(format!(
                "association '{}' crosses user ownership",
                association.assoc_id
            )));
        }
        if export
            .user_id
            .as_deref()
            .is_some_and(|user_id| user_id != *from_user)
        {
            return Err(MemoryError::Config(format!(
                "association '{}' does not belong to exported user",
                association.assoc_id
            )));
        }
    }
    Ok(())
}

fn reject_existing_record(table: &str, id: &str) -> Result<()> {
    Err(MemoryError::Config(format!(
        "{table} id '{id}' already exists; full import requires a collision-free target"
    )))
}

fn reject_target_collision(
    storage: &Storage,
    table: &str,
    id_column: &str,
    id: &str,
) -> Result<()> {
    let sql = format!("SELECT COUNT(*) FROM {table} WHERE {id_column} = $1");
    if storage
        .backend
        .query_count(&sql, &[SqlParam::Text(id.to_string())])?
        > 0
    {
        reject_existing_record(table, id)
    } else {
        Ok(())
    }
}

fn ensure_owned_reference(
    tx: &BackendTransaction<'_>,
    table: &str,
    id_column: &str,
    id: &str,
    expected_user: &str,
    context: &str,
) -> Result<()> {
    let owner = tx.query_one(
        &format!("SELECT user_id FROM {table} WHERE {id_column} = $1"),
        &[SqlParam::Text(id.to_string())],
        |row| row.get_string(0),
    )?;
    match owner.as_deref() {
        Some(owner) if owner == expected_user => Ok(()),
        Some(_) => Err(MemoryError::Config(format!(
            "{context} references {table} record '{id}' owned by another user"
        ))),
        None => Err(MemoryError::Config(format!(
            "{context} references missing {table} record '{id}'"
        ))),
    }
}

fn owned_layer_user(
    tx: &BackendTransaction<'_>,
    collection: &str,
    layer: &str,
    id: &str,
) -> Result<String> {
    let (table, id_column) = match layer {
        "memory" => ("memories".to_string(), "id"),
        "event" => ("events".to_string(), "event_id"),
        "episode" => ("episodes".to_string(), "episode_id"),
        "identity" => ("identity".to_string(), "trait_id"),
        "entity" => (format!("entities_{collection}"), "id"),
        _ => {
            return Err(MemoryError::Config(format!(
                "unsupported association layer '{layer}'"
            )))
        }
    };
    tx.query_one(
        &format!("SELECT user_id FROM {table} WHERE {id_column} = $1"),
        &[SqlParam::Text(id.to_string())],
        |row| row.get_string(0),
    )?
    .ok_or_else(|| {
        MemoryError::Config(format!(
            "association references missing {layer} record '{id}'"
        ))
    })
}

impl Storage {
    /// Reject target collisions before making any remote embedding calls.
    /// The import transaction repeats these checks when inserting, so a race
    /// cannot turn a preflight success into an overwrite.
    pub(crate) fn preflight_full_import_target(&self, export: &FullExport) -> Result<()> {
        for row in &export.sources {
            reject_target_collision(self, "sources", "source_id", &row.source_id)?;
        }
        for row in &export.sessions {
            reject_target_collision(self, "sessions", "session_id", &row.session_id)?;
        }
        for row in &export.events {
            reject_target_collision(self, "events", "event_id", &row.event_id)?;
        }
        for row in &export.episodes {
            reject_target_collision(self, "episodes", "episode_id", &row.episode_id)?;
        }
        for row in &export.memories {
            reject_target_collision(self, "memories", "id", &row.id)?;
        }
        let entity_table = format!("entities_{}", self.config.collection_name);
        for row in &export.entities {
            reject_target_collision(self, &entity_table, "id", &row.id)?;
        }
        let relation_table = format!("relationships_{}", self.config.collection_name);
        for row in &export.relations {
            reject_target_collision(self, &relation_table, "id", &row.id)?;
        }
        for row in &export.identity_traits {
            reject_target_collision(self, "identity", "trait_id", &row.trait_id)?;
        }
        for row in &export.history {
            reject_target_collision(self, "history", "id", &row.id)?;
            // A delete history row may legitimately refer to a memory omitted
            // from the package, but it must not attach to a target-side row.
            if row
                .event
                .as_deref()
                .is_some_and(|event| event.eq_ignore_ascii_case("delete"))
                && row
                    .memory_id
                    .as_deref()
                    .is_some_and(|id| !export.memories.iter().any(|memory| memory.id == id))
            {
                if let Some(memory_id) = row.memory_id.as_deref() {
                    reject_target_collision(self, "memories", "id", memory_id)?;
                }
            }
        }
        for row in &export.procedures {
            reject_target_collision(self, "procedures", "id", &row.id)?;
        }
        for row in &export.meditations {
            reject_target_collision(self, "meditations", "meditation_id", &row.meditation_id)?;
        }
        for row in &export.recalls {
            reject_target_collision(self, "recalls", "recall_id", &row.recall_id)?;
        }
        for row in &export.associations {
            reject_target_collision(self, "associations", "assoc_id", &row.assoc_id)?;
        }
        for row in &export.memory_entities {
            let count = self.backend.query_count(
                "SELECT COUNT(*) FROM memory_entities WHERE memory_id = $1 AND entity_id = $2",
                &[
                    SqlParam::Text(row.memory_id.clone()),
                    SqlParam::Text(row.entity_id.clone()),
                ],
            )?;
            if count > 0 {
                reject_existing_record(
                    "memory_entities",
                    &format!("{}:{}", row.memory_id, row.entity_id),
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn import_memories_atomic(
        &self,
        memories: &[MemoryExport],
        embeddings: &[Vec<f32>],
    ) -> Result<u64> {
        if memories.len() != embeddings.len() {
            return Err(MemoryError::Config(
                "memory embedding count does not match import".into(),
            ));
        }
        self.backend.transaction(|tx| {
            let count = self.import_memories_on(tx, memories, embeddings)?;
            self.rebuild_fts_on(tx)?;
            Ok(count)
        })
    }

    pub(crate) fn import_full_atomic(
        &self,
        export: &FullExport,
        embeddings: FullImportEmbeddings<'_>,
    ) -> Result<FullImportResult> {
        validate_full_import_envelope(
            export,
            self.config.embedding_dims,
            &self.config.collection_name,
        )?;
        validate_embedding_counts(export, &embeddings)?;
        self.backend.transaction(|tx| {
            let sources = self.import_sources_on(tx, export)?;
            let sessions = self.import_sessions_on(tx, export)?;
            let events = self.import_events_on(tx, &export.events, embeddings.events)?;
            let episodes = self.import_episodes_on(tx, &export.episodes, embeddings.episodes)?;
            let memories = self.import_memories_on(tx, &export.memories, embeddings.memories)?;
            self.validate_memory_references_on(tx, &export.memories)?;
            let entities = self.import_entities_on(tx, export)?;
            let relations = self.import_relations_on(tx, export)?;
            let identity_traits =
                self.import_identity_on(tx, &export.identity_traits, embeddings.identity_traits)?;
            let history = self.import_history_on(tx, export)?;
            let procedures = self.import_procedures_on(tx, export)?;
            let meditations = self.import_meditations_on(tx, export)?;
            let recalls = self.import_recalls_on(tx, export)?;
            let memory_entities = self.import_memory_entities_on(tx, export)?;
            let associations = self.import_associations_on(tx, export)?;
            self.rebuild_fts_on(tx)?;
            Ok(FullImportResult {
                sources,
                sessions,
                events,
                episodes,
                memories,
                entities,
                relations,
                identity_traits,
                history,
                procedures,
                meditations,
                recalls,
                memory_entities,
                associations,
            })
        })
    }

    fn import_sources_on(&self, tx: &BackendTransaction<'_>, export: &FullExport) -> Result<u64> {
        let mut count = 0;
        for row in &export.sources {
            let affected = tx.execute(
                r#"INSERT INTO sources (source_id, source_type, name, registered_at, metadata, user_id)
                   VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (source_id) DO NOTHING"#,
                &[
                    SqlParam::Text(row.source_id.clone()),
                    SqlParam::Text(row.source_type.clone()),
                    opt_text(row.name.as_deref()),
                    SqlParam::Text(row.registered_at.clone()),
                    json_param(row.metadata.as_ref())?,
                    opt_text(row.user_id.as_deref()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("sources", &row.source_id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_sessions_on(&self, tx: &BackendTransaction<'_>, export: &FullExport) -> Result<u64> {
        let mut count = 0;
        for row in &export.sessions {
            let affected = tx.execute(
                r#"INSERT INTO sessions (session_id, user_id, source_id, started_at, ended_at,
                                          metadata, created_at, structured_notes, queried_count,
                                          last_queried_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                   ON CONFLICT (session_id) DO NOTHING"#,
                &[
                    SqlParam::Text(row.session_id.clone()),
                    SqlParam::Text(row.user_id.clone()),
                    opt_text(row.source_id.as_deref()),
                    SqlParam::Text(row.started_at.clone()),
                    opt_text(row.ended_at.as_deref()),
                    json_param(row.metadata.as_ref())?,
                    SqlParam::Text(row.created_at.clone()),
                    opt_text(row.structured_notes.as_deref()),
                    SqlParam::Int(row.queried_count as i64),
                    opt_text(row.last_queried_at.as_deref()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("sessions", &row.session_id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_events_on(
        &self,
        tx: &BackendTransaction<'_>,
        events: &[Event],
        embeddings: &[Vec<f32>],
    ) -> Result<u64> {
        let mut count = 0;
        for (row, embedding) in events.iter().zip(embeddings) {
            if let Some(session_id) = row.session_id.as_deref() {
                let stored = tx.query_one(
                    "SELECT user_id, source_id, metadata, started_at, created_at FROM sessions WHERE session_id = $1",
                    &[SqlParam::Text(session_id.to_string())],
                    |value| {
                        Ok(Session {
                            session_id: session_id.to_string(),
                            user_id: value.get_string(0)?,
                            source_id: value.get_opt_string(1)?,
                            metadata: value
                                .get_opt_string(2)?
                                .map(|metadata| serde_json::from_str(&metadata))
                                .transpose()?,
                            started_at: value.get_string(3)?,
                            created_at: value.get_string(4)?,
                            ended_at: None,
                            event_count: 0,
                            structured_notes: None,
                            queried_count: 0,
                            last_queried_at: None,
                        })
                    },
                )?;
                let session = stored.ok_or_else(|| {
                    MemoryError::Config(format!(
                        "event '{}' references missing session '{session_id}'",
                        row.event_id
                    ))
                })?;
                validate_event_session(row, &session)?;
            }
            let literal = self.format_embedding(embedding, self.config.embedding_dims)?;
            let affected = tx.execute(
                &format!(r#"INSERT INTO events (event_id, source_id, session_id, agent_id, app_id,
                                                run_id, timestamp, event_type, content, content_vec,
                                                parent_id, metadata, user_id, processed, processed_at,
                                                purified_content, purified, event_time, location)
                              VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, {literal},
                                      $10, $11, $12, $13, $14, $15, $16, $17, $18)
                              ON CONFLICT (event_id) DO NOTHING"#),
                &[
                    SqlParam::Text(row.event_id.clone()),
                    opt_text(row.source_id.as_deref()),
                    opt_text(row.session_id.as_deref()),
                    opt_text(event_scope(row, "agent_id")?.as_deref()),
                    opt_text(event_scope(row, "app_id")?.as_deref()),
                    opt_text(event_scope(row, "run_id")?.as_deref()),
                    SqlParam::Text(row.timestamp.clone()),
                    SqlParam::Text(row.event_type.as_str().to_string()),
                    SqlParam::Text(row.content.clone()),
                    opt_text(row.parent_id.as_deref()),
                    json_param(row.metadata.as_ref())?,
                    SqlParam::Text(row.user_id.clone()),
                    SqlParam::Bool(row.processed),
                    opt_text(row.processed_at.as_deref()),
                    opt_text(row.purified_content.as_deref()),
                    SqlParam::Bool(row.purified),
                    opt_text(row.event_time.as_deref()),
                    opt_text(row.location.as_deref()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("events", &row.event_id)?;
            } else if let Some(sql) = self.dialect().vector_event_insert_sql("$1", &literal) {
                tx.execute(
                    &sql,
                    &[
                        SqlParam::Text(row.event_id.clone()),
                        SqlParam::Text(row.user_id.clone()),
                        opt_text(event_scope(row, "agent_id")?.as_deref()),
                    ],
                )?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_episodes_on(
        &self,
        tx: &BackendTransaction<'_>,
        rows: &[crate::types::Episode],
        embeddings: &[Vec<f32>],
    ) -> Result<u64> {
        let mut count = 0;
        for (row, embedding) in rows.iter().zip(embeddings) {
            let literal = self.format_embedding(embedding, self.config.embedding_dims)?;
            let affected = tx.execute(
                &format!(r#"INSERT INTO episodes (episode_id, title, summary, summary_vec, started_at,
                                                  ended_at, significance, outcome, source_id, event_ids,
                                                  user_id, created_at, last_recalled, recall_count,
                                                  storage_strength, retrieval_strength, session_ids,
                                                  last_meditated_at)
                              VALUES ($1, $2, $3, {literal}, $4, $5, $6, $7, $8, $9, $10,
                                      $11, $12, $13, $14, $15, $16, $17)
                              ON CONFLICT (episode_id) DO NOTHING"#),
                &[
                    SqlParam::Text(row.episode_id.clone()), SqlParam::Text(row.title.clone()),
                    SqlParam::Text(row.summary.clone()), SqlParam::Text(row.started_at.clone()),
                    opt_text(row.ended_at.as_deref()), SqlParam::Float(row.significance as f64),
                    opt_text(row.outcome.as_deref()), opt_text(row.source_id.as_deref()),
                    SqlParam::Text(serde_json::to_string(&row.event_ids)?),
                    SqlParam::Text(row.user_id.clone()), SqlParam::Text(row.created_at.clone()),
                    opt_text(row.last_recalled.as_deref()), SqlParam::Int(row.recall_count as i64),
                    SqlParam::Float(row.storage_strength as f64),
                    SqlParam::Float(row.retrieval_strength as f64),
                    SqlParam::Text(serde_json::to_string(&row.session_ids)?),
                    opt_text(row.last_meditated_at.as_deref()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("episodes", &row.episode_id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_memories_on(
        &self,
        tx: &BackendTransaction<'_>,
        rows: &[MemoryExport],
        embeddings: &[Vec<f32>],
    ) -> Result<u64> {
        let mut count = 0;
        for (row, embedding) in rows.iter().zip(embeddings) {
            let literal = self.format_embedding(embedding, self.config.embedding_dims)?;
            let categories = self.format_categories(row.categories.as_deref())?;
            let hash = crate::memory::content_hash(&row.content);
            let affected = tx.execute(
                &format!(r#"INSERT INTO memories (id, content, embedding, user_id, agent_id, run_id,
                                                  app_id, actor_id, importance, access_count, hash,
                                                  created_at, updated_at, metadata, immutable,
                                                  expiration_date, categories, memory_type, stability,
                                                  privacy, event_time, episode_id, session_id, resolution,
                                                  storage_strength, retrieval_strength, superseded_by,
                                                  valid_from, valid_until, confidence, evidence, episode_ids,
                                                  ingestion_time, sync_version, device_id, sync_status, pinned)
                              VALUES ($1, $2, {literal}, $3, $4, $5, $6, $7, $8,
                                      COALESCE($9, 0), $10, $11, $12, $13, $14, $15,
                                      {categories}, $16, $17, $18, $19, $20, $21, $22,
                                      $23, $24, $25, $26, $27, $28, $29, $30, $31,
                                      COALESCE($32, 0), $33, $34, COALESCE($35, 0))
                              ON CONFLICT (id) DO NOTHING"#),
                &[
                    SqlParam::Text(row.id.clone()), SqlParam::Text(row.content.clone()),
                    SqlParam::Text(row.user_id.clone()), opt_text(row.agent_id.as_deref()),
                    opt_text(row.run_id.as_deref()), opt_text(row.app_id.as_deref()),
                    opt_text(row.actor_id.as_deref()), SqlParam::Float(row.importance as f64),
                    row.access_count.map(|v| SqlParam::Int(v as i64)).unwrap_or(SqlParam::Null),
                    SqlParam::Text(hash), SqlParam::Text(row.created_at.clone()),
                    SqlParam::Text(row.updated_at.clone()), json_param(row.metadata.as_ref())?,
                    SqlParam::Bool(row.immutable), opt_text(row.expiration_date.as_deref()),
                    opt_text(row.memory_type.as_deref()),
                    row.stability.map(|v| SqlParam::Float(v as f64)).unwrap_or(SqlParam::Null),
                    opt_text(row.privacy.as_deref()), opt_text(row.event_time.as_deref()),
                    opt_text(row.episode_id.as_deref()), opt_text(row.session_id.as_deref()),
                    opt_text(row.resolution.as_deref()),
                    row.storage_strength.map(|v| SqlParam::Float(v as f64)).unwrap_or(SqlParam::Null),
                    row.retrieval_strength.map(|v| SqlParam::Float(v as f64)).unwrap_or(SqlParam::Null),
                    opt_text(row.superseded_by.as_deref()), opt_text(row.valid_from.as_deref()),
                    opt_text(row.valid_until.as_deref()),
                    row.confidence.map(|v| SqlParam::Float(v as f64)).unwrap_or(SqlParam::Null),
                    json_param(row.evidence.as_ref())?, json_vec_param(row.episode_ids.as_ref())?,
                    opt_text(row.ingestion_time.as_deref()),
                    row.sync_version.map(|v| SqlParam::Int(v as i64)).unwrap_or(SqlParam::Null),
                    opt_text(row.device_id.as_deref()), opt_text(row.sync_status.as_deref()),
                    row.pinned.map(SqlParam::Bool).unwrap_or(SqlParam::Null),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("memories", &row.id)?;
            } else if let Some(sql) = self.dialect().vector_insert_sql("$1", &literal) {
                tx.execute(
                    &sql,
                    &[
                        SqlParam::Text(row.id.clone()),
                        SqlParam::Text(row.user_id.clone()),
                        opt_text(row.agent_id.as_deref()),
                        opt_text(row.run_id.as_deref()),
                        opt_text(row.app_id.as_deref()),
                    ],
                )?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn validate_memory_references_on(
        &self,
        tx: &BackendTransaction<'_>,
        rows: &[MemoryExport],
    ) -> Result<()> {
        for row in rows {
            let context = format!("memory '{}'", row.id);
            if let Some(session_id) = row.session_id.as_deref() {
                ensure_owned_reference(
                    tx,
                    "sessions",
                    "session_id",
                    session_id,
                    &row.user_id,
                    &context,
                )?;
            }
            if let Some(episode_id) = row.episode_id.as_deref() {
                ensure_owned_reference(
                    tx,
                    "episodes",
                    "episode_id",
                    episode_id,
                    &row.user_id,
                    &context,
                )?;
            }
            if let Some(episode_ids) = row.episode_ids.as_deref() {
                for episode_id in episode_ids {
                    ensure_owned_reference(
                        tx,
                        "episodes",
                        "episode_id",
                        episode_id,
                        &row.user_id,
                        &context,
                    )?;
                }
            }
            if let Some(superseded_by) = row.superseded_by.as_deref() {
                ensure_owned_reference(
                    tx,
                    "memories",
                    "id",
                    superseded_by,
                    &row.user_id,
                    &context,
                )?;
            }
        }
        Ok(())
    }

    fn import_entities_on(&self, tx: &BackendTransaction<'_>, export: &FullExport) -> Result<u64> {
        let table = format!("entities_{}", self.config.collection_name);
        let mut count = 0;
        for row in &export.entities {
            let affected = tx.execute(
                &format!("INSERT INTO {table} (id, name, entity_type, user_id, created_at, updated_at) VALUES ($1, $2, $3, $4, COALESCE($5, datetime('now')), COALESCE($6, datetime('now'))) ON CONFLICT (id) DO NOTHING"),
                &[
                    SqlParam::Text(row.id.clone()), SqlParam::Text(row.name.clone()),
                    opt_text(row.entity_type.as_deref()), SqlParam::Text(row.user_id.clone()),
                    opt_text(row.created_at.as_deref()), opt_text(row.updated_at.as_deref()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record(&table, &row.id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_relations_on(&self, tx: &BackendTransaction<'_>, export: &FullExport) -> Result<u64> {
        let table = format!("relationships_{}", self.config.collection_name);
        let entity_table = format!("entities_{}", self.config.collection_name);
        let mut count = 0;
        for row in &export.relations {
            ensure_owned_reference(
                tx,
                &entity_table,
                "id",
                &row.source_id,
                &row.user_id,
                "relationship source",
            )?;
            ensure_owned_reference(
                tx,
                &entity_table,
                "id",
                &row.target_id,
                &row.user_id,
                "relationship target",
            )?;
            let affected = tx.execute(
                &format!("INSERT INTO {table} (id, source_id, target_id, relation_type, user_id, created_at, strength, context, description, episode_ids) VALUES ($1, $2, $3, $4, $5, COALESCE($6, datetime('now')), COALESCE($7, 0.5), $8, $9, $10) ON CONFLICT (id) DO NOTHING"),
                &[
                    SqlParam::Text(row.id.clone()), SqlParam::Text(row.source_id.clone()),
                    SqlParam::Text(row.target_id.clone()), SqlParam::Text(row.relation_type.clone()),
                    SqlParam::Text(row.user_id.clone()), opt_text(row.created_at.as_deref()),
                    row.strength.map(|v| SqlParam::Float(v as f64)).unwrap_or(SqlParam::Null),
                    json_param(row.context.as_ref())?, opt_text(row.description.as_deref()),
                    json_vec_param(row.episode_ids.as_ref())?,
                ],
            )?;
            if affected == 0 {
                reject_existing_record(&table, &row.id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_identity_on(
        &self,
        tx: &BackendTransaction<'_>,
        rows: &[crate::types::IdentityTrait],
        embeddings: &[Vec<f32>],
    ) -> Result<u64> {
        let mut count = 0;
        for (row, embedding) in rows.iter().zip(embeddings) {
            let literal = self.format_embedding(embedding, self.config.embedding_dims)?;
            let affected = tx.execute(
                &format!("INSERT INTO identity (trait_id, trait_type, content, content_vec, confidence, evidence_ids, user_id, created_at, updated_at) VALUES ($1, $2, $3, {literal}, $4, $5, $6, $7, $8) ON CONFLICT (trait_id) DO NOTHING"),
                &[
                    SqlParam::Text(row.trait_id.clone()),
                    SqlParam::Text(row.trait_type.as_str().to_string()),
                    SqlParam::Text(row.content.clone()), SqlParam::Float(row.confidence as f64),
                    SqlParam::Text(serde_json::to_string(&row.evidence_ids)?),
                    SqlParam::Text(row.user_id.clone()), SqlParam::Text(row.created_at.clone()),
                    opt_text(row.updated_at.as_deref()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("identity", &row.trait_id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_history_on(&self, tx: &BackendTransaction<'_>, export: &FullExport) -> Result<u64> {
        let mut count = 0;
        for row in &export.history {
            let affected = tx.execute(
                "INSERT INTO history (id, memory_id, user_id, old_memory, new_memory, event, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (id) DO NOTHING",
                &[
                    SqlParam::Text(row.id.clone()), opt_text(row.memory_id.as_deref()),
                    SqlParam::Text(row.user_id.clone()), opt_text(row.old_memory.as_deref()),
                    opt_text(row.new_memory.as_deref()), opt_text(row.event.as_deref()),
                    SqlParam::Text(row.created_at.clone()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("history", &row.id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_procedures_on(
        &self,
        tx: &BackendTransaction<'_>,
        export: &FullExport,
    ) -> Result<u64> {
        let mut count = 0;
        for row in &export.procedures {
            let affected = tx.execute(
                "INSERT INTO procedures (id, name, description, steps, user_id, trigger_pattern, confidence, usage_count, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) ON CONFLICT (id) DO NOTHING",
                &[
                    SqlParam::Text(row.id.clone()), SqlParam::Text(row.name.clone()),
                    SqlParam::Text(row.description.clone()),
                    SqlParam::Text(serde_json::to_string(&row.steps)?),
                    SqlParam::Text(row.user_id.clone()), opt_text(row.trigger.as_deref()),
                    SqlParam::Float(row.confidence as f64), SqlParam::Int(row.usage_count as i64),
                    SqlParam::Text(row.created_at.clone()), SqlParam::Text(row.updated_at.clone()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("procedures", &row.id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_meditations_on(
        &self,
        tx: &BackendTransaction<'_>,
        export: &FullExport,
    ) -> Result<u64> {
        let mut count = 0;
        for row in &export.meditations {
            let affected = tx.execute(
                "INSERT INTO meditations (meditation_id, triggered_by, started_at, finished_at, status, user_id, events_processed, episodes_created, memories_created, memories_updated, memories_decayed, entities_created, relations_created, conflicts_found, journal, metadata) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16) ON CONFLICT (meditation_id) DO NOTHING",
                &[
                    SqlParam::Text(row.meditation_id.clone()),
                    SqlParam::Text(row.triggered_by.clone()), SqlParam::Text(row.started_at.clone()),
                    opt_text(row.finished_at.as_deref()),
                    SqlParam::Text(row.status.as_str().to_string()),
                    SqlParam::Text(row.user_id.clone()), SqlParam::Int(row.events_processed as i64),
                    SqlParam::Int(row.episodes_created as i64),
                    SqlParam::Int(row.memories_created as i64),
                    SqlParam::Int(row.memories_updated as i64),
                    SqlParam::Int(row.memories_decayed as i64),
                    SqlParam::Int(row.entities_created as i64),
                    SqlParam::Int(row.relations_created as i64),
                    SqlParam::Int(row.conflicts_found as i64), opt_text(row.journal.as_deref()),
                    json_param(row.metadata.as_ref())?,
                ],
            )?;
            if affected == 0 {
                reject_existing_record("meditations", &row.meditation_id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_recalls_on(&self, tx: &BackendTransaction<'_>, export: &FullExport) -> Result<u64> {
        let mut count = 0;
        for row in &export.recalls {
            let affected = tx.execute(
                "INSERT INTO recalls (recall_id, query, timestamp, source_id, user_id, results, feedback) VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (recall_id) DO NOTHING",
                &[
                    SqlParam::Text(row.recall_id.clone()), SqlParam::Text(row.query.clone()),
                    SqlParam::Text(row.timestamp.clone()), opt_text(row.source_id.as_deref()),
                    SqlParam::Text(row.user_id.clone()), json_param(row.results.as_ref())?,
                    opt_text(row.feedback.as_deref()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("recalls", &row.recall_id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_memory_entities_on(
        &self,
        tx: &BackendTransaction<'_>,
        export: &FullExport,
    ) -> Result<u64> {
        let mut count = 0;
        let entity_table = format!("entities_{}", self.config.collection_name);
        for row in &export.memory_entities {
            ensure_owned_reference(
                tx,
                "memories",
                "id",
                &row.memory_id,
                &row.user_id,
                "memory-entity link",
            )?;
            ensure_owned_reference(
                tx,
                &entity_table,
                "id",
                &row.entity_id,
                &row.user_id,
                "memory-entity link",
            )?;
            let affected = tx.execute(
                "INSERT INTO memory_entities (memory_id, entity_id, entity_name, user_id) VALUES ($1, $2, $3, $4) ON CONFLICT (memory_id, entity_id) DO NOTHING",
                &[
                    SqlParam::Text(row.memory_id.clone()), SqlParam::Text(row.entity_id.clone()),
                    SqlParam::Text(row.entity_name.clone()), SqlParam::Text(row.user_id.clone()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record(
                    "memory_entities",
                    &format!("{}:{}", row.memory_id, row.entity_id),
                )?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn import_associations_on(
        &self,
        tx: &BackendTransaction<'_>,
        export: &FullExport,
    ) -> Result<u64> {
        let mut count = 0;
        for row in &export.associations {
            let from_user = owned_layer_user(
                tx,
                &self.config.collection_name,
                &row.from_layer,
                &row.from_id,
            )?;
            let to_user =
                owned_layer_user(tx, &self.config.collection_name, &row.to_layer, &row.to_id)?;
            if from_user != to_user {
                return Err(MemoryError::Config(format!(
                    "association '{}' crosses user ownership",
                    row.assoc_id
                )));
            }
            if export
                .user_id
                .as_deref()
                .is_some_and(|user_id| user_id != from_user)
            {
                return Err(MemoryError::Config(format!(
                    "association '{}' does not belong to exported user",
                    row.assoc_id
                )));
            }
            let affected = tx.execute(
                "INSERT INTO associations (assoc_id, from_id, from_layer, to_id, to_layer, assoc_type, strength, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) ON CONFLICT (assoc_id) DO NOTHING",
                &[
                    SqlParam::Text(row.assoc_id.clone()), SqlParam::Text(row.from_id.clone()),
                    SqlParam::Text(row.from_layer.clone()), SqlParam::Text(row.to_id.clone()),
                    SqlParam::Text(row.to_layer.clone()), SqlParam::Text(row.assoc_type.clone()),
                    SqlParam::Float(row.strength as f64), SqlParam::Text(row.created_at.clone()),
                ],
            )?;
            if affected == 0 {
                reject_existing_record("associations", &row.assoc_id)?;
            }
            count += affected as u64;
        }
        Ok(count)
    }

    fn rebuild_fts_on(&self, tx: &BackendTransaction<'_>) -> Result<()> {
        tx.execute_batch(
            &self
                .dialect()
                .create_fts_index_sql("memories", "id", &["content"]),
        )?;
        tx.execute_batch(&self.dialect().create_fts_index_sql(
            "episodes",
            "episode_id",
            &["title", "summary"],
        ))?;
        tx.execute_batch(
            "DROP TABLE IF EXISTS events_fts;
             CREATE VIRTUAL TABLE events_fts USING fts5(event_id UNINDEXED, content, tokenize='unicode61');
             INSERT INTO events_fts(event_id, content)
             SELECT event_id, COALESCE(NULLIF(purified_content, ''), content) FROM events;",
        )?;
        Ok(())
    }
}
