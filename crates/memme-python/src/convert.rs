//! Python dict conversion helpers for MemMe types.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert a MemoryResult to a Python dict.
pub(crate) fn memory_result_to_dict(
    py: Python<'_>,
    r: &memme_core::types::MemoryResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("id", &r.id)?;
    dict.set_item("content", &r.content)?;
    dict.set_item("user_id", &r.user_id)?;
    dict.set_item("agent_id", &r.agent_id)?;
    dict.set_item("app_id", &r.app_id)?;
    dict.set_item("run_id", &r.run_id)?;
    dict.set_item("score", r.score)?;
    dict.set_item("created_at", &r.created_at)?;
    dict.set_item("updated_at", &r.updated_at)?;
    match &r.metadata {
        Some(meta) => {
            dict.set_item("metadata", meta.to_string())?;
        }
        None => {
            dict.set_item("metadata", py.None())?;
        }
    }
    dict.set_item("importance", r.importance)?;
    dict.set_item("access_count", r.access_count)?;
    dict.set_item("immutable", r.immutable)?;
    dict.set_item("expiration_date", &r.expiration_date)?;
    match &r.categories {
        Some(cats) => {
            dict.set_item("categories", cats.clone())?;
        }
        None => {
            dict.set_item("categories", py.None())?;
        }
    }
    dict.set_item("memory_type", &r.memory_type)?;
    dict.set_item("retention", r.retention)?;
    dict.set_item("stability", r.stability)?;
    dict.set_item("privacy", &r.privacy)?;
    dict.set_item("event_time", &r.event_time)?;
    dict.set_item("episode_id", &r.episode_id)?;
    dict.set_item("session_id", &r.session_id)?;
    Ok(dict.into())
}

/// Convert a HistoryRecord to a Python dict.
pub(crate) fn history_record_to_dict(
    py: Python<'_>,
    r: &memme_core::types::HistoryRecord,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("id", &r.id)?;
    dict.set_item("memory_id", &r.memory_id)?;
    dict.set_item("old_memory", &r.old_memory)?;
    dict.set_item("new_memory", &r.new_memory)?;
    dict.set_item("event", &r.event)?;
    dict.set_item("created_at", &r.created_at)?;
    Ok(dict.into())
}

/// Convert a GraphSearchResult to a Python dict.
pub(crate) fn graph_result_to_dict(
    py: Python<'_>,
    r: &memme_core::types::GraphSearchResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    let entities: Vec<PyObject> = r
        .entities
        .iter()
        .map(|e| {
            let d = PyDict::new(py);
            d.set_item("id", &e.id)?;
            d.set_item("name", &e.name)?;
            d.set_item("entity_type", &e.entity_type)?;
            d.set_item("user_id", &e.user_id)?;
            Ok(d.into())
        })
        .collect::<PyResult<Vec<_>>>()?;

    let relations: Vec<PyObject> = r
        .relations
        .iter()
        .map(|rel| {
            let d = PyDict::new(py);
            d.set_item("id", &rel.id)?;
            d.set_item("source", &rel.source)?;
            d.set_item("source_id", &rel.source_id)?;
            d.set_item("target", &rel.target)?;
            d.set_item("target_id", &rel.target_id)?;
            d.set_item("relation_type", &rel.relation_type)?;
            d.set_item("user_id", &rel.user_id)?;
            Ok(d.into())
        })
        .collect::<PyResult<Vec<_>>>()?;

    dict.set_item("entities", entities)?;
    dict.set_item("relations", relations)?;
    Ok(dict.into())
}

/// Convert an Episode to a Python dict.
pub(crate) fn episode_to_dict(
    py: Python<'_>,
    ep: &memme_core::types::Episode,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("episode_id", &ep.episode_id)?;
    dict.set_item("title", &ep.title)?;
    dict.set_item("summary", &ep.summary)?;
    dict.set_item("started_at", &ep.started_at)?;
    dict.set_item("ended_at", &ep.ended_at)?;
    dict.set_item("significance", ep.significance)?;
    dict.set_item("outcome", &ep.outcome)?;
    dict.set_item("source_id", &ep.source_id)?;
    dict.set_item("event_ids", &ep.event_ids)?;
    dict.set_item("user_id", &ep.user_id)?;
    dict.set_item("created_at", &ep.created_at)?;
    dict.set_item("recall_count", ep.recall_count)?;
    dict.set_item("score", ep.score)?;
    Ok(dict.into())
}

/// Convert an Event to a Python dict (all fields).
pub(crate) fn event_to_dict(py: Python<'_>, ev: &memme_core::types::Event) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("event_id", &ev.event_id)?;
    dict.set_item("event_type", ev.event_type.as_str())?;
    dict.set_item("content", &ev.content)?;
    dict.set_item("timestamp", &ev.timestamp)?;
    dict.set_item("session_id", &ev.session_id)?;
    dict.set_item("user_id", &ev.user_id)?;
    dict.set_item("purified_content", &ev.purified_content)?;
    dict.set_item("purified", ev.purified)?;
    dict.set_item("event_time", &ev.event_time)?;
    dict.set_item("location", &ev.location)?;
    Ok(dict.into())
}
