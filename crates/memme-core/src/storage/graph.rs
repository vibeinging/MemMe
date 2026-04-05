use duckdb::params;

use crate::error::Result;
use crate::types::GraphRelation;

use super::util::opt_text;
use super::Storage;

impl Storage {
    // ── Entity CRUD ──

    /// Upsert an entity: insert or update name/type if it already exists.
    pub(crate) fn upsert_entity(
        &self,
        id: &str,
        name: &str,
        entity_type: Option<&str>,
        user_id: &str,
    ) -> Result<()> {
        let collection = &self.config.collection_name;
        let type_val = opt_text(entity_type);

        // Try insert first; if conflict on id, update
        let sql = format!(
            r#"INSERT INTO entities_{collection} (id, name, entity_type, user_id)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (id) DO UPDATE SET
                   name = EXCLUDED.name,
                   entity_type = EXCLUDED.entity_type,
                   updated_at = now()::TIMESTAMP"#
        );
        let conn = self.write_conn();
        conn.execute(&sql, params![id, name, type_val, user_id])?;
        Ok(())
    }

    /// Find an entity by name (case-insensitive) for a given user.
    /// Returns `(id, name, entity_type)` if found.
    pub(crate) fn find_entity_by_name(
        &self,
        name: &str,
        user_id: &str,
    ) -> Result<Option<(String, String, Option<String>)>> {
        let collection = &self.config.collection_name;
        let sql = format!(
            r#"SELECT id, name, entity_type
               FROM entities_{collection}
               WHERE LOWER(name) = LOWER($1) AND user_id = $2
               LIMIT 1"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![name, user_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// List all entities for a given user.
    #[allow(dead_code)]
    pub(crate) fn list_entities(
        &self,
        user_id: &str,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        let collection = &self.config.collection_name;
        let sql = format!(
            r#"SELECT id, name, entity_type
               FROM entities_{collection}
               WHERE user_id = $1
               ORDER BY created_at"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![user_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Delete an entity by ID, also removing all its relationships atomically.
    #[allow(dead_code)]
    pub(crate) fn delete_entity(&self, id: &str) -> Result<()> {
        let collection = &self.config.collection_name;
        let conn = self.write_conn();
        conn.execute(
            &format!(
                "DELETE FROM relationships_{collection} WHERE source_id = $1 OR target_id = $1"
            ),
            params![id],
        )?;
        conn.execute(
            &format!("DELETE FROM entities_{collection} WHERE id = $1"),
            params![id],
        )?;
        Ok(())
    }

    // ── Relationship CRUD ──

    /// Insert a relationship between two entities (skips if duplicate).
    ///
    /// A relationship is considered duplicate if (source_id, target_id, relation_type, user_id)
    /// already exists. In that case the existing row is kept unchanged.
    pub(crate) fn insert_relationship(
        &self,
        id: &str,
        source_id: &str,
        target_id: &str,
        relation_type: &str,
        user_id: &str,
        description: Option<&str>,
    ) -> Result<()> {
        let collection = &self.config.collection_name;
        let conn = self.write_conn();

        // Check for existing duplicate
        let check_sql = format!(
            "SELECT 1 FROM relationships_{collection} WHERE source_id = $1 AND target_id = $2 AND relation_type = $3 AND user_id = $4 LIMIT 1"
        );
        let mut stmt = conn.prepare(&check_sql)?;
        let exists = stmt
            .query_map(
                params![source_id, target_id, relation_type, user_id],
                |_| Ok(()),
            )?
            .next()
            .is_some();
        if exists {
            return Ok(());
        }

        let desc_val = opt_text(description);

        let sql = format!(
            r#"INSERT INTO relationships_{collection} (id, source_id, target_id, relation_type, user_id, description)
               VALUES ($1, $2, $3, $4, $5, $6)"#
        );
        conn.execute(
            &sql,
            params![id, source_id, target_id, relation_type, user_id, desc_val],
        )?;
        Ok(())
    }

    /// Find all relationships involving an entity (as source or target).
    #[allow(dead_code)]
    pub(crate) fn find_relationships(
        &self,
        entity_id: &str,
        user_id: &str,
    ) -> Result<Vec<GraphRelation>> {
        let collection = &self.config.collection_name;
        let sql = format!(
            r#"SELECT r.id, r.source_id, r.target_id, r.relation_type, r.user_id,
                      s.name AS source_name, t.name AS target_name, r.description
               FROM relationships_{collection} r
               JOIN entities_{collection} s ON r.source_id = s.id
               JOIN entities_{collection} t ON r.target_id = t.id
               WHERE (r.source_id = $1 OR r.target_id = $1) AND r.user_id = $2"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![entity_id, user_id], |row| {
                Ok(GraphRelation {
                    id: row.get(0)?,
                    source_id: row.get(1)?,
                    target_id: row.get(2)?,
                    relation_type: row.get(3)?,
                    user_id: row.get(4)?,
                    source: row.get(5)?,
                    target: row.get(6)?,
                    description: row.get::<_, Option<String>>(7)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Delete all relationships involving an entity (as source or target).
    #[allow(dead_code)]
    pub(crate) fn delete_relationships_for_entity(&self, entity_id: &str) -> Result<()> {
        let collection = &self.config.collection_name;
        let sql = format!(
            "DELETE FROM relationships_{collection} WHERE source_id = $1 OR target_id = $1"
        );
        let conn = self.write_conn();
        conn.execute(&sql, params![entity_id])?;
        Ok(())
    }

    /// Delete all entities and relationships for a user.
    #[allow(dead_code)] // planned API: user data cleanup
    pub(crate) fn delete_user_entities(&self, user_id: &str) -> Result<()> {
        let collection = &self.config.collection_name;
        let conn = self.write_conn();
        conn.execute(
            &format!("DELETE FROM relationships_{collection} WHERE user_id = $1"),
            params![user_id],
        )?;
        conn.execute(
            &format!("DELETE FROM entities_{collection} WHERE user_id = $1"),
            params![user_id],
        )?;
        conn.execute(
            "DELETE FROM memory_entities WHERE user_id = $1",
            params![user_id],
        )?;
        Ok(())
    }

    // ── Graph Search ──

    /// Search entities by name using SQL LIKE (case-insensitive).
    pub(crate) fn search_entities_by_name(
        &self,
        query: &str,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        let collection = &self.config.collection_name;
        let escaped = query.replace('%', "\\%").replace('_', "\\_");
        let pattern = format!("%{escaped}%");
        let sql = format!(
            r#"SELECT id, name, entity_type
               FROM entities_{collection}
               WHERE LOWER(name) LIKE LOWER($1) ESCAPE '\' AND user_id = $2
               LIMIT {limit}"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![pattern, user_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get the neighborhood of an entity up to `depth` hops using a recursive CTE.
    pub(crate) fn get_entity_neighborhood(
        &self,
        entity_id: &str,
        user_id: &str,
        depth: usize,
    ) -> Result<Vec<GraphRelation>> {
        let depth = depth.min(10);
        let collection = &self.config.collection_name;
        let sql = format!(
            r#"WITH RECURSIVE neighborhood AS (
                -- Base case: relationships directly connected to the starting entity
                SELECT r.id, r.source_id, r.target_id, r.relation_type, r.user_id, r.description, 1 AS depth
                FROM relationships_{collection} r
                WHERE (r.source_id = $1 OR r.target_id = $1) AND r.user_id = $2

                UNION

                -- Recursive case: expand from the frontier only
                SELECT r.id, r.source_id, r.target_id, r.relation_type, r.user_id, r.description, n.depth + 1
                FROM relationships_{collection} r
                JOIN neighborhood n ON (
                    r.source_id = n.target_id OR r.target_id = n.source_id
                )
                WHERE r.user_id = $2
                AND n.depth < {depth}
                AND r.id != n.id
            )
            SELECT DISTINCT nb.id, nb.source_id, nb.target_id, nb.relation_type, nb.user_id,
                   s.name AS source_name, t.name AS target_name, nb.description
            FROM neighborhood nb
            JOIN entities_{collection} s ON nb.source_id = s.id
            JOIN entities_{collection} t ON nb.target_id = t.id
            LIMIT 1000"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![entity_id, user_id], |row| {
                Ok(GraphRelation {
                    id: row.get(0)?,
                    source_id: row.get(1)?,
                    target_id: row.get(2)?,
                    relation_type: row.get(3)?,
                    user_id: row.get(4)?,
                    source: row.get(5)?,
                    target: row.get(6)?,
                    description: row.get::<_, Option<String>>(7)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}
