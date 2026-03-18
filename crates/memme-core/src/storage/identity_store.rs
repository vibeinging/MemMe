use duckdb::params;

use crate::error::Result;
use crate::types::{IdentityTrait, TraitType};

use super::Storage;

impl Storage {
    /// Insert a new identity trait.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert_identity_trait(
        &self,
        trait_id: &str,
        trait_type: &str,
        content: &str,
        content_vec: &[f32],
        confidence: f32,
        evidence_ids: &[String],
        user_id: &str,
    ) -> Result<()> {
        let emb_literal = Self::format_embedding(content_vec, self.config.embedding_dims)?;
        let evidence_str = serde_json::to_string(evidence_ids).unwrap_or_default();

        let sql = format!(
            r#"INSERT INTO identity (trait_id, trait_type, content, content_vec, confidence, evidence_ids, user_id)
               VALUES ($1, $2, $3, {emb_literal}, $4, $5, $6)"#
        );
        let conn = self.write_conn();
        conn.execute(
            &sql,
            params![
                trait_id,
                trait_type,
                content,
                confidence as f64,
                evidence_str,
                user_id
            ],
        )?;
        Ok(())
    }

    /// Get an identity trait by ID.
    pub(crate) fn get_identity_trait(&self, trait_id: &str) -> Result<Option<IdentityTrait>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT trait_id, trait_type, content, confidence, evidence_ids, user_id,
                      CAST(created_at AS VARCHAR), CAST(updated_at AS VARCHAR)
               FROM identity WHERE trait_id = $1"#,
        )?;
        let mut rows = stmt.query_map(params![trait_id], map_identity_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// List all identity traits for a user.
    pub(crate) fn list_identity_traits(&self, user_id: &str) -> Result<Vec<IdentityTrait>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT trait_id, trait_type, content, confidence, evidence_ids, user_id,
                      CAST(created_at AS VARCHAR), CAST(updated_at AS VARCHAR)
               FROM identity WHERE user_id = $1 ORDER BY confidence DESC"#,
        )?;
        let rows = stmt
            .query_map(params![user_id], map_identity_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Update an identity trait.
    #[allow(dead_code)] // planned API: identity trait management
    pub(crate) fn update_identity_trait(
        &self,
        trait_id: &str,
        content: &str,
        content_vec: &[f32],
        confidence: f32,
        evidence_ids: &[String],
    ) -> Result<()> {
        let emb_literal = Self::format_embedding(content_vec, self.config.embedding_dims)?;
        let evidence_str = serde_json::to_string(evidence_ids).unwrap_or_default();

        let sql = format!(
            r#"UPDATE identity
               SET content = $1,
                   content_vec = {emb_literal},
                   confidence = $2,
                   evidence_ids = $3,
                   updated_at = current_timestamp
               WHERE trait_id = $4"#
        );
        let conn = self.write_conn();
        conn.execute(
            &sql,
            params![content, confidence as f64, evidence_str, trait_id],
        )?;
        Ok(())
    }

    /// Delete an identity trait.
    #[allow(dead_code)] // planned API: identity trait management
    pub(crate) fn delete_identity_trait(&self, trait_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "DELETE FROM identity WHERE trait_id = $1",
            params![trait_id],
        )?;
        Ok(())
    }

    /// Search identity traits by vector similarity.
    #[allow(dead_code)] // planned API: identity search
    pub(crate) fn search_identity_by_vector(
        &self,
        query_vec: &[f32],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<IdentityTrait>> {
        let emb_literal = Self::format_embedding(query_vec, self.config.embedding_dims)?;
        let sql = format!(
            r#"SELECT trait_id, trait_type, content, confidence, evidence_ids, user_id,
                      CAST(created_at AS VARCHAR), CAST(updated_at AS VARCHAR)
               FROM identity
               WHERE user_id = $1
               ORDER BY array_cosine_distance(content_vec, {emb_literal}) ASC
               LIMIT {limit}"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![user_id], map_identity_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

fn map_identity_row(row: &duckdb::Row<'_>) -> duckdb::Result<IdentityTrait> {
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
}
