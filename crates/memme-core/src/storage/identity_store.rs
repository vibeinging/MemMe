use crate::error::Result;
use crate::types::{IdentityTrait, SqlParam, TraitType};

use super::backend::RowAccess;
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
        let emb_literal = self.format_embedding(content_vec, self.config.embedding_dims)?;
        let evidence_str = serde_json::to_string(evidence_ids).unwrap_or_default();

        let sql = format!(
            r#"INSERT INTO identity (trait_id, trait_type, content, content_vec, confidence, evidence_ids, user_id)
               VALUES ($1, $2, $3, {emb_literal}, $4, $5, $6)"#
        );
        self.backend.execute(
            &sql,
            &[
                SqlParam::Text(trait_id.to_string()),
                SqlParam::Text(trait_type.to_string()),
                SqlParam::Text(content.to_string()),
                SqlParam::Float(confidence as f64),
                SqlParam::Text(evidence_str),
                SqlParam::Text(user_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Get an identity trait by ID.
    pub(crate) fn get_identity_trait(&self, trait_id: &str) -> Result<Option<IdentityTrait>> {
        self.backend.query_one(
            r#"SELECT trait_id, trait_type, content, confidence, evidence_ids, user_id,
                      created_at, updated_at
               FROM identity WHERE trait_id = $1"#,
            &[SqlParam::Text(trait_id.to_string())],
            map_identity_row,
        )
    }

    /// List all identity traits for a user.
    pub(crate) fn list_identity_traits(&self, user_id: &str) -> Result<Vec<IdentityTrait>> {
        self.backend.query_read(
            r#"SELECT trait_id, trait_type, content, confidence, evidence_ids, user_id,
                      created_at, updated_at
               FROM identity WHERE user_id = $1 ORDER BY confidence DESC"#,
            &[SqlParam::Text(user_id.to_string())],
            map_identity_row,
        )
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
        let emb_literal = self.format_embedding(content_vec, self.config.embedding_dims)?;
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
        self.backend.execute(
            &sql,
            &[
                SqlParam::Text(content.to_string()),
                SqlParam::Float(confidence as f64),
                SqlParam::Text(evidence_str),
                SqlParam::Text(trait_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Delete an identity trait.
    #[allow(dead_code)] // planned API: identity trait management
    pub(crate) fn delete_identity_trait(&self, trait_id: &str) -> Result<()> {
        self.backend.execute(
            "DELETE FROM identity WHERE trait_id = $1",
            &[SqlParam::Text(trait_id.to_string())],
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
        let emb_literal = self.format_embedding(query_vec, self.config.embedding_dims)?;
        let distance_expr = self
            .dialect()
            .cosine_distance_expr("content_vec", &emb_literal);
        let sql = format!(
            r#"SELECT trait_id, trait_type, content, confidence, evidence_ids, user_id,
                      created_at, updated_at
               FROM identity
               WHERE user_id = $1
               ORDER BY {distance_expr} ASC
               LIMIT {limit}"#
        );
        self.backend.query_read(
            &sql,
            &[SqlParam::Text(user_id.to_string())],
            map_identity_row,
        )
    }
}

fn map_identity_row(row: &dyn RowAccess) -> Result<IdentityTrait> {
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
}
