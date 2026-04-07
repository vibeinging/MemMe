use uuid::Uuid;

use crate::error::{MemoryError, Result};
use crate::types::*;

impl super::MemoryStore {
    /// Add or update an identity trait.
    pub fn add_identity_trait(&self, options: AddIdentityTraitOptions) -> Result<IdentityTrait> {
        let trait_id = Uuid::new_v4().to_string();
        let embedding = self
            .embedder
            .embed(&options.content)
            .map_err(MemoryError::Embedding)?;
        let confidence = options.confidence.unwrap_or(0.5);

        self.storage.insert_identity_trait(
            &trait_id,
            &options.trait_type,
            &options.content,
            &embedding,
            confidence,
            &options.evidence_ids,
            &options.user_id,
        )?;
        self.storage
            .get_identity_trait(&trait_id)?
            .ok_or_else(|| MemoryError::NotFound(trait_id))
    }

    /// List all identity traits for a user.
    pub fn list_identity_traits(&self, user_id: &str) -> Result<Vec<IdentityTrait>> {
        self.storage.list_identity_traits(user_id)
    }

    /// Search identity traits by semantic similarity.
    #[allow(dead_code)] // planned API: identity search
    pub(crate) fn search_identity(
        &self,
        query: &str,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<IdentityTrait>> {
        let embedding = self.embedder.embed(query).map_err(MemoryError::Embedding)?;
        self.storage
            .search_identity_by_vector(&embedding, user_id, limit)
    }
}
