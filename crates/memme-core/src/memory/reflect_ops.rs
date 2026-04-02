use std::sync::Arc;

use memme_llm::prompts::{
    get_feedback_messages, get_reflect_messages, parse_feedback_response, parse_reflect_response,
    FeedbackItem, LearnedPrinciple, ReflectResponse,
};
use memme_llm::{generate_structured, LlmProvider, ResponseFormat, StructuredGenConfig};

use crate::error::{MemoryError, Result};
use crate::types::*;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Options for generating a reflection.
#[derive(Debug, Clone)]
pub struct ReflectOptions {
    /// Owner whose memories to reflect on (required).
    pub user_id: String,
    /// Maximum number of recent memories to include.
    pub limit: usize,
    /// Optional additional context (e.g., "focus on work-related memories").
    pub context: Option<String>,
}

impl ReflectOptions {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            limit: 50,
            context: None,
        }
    }

    pub fn limit(mut self, v: usize) -> Self {
        self.limit = v;
        self
    }

    pub fn context(mut self, v: impl Into<String>) -> Self {
        self.context = Some(v.into());
        self
    }
}

/// Result of a reflection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReflectResult {
    /// The generated reflection text.
    pub reflection: String,
    /// Key themes identified.
    pub themes: Vec<String>,
    /// Suggested areas of focus.
    pub focus_suggestions: Vec<String>,
    /// Number of memories considered.
    pub memories_considered: usize,
}

/// Options for learning from feedback.
#[derive(Debug, Clone)]
pub struct LearnFromFeedbackOptions {
    /// Owner (required).
    pub user_id: String,
    /// Corrections/feedback items to learn from.
    pub feedback: Vec<FeedbackItem>,
}

impl LearnFromFeedbackOptions {
    pub fn new(user_id: impl Into<String>, feedback: Vec<FeedbackItem>) -> Self {
        Self {
            user_id: user_id.into(),
            feedback,
        }
    }
}

/// Result of learning from feedback.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LearnFromFeedbackResult {
    /// Principles extracted from feedback.
    pub principles: Vec<LearnedPrinciple>,
    /// Number of principles stored as memories.
    pub memories_created: usize,
    /// Number of identity traits updated.
    pub traits_updated: usize,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl super::MemoryStore {
    /// Generate a reflection based on recent memories.
    ///
    /// Requires an LLM provider to be configured.
    /// Retrieves recent memories and identity traits, then asks the LLM
    /// to synthesize a thoughtful reflection.
    pub fn reflect(&self, options: ReflectOptions) -> Result<ReflectResult> {
        let llm = self
            .llm()
            .ok_or_else(|| MemoryError::Llm("No LLM provider configured".to_string()))?;

        // Gather recent memories
        let list_opts = ListOptions::new(&options.user_id).limit(options.limit);
        let memories = self.list_traces(list_opts)?;
        if memories.is_empty() {
            return Ok(ReflectResult {
                reflection: "还没有足够的记忆来生成反思。".to_string(),
                themes: vec![],
                focus_suggestions: vec![],
                memories_considered: 0,
            });
        }

        let memory_texts: Vec<String> = memories.iter().map(|m| m.content.clone()).collect();

        // Gather identity traits
        let traits = self
            .list_identity_traits(&options.user_id)
            .unwrap_or_default();
        let trait_texts: Vec<String> = traits
            .iter()
            .map(|t| format!("[{}] {}", t.trait_type.as_str(), t.content))
            .collect();

        // Call LLM
        let messages =
            get_reflect_messages(&memory_texts, &trait_texts, options.context.as_deref());

        let config = StructuredGenConfig {
            base_temperature: Some(0.3),
            max_tokens: Some(2048),
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };

        let response = generate_structured(llm.as_ref(), &messages, &config, |raw| {
            parse_reflect_response(raw)
        })
        .map_err(|e| MemoryError::Llm(e.to_string()))?;

        Ok(ReflectResult {
            reflection: response.reflection,
            themes: response.themes,
            focus_suggestions: response.focus_suggestions,
            memories_considered: memories.len(),
        })
    }

    /// Learn from user feedback/corrections.
    ///
    /// Takes a list of corrections and synthesizes them into behavioral
    /// principles using LLM. Each principle is stored as a high-importance
    /// memory and optionally as an identity trait (style type).
    pub fn learn_from_feedback(
        &self,
        options: LearnFromFeedbackOptions,
    ) -> Result<LearnFromFeedbackResult> {
        if options.feedback.is_empty() {
            return Ok(LearnFromFeedbackResult {
                principles: vec![],
                memories_created: 0,
                traits_updated: 0,
            });
        }

        let llm = self
            .llm()
            .ok_or_else(|| MemoryError::Llm("No LLM provider configured".to_string()))?;

        // Call LLM to synthesize principles
        let messages = get_feedback_messages(&options.feedback);

        let config = StructuredGenConfig {
            base_temperature: Some(0.1),
            max_tokens: Some(2048),
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };

        let response = generate_structured(llm.as_ref(), &messages, &config, |raw| {
            parse_feedback_response(raw)
        })
        .map_err(|e| MemoryError::Llm(e.to_string()))?;

        let mut memories_created = 0;
        let mut traits_updated = 0;

        for principle in &response.principles {
            // Store as high-importance memory
            let add_opts = AddOptions::new(&options.user_id)
                .importance(0.9)
                .categories(vec!["principle".to_string()]);

            match self.add(&principle.content, add_opts) {
                Ok(_) => memories_created += 1,
                Err(e) => tracing::warn!("Failed to store principle as memory: {e}"),
            }

            // Store high-confidence principles as identity traits (style)
            if principle.confidence >= 0.7 {
                let trait_opts = AddIdentityTraitOptions::new(
                    TraitType::Style.as_str(),
                    &principle.content,
                    &options.user_id,
                )
                .confidence(principle.confidence);

                match self.add_identity_trait(trait_opts) {
                    Ok(_) => traits_updated += 1,
                    Err(e) => tracing::warn!("Failed to store principle as trait: {e}"),
                }
            }
        }

        Ok(LearnFromFeedbackResult {
            principles: response.principles,
            memories_created,
            traits_updated,
        })
    }
}
