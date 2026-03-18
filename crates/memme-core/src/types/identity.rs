use serde::{Deserialize, Serialize};

/// Classification of an identity trait, representing what aspect of the user it describes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TraitType {
    /// A role the user plays (e.g., "software engineer", "parent").
    Role,
    /// A core belief or conviction.
    Belief,
    /// A personal value or principle.
    Value,
    /// A communication or behavioral style.
    Style,
    /// A goal or aspiration.
    Goal,
}

impl TraitType {
    /// Return the string representation of this trait type.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Role => "role",
            Self::Belief => "belief",
            Self::Value => "value",
            Self::Style => "style",
            Self::Goal => "goal",
        }
    }
    /// Parse a trait type from its string representation (case-insensitive).
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "role" => Self::Role,
            "belief" => Self::Belief,
            "value" => Self::Value,
            "style" => Self::Style,
            "goal" => Self::Goal,
            _ => Self::Goal,
        }
    }
}

/// An identity trait -- the highest-level abstraction about the user.
///
/// Identity traits are distilled from many memories and represent stable
/// personality characteristics, goals, or values. Confidence grows slowly
/// as more evidence accumulates.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityTrait {
    /// Unique trait ID (UUID v4).
    pub trait_id: String,
    /// What aspect of the user this trait describes.
    pub trait_type: TraitType,
    /// Human-readable description of the trait.
    pub content: String,
    /// Confidence score (0.0-1.0). Grows slowly as evidence accumulates.
    pub confidence: f32,
    /// Memory IDs that serve as evidence for this trait.
    pub evidence_ids: Vec<String>,
    /// Owner of this trait.
    pub user_id: String,
    /// When this trait was first created (ISO 8601).
    pub created_at: String,
    /// When this trait was last updated (ISO 8601).
    pub updated_at: Option<String>,
}

/// Options for adding or updating an identity trait.
#[derive(Debug, Clone)]
pub struct AddIdentityTraitOptions {
    /// Trait type string (e.g., "role", "belief", "value", "style", "goal").
    pub trait_type: String,
    /// Human-readable description of the trait.
    pub content: String,
    /// Owner of this trait (required).
    pub user_id: String,
    /// Initial confidence score (0.0-1.0). Defaults to a low value.
    pub confidence: Option<f32>,
    /// Memory IDs that serve as evidence for this trait.
    pub evidence_ids: Vec<String>,
}

impl AddIdentityTraitOptions {
    /// Create new identity trait options with required fields.
    pub fn new(
        trait_type: impl Into<String>,
        content: impl Into<String>,
        user_id: impl Into<String>,
    ) -> Self {
        Self {
            trait_type: trait_type.into(),
            content: content.into(),
            user_id: user_id.into(),
            confidence: None,
            evidence_ids: Vec::new(),
        }
    }
    /// Set the initial confidence score.
    pub fn confidence(mut self, v: f32) -> Self {
        self.confidence = Some(v);
        self
    }
    /// Set the memory IDs that serve as evidence.
    pub fn evidence_ids(mut self, v: Vec<String>) -> Self {
        self.evidence_ids = v;
        self
    }
}
