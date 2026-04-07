use serde::{Deserialize, Serialize};

/// A procedure (skill/habit/workflow) learned from interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub id: String,
    pub name: String,
    pub description: String,
    pub steps: Vec<ProcedureStep>,
    pub user_id: String,
    /// When to invoke this procedure.
    pub trigger: Option<String>,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
    /// Number of times this procedure has been used.
    pub usage_count: u32,
    pub created_at: String,
    pub updated_at: String,
}

/// A single step in a procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureStep {
    pub order: u32,
    pub action: String,
    pub parameters: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_procedure_step_serialize() {
        let step = ProcedureStep {
            order: 1,
            action: "open_file".to_string(),
            parameters: Some(serde_json::json!({"path": "/tmp/test.txt"})),
        };
        let json = serde_json::to_string(&step).unwrap();
        assert!(json.contains("open_file"));
        assert!(json.contains("/tmp/test.txt"));
    }

    #[test]
    fn test_procedure_serialize() {
        let proc = Procedure {
            id: "proc-1".to_string(),
            name: "Daily Backup".to_string(),
            description: "Backs up important files".to_string(),
            steps: vec![
                ProcedureStep {
                    order: 1,
                    action: "compress".to_string(),
                    parameters: None,
                },
                ProcedureStep {
                    order: 2,
                    action: "upload".to_string(),
                    parameters: Some(serde_json::json!({"target": "cloud"})),
                },
            ],
            user_id: "user1".to_string(),
            trigger: Some("daily at 3am".to_string()),
            confidence: 0.8,
            usage_count: 5,
            created_at: "2026-01-01".to_string(),
            updated_at: "2026-03-17".to_string(),
        };
        let json = serde_json::to_string(&proc).unwrap();
        let deserialized: Procedure = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "Daily Backup");
        assert_eq!(deserialized.steps.len(), 2);
        assert_eq!(deserialized.trigger.as_deref(), Some("daily at 3am"));
    }
}
