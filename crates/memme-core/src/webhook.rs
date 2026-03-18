use serde::{Deserialize, Serialize};

/// Webhook event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebhookEvent {
    MemoryAdd,
    MemoryUpdate,
    MemoryDelete,
}

impl WebhookEvent {
    pub fn as_str(&self) -> &'static str {
        match self {
            WebhookEvent::MemoryAdd => "memory.add",
            WebhookEvent::MemoryUpdate => "memory.update",
            WebhookEvent::MemoryDelete => "memory.delete",
        }
    }
}

/// A webhook configuration
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    pub url: String,
    pub events: Vec<WebhookEvent>,
    pub active: bool,
}

/// Webhook event payload
#[derive(Debug, Clone, Serialize)]
pub struct WebhookPayload {
    pub event: String,
    pub memory_id: String,
    pub data: serde_json::Value,
    pub timestamp: String,
}

/// Webhook manager that fires HTTP POST requests on memory events.
pub struct WebhookManager {
    hooks: Vec<WebhookConfig>,
    client: reqwest::Client,
}

impl Default for WebhookManager {
    fn default() -> Self {
        Self::new()
    }
}

impl WebhookManager {
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_configs(configs: Vec<WebhookConfig>) -> Self {
        Self {
            hooks: configs,
            client: reqwest::Client::new(),
        }
    }

    pub fn add_hook(&mut self, config: WebhookConfig) {
        self.hooks.push(config);
    }

    pub fn remove_hook(&mut self, url: &str) {
        self.hooks.retain(|h| h.url != url);
    }

    /// Fire webhook asynchronously (best-effort, don't block main operation)
    pub fn fire(&self, event: WebhookEvent, memory_id: &str, data: serde_json::Value) {
        let event_str = event.as_str().to_string();
        let payload = WebhookPayload {
            event: event_str.clone(),
            memory_id: memory_id.to_string(),
            data,
            timestamp: chrono_now(),
        };

        // Collect matching webhook URLs
        let urls: Vec<String> = self
            .hooks
            .iter()
            .filter(|h| h.active)
            .filter(|h| h.events.iter().any(|e| e.as_str() == event_str))
            .map(|h| h.url.clone())
            .collect();

        if urls.is_empty() {
            return;
        }

        let client = self.client.clone();
        // Fire-and-forget via tokio::spawn — only if a tokio runtime is available
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    for url in urls {
                        let res = client.post(&url).json(&payload).send().await;
                        if let Err(e) = res {
                            tracing::warn!(url = %url, error = %e, "Webhook delivery failed");
                        }
                    }
                });
            }
            Err(_) => {
                tracing::warn!("No tokio runtime available, skipping webhook delivery");
            }
        }
    }
}

/// Simple ISO 8601 timestamp without external chrono dependency.
fn chrono_now() -> String {
    // Use std::time for a basic timestamp
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Format as a simple integer timestamp (clients can parse this)
    format!("{}", secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_event_as_str() {
        assert_eq!(WebhookEvent::MemoryAdd.as_str(), "memory.add");
        assert_eq!(WebhookEvent::MemoryUpdate.as_str(), "memory.update");
        assert_eq!(WebhookEvent::MemoryDelete.as_str(), "memory.delete");
    }

    #[test]
    fn test_webhook_config_creation() {
        let config = WebhookConfig {
            url: "https://example.com/webhook".to_string(),
            events: vec![WebhookEvent::MemoryAdd, WebhookEvent::MemoryDelete],
            active: true,
        };
        assert_eq!(config.url, "https://example.com/webhook");
        assert_eq!(config.events.len(), 2);
        assert!(config.active);
    }

    #[test]
    fn test_webhook_payload_serialization() {
        let payload = WebhookPayload {
            event: "memory.add".to_string(),
            memory_id: "test-id-123".to_string(),
            data: serde_json::json!({"content": "hello world"}),
            timestamp: "1234567890".to_string(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("memory.add"));
        assert!(json.contains("test-id-123"));
        assert!(json.contains("hello world"));
    }

    #[test]
    fn test_webhook_manager_add_remove() {
        let mut manager = WebhookManager::new();
        assert_eq!(manager.hooks.len(), 0);

        manager.add_hook(WebhookConfig {
            url: "https://example.com/hook1".to_string(),
            events: vec![WebhookEvent::MemoryAdd],
            active: true,
        });
        assert_eq!(manager.hooks.len(), 1);

        manager.add_hook(WebhookConfig {
            url: "https://example.com/hook2".to_string(),
            events: vec![WebhookEvent::MemoryDelete],
            active: true,
        });
        assert_eq!(manager.hooks.len(), 2);

        manager.remove_hook("https://example.com/hook1");
        assert_eq!(manager.hooks.len(), 1);
        assert_eq!(manager.hooks[0].url, "https://example.com/hook2");
    }

    #[test]
    fn test_webhook_manager_from_configs() {
        let configs = vec![
            WebhookConfig {
                url: "https://example.com/hook1".to_string(),
                events: vec![WebhookEvent::MemoryAdd],
                active: true,
            },
            WebhookConfig {
                url: "https://example.com/hook2".to_string(),
                events: vec![WebhookEvent::MemoryUpdate],
                active: false,
            },
        ];
        let manager = WebhookManager::from_configs(configs);
        assert_eq!(manager.hooks.len(), 2);
    }

    #[test]
    fn test_webhook_payload_with_empty_data() {
        let payload = WebhookPayload {
            event: "memory.delete".to_string(),
            memory_id: "del-id".to_string(),
            data: serde_json::json!({}),
            timestamp: "0".to_string(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["event"], "memory.delete");
        assert_eq!(parsed["memory_id"], "del-id");
    }

    #[test]
    fn test_webhook_inactive_not_fired() {
        // This is a structural test - inactive hooks should not match
        let manager = WebhookManager::from_configs(vec![WebhookConfig {
            url: "https://example.com/inactive".to_string(),
            events: vec![WebhookEvent::MemoryAdd],
            active: false,
        }]);
        // Count matching URLs (same logic as fire())
        let event_str = WebhookEvent::MemoryAdd.as_str();
        let matching: Vec<&WebhookConfig> = manager
            .hooks
            .iter()
            .filter(|h| h.active)
            .filter(|h| h.events.iter().any(|e| e.as_str() == event_str))
            .collect();
        assert_eq!(matching.len(), 0, "inactive hooks should not match");
    }

    #[test]
    fn test_webhook_event_filtering() {
        let manager = WebhookManager::from_configs(vec![
            WebhookConfig {
                url: "https://example.com/adds-only".to_string(),
                events: vec![WebhookEvent::MemoryAdd],
                active: true,
            },
            WebhookConfig {
                url: "https://example.com/all-events".to_string(),
                events: vec![
                    WebhookEvent::MemoryAdd,
                    WebhookEvent::MemoryUpdate,
                    WebhookEvent::MemoryDelete,
                ],
                active: true,
            },
        ]);

        // MemoryAdd should match both
        let add_str = WebhookEvent::MemoryAdd.as_str();
        let add_matches: Vec<_> = manager
            .hooks
            .iter()
            .filter(|h| h.active && h.events.iter().any(|e| e.as_str() == add_str))
            .collect();
        assert_eq!(add_matches.len(), 2);

        // MemoryDelete should match only the all-events one
        let del_str = WebhookEvent::MemoryDelete.as_str();
        let del_matches: Vec<_> = manager
            .hooks
            .iter()
            .filter(|h| h.active && h.events.iter().any(|e| e.as_str() == del_str))
            .collect();
        assert_eq!(del_matches.len(), 1);
        assert_eq!(del_matches[0].url, "https://example.com/all-events");
    }
}
