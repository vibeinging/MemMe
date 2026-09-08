//! Ollama LLM provider.
//!
//! Connects to a local Ollama instance via its HTTP API (`/api/chat`).
//! Requires the `ollama` feature to be enabled (on by default).
//!
//! Uses `reqwest::blocking` for synchronous HTTP — no tokio runtime required.

use crate::error::LlmError;
use crate::{GenerateOptions, LlmProvider, Message, MessageRole, ResponseFormat};
use serde::Deserialize;

/// Configuration for the Ollama provider.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub host: String,
    pub model: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost:11434".to_string(),
            model: "llama3.2".to_string(),
        }
    }
}

/// An LLM provider backed by a local Ollama instance.
pub struct OllamaProvider {
    config: OllamaConfig,
    client: crate::http_client::SafeBlockingClient,
}

impl OllamaProvider {
    pub fn new(config: OllamaConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .connect_timeout(std::time::Duration::from_secs(10))
            .tcp_keepalive(std::time::Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("valid Ollama HTTP client configuration");
        Self {
            config,
            client: client.into(),
        }
    }
}

impl LlmProvider for OllamaProvider {
    fn generate(
        &self,
        messages: &[Message],
        options: &GenerateOptions,
    ) -> Result<String, LlmError> {
        let url = format!("{}/api/chat", self.config.host);

        let ollama_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        MessageRole::System => "system",
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                    },
                    "content": m.content,
                })
            })
            .collect();

        let mut ollama_options = serde_json::Map::new();
        if let Some(temp) = options.temperature {
            ollama_options.insert(
                "temperature".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(temp as f64)
                        .unwrap_or(serde_json::Number::from(0)),
                ),
            );
        }
        if let Some(max_tokens) = options.max_tokens {
            ollama_options.insert(
                "num_predict".to_string(),
                serde_json::Value::Number(serde_json::Number::from(max_tokens)),
            );
        }

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": false,
            "think": false,
            "options": ollama_options,
        });

        if let Some(ResponseFormat::Json) = options.response_format {
            body["format"] = serde_json::json!("json");
        }

        tracing::debug!(url = %url, model = %self.config.model, "Ollama request");

        // Retry with backoff
        let mut last_err = LlmError::RequestFailed("no attempts".into());
        for attempt in 0..3u64 {
            if attempt > 0 {
                tracing::warn!(attempt, "Retrying Ollama request after: {last_err}");
                std::thread::sleep(std::time::Duration::from_secs(2u64.pow(attempt as u32)));
            }

            match self.client.post(&url).json(&body).send() {
                Ok(resp) if resp.status().is_success() => match resp.json::<OllamaChatResponse>() {
                    Ok(r) => return Ok(r.message.content),
                    Err(e) => {
                        last_err = LlmError::ParseError(e.to_string());
                        continue;
                    }
                },
                Ok(resp) => {
                    let status = resp.status();
                    let text = resp.text().unwrap_or_default();
                    last_err = LlmError::RequestFailed(format!("Ollama HTTP {status}: {text}"));
                    // Only retry on 5xx server errors; 4xx are permanent
                    if status.is_client_error() {
                        return Err(last_err);
                    }
                    continue;
                }
                Err(e) => {
                    last_err = if e.is_timeout() {
                        LlmError::Timeout
                    } else {
                        LlmError::RequestFailed(e.to_string())
                    };
                    continue;
                }
            }
        }

        Err(last_err)
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

// ── Ollama API types ──

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponseMessage {
    content: String,
}
