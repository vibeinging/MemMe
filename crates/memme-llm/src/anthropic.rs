//! Anthropic Claude LLM provider.
//!
//! Connects to the Anthropic Messages API (`/v1/messages`).
//! Auth via `x-api-key` header (not Bearer token).
//!
//! Uses `reqwest::blocking` for synchronous HTTP — no tokio runtime required.

use crate::error::LlmError;
use crate::{GenerateOptions, LlmProvider, Message, MessageRole, ResponseFormat};
use serde::Deserialize;

/// Configuration for the Anthropic provider.
#[derive(Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl std::fmt::Debug for AnthropicConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish()
    }
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

/// An LLM provider backed by the Anthropic Messages API.
pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: reqwest::blocking::Client,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(90))
            .connect_timeout(std::time::Duration::from_secs(10))
            .tcp_keepalive(std::time::Duration::from_secs(15))
            .pool_max_idle_per_host(5)
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self { config, client }
    }

    fn do_request(&self, url: &str, body: &serde_json::Value) -> Result<String, LlmError> {
        let response = self
            .client
            .post(url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(body)
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    LlmError::Timeout
                } else if e.is_connect() {
                    LlmError::RequestFailed(format!("Connection failed: {e}"))
                } else {
                    LlmError::RequestFailed(format!("Network error: {e}"))
                }
            })?;

        let status = response.status();
        if status.is_success() {
            return response
                .text()
                .map_err(|e| LlmError::ParseError(format!("Failed to read body: {e}")));
        }

        let body_text = response.text().unwrap_or_default();

        if status.as_u16() == 429 {
            Err(LlmError::RequestFailed(format!(
                "Rate limited (429): {body_text}"
            )))
        } else if status.as_u16() == 400 && body_text.contains("max_tokens") {
            Err(LlmError::InvalidFormat(format!(
                "Context length exceeded: {body_text}"
            )))
        } else if status.is_server_error() || status.as_u16() == 529 {
            Err(LlmError::RequestFailed(format!(
                "Server error ({status}): {body_text}"
            )))
        } else if status.as_u16() == 401 || status.as_u16() == 403 {
            Err(LlmError::ConfigError(format!(
                "Authentication failed ({status}): check your API key"
            )))
        } else {
            Err(LlmError::RequestFailed(format!(
                "HTTP {status}: {body_text}"
            )))
        }
    }

    fn parse_response(&self, body: &str) -> Result<String, LlmError> {
        let resp: AnthropicResponse = serde_json::from_str(body)
            .map_err(|e| LlmError::ParseError(format!("Invalid JSON response: {e}")))?;

        resp.content
            .into_iter()
            .find(|b| b.block_type == "text")
            .map(|b| b.text)
            .ok_or_else(|| LlmError::ParseError("No text block in response".to_string()))
    }

    fn build_body(&self, messages: &[Message], options: &GenerateOptions) -> serde_json::Value {
        // Anthropic separates system prompt from messages
        let system: Option<String> = messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .map(|m| m.content.clone())
            .reduce(|a, b| format!("{a}\n\n{b}"));

        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                        MessageRole::System => unreachable!(),
                    },
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": options.max_tokens.unwrap_or(4096),
        });

        if let Some(ref sys) = system {
            body["system"] = serde_json::json!(sys);
        }
        if let Some(temp) = options.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        body
    }

    fn is_retriable(err: &LlmError) -> bool {
        matches!(err, LlmError::RequestFailed(_) | LlmError::Timeout)
    }
}

impl LlmProvider for AnthropicProvider {
    fn generate(
        &self,
        messages: &[Message],
        options: &GenerateOptions,
    ) -> Result<String, LlmError> {
        let url = format!("{}/v1/messages", self.config.base_url);
        let body = self.build_body(messages, options);

        tracing::debug!(url = %url, model = %self.config.model, "Anthropic request");

        let mut last_err = LlmError::RequestFailed("no attempts made".into());
        for attempt in 0..3u64 {
            if attempt > 0 {
                let wait = std::time::Duration::from_secs(2u64.pow(attempt as u32));
                tracing::warn!(attempt, "Retrying Anthropic request after: {last_err}");
                std::thread::sleep(wait);
            }

            match self.do_request(&url, &body) {
                Ok(resp_body) => match self.parse_response(&resp_body) {
                    Ok(content) => return Ok(content),
                    Err(e) => {
                        tracing::warn!("Anthropic response parse error: {e}");
                        last_err = e;
                        continue;
                    }
                },
                Err(e) if Self::is_retriable(&e) => {
                    last_err = e;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_err)
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

// ── Anthropic API types ──

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}
