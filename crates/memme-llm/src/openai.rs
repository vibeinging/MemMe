//! OpenAI-compatible LLM provider.
//!
//! Connects to the OpenAI chat completions API (`/v1/chat/completions`).
//! Also works with any OpenAI-compatible API (e.g., DashScope, vLLM, LM Studio)
//! by configuring `base_url`.
//!
//! Uses `reqwest::blocking` for synchronous HTTP — no tokio runtime required,
//! safe to call from Python/Node FFI contexts.
//!
//! Three-layer error handling:
//! - Transport: 429/5xx → exponential backoff retry
//! - Parse: invalid JSON → deterministic repair (via `try_repair_json`)
//! - Content: context_length_exceeded → permanent error, no retry

use crate::error::LlmError;
use crate::{GenerateOptions, LlmProvider, Message, MessageRole, ResponseFormat};
use serde::Deserialize;

/// Protocol adapter for custom LLM chat completions API formats.
///
/// Implement this trait to support non-OpenAI chat API formats.
/// The HTTP transport layer (retry, backoff, auth, error classification) is
/// handled by `OpenAIProvider`; you only define request/response format.
///
/// # Example
/// ```no_run
/// use memme_llm::openai::ChatProtocol;
/// use memme_llm::{Message, GenerateOptions, LlmError};
///
/// struct CustomLlmFormat;
///
/// impl ChatProtocol for CustomLlmFormat {
///     fn build_request(&self, model: &str, messages: &[Message], options: &GenerateOptions) -> serde_json::Value {
///         serde_json::json!({
///             "model": model,
///             "prompt": messages.last().map(|m| m.content.as_str()).unwrap_or(""),
///         })
///     }
///
///     fn parse_response(&self, raw: &str) -> Result<String, LlmError> {
///         let v: serde_json::Value = serde_json::from_str(raw)
///             .map_err(|e| LlmError::ParseError(e.to_string()))?;
///         v["output"]["text"].as_str()
///             .map(|s| s.to_string())
///             .ok_or_else(|| LlmError::ParseError("missing output.text".into()))
///     }
/// }
/// ```
pub trait ChatProtocol: Send + Sync {
    /// Build the JSON request body for the chat completions API.
    fn build_request(
        &self,
        model: &str,
        messages: &[Message],
        options: &GenerateOptions,
    ) -> serde_json::Value;

    /// Parse the raw response body into the generated text content.
    fn parse_response(&self, raw: &str) -> Result<String, LlmError>;
}

/// Configuration for the OpenAI provider.
#[derive(Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl std::fmt::Debug for OpenAIConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish()
    }
}

impl OpenAIConfig {
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }
}

/// An LLM provider backed by the OpenAI chat completions API.
pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: reqwest::blocking::Client,
    /// Custom protocol adapter for non-OpenAI API formats.
    protocol: Option<Box<dyn ChatProtocol>>,
}

impl OpenAIProvider {
    pub fn new(mut config: OpenAIConfig) -> Self {
        config.base_url = config.base_url.trim_end_matches('/').to_string();
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .connect_timeout(std::time::Duration::from_secs(15))
            .pool_max_idle_per_host(5)
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self {
            config,
            client,
            protocol: None,
        }
    }

    /// Set a custom protocol adapter for non-OpenAI API formats.
    ///
    /// When set, the default OpenAI request/response format is bypassed entirely.
    /// The HTTP transport (retry, backoff, auth, error classification) remains unchanged.
    pub fn with_protocol(mut self, protocol: impl ChatProtocol + 'static) -> Self {
        self.protocol = Some(Box::new(protocol));
        self
    }

    /// Make a single HTTP request and classify the result.
    fn do_request(&self, url: &str, body: &serde_json::Value) -> Result<String, LlmError> {
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
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
        } else if status.as_u16() == 400 && body_text.contains("context_length_exceeded") {
            Err(LlmError::InvalidFormat(format!(
                "Context length exceeded: {body_text}"
            )))
        } else if status.as_u16() == 400 && body_text.contains("content_filter") {
            Err(LlmError::InvalidFormat(format!(
                "Content filtered: {body_text}"
            )))
        } else if status.is_server_error() {
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
        if let Some(ref proto) = self.protocol {
            return proto.parse_response(body);
        }

        let resp: OpenAIChatCompletionResponse = serde_json::from_str(body)
            .map_err(|e| LlmError::ParseError(format!("Invalid JSON response: {e}")))?;

        resp.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| LlmError::ParseError("No choices in response".to_string()))
    }

    fn is_retriable(err: &LlmError) -> bool {
        matches!(err, LlmError::RequestFailed(_) | LlmError::Timeout)
    }

    fn build_body(&self, messages: &[Message], options: &GenerateOptions) -> serde_json::Value {
        let openai_messages: Vec<serde_json::Value> = messages
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

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": openai_messages,
        });

        if let Some(temp) = options.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(ResponseFormat::Json) = options.response_format {
            body["response_format"] = serde_json::json!({"type": "json_object"});
        }

        body
    }
}

impl LlmProvider for OpenAIProvider {
    fn generate(
        &self,
        messages: &[Message],
        options: &GenerateOptions,
    ) -> Result<String, LlmError> {
        let url: &str = &self.config.base_url;
        let body = if let Some(ref proto) = self.protocol {
            proto.build_request(&self.config.model, messages, options)
        } else {
            self.build_body(messages, options)
        };

        tracing::debug!(url = %url, model = %self.config.model, "LLM request");

        let mut last_err = LlmError::RequestFailed("no attempts made".into());
        for attempt in 0..3u64 {
            if attempt > 0 {
                let wait = std::time::Duration::from_secs(2u64.pow(attempt as u32));
                tracing::warn!(attempt, "Retrying LLM request after: {last_err}");
                std::thread::sleep(wait);
            }

            match self.do_request(url, &body) {
                Ok(resp_body) => match self.parse_response(&resp_body) {
                    Ok(content) => return Ok(content),
                    Err(e) => {
                        tracing::warn!("LLM response parse error: {e}");
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
        "openai"
    }
}

// ── OpenAI API types ──

#[derive(Debug, Deserialize)]
struct OpenAIChatCompletionResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoiceMessage {
    content: String,
}
