//! Google Gemini LLM provider.
//!
//! Connects to the Gemini generateContent API.
//! Auth via `x-goog-api-key` header.
//!
//! Uses `reqwest::blocking` for synchronous HTTP — no tokio runtime required.

use crate::error::LlmError;
use crate::{GenerateOptions, LlmProvider, Message, MessageRole, ResponseFormat};
use serde::Deserialize;

/// Configuration for the Gemini provider.
#[derive(Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl std::fmt::Debug for GeminiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish()
    }
}

impl GeminiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
            model: "gemini-2.0-flash".to_string(),
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

/// An LLM provider backed by the Google Gemini API.
pub struct GeminiProvider {
    config: GeminiConfig,
    client: crate::http_client::SafeBlockingClient,
}

impl GeminiProvider {
    pub fn new(config: GeminiConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(90))
            .connect_timeout(std::time::Duration::from_secs(10))
            .tcp_keepalive(std::time::Duration::from_secs(15))
            .pool_max_idle_per_host(5)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("valid Gemini HTTP client configuration");
        Self {
            config,
            client: client.into(),
        }
    }

    fn do_request(&self, url: &str, body: &serde_json::Value) -> Result<String, LlmError> {
        let response = self
            .client
            .post(url)
            .header("x-goog-api-key", &self.config.api_key)
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

        let _body_text = response.text().unwrap_or_default();

        if status.as_u16() == 429 {
            Err(LlmError::RequestFailed("Rate limited (429)".to_string()))
        } else if status.as_u16() == 400 {
            Err(LlmError::InvalidFormat("Bad request".to_string()))
        } else if status.is_server_error() {
            Err(LlmError::RequestFailed(format!("Server error ({status})")))
        } else if status.as_u16() == 401 || status.as_u16() == 403 {
            Err(LlmError::ConfigError(format!(
                "Authentication failed ({status}): check your API key"
            )))
        } else {
            Err(LlmError::RequestFailed(format!("HTTP {status}")))
        }
    }

    fn parse_response(&self, body: &str) -> Result<String, LlmError> {
        let resp: GeminiResponse = serde_json::from_str(body)
            .map_err(|e| LlmError::ParseError(format!("Invalid JSON response: {e}")))?;

        resp.candidates
            .into_iter()
            .next()
            .and_then(|c| c.content.parts.into_iter().find_map(|p| p.text))
            .ok_or_else(|| LlmError::ParseError("No text in response candidates".to_string()))
    }

    fn build_body(&self, messages: &[Message], options: &GenerateOptions) -> serde_json::Value {
        // Gemini separates system instruction from contents
        let system: Option<String> = messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .map(|m| m.content.clone())
            .reduce(|a, b| format!("{a}\n\n{b}"));

        let contents: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        MessageRole::User => "user",
                        MessageRole::Assistant => "model",
                        MessageRole::System => unreachable!(),
                    },
                    "parts": [{"text": m.content}],
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "contents": contents,
        });

        if let Some(ref sys) = system {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{"text": sys}]
            });
        }

        // generationConfig
        let mut gen_config = serde_json::Map::new();
        if let Some(temp) = options.temperature {
            gen_config.insert("temperature".to_string(), serde_json::json!(temp));
        }
        if let Some(max_tokens) = options.max_tokens {
            gen_config.insert("maxOutputTokens".to_string(), serde_json::json!(max_tokens));
        }
        if let Some(ResponseFormat::Json) = options.response_format {
            gen_config.insert(
                "responseMimeType".to_string(),
                serde_json::json!("application/json"),
            );
        }
        if !gen_config.is_empty() {
            body["generationConfig"] = serde_json::Value::Object(gen_config);
        }

        body
    }

    fn is_retriable(err: &LlmError) -> bool {
        matches!(err, LlmError::RequestFailed(_) | LlmError::Timeout)
    }
}

impl LlmProvider for GeminiProvider {
    fn generate(
        &self,
        messages: &[Message],
        options: &GenerateOptions,
    ) -> Result<String, LlmError> {
        // Gemini URL includes model name
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.config.base_url, self.config.model
        );
        let body = self.build_body(messages, options);

        tracing::debug!(url = %url, model = %self.config.model, "Gemini request");

        let mut last_err = LlmError::RequestFailed("no attempts made".into());
        for attempt in 0..3u64 {
            if attempt > 0 {
                let wait = std::time::Duration::from_secs(2u64.pow(attempt as u32));
                tracing::warn!(attempt, "Retrying Gemini request after: {last_err}");
                std::thread::sleep(wait);
            }

            match self.do_request(&url, &body) {
                Ok(resp_body) => match self.parse_response(&resp_body) {
                    Ok(content) => return Ok(content),
                    Err(e) => {
                        tracing::warn!("Gemini response parse error: {e}");
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
        "gemini"
    }
}

// ── Gemini API types ──

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: Option<String>,
}
