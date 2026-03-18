//! Shared retry logic for LLM providers.
//!
//! Provides configurable exponential backoff with jitter, rate-limit awareness
//! (reads `retry-after` header), and context-length-exceeded detection.

use crate::error::LlmError;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of attempts (including the first). Default: 3.
    pub max_attempts: u32,
    /// Base delay in milliseconds for exponential backoff. Default: 500.
    pub base_delay_ms: u64,
    /// Maximum delay in milliseconds (cap for backoff). Default: 30_000.
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 500,
            max_delay_ms: 30_000,
        }
    }
}

/// Compute the sleep duration for a given attempt, respecting rate-limit headers.
///
/// Uses exponential backoff with jitter: `base * 2^attempt + random(0..base)`.
/// If the error is `RateLimit(Some(secs))`, uses that value instead (capped by max_delay).
pub fn retry_delay(config: &RetryConfig, attempt: u32, err: &LlmError) -> std::time::Duration {
    // If rate-limited with explicit retry-after, use it
    if let LlmError::RateLimit(Some(secs)) = err {
        let ms = (*secs * 1000).min(config.max_delay_ms);
        return std::time::Duration::from_millis(ms);
    }

    // Exponential backoff with jitter
    let exp = config.base_delay_ms.saturating_mul(1u64 << attempt.min(10));
    let jitter = fastrand::u64(0..config.base_delay_ms.max(1));
    let delay = (exp + jitter).min(config.max_delay_ms);
    std::time::Duration::from_millis(delay)
}

/// Parse the `retry-after` header value (seconds) from a response.
/// Returns `None` if the header is missing or unparseable.
#[cfg(feature = "reqwest")]
pub fn parse_retry_after(response: &reqwest::blocking::Response) -> Option<u64> {
    response
        .headers()
        .get("retry-after")
        .and_then(|v: &reqwest::header::HeaderValue| v.to_str().ok())
        .and_then(|s: &str| s.trim().parse::<u64>().ok())
}

/// Check if an HTTP error body indicates context length exceeded.
///
/// Works across providers:
/// - OpenAI: `"context_length_exceeded"`
/// - Anthropic: body mentions `"max_tokens"` in a 400 error
/// - Gemini: body mentions `"token"` limits
pub fn is_context_too_long(status: u16, body: &str) -> bool {
    if status != 400 {
        return false;
    }
    // Match known exact markers without allocating a lowercase copy.
    // These strings come from API error responses and are stable.
    contains_ignore_ascii_case(body, "context_length_exceeded")
        || contains_ignore_ascii_case(body, "maximum context length")
        || (contains_ignore_ascii_case(body, "token")
            && (contains_ignore_ascii_case(body, "limit")
                || contains_ignore_ascii_case(body, "exceed")))
}

/// Case-insensitive ASCII substring search without allocation.
fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_delay_exponential() {
        let cfg = RetryConfig {
            base_delay_ms: 500,
            max_delay_ms: 30_000,
            ..Default::default()
        };
        let err = LlmError::RequestFailed("server error".into());

        let d0 = retry_delay(&cfg, 0, &err);
        let d1 = retry_delay(&cfg, 1, &err);
        // attempt 0: 500 + jitter(0..500), attempt 1: 1000 + jitter(0..500)
        assert!(d0.as_millis() >= 500);
        assert!(d0.as_millis() < 1100);
        assert!(d1.as_millis() >= 1000);
    }

    #[test]
    fn test_retry_delay_capped() {
        let cfg = RetryConfig {
            base_delay_ms: 500,
            max_delay_ms: 2000,
            ..Default::default()
        };
        let err = LlmError::Timeout;

        let d = retry_delay(&cfg, 20, &err);
        assert!(d.as_millis() <= 2000);
    }

    #[test]
    fn test_retry_delay_rate_limit() {
        let cfg = RetryConfig::default();
        let err = LlmError::RateLimit(Some(5));

        let d = retry_delay(&cfg, 0, &err);
        assert_eq!(d.as_millis(), 5000);
    }

    #[test]
    fn test_retry_delay_rate_limit_capped() {
        let cfg = RetryConfig {
            max_delay_ms: 3000,
            ..Default::default()
        };
        let err = LlmError::RateLimit(Some(60));

        let d = retry_delay(&cfg, 0, &err);
        assert_eq!(d.as_millis(), 3000);
    }

    #[test]
    fn test_is_context_too_long() {
        assert!(is_context_too_long(
            400,
            "This model's maximum context length is 128000 tokens. However, your messages resulted in context_length_exceeded."
        ));
        assert!(is_context_too_long(
            400,
            "token limit exceeded"
        ));
        assert!(!is_context_too_long(
            429,
            "context_length_exceeded"
        ));
        assert!(!is_context_too_long(400, "invalid temperature"));
    }
}
