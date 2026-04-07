// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return today's date as `YYYY-MM-DD`.
pub(crate) fn today_string() -> String {
    // Use a simple approach that works without the `chrono` crate.
    // We rely on the system time via std.
    let now = std::time::SystemTime::now();
    let since_epoch = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = since_epoch.as_secs();

    // Convert epoch seconds to a date. This is a minimal implementation
    // that avoids pulling in the chrono dependency.
    let days = (secs / 86400) as i64;
    let (year, month, day) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// Convert days since Unix epoch (1970-01-01) to (year, month, day).
fn days_to_ymd(days_since_epoch: i64) -> (i64, u32, u32) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days_since_epoch + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Strip reasoning model think blocks and markdown code fences from LLM output.
///
/// Handles all known reasoning model tag formats:
/// - `<think>...</think>` — Qwen3, DeepSeek R1, Kimi k1.5, GLM-4, open-source reasoning models
/// - Case-insensitive matching (`<Think>`, `<THINK>`, etc.)
/// - Tags with attributes (`<think type="reasoning">`)
/// - Multiple think blocks in one response
/// - Unclosed `<think>` tags (streaming cut-off) — strips from `<think>` to end
///
/// Also strips markdown code fences: ` ```json ... ``` ` or ` ``` ... ``` `.
///
/// Note: OpenAI o1/o3, Claude, and Gemini expose thinking in separate response
/// fields, not inline tags — no stripping needed for those providers.
pub(crate) fn strip_code_fences(s: &str) -> String {
    use regex_lite::Regex;
    use std::sync::OnceLock;

    // Compiled once, reused across calls.
    static THINK_RE: OnceLock<Regex> = OnceLock::new();
    let re = THINK_RE.get_or_init(|| {
        // Case-insensitive, matches <think>, <think ...attributes>, and </think>
        // [\s\S] matches everything including newlines.
        Regex::new(r"(?i)<think[^>]*>[\s\S]*?</think>").unwrap()
    });

    let mut cleaned = re.replace_all(s, "").to_string();

    // Handle unclosed <think> (streaming cut-off): strip from <think> to end
    if let Some(pos) = cleaned.to_lowercase().find("<think") {
        cleaned.truncate(pos);
    }

    let trimmed = cleaned.trim();
    let without_prefix = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    let without_suffix = without_prefix.strip_suffix("```").unwrap_or(without_prefix);
    without_suffix.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_today_string_format() {
        let today = today_string();
        assert_eq!(today.len(), 10);
        assert_eq!(&today[4..5], "-");
        assert_eq!(&today[7..8], "-");
    }

    #[test]
    fn test_strip_think_tags() {
        // Basic <think>...</think>
        let input = "<think>reasoning here</think>{\"result\": 1}";
        assert_eq!(strip_code_fences(input), "{\"result\": 1}");
    }

    #[test]
    fn test_strip_think_multiline() {
        let input = "<think>\nI need to think about this...\nLet me consider the options.\n</think>\n{\"answer\": 42}";
        assert_eq!(strip_code_fences(input), "{\"answer\": 42}");
    }

    #[test]
    fn test_strip_think_case_insensitive() {
        let input = "<Think>reasoning</Think>{\"ok\": true}";
        assert_eq!(strip_code_fences(input), "{\"ok\": true}");

        let input2 = "<THINK>REASONING</THINK>{\"ok\": true}";
        assert_eq!(strip_code_fences(input2), "{\"ok\": true}");
    }

    #[test]
    fn test_strip_think_with_attributes() {
        let input = "<think type=\"reasoning\" depth=\"deep\">long reasoning</think>{\"data\": 1}";
        assert_eq!(strip_code_fences(input), "{\"data\": 1}");
    }

    #[test]
    fn test_strip_multiple_think_blocks() {
        let input =
            "<think>first thought</think>middle<think>second thought</think>{\"end\": true}";
        assert_eq!(strip_code_fences(input), "middle{\"end\": true}");
    }

    #[test]
    fn test_strip_unclosed_think() {
        // Streaming cut-off: <think> without </think>
        let input = "{\"partial\": true}<think>still thinking about this and the stream got cut";
        assert_eq!(strip_code_fences(input), "{\"partial\": true}");
    }

    #[test]
    fn test_strip_think_plus_code_fences() {
        let input = "<think>reasoning</think>\n```json\n{\"combined\": true}\n```";
        assert_eq!(strip_code_fences(input), "{\"combined\": true}");
    }

    #[test]
    fn test_strip_no_think_tags() {
        // No think tags, just code fences
        let input = "```json\n{\"plain\": true}\n```";
        assert_eq!(strip_code_fences(input), "{\"plain\": true}");

        // No tags at all
        let input2 = "{\"raw\": true}";
        assert_eq!(strip_code_fences(input2), "{\"raw\": true}");
    }
}
