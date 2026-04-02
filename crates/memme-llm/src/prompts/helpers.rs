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

/// Strip optional markdown code fences (```json ... ``` or ``` ... ```)
/// and reasoning model think blocks (`<think>...</think>`).
pub(crate) fn strip_code_fences(s: &str) -> String {
    // Strip <think>...</think> blocks first (reasoning models like qwen3, o1)
    let mut cleaned = s.to_string();
    while let Some(start) = cleaned.find("<think>") {
        if let Some(end) = cleaned.find("</think>") {
            cleaned = format!(
                "{}{}",
                &cleaned[..start],
                &cleaned[end + "</think>".len()..]
            );
        } else {
            // Unclosed <think> — strip from <think> to end
            cleaned = cleaned[..start].to_string();
            break;
        }
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
        // Should match YYYY-MM-DD
        assert_eq!(today.len(), 10);
        assert_eq!(&today[4..5], "-");
        assert_eq!(&today[7..8], "-");
    }
}
