//! Rule-based contradiction detection between memories.
//!
//! Detects contradictions using four signal types:
//! - Negation asymmetry (one statement negates the other)
//! - Antonym pairs (built-in EN + ZH lexicon)
//! - Preference change markers ("switched to", "changed to", etc.)
//! - Temporal override (same topic, newer timestamp)
//!
//! Keyword patterns are sourced from [`crate::locale`] for centralized management.

use crate::locale::{get_locale_patterns, LocalePatterns};

/// Result of a contradiction check between two memory texts.
#[derive(Debug, Clone)]
pub(crate) struct ContradictionResult {
    /// Whether the overall score exceeds the contradiction threshold.
    pub is_contradiction: bool,
    /// Combined weighted score (0.0 to 1.0).
    pub score: f32,
    /// Human-readable signal descriptions for debugging.
    pub signals: Vec<String>,
}

/// Weight allocation for each signal type (must sum to 1.0).
const NEGATION_WEIGHT: f32 = 0.30;
const ANTONYM_WEIGHT: f32 = 0.20;
const PREFERENCE_WEIGHT: f32 = 0.20;
const NUMERIC_WEIGHT: f32 = 0.15;
const TEMPORAL_WEIGHT: f32 = 0.15;

/// Default threshold above which a contradiction is flagged.
pub(crate) const DEFAULT_CONTRADICTION_THRESHOLD: f32 = 0.5;

/// Whether a new fact explicitly says that an earlier fact changed.
/// This gate keeps the vector fallback conservative and avoids an extra search
/// for ordinary independent facts.
pub(crate) fn has_explicit_override_marker(text: &str) -> bool {
    let lower = text.to_lowercase();
    contains_any(&lower, &get_locale_patterns("auto").preference_markers)
}

/// Check whether `text` contains any of the given markers (case-insensitive for EN).
fn contains_any(text_lower: &str, markers: &[&str]) -> bool {
    markers.iter().any(|m| text_lower.contains(m))
}

/// Detect negation asymmetry: one text contains negation markers while the other does not.
fn negation_signal(
    old_lower: &str,
    new_lower: &str,
    patterns: &LocalePatterns,
) -> (f32, Option<String>) {
    let old_has = contains_any(old_lower, &patterns.negation_words);
    let new_has = contains_any(new_lower, &patterns.negation_words);

    if old_has != new_has {
        (1.0, Some("negation_asymmetry".to_string()))
    } else {
        (0.0, None)
    }
}

/// Detect antonym pairs: one text contains one side, the other text contains the opposite.
fn antonym_signal(
    old_lower: &str,
    new_lower: &str,
    patterns: &LocalePatterns,
) -> (f32, Option<String>) {
    for &(a, b) in &patterns.antonym_pairs {
        let old_a = old_lower.contains(a);
        let old_b = old_lower.contains(b);
        let new_a = new_lower.contains(a);
        let new_b = new_lower.contains(b);
        // One text has word A (but not B), the other has word B (but not A)
        if (old_a && !old_b && new_b && !new_a) || (old_b && !old_a && new_a && !new_b) {
            return (1.0, Some(format!("antonym_pair({a}/{b})")));
        }
    }
    (0.0, None)
}

/// Detect preference change: the new text contains markers indicating a change in preference.
fn preference_signal(new_lower: &str, patterns: &LocalePatterns) -> (f32, Option<String>) {
    if contains_any(new_lower, &patterns.preference_markers) {
        (1.0, Some("preference_change".to_string()))
    } else {
        (0.0, None)
    }
}

/// Detect numeric value changes: same context but different numbers.
/// Catches "37 coins" → "38 coins", "$350,000" → "$400,000", "7pm" → "6pm".
fn numeric_change_signal(old_lower: &str, new_lower: &str) -> (f32, Option<String>) {
    // Extract numbers from both texts
    let old_nums: Vec<f64> = extract_numbers(old_lower);
    let new_nums: Vec<f64> = extract_numbers(new_lower);

    if old_nums.is_empty() || new_nums.is_empty() {
        return (0.0, None);
    }

    // If both have numbers and they differ, it's a potential update
    // Check if there's at least one number that changed
    for &old_n in &old_nums {
        for &new_n in &new_nums {
            // Same order of magnitude but different value → likely an update
            if old_n != new_n && old_n != 0.0 && new_n != 0.0 {
                let ratio = if old_n > new_n {
                    old_n / new_n
                } else {
                    new_n / old_n
                };
                // Within 10x of each other → plausible update (not random numbers)
                if ratio < 10.0 {
                    return (1.0, Some(format!("numeric_change({old_n}→{new_n})")));
                }
            }
        }
    }

    (0.0, None)
}

/// Extract numbers from text (integers and decimals, including currency).
fn extract_numbers(text: &str) -> Vec<f64> {
    let mut nums = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c == '$' || c == '¥' || c == '€' || c == '£' {
            chars.next();
            continue;
        }
        if c.is_ascii_digit() {
            let mut num_str = String::new();
            while let Some(&nc) = chars.peek() {
                if nc.is_ascii_digit() || nc == '.' || nc == ',' {
                    num_str.push(nc);
                    chars.next();
                } else {
                    break;
                }
            }
            // Remove commas (e.g., "400,000" → "400000")
            let clean = num_str.replace(',', "");
            if let Ok(n) = clean.parse::<f64>() {
                if n > 0.0 {
                    nums.push(n);
                }
            }
        } else {
            chars.next();
        }
    }

    let mut chinese = String::new();
    for ch in text.chars().chain(std::iter::once(' ')) {
        if matches!(
            ch,
            '零' | '一'
                | '二'
                | '两'
                | '三'
                | '四'
                | '五'
                | '六'
                | '七'
                | '八'
                | '九'
                | '十'
                | '百'
        ) {
            chinese.push(ch);
        } else if !chinese.is_empty() {
            if let Some(value) = parse_chinese_number(&chinese) {
                nums.push(value as f64);
            }
            chinese.clear();
        }
    }
    nums
}

/// Parse the common Chinese number forms used in dates, times, counts, and doses.
/// This intentionally stops at hundreds; larger financial values should use the
/// LLM reconciliation path or Arabic digits.
fn parse_chinese_number(text: &str) -> Option<u32> {
    let digit = |ch| match ch {
        '零' => Some(0),
        '一' => Some(1),
        '二' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    };

    let mut total = 0_u32;
    let mut current = 0_u32;
    for ch in text.chars() {
        match ch {
            '十' => {
                total += current.max(1) * 10;
                current = 0;
            }
            '百' => {
                total += current.max(1) * 100;
                current = 0;
            }
            _ => current = digit(ch)?,
        }
    }
    Some(total + current)
}

/// Detect temporal override: both texts exist and the old one is semantically similar
/// (assumed by the caller having already checked cosine similarity >= threshold),
/// so a newer memory on the same topic is a potential override.
///
/// `is_newer` indicates whether the new memory has a more recent timestamp.
fn temporal_signal(is_newer: bool) -> (f32, Option<String>) {
    if is_newer {
        (1.0, Some("temporal_override".to_string()))
    } else {
        (0.0, None)
    }
}

/// Run all contradiction signals and return a combined result.
///
/// `old_content` — content of the existing memory.
/// `new_content` — content of the newly added memory.
/// `is_newer` — true if the new memory has a more recent timestamp than the old one.
pub(crate) fn detect_contradiction(
    old_content: &str,
    new_content: &str,
    is_newer: bool,
) -> ContradictionResult {
    let old_lower = old_content.to_lowercase();
    let new_lower = new_content.to_lowercase();
    let patterns = get_locale_patterns("auto");

    let mut total_score = 0.0f32;
    let mut signals = Vec::new();

    // 1. Negation asymmetry (35%)
    let (neg_s, neg_desc) = negation_signal(&old_lower, &new_lower, &patterns);
    total_score += neg_s * NEGATION_WEIGHT;
    if let Some(d) = neg_desc {
        signals.push(d);
    }

    // 2. Antonym pairs (25%)
    let (ant_s, ant_desc) = antonym_signal(&old_lower, &new_lower, &patterns);
    total_score += ant_s * ANTONYM_WEIGHT;
    if let Some(d) = ant_desc {
        signals.push(d);
    }

    // 3. Preference change (20%)
    let (pref_s, pref_desc) = preference_signal(&new_lower, &patterns);
    total_score += pref_s * PREFERENCE_WEIGHT;
    if let Some(d) = pref_desc {
        signals.push(d);
    }

    // 4. Numeric value change (15%)
    let (num_s, num_desc) = numeric_change_signal(&old_lower, &new_lower);
    total_score += num_s * NUMERIC_WEIGHT;
    if let Some(d) = num_desc {
        signals.push(d);
    }

    // 5. Temporal override (15%)
    let (temp_s, temp_desc) = temporal_signal(is_newer);
    total_score += temp_s * TEMPORAL_WEIGHT;
    if let Some(d) = temp_desc {
        signals.push(d);
    }

    ContradictionResult {
        is_contradiction: total_score >= DEFAULT_CONTRADICTION_THRESHOLD,
        score: total_score,
        signals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_contradiction() {
        let r = detect_contradiction("I like pizza", "I enjoy hiking", true);
        assert!(!r.is_contradiction);
        assert!(r.signals.is_empty() || r.score < DEFAULT_CONTRADICTION_THRESHOLD);
    }

    #[test]
    fn test_negation_asymmetry() {
        let r = detect_contradiction("I like coffee", "I don't like coffee", true);
        assert!(r.score >= NEGATION_WEIGHT);
        assert!(r.signals.iter().any(|s| s.contains("negation")));
    }

    #[test]
    fn test_antonym_en() {
        let r = detect_contradiction("I love this restaurant", "I hate this restaurant", true);
        assert!(r.score >= ANTONYM_WEIGHT);
        assert!(r.signals.iter().any(|s| s.contains("antonym")));
    }

    #[test]
    fn test_antonym_zh() {
        let r = detect_contradiction("我喜欢这家餐厅", "我讨厌这家餐厅", true);
        assert!(r.score >= ANTONYM_WEIGHT);
        assert!(r.signals.iter().any(|s| s.contains("antonym")));
    }

    #[test]
    fn test_preference_change_en() {
        let r = detect_contradiction("I use VS Code", "I switched to Cursor", true);
        assert!(r.score >= PREFERENCE_WEIGHT);
        assert!(r.signals.iter().any(|s| s.contains("preference")));
    }

    #[test]
    fn test_preference_change_zh() {
        let r = detect_contradiction("我用VS Code", "我改成了Cursor", true);
        assert!(r.score >= PREFERENCE_WEIGHT);
        assert!(r.signals.iter().any(|s| s.contains("preference")));
    }

    #[test]
    fn test_temporal_override() {
        let r = detect_contradiction("I work at Google", "I work at Google", true);
        // Only temporal signal, which is 0.15 — below threshold
        assert_eq!(r.score, TEMPORAL_WEIGHT);
        assert!(!r.is_contradiction);
    }

    #[test]
    fn test_combined_signals_trigger_contradiction() {
        // negation + temporal = 0.35 + 0.15 = 0.50 — at threshold
        // negation + antonym = 0.35 + 0.25 = 0.60 — above threshold
        let r = detect_contradiction(
            "I always buy coffee in the morning",
            "I never buy coffee in the morning",
            true,
        );
        // "always" vs "never" is an antonym pair, and "never" is also a negation
        assert!(
            r.is_contradiction,
            "score={} signals={:?}",
            r.score, r.signals
        );
    }

    #[test]
    fn test_negation_zh() {
        let r = detect_contradiction("我会做这件事", "我不会做这件事", true);
        assert!(r.score >= NEGATION_WEIGHT);
        assert!(r.signals.iter().any(|s| s.contains("negation")));
    }

    #[test]
    fn test_full_contradiction_all_signals() {
        // negation + antonym + preference + temporal
        let r = detect_contradiction(
            "I always like this tool",
            "I never hate this tool, switched to another",
            true,
        );
        // New has negation ("never"), old doesn't have typical negation
        // BUT: old has "always", new has "never" → antonym (always/never)
        // New has "switched to" → preference
        // is_newer = true → temporal
        assert!(
            r.score > DEFAULT_CONTRADICTION_THRESHOLD,
            "score={} signals={:?}",
            r.score,
            r.signals
        );
    }

    #[test]
    fn test_numeric_change() {
        let r = detect_contradiction(
            "I have 37 pre-1920 American coins",
            "I have 38 pre-1920 American coins",
            true,
        );
        assert!(r.signals.iter().any(|s| s.contains("numeric_change")));
        // numeric (0.15) + temporal (0.15) = 0.30, below threshold alone
        // But with entity neighbor context this would be flagged
    }

    #[test]
    fn test_explicit_numeric_override_is_contradiction() {
        let r = detect_contradiction("明天遛狗时间是上午九点", "明天遛狗时间改成上午十点", true);
        assert!(
            r.is_contradiction,
            "score={} signals={:?}",
            r.score, r.signals
        );
        assert!(has_explicit_override_marker("明天改成上午十点"));
    }

    #[test]
    fn test_numeric_change_currency() {
        let r = detect_contradiction(
            "Pre-approved for $350,000",
            "Pre-approved for $400,000",
            true,
        );
        assert!(
            r.signals.iter().any(|s| s.contains("numeric_change")),
            "signals: {:?}",
            r.signals
        );
    }

    #[test]
    fn test_chinese_numeric_change() {
        let r = detect_contradiction("上午九点遛狗", "改成上午十点遛狗", true);
        assert!(r.signals.iter().any(|s| s.contains("numeric_change")));
        assert!(
            r.is_contradiction,
            "score={} signals={:?}",
            r.score, r.signals
        );
    }

    #[test]
    fn test_same_negation_no_signal() {
        // Both have negation — no asymmetry
        let r = detect_contradiction("I don't like A", "I don't like B", false);
        let has_negation = r.signals.iter().any(|s| s.contains("negation"));
        assert!(!has_negation);
    }

    #[test]
    fn test_threshold_boundary() {
        // Only preference (0.25) + temporal (0.15) = 0.40 — below 0.5
        let r = detect_contradiction("I use Vim", "I switched to Emacs", true);
        assert!(
            !r.is_contradiction,
            "score={} signals={:?}",
            r.score, r.signals
        );
    }
}
