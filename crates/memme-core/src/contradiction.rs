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
const NEGATION_WEIGHT: f32 = 0.35;
const ANTONYM_WEIGHT: f32 = 0.25;
const PREFERENCE_WEIGHT: f32 = 0.25;
const TEMPORAL_WEIGHT: f32 = 0.15;

/// Default threshold above which a contradiction is flagged.
pub(crate) const DEFAULT_CONTRADICTION_THRESHOLD: f32 = 0.5;

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

    // 3. Preference change (25%)
    let (pref_s, pref_desc) = preference_signal(&new_lower, &patterns);
    total_score += pref_s * PREFERENCE_WEIGHT;
    if let Some(d) = pref_desc {
        signals.push(d);
    }

    // 4. Temporal override (15%)
    let (temp_s, temp_desc) = temporal_signal(is_newer);
    total_score += temp_s * TEMPORAL_WEIGHT;
    if let Some(d) = temp_desc {
        signals.push(d);
    }

    ContradictionResult {
        is_contradiction: total_score > DEFAULT_CONTRADICTION_THRESHOLD,
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
