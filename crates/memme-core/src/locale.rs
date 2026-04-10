//! Locale-aware keyword patterns for multilingual contradiction detection.
//!
//! Centralizes EN + ZH keyword lists so they can be shared across modules.
//! When locale is "auto" or mixed usage is expected, all patterns are merged.

/// Locale-specific keyword patterns used by the contradiction detector.
pub(crate) struct LocalePatterns {
    pub negation_words: Vec<&'static str>,
    pub antonym_pairs: Vec<(&'static str, &'static str)>,
    pub preference_markers: Vec<&'static str>,
}

// ── English patterns ──

const EN_NEGATIONS: &[&str] = &[
    "not",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "can't",
    "cannot",
    "never",
    "no longer",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "shouldn't",
    "couldn't",
];

const ZH_NEGATIONS: &[&str] = &[
    "不", "没有", "没", "从不", "并非", "未", "别", "不再", "不会", "无法",
];

const EN_ANTONYMS: &[(&str, &str)] = &[
    ("like", "dislike"),
    ("love", "hate"),
    ("good", "bad"),
    ("open", "close"),
    ("start", "stop"),
    ("buy", "sell"),
    ("agree", "disagree"),
    ("always", "never"),
    ("happy", "sad"),
    ("fast", "slow"),
    ("hot", "cold"),
    ("big", "small"),
    ("easy", "hard"),
    ("right", "wrong"),
    ("win", "lose"),
    ("accept", "reject"),
    ("allow", "forbid"),
    ("connect", "disconnect"),
    ("include", "exclude"),
    ("increase", "decrease"),
];

const ZH_ANTONYMS: &[(&str, &str)] = &[
    ("喜欢", "讨厌"),
    ("好", "坏"),
    ("开", "关"),
    ("买", "卖"),
    ("同意", "反对"),
    ("总是", "从不"),
    ("快乐", "悲伤"),
    ("快", "慢"),
    ("热", "冷"),
    ("大", "小"),
    ("容易", "困难"),
    ("对", "错"),
    ("赢", "输"),
    ("接受", "拒绝"),
    ("允许", "禁止"),
    ("连接", "断开"),
    ("包含", "排除"),
    ("增加", "减少"),
    ("开始", "结束"),
    ("爱", "恨"),
];

const EN_PREFERENCE_MARKERS: &[&str] = &[
    "switched to",
    "changed to",
    "now prefer",
    "now prefers",
    "moved to",
    "replaced with",
    "no longer use",
    "no longer uses",
    "stopped using",
    "instead of",
    "gave up",
    "switched from",
    "transitioned to",
];

const ZH_PREFERENCE_MARKERS: &[&str] = &[
    "改成了",
    "换成了",
    "现在用",
    "不再用",
    "转向了",
    "替换成",
    "放弃了",
    "改用了",
    "从此用",
    "不用了",
];

/// Get locale patterns for contradiction detection.
///
/// Always returns merged EN + ZH patterns, because users often mix languages.
/// The `locale` parameter is accepted for future per-locale optimization.
pub(crate) fn get_locale_patterns(_locale: &str) -> LocalePatterns {
    // Merge EN + ZH for all locales — users may mix languages.
    let mut negation_words = Vec::with_capacity(EN_NEGATIONS.len() + ZH_NEGATIONS.len());
    negation_words.extend_from_slice(EN_NEGATIONS);
    negation_words.extend_from_slice(ZH_NEGATIONS);

    let mut antonym_pairs = Vec::with_capacity(EN_ANTONYMS.len() + ZH_ANTONYMS.len());
    antonym_pairs.extend_from_slice(EN_ANTONYMS);
    antonym_pairs.extend_from_slice(ZH_ANTONYMS);

    let mut preference_markers =
        Vec::with_capacity(EN_PREFERENCE_MARKERS.len() + ZH_PREFERENCE_MARKERS.len());
    preference_markers.extend_from_slice(EN_PREFERENCE_MARKERS);
    preference_markers.extend_from_slice(ZH_PREFERENCE_MARKERS);

    LocalePatterns {
        negation_words,
        antonym_pairs,
        preference_markers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_locale_patterns_returns_both_languages() {
        let patterns = get_locale_patterns("auto");
        // Should contain English negations
        assert!(patterns.negation_words.contains(&"not"));
        assert!(patterns.negation_words.contains(&"don't"));
        // Should contain Chinese negations
        assert!(patterns.negation_words.contains(&"不"));
        assert!(patterns.negation_words.contains(&"没有"));
    }

    #[test]
    fn test_get_locale_patterns_has_antonyms() {
        let patterns = get_locale_patterns("auto");
        assert!(patterns.antonym_pairs.contains(&("love", "hate")));
        assert!(patterns.antonym_pairs.contains(&("喜欢", "讨厌")));
    }

    #[test]
    fn test_get_locale_patterns_has_preference_markers() {
        let patterns = get_locale_patterns("auto");
        assert!(patterns.preference_markers.contains(&"switched to"));
        assert!(patterns.preference_markers.contains(&"改成了"));
    }

    #[test]
    fn test_locale_en_still_returns_merged() {
        let patterns = get_locale_patterns("en");
        // Even with "en" locale, we return merged patterns
        assert!(patterns.negation_words.contains(&"不"));
    }

    #[test]
    fn test_locale_zh_still_returns_merged() {
        let patterns = get_locale_patterns("zh");
        assert!(patterns.negation_words.contains(&"not"));
    }
}
