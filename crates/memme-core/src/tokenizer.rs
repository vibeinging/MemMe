//! Tokenizer trait and implementations for multilingual text tokenization.
//!
//! Provides pluggable tokenizers:
//! - `UnicodeTokenizer` — whitespace + punctuation splitting, lowercase (English/Latin)
//! - `CJKTokenizer` — CJK character-level splitting, non-CJK by whitespace
//! - `AutoTokenizer` — auto-detects CJK presence and delegates accordingly

/// Trait for text tokenization.
pub(crate) trait Tokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// Default: splits on whitespace + punctuation, lowercases.
/// Suitable for English and other Latin-script languages.
pub(crate) struct UnicodeTokenizer;

impl Tokenizer for UnicodeTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect()
    }
}

/// Returns true if a character falls in CJK Unified Ideographs ranges.
fn is_cjk_char(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Unified Ideographs Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{3000}'..='\u{303F}' // CJK Symbols and Punctuation
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
    )
}

/// Returns true if a character is CJK punctuation (should be treated as separator).
fn is_cjk_punctuation(c: char) -> bool {
    matches!(c,
        '\u{3000}'..='\u{303F}'   // CJK Symbols and Punctuation
        | '\u{FF00}'..='\u{FFEF}' // Halfwidth and Fullwidth Forms (fullwidth punctuation)
        | '\u{FE30}'..='\u{FE4F}' // CJK Compatibility Forms
    )
}

/// CJK-aware tokenizer: splits CJK characters individually, non-CJK text by whitespace.
/// Each CJK character becomes an independent token. Non-CJK segments are lowercased
/// and split by whitespace/punctuation.
pub(crate) struct CJKTokenizer;

impl Tokenizer for CJKTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut non_cjk_buf = String::new();

        for c in text.chars() {
            if is_cjk_char(c) && !is_cjk_punctuation(c) {
                // Flush any accumulated non-CJK text
                if !non_cjk_buf.is_empty() {
                    flush_non_cjk(&non_cjk_buf, &mut tokens);
                    non_cjk_buf.clear();
                }
                // Each CJK character is a separate token
                tokens.push(c.to_string());
            } else if c.is_whitespace() || c.is_ascii_punctuation() || is_cjk_punctuation(c) {
                // Delimiter: flush non-CJK buffer
                if !non_cjk_buf.is_empty() {
                    flush_non_cjk(&non_cjk_buf, &mut tokens);
                    non_cjk_buf.clear();
                }
            } else {
                non_cjk_buf.push(c);
            }
        }
        // Flush remaining
        if !non_cjk_buf.is_empty() {
            flush_non_cjk(&non_cjk_buf, &mut tokens);
        }
        tokens
    }
}

/// Flush a non-CJK buffer: lowercase and split by whitespace/punctuation.
fn flush_non_cjk(buf: &str, tokens: &mut Vec<String>) {
    for word in buf
        .to_lowercase()
        .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
    {
        if !word.is_empty() {
            tokens.push(word.to_string());
        }
    }
}

/// Returns true if text contains any CJK characters.
pub(crate) fn contains_cjk(text: &str) -> bool {
    text.chars()
        .any(|c| is_cjk_char(c) && !is_cjk_punctuation(c))
}

/// Auto-detecting tokenizer: inspects the input text for CJK characters
/// and delegates to the appropriate tokenizer.
pub(crate) struct AutoTokenizer;

impl Tokenizer for AutoTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        if contains_cjk(text) {
            CJKTokenizer.tokenize(text)
        } else {
            UnicodeTokenizer.tokenize(text)
        }
    }
}

/// Select the appropriate tokenizer based on the locale string.
///
/// - `"zh"`, `"ja"`, `"ko"` -> `CJKTokenizer`
/// - `"en"` -> `UnicodeTokenizer`
/// - `"auto"` or anything else -> `AutoTokenizer`
pub(crate) fn select_tokenizer(locale: &str) -> Box<dyn Tokenizer> {
    match locale {
        "zh" | "ja" | "ko" => Box::new(CJKTokenizer),
        "en" => Box::new(UnicodeTokenizer),
        _ => Box::new(AutoTokenizer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── UnicodeTokenizer ──

    #[test]
    fn test_unicode_tokenizer_basic() {
        let t = UnicodeTokenizer;
        let tokens = t.tokenize("Hello, World! How are you?");
        assert_eq!(tokens, vec!["hello", "world", "how", "are", "you"]);
    }

    #[test]
    fn test_unicode_tokenizer_empty() {
        let t = UnicodeTokenizer;
        assert!(t.tokenize("").is_empty());
    }

    #[test]
    fn test_unicode_tokenizer_punctuation() {
        let t = UnicodeTokenizer;
        let tokens = t.tokenize("don't stop! it's ok.");
        assert_eq!(tokens, vec!["don", "t", "stop", "it", "s", "ok"]);
    }

    // ── CJKTokenizer ──

    #[test]
    fn test_cjk_tokenizer_chinese() {
        let t = CJKTokenizer;
        let tokens = t.tokenize("我喜欢咖啡");
        assert_eq!(tokens, vec!["我", "喜", "欢", "咖", "啡"]);
    }

    #[test]
    fn test_cjk_tokenizer_mixed() {
        let t = CJKTokenizer;
        let tokens = t.tokenize("我喜欢coffee");
        assert_eq!(tokens, vec!["我", "喜", "欢", "coffee"]);
    }

    #[test]
    fn test_cjk_tokenizer_japanese() {
        let t = CJKTokenizer;
        let tokens = t.tokenize("東京タワー");
        // Should tokenize each character individually
        assert_eq!(tokens, vec!["東", "京", "タ", "ワ", "ー"]);
    }

    #[test]
    fn test_cjk_tokenizer_pure_english() {
        let t = CJKTokenizer;
        let tokens = t.tokenize("Hello World");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_cjk_tokenizer_with_punctuation() {
        let t = CJKTokenizer;
        let tokens = t.tokenize("你好，世界！");
        // Fullwidth punctuation should be treated as separators
        assert_eq!(tokens, vec!["你", "好", "世", "界"]);
    }

    #[test]
    fn test_cjk_tokenizer_empty() {
        let t = CJKTokenizer;
        assert!(t.tokenize("").is_empty());
    }

    // ── AutoTokenizer ──

    #[test]
    fn test_auto_tokenizer_english() {
        let t = AutoTokenizer;
        let tokens = t.tokenize("Hello, World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_auto_tokenizer_chinese() {
        let t = AutoTokenizer;
        let tokens = t.tokenize("我喜欢咖啡");
        assert_eq!(tokens, vec!["我", "喜", "欢", "咖", "啡"]);
    }

    #[test]
    fn test_auto_tokenizer_mixed() {
        let t = AutoTokenizer;
        let tokens = t.tokenize("我喜欢coffee shop");
        // Has CJK -> uses CJKTokenizer
        assert_eq!(tokens, vec!["我", "喜", "欢", "coffee", "shop"]);
    }

    // ── select_tokenizer ──

    #[test]
    fn test_select_tokenizer_zh() {
        let t = select_tokenizer("zh");
        let tokens = t.tokenize("我喜欢咖啡");
        assert_eq!(tokens, vec!["我", "喜", "欢", "咖", "啡"]);
    }

    #[test]
    fn test_select_tokenizer_en() {
        let t = select_tokenizer("en");
        let tokens = t.tokenize("Hello World");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_select_tokenizer_auto() {
        let t = select_tokenizer("auto");
        // English text
        let tokens = t.tokenize("Hello World");
        assert_eq!(tokens, vec!["hello", "world"]);
        // Chinese text
        let tokens = t.tokenize("我喜欢咖啡");
        assert_eq!(tokens, vec!["我", "喜", "欢", "咖", "啡"]);
    }

    // ── contains_cjk ──

    #[test]
    fn test_contains_cjk_true() {
        assert!(contains_cjk("我喜欢coffee"));
        assert!(contains_cjk("東京Tower"));
    }

    #[test]
    fn test_contains_cjk_false() {
        assert!(!contains_cjk("Hello World"));
        assert!(!contains_cjk("12345"));
        assert!(!contains_cjk(""));
    }

    // ── Korean support ──

    #[test]
    fn test_cjk_tokenizer_korean() {
        let t = CJKTokenizer;
        let tokens = t.tokenize("서울타워");
        assert_eq!(tokens, vec!["서", "울", "타", "워"]);
    }

    // ── Word overlap with CJK ──

    #[test]
    fn test_cjk_word_overlap_chinese() {
        let t = CJKTokenizer;
        let query_tokens = t.tokenize("咖啡");
        // query: ["咖", "啡"]
        assert_eq!(query_tokens, vec!["咖", "啡"]);

        // Candidate contains both characters
        let candidate = "我喜欢咖啡";
        let candidate_tokens = t.tokenize(candidate);
        // candidate: ["我", "喜", "欢", "咖", "啡"]
        let matched = query_tokens
            .iter()
            .filter(|qt| candidate_tokens.contains(qt))
            .count();
        assert_eq!(matched, 2); // both "咖" and "啡" found
    }
}
