use aho_corasick::AhoCorasick;

/// Fast entity name extractor using Aho-Corasick automaton.
/// Built from the entities table, used at search time for <1ms entity detection.
pub struct EntityIndex {
    automaton: Option<AhoCorasick>,
    patterns: Vec<String>, // lowercase entity names
}

impl EntityIndex {
    /// Build from a list of entity names.
    pub fn build(entity_names: &[String]) -> Self {
        if entity_names.is_empty() {
            return Self {
                automaton: None,
                patterns: Vec::new(),
            };
        }

        let patterns: Vec<String> = entity_names.iter().map(|n| n.to_lowercase()).collect();

        let automaton = match AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
        {
            Ok(ac) => Some(ac),
            Err(e) => {
                tracing::error!(
                    "Failed to build EntityIndex ({} patterns): {e}",
                    patterns.len()
                );
                None
            }
        };

        Self {
            automaton,
            patterns,
        }
    }

    /// Extract entity names from a query string. Returns matched entity names.
    /// Only matches at word boundaries to avoid "art" matching inside "start".
    pub fn extract(&self, query: &str) -> Vec<String> {
        let Some(ref ac) = self.automaton else {
            return Vec::new();
        };

        let bytes = query.as_bytes();
        let mut matched = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for mat in ac.find_iter(query) {
            // Check word boundaries: char before match start and after match end
            // must not be alphanumeric (or match must be at string boundary)
            let start = mat.start();
            let end = mat.end();
            let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
            let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
            if before_ok && after_ok {
                let idx = mat.pattern().as_usize();
                if seen.insert(idx) {
                    matched.push(self.patterns[idx].clone());
                }
            }
        }

        matched
    }

    /// Check if the index has any patterns.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let index = EntityIndex::build(&[]);
        assert!(index.is_empty());
        assert!(index.extract("hello world").is_empty());
    }

    #[test]
    fn test_basic_extraction() {
        let names = vec!["Alice".to_string(), "Google".to_string()];
        let index = EntityIndex::build(&names);
        assert!(!index.is_empty());

        let matched = index.extract("Alice works at Google");
        assert_eq!(matched.len(), 2);
        assert!(matched.contains(&"alice".to_string()));
        assert!(matched.contains(&"google".to_string()));
    }

    #[test]
    fn test_case_insensitive() {
        let names = vec!["Alice".to_string()];
        let index = EntityIndex::build(&names);

        let matched = index.extract("ALICE is here");
        assert_eq!(matched.len(), 1);
        assert!(matched.contains(&"alice".to_string()));
    }

    #[test]
    fn test_no_match() {
        let names = vec!["Alice".to_string()];
        let index = EntityIndex::build(&names);

        let matched = index.extract("Bob is here");
        assert!(matched.is_empty());
    }

    #[test]
    fn test_word_boundary() {
        let names = vec!["art".to_string(), "AI".to_string()];
        let index = EntityIndex::build(&names);

        // "art" should NOT match inside "start"
        let matched = index.extract("I want to start painting");
        assert!(matched.is_empty(), "should not match 'art' inside 'start'");

        // "art" SHOULD match as a standalone word
        let matched = index.extract("I love art and AI");
        assert_eq!(matched.len(), 2);

        // "AI" should NOT match inside "tail"
        let matched = index.extract("The cat's tail is fluffy");
        assert!(matched.is_empty(), "should not match 'AI' inside 'tail'");
    }

    #[test]
    fn test_dedup_matches() {
        let names = vec!["Alice".to_string()];
        let index = EntityIndex::build(&names);

        let matched = index.extract("Alice met Alice");
        assert_eq!(matched.len(), 1);
    }
}
