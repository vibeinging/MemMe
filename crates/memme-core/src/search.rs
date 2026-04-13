use std::collections::HashMap;

use crate::tokenizer::Tokenizer;
use crate::types::{Episode, Event, MemoryResult};

/// Compute confidence score for a search channel based on its score distribution.
///
/// Confidence = mean(top_k_similarities) / (1 + stddev). Higher values indicate
/// the channel returned tightly clustered, high-quality results.
///
/// - `is_distance`: true if scores are cosine distances (lower = better),
///   false if scores are relevance (higher = better).
/// - Returns 1.0 if no scores are available (no adjustment).
#[allow(dead_code)] // available for adaptive RRF when enabled
pub(crate) fn compute_channel_confidence(
    results: &[MemoryResult],
    is_distance: bool,
    top_k: usize,
) -> f64 {
    let sims: Vec<f64> = results
        .iter()
        .take(top_k)
        .filter_map(|r| {
            r.score.map(|s| {
                let s = s as f64;
                if is_distance {
                    // Cosine distance [0, 2] → similarity [0, 1]
                    (1.0 - s / 2.0).clamp(0.0, 1.0)
                } else {
                    // Unbounded relevance scores (e.g. BM25) → [0, 1] via sigmoid
                    (s / (1.0 + s)).clamp(0.0, 1.0)
                }
            })
        })
        .collect();

    if sims.is_empty() {
        return 1.0; // no scores → no adjustment
    }

    let n = sims.len() as f64;
    let mean = sims.iter().sum::<f64>() / n;

    if n < 2.0 {
        return mean.clamp(0.0, 1.0);
    }

    let variance = sims.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let stddev = variance.sqrt();

    (mean / (1.0 + stddev)).clamp(0.0, 1.0)
}

/// Reciprocal Rank Fusion: combines multiple ranked lists into one.
///
/// Formula: score(d) = Σ weight_i / (k + rank_i)
/// where rank_i is the 1-based rank of document d in list i.
pub fn rrf_fuse(
    ranked_lists: &[(&[MemoryResult], f64)], // (results, weight) pairs
    k: usize,
    limit: usize,
) -> Vec<MemoryResult> {
    let mut scores: HashMap<String, (f64, MemoryResult)> = HashMap::new();

    for (results, weight) in ranked_lists {
        for (rank, result) in results.iter().enumerate() {
            let rrf_score = weight / (k as f64 + (rank + 1) as f64);
            scores
                .entry(result.id.clone())
                .and_modify(|(s, _)| *s += rrf_score)
                .or_insert((rrf_score, result.clone()));
        }
    }

    let mut fused: Vec<(f64, MemoryResult)> = scores.into_values().collect();
    fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);

    fused
        .into_iter()
        .map(|(score, mut result)| {
            result.score = Some(score as f32);
            result
        })
        .collect()
}

#[allow(dead_code)] // available for inference-mode queries (semantic disconnect)
/// CombMAX fusion: for each document, take the max score across all channels.
/// "Strong signals should not be diluted" (Kumiho philosophy).
/// Best for queries where only one channel finds the relevant result (semantic disconnect).
pub fn combmax_fuse(
    ranked_lists: &[(&[MemoryResult], f64)], // (results, weight) pairs
    limit: usize,
) -> Vec<MemoryResult> {
    let mut scores: HashMap<String, (f64, MemoryResult)> = HashMap::new();

    for (results, weight) in ranked_lists {
        for (rank, result) in results.iter().enumerate() {
            // Normalize: higher weight + lower rank = higher score
            let channel_score = weight / (1.0 + rank as f64);
            scores
                .entry(result.id.clone())
                .and_modify(|(s, _)| {
                    if channel_score > *s {
                        *s = channel_score;
                    }
                })
                .or_insert((channel_score, result.clone()));
        }
    }

    let mut fused: Vec<(f64, MemoryResult)> = scores.into_values().collect();
    fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);

    fused
        .into_iter()
        .map(|(score, mut result)| {
            result.score = Some(score as f32);
            result
        })
        .collect()
}

/// RRF fusion for Episode lists.
pub fn rrf_fuse_episodes(
    ranked_lists: &[(&[Episode], f64)],
    k: usize,
    limit: usize,
) -> Vec<Episode> {
    let mut scores: HashMap<String, (f64, Episode)> = HashMap::new();

    for (results, weight) in ranked_lists {
        for (rank, result) in results.iter().enumerate() {
            let rrf_score = weight / (k as f64 + (rank + 1) as f64);
            scores
                .entry(result.episode_id.clone())
                .and_modify(|(s, _)| *s += rrf_score)
                .or_insert((rrf_score, result.clone()));
        }
    }

    let mut fused: Vec<(f64, Episode)> = scores.into_values().collect();
    fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);

    fused
        .into_iter()
        .map(|(score, mut result)| {
            result.score = Some(score as f32);
            result
        })
        .collect()
}

/// Tokenize text into lowercase words by splitting on whitespace and punctuation.
///
/// This is the default tokenization (UnicodeTokenizer) kept for backward compatibility.
/// For locale-aware tokenization, use [`tokenize_words_with`].
#[allow(dead_code)] // used in tests; kept as convenience API
pub(crate) fn tokenize_words(text: &str) -> Vec<String> {
    crate::tokenizer::UnicodeTokenizer.tokenize(text)
}

/// Tokenize text using the given locale-aware tokenizer.
pub(crate) fn tokenize_words_with(text: &str, tokenizer: &dyn Tokenizer) -> Vec<String> {
    tokenizer.tokenize(text)
}

/// Compute word overlap score between a query and a candidate text.
///
/// Returns the fraction of query tokens that appear in the candidate (0.0 to 1.0).
pub(crate) fn word_overlap_score(query_tokens: &[String], candidate: &str) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let candidate_lower = candidate.to_lowercase();
    let matched = query_tokens
        .iter()
        .filter(|qt| candidate_lower.contains(qt.as_str()))
        .count();
    matched as f32 / query_tokens.len() as f32
}

/// Score all candidates by word overlap with the query, returning them sorted
/// by overlap score (descending). Only candidates with score > 0 are returned.
///
/// Uses the default UnicodeTokenizer. For locale-aware overlap, use
/// [`word_overlap_rank_with`].
#[allow(dead_code)] // used in tests; kept as convenience API
pub(crate) fn word_overlap_rank(query: &str, candidates: &[MemoryResult]) -> Vec<MemoryResult> {
    word_overlap_rank_with(query, candidates, &crate::tokenizer::UnicodeTokenizer)
}

/// Score all candidates by word overlap using a locale-aware tokenizer.
pub(crate) fn word_overlap_rank_with(
    query: &str,
    candidates: &[MemoryResult],
    tokenizer: &dyn Tokenizer,
) -> Vec<MemoryResult> {
    let query_tokens = tokenize_words_with(query, tokenizer);
    if query_tokens.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<(f32, MemoryResult)> = candidates
        .iter()
        .filter_map(|c| {
            let score = word_overlap_score(&query_tokens, &c.content);
            if score > 0.0 {
                let mut r = c.clone();
                r.score = Some(score);
                Some((score, r))
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, r)| r).collect()
}

/// RRF fusion for Event lists.
#[allow(dead_code)]
pub fn rrf_fuse_events(ranked_lists: &[(&[Event], f64)], k: usize, limit: usize) -> Vec<Event> {
    let mut scores: HashMap<String, (f64, Event)> = HashMap::new();

    for (results, weight) in ranked_lists {
        for (rank, result) in results.iter().enumerate() {
            let rrf_score = weight / (k as f64 + (rank + 1) as f64);
            scores
                .entry(result.event_id.clone())
                .and_modify(|(s, _)| *s += rrf_score)
                .or_insert((rrf_score, result.clone()));
        }
    }

    let mut fused: Vec<(f64, Event)> = scores.into_values().collect();
    fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);

    fused.into_iter().map(|(_, result)| result).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, content: &str) -> MemoryResult {
        MemoryResult {
            id: id.to_string(),
            content: content.to_string(),
            user_id: "user1".to_string(),
            agent_id: None,
            app_id: None,
            run_id: None,
            score: None,
            created_at: "2024-01-01".to_string(),
            updated_at: "2024-01-01".to_string(),
            metadata: None,
            importance: None,
            access_count: None,
            immutable: false,
            expiration_date: None,
            categories: None,
            memory_type: None,
            retention: None,
            stability: None,
            privacy: "syncable".to_string(),
            event_time: None,
            episode_id: None,
            session_id: None,
            resolution: crate::types::Resolution::Granular,
        }
    }

    #[test]
    fn test_rrf_single_list() {
        let results = vec![make_result("a", "alpha"), make_result("b", "beta")];
        let lists: Vec<(&[MemoryResult], f64)> = vec![(results.as_slice(), 1.0)];
        let fused = rrf_fuse(&lists, 60, 10);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].id, "a");
        assert_eq!(fused[1].id, "b");

        // score(a) = 1.0 / (60 + 1) = 1/61
        let expected_a = 1.0_f64 / 61.0;
        assert!((fused[0].score.unwrap() as f64 - expected_a).abs() < 1e-6);

        // score(b) = 1.0 / (60 + 2) = 1/62
        let expected_b = 1.0_f64 / 62.0;
        assert!((fused[1].score.unwrap() as f64 - expected_b).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_two_lists_overlap() {
        let list1 = vec![make_result("a", "alpha"), make_result("b", "beta")];
        let list2 = vec![make_result("b", "beta"), make_result("a", "alpha")];

        let lists: Vec<(&[MemoryResult], f64)> =
            vec![(list1.as_slice(), 0.7), (list2.as_slice(), 0.3)];
        let fused = rrf_fuse(&lists, 60, 10);

        assert_eq!(fused.len(), 2);

        // score(a) = 0.7/(60+1) + 0.3/(60+2) = 0.7/61 + 0.3/62
        let expected_a = 0.7 / 61.0 + 0.3 / 62.0;
        // score(b) = 0.7/(60+2) + 0.3/(60+1) = 0.7/62 + 0.3/61
        let expected_b = 0.7 / 62.0 + 0.3 / 61.0;

        // "a" should score higher since it's rank 1 in the higher-weight list
        assert!(expected_a > expected_b);

        let a_result = fused.iter().find(|r| r.id == "a").unwrap();
        let b_result = fused.iter().find(|r| r.id == "b").unwrap();
        assert!((a_result.score.unwrap() as f64 - expected_a).abs() < 1e-6);
        assert!((b_result.score.unwrap() as f64 - expected_b).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_two_lists_no_overlap() {
        let list1 = vec![make_result("a", "alpha")];
        let list2 = vec![make_result("b", "beta")];

        let lists: Vec<(&[MemoryResult], f64)> =
            vec![(list1.as_slice(), 0.7), (list2.as_slice(), 0.3)];
        let fused = rrf_fuse(&lists, 60, 10);

        assert_eq!(fused.len(), 2);

        // "a" only in list1: score = 0.7/61
        // "b" only in list2: score = 0.3/61
        let a_result = fused.iter().find(|r| r.id == "a").unwrap();
        let b_result = fused.iter().find(|r| r.id == "b").unwrap();
        assert!((a_result.score.unwrap() as f64 - 0.7 / 61.0).abs() < 1e-6);
        assert!((b_result.score.unwrap() as f64 - 0.3 / 61.0).abs() < 1e-6);

        // "a" should be ranked first
        assert_eq!(fused[0].id, "a");
    }

    #[test]
    fn test_tokenize_words_basic() {
        let tokens = tokenize_words("Hello, World! How are you?");
        assert_eq!(tokens, vec!["hello", "world", "how", "are", "you"]);
    }

    #[test]
    fn test_tokenize_words_empty() {
        let tokens = tokenize_words("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_word_overlap_exact_match() {
        let tokens = tokenize_words("coffee shop");
        let score = word_overlap_score(&tokens, "I went to a coffee shop");
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_overlap_partial_match() {
        let tokens = tokenize_words("coffee shop tokyo");
        let score = word_overlap_score(&tokens, "I went to a coffee shop");
        // 2 out of 3 tokens match
        assert!((score - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_word_overlap_no_match() {
        let tokens = tokenize_words("xyz123");
        let score = word_overlap_score(&tokens, "I went to a coffee shop");
        assert!((score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_overlap_rank_ordering() {
        let candidates = vec![
            make_result("a", "quantum physics lecture notes"),
            make_result("b", "the cat sat on the mat"),
            make_result("c", "the cat loves fish and the cat plays"),
        ];
        let ranked = word_overlap_rank("the cat", &candidates);
        // "b" and "c" both contain "the" and "cat"; "a" may contain partial
        assert!(!ranked.is_empty());
        // All returned should have score > 0
        for r in &ranked {
            assert!(r.score.unwrap() > 0.0);
        }
    }

    #[test]
    fn test_word_overlap_empty_query() {
        let candidates = vec![make_result("a", "hello world")];
        let ranked = word_overlap_rank("", &candidates);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_tokenize_words_with_cjk_tokenizer() {
        let t = crate::tokenizer::CJKTokenizer;
        let tokens = tokenize_words_with("我喜欢coffee", &t);
        assert_eq!(tokens, vec!["我", "喜", "欢", "coffee"]);
    }

    #[test]
    fn test_word_overlap_cjk() {
        let t = crate::tokenizer::CJKTokenizer;
        let query_tokens = tokenize_words_with("咖啡", &t);
        // query tokens: ["咖", "啡"]
        let score = word_overlap_score(&query_tokens, "我喜欢咖啡");
        // Both "咖" and "啡" are in the candidate via contains()
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_overlap_rank_with_cjk() {
        let candidates = vec![
            make_result("a", "我喜欢咖啡和茶"),
            make_result("b", "the cat sat on the mat"),
            make_result("c", "今天天气很好"),
        ];
        let t = crate::tokenizer::CJKTokenizer;
        let ranked = word_overlap_rank_with("咖啡", &candidates, &t);
        // Only "a" contains 咖 and 啡
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].id, "a");
    }

    #[test]
    fn test_word_overlap_rank_with_auto() {
        let candidates = vec![
            make_result("a", "我喜欢咖啡"),
            make_result("b", "I like coffee"),
        ];
        let t = crate::tokenizer::AutoTokenizer;
        let ranked = word_overlap_rank_with("咖啡", &candidates, &t);
        // "咖" and "啡" should match in candidate "a"
        assert!(!ranked.is_empty());
        assert_eq!(ranked[0].id, "a");
    }

    #[test]
    fn test_rrf_limit() {
        let results = vec![
            make_result("a", "alpha"),
            make_result("b", "beta"),
            make_result("c", "gamma"),
        ];
        let lists: Vec<(&[MemoryResult], f64)> = vec![(results.as_slice(), 1.0)];
        let fused = rrf_fuse(&lists, 60, 2);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].id, "a");
        assert_eq!(fused[1].id, "b");
    }
}
