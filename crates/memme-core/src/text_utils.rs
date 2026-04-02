/// Split text into overlapping chunks at sentence boundaries.
///
/// `chunk_size` is the target chunk size in characters.
/// `overlap` is the number of overlap characters between consecutive chunks.
#[allow(dead_code)] // reserved for future long-text chunking in meditate
pub(crate) fn split_into_chunks(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    // Split into sentences (by `. `, `? `, `! `, or newlines)
    let mut sentences: Vec<&str> = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        let is_boundary = matches!(bytes[i], b'.' | b'?' | b'!')
            && i + 1 < bytes.len()
            && (bytes[i + 1] == b' ' || bytes[i + 1] == b'\n');
        let is_newline = bytes[i] == b'\n';
        if is_boundary || is_newline {
            let end = if is_newline { i } else { i + 1 };
            let sentence = &text[start..=end];
            let trimmed = sentence.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            start = end + 1;
        }
    }
    // Remainder
    if start < text.len() {
        let remainder = text[start..].trim();
        if !remainder.is_empty() {
            sentences.push(remainder);
        }
    }

    if sentences.is_empty() {
        return vec![text.to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();
    let mut chunk_start_idx = 0;

    for (idx, sentence) in sentences.iter().enumerate() {
        if current_chunk.is_empty() {
            current_chunk.push_str(sentence);
        } else if current_chunk.len() + sentence.len() < chunk_size {
            current_chunk.push(' ');
            current_chunk.push_str(sentence);
        } else {
            // Current chunk is full, save it
            chunks.push(current_chunk.clone());

            // Build overlap: go backwards from current position to find overlap sentences
            current_chunk.clear();
            let mut overlap_text = String::new();
            let mut j = idx;
            while j > chunk_start_idx {
                j -= 1;
                let candidate = format!(
                    "{}{}",
                    if overlap_text.is_empty() { "" } else { " " },
                    sentences[j]
                );
                if overlap_text.len() + candidate.len() > overlap {
                    break;
                }
                overlap_text = if overlap_text.is_empty() {
                    sentences[j].to_string()
                } else {
                    format!("{} {}", sentences[j], overlap_text)
                };
            }
            if !overlap_text.is_empty() {
                current_chunk.push_str(&overlap_text);
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
            chunk_start_idx = idx;
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

/// Extract a date prefix from text like `[Date: 1:56 pm on 8 May, 2023]`.
/// Returns the date string if found.
#[allow(dead_code)] // reserved for future [Date:] prefix handling
pub(crate) fn extract_date_prefix(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if let Some(rest) = trimmed
        .strip_prefix("[Date:")
        .or_else(|| trimmed.strip_prefix("[date:"))
    {
        if let Some(end) = rest.find(']') {
            let date_str = rest[..end].trim();
            if !date_str.is_empty() {
                return Some(date_str.to_string());
            }
        }
    }
    None
}

/// Extract an ISO date pattern (YYYY-MM-DD, YYYY-MM, or YYYY) from text.
/// Returns the first match found.
pub(crate) fn extract_iso_date_from_text(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    for i in 0..len.saturating_sub(3) {
        // Look for 4-digit year starting with 19 or 20
        if i + 3 < len
            && (bytes[i] == b'1' && bytes[i + 1] == b'9'
                || bytes[i] == b'2' && bytes[i + 1] == b'0')
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
        {
            // Check it's not part of a longer number
            if i > 0 && bytes[i - 1].is_ascii_digit() {
                continue;
            }
            // Try YYYY-MM-DD
            if i + 9 < len
                && bytes[i + 4] == b'-'
                && bytes[i + 7] == b'-'
                && bytes[i + 5].is_ascii_digit()
                && bytes[i + 6].is_ascii_digit()
                && bytes[i + 8].is_ascii_digit()
                && bytes[i + 9].is_ascii_digit()
            {
                let date = &text[i..i + 10];
                // Verify not followed by more digits
                if i + 10 >= len || !bytes[i + 10].is_ascii_digit() {
                    return Some(date.to_string());
                }
            }
            // Try YYYY-MM
            if i + 6 < len
                && bytes[i + 4] == b'-'
                && bytes[i + 5].is_ascii_digit()
                && bytes[i + 6].is_ascii_digit()
                && (i + 7 >= len || !bytes[i + 7].is_ascii_digit())
            {
                let date = &text[i..i + 7];
                return Some(date.to_string());
            }
            // Just YYYY
            if i + 4 >= len || !bytes[i + 4].is_ascii_digit() {
                let year = &text[i..i + 4];
                return Some(year.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_into_chunks_short() {
        let text = "Hello world.";
        let chunks = split_into_chunks(text, 2500, 300);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_extract_date_prefix_standard() {
        let text = "[Date: 1:56 pm on 8 May, 2023]\nAlice: Hello";
        assert_eq!(
            extract_date_prefix(text),
            Some("1:56 pm on 8 May, 2023".to_string())
        );
    }

    #[test]
    fn test_extract_date_prefix_none() {
        assert_eq!(extract_date_prefix("Just some text"), None);
    }

    #[test]
    fn test_extract_iso_date_full() {
        assert_eq!(
            extract_iso_date_from_text("moved on 2023-05-07"),
            Some("2023-05-07".to_string())
        );
    }

    #[test]
    fn test_extract_iso_date_year_month() {
        assert_eq!(
            extract_iso_date_from_text("in 2020-03"),
            Some("2020-03".to_string())
        );
    }

    #[test]
    fn test_extract_iso_date_year_only() {
        assert_eq!(
            extract_iso_date_from_text("since 2019"),
            Some("2019".to_string())
        );
    }
}
