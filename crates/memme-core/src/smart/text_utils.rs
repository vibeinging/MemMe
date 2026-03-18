/// A single dialogue turn parsed from conversation text.
pub(crate) struct DialogueTurn {
    pub speaker: String,
    pub text: String,
    pub timestamp: Option<String>,
}

/// A chunk of dialogue turns with context for extraction.
pub(crate) struct DialogueChunk {
    /// Previous 2 turns for reference (not extracted).
    pub context: String,
    /// 3-5 turns to extract from.
    pub content: String,
    /// Timestamp from the first turn if available.
    pub timestamp: Option<String>,
}

/// Split text into overlapping chunks at sentence boundaries.
///
/// `chunk_size` is the target chunk size in characters.
/// `overlap` is the number of overlap characters between consecutive chunks.
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

/// Check if a line of text is low-information (greetings, pleasantries).
pub(crate) fn is_low_information(text: &str) -> bool {
    let lower = text.to_lowercase();
    let pleasantries = [
        "how are you",
        "i'm fine",
        "good morning",
        "hello",
        "hi there",
        "see you",
        "bye",
        "take care",
        "talk to you later",
        "sounds good",
        "okay",
        "thanks",
        "thank you",
        "you're welcome",
        "no problem",
        "good night",
        "good evening",
        "good afternoon",
    ];
    text.len() < 50 && pleasantries.iter().any(|p| lower.contains(p))
}

/// Extract a date prefix from text like `[Date: 1:56 pm on 8 May, 2023]`.
/// Returns the date string if found.
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

/// Check whether text looks like a dialogue (has "Speaker: text" patterns).
pub(crate) fn looks_like_dialogue(text: &str) -> bool {
    let mut speaker_lines = 0;
    let mut total_lines = 0;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        total_lines += 1;
        // Match "word: text" pattern (speaker turns)
        if let Some(colon_pos) = trimmed.find(':') {
            let prefix = &trimmed[..colon_pos];
            // Speaker name should be 1-30 chars, no newlines, mostly alpha
            if !prefix.is_empty()
                && prefix.chars().count() <= 30
                && prefix
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == ' ')
                && colon_pos + 1 < trimmed.len()
            {
                speaker_lines += 1;
            }
        }
    }
    total_lines >= 2 && speaker_lines as f32 / total_lines as f32 > 0.4
}

/// Split dialogue text into turn-aware chunks.
/// Groups 3-5 speaker turns together, with 2-turn read-only context prefix.
/// Filters out groups where all turns are low-information.
pub(crate) fn split_dialogue_turns(text: &str) -> Vec<DialogueChunk> {
    // Parse turns: detect "Speaker: text" patterns
    let mut turns: Vec<DialogueTurn> = Vec::new();
    let mut current_speaker = String::new();
    let mut current_text = String::new();
    let mut current_timestamp: Option<String> = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current_text.is_empty() {
                current_text.push('\n');
            }
            continue;
        }

        // Try to match "Speaker: text" pattern
        if let Some(colon_pos) = trimmed.find(':') {
            let prefix = &trimmed[..colon_pos];
            if !prefix.is_empty()
                && prefix.chars().count() <= 30
                && prefix
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == ' ')
                && colon_pos + 1 < trimmed.len()
            {
                // Save previous turn
                if !current_speaker.is_empty() && !current_text.trim().is_empty() {
                    turns.push(DialogueTurn {
                        speaker: current_speaker.clone(),
                        text: current_text.trim().to_string(),
                        timestamp: current_timestamp.take(),
                    });
                }
                current_speaker = prefix.to_string();
                current_text = trimmed[colon_pos + 1..].trim().to_string();
                continue;
            }
        }
        // Continuation of current turn
        if !current_text.is_empty() {
            current_text.push(' ');
        }
        current_text.push_str(trimmed);
    }
    // Save last turn
    if !current_speaker.is_empty() && !current_text.trim().is_empty() {
        turns.push(DialogueTurn {
            speaker: current_speaker,
            text: current_text.trim().to_string(),
            timestamp: current_timestamp,
        });
    }

    if turns.is_empty() {
        return vec![];
    }

    // Group into chunks of 3-5 turns
    let chunk_size = 4; // target turns per chunk
    let mut chunks: Vec<DialogueChunk> = Vec::new();

    let mut i = 0;
    while i < turns.len() {
        let end = (i + chunk_size).min(turns.len());
        // If remaining turns are less than 3 after this chunk, include them
        let end = if turns.len() - end < 3 {
            turns.len()
        } else {
            end
        };

        // Build context from 2 previous turns
        let context = if i >= 2 {
            let ctx_turns: Vec<String> = turns[i - 2..i]
                .iter()
                .map(|t| format!("[CONTEXT] {}: {}", t.speaker, t.text))
                .collect();
            ctx_turns.join("\n")
        } else if i == 1 {
            format!("[CONTEXT] {}: {}", turns[0].speaker, turns[0].text)
        } else {
            String::new()
        };

        // Build content from current chunk turns
        let chunk_turns = &turns[i..end];
        let content: String = chunk_turns
            .iter()
            .map(|t| format!("{}: {}", t.speaker, t.text))
            .collect::<Vec<_>>()
            .join("\n");

        // Check if ALL turns in this chunk are low-information
        let all_low_info = chunk_turns.iter().all(|t| is_low_information(&t.text));

        if !all_low_info {
            let timestamp = chunk_turns.first().and_then(|t| t.timestamp.clone());

            chunks.push(DialogueChunk {
                context,
                content,
                timestamp,
            });
        }

        i = end;
    }

    chunks
}
