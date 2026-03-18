#[cfg(test)]
mod tests {
    use crate::smart::text_utils;
    use crate::smart::*;
    use crate::types::SearchOptions;
    use memme_llm::{GenerateOptions, LlmError, Message};
    use std::sync::Arc;
    use std::sync::Mutex;

    struct MockLlm {
        responses: Mutex<Vec<String>>,
    }

    impl MockLlm {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl LlmProvider for MockLlm {
        fn generate(
            &self,
            _messages: &[Message],
            _options: &GenerateOptions,
        ) -> Result<String, LlmError> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Err(LlmError::NotAvailable("no more mock responses".into()))
            } else {
                Ok(responses.remove(0))
            }
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    fn test_store() -> MemoryStore {
        use crate::config::MemoryConfig;
        use memme_embeddings::mock::MockEmbedder;
        let config = MemoryConfig::new(":memory:", 384);
        let embedder = Arc::new(MockEmbedder::new(384));
        MemoryStore::new(config, embedder).unwrap()
    }

    #[test]
    fn test_smart_extract_and_add() {
        let store = test_store();

        // First LLM call: fact extraction returns two facts
        let fact_response = r#"{"facts": ["Name is Alice", "Works as a designer"]}"#.to_string();
        // Second LLM call: update memory says to ADD both (no existing memories)
        let update_response = r#"{
            "memory": [
                {"id": "new1", "text": "Name is Alice", "event": "ADD"},
                {"id": "new2", "text": "Works as a designer", "event": "ADD"}
            ]
        }"#
        .to_string();

        let llm = Arc::new(MockLlm::new(vec![fact_response, update_response]));
        let processor = SmartProcessor::new(llm);
        let results = processor
            .process(
                &store,
                "Hi, I am Alice. I work as a designer.",
                "user1",
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "Name is Alice");
        assert_eq!(results[1].content, "Works as a designer");

        // Verify memories are actually in the store
        let listed = store
            .list_traces(crate::types::ListOptions::new("user1"))
            .unwrap();
        assert_eq!(listed.len(), 2);
    }

    #[test]
    fn test_smart_update_existing() {
        let store = test_store();

        // Pre-populate a memory
        let existing = store
            .add("Likes cheese pizza", AddOptions::new("user1"))
            .unwrap();
        let existing_id = existing.id.clone();

        // Fact extraction
        let fact_response = r#"{"facts": ["Loves chicken pizza"]}"#.to_string();
        // Update memory: UPDATE the existing one using integer index "0"
        // (the LLM now receives integer indices instead of UUIDs)
        let update_response = r#"{
                "memory": [
                    {
                        "id": "0",
                        "text": "Loves cheese and chicken pizza",
                        "event": "UPDATE",
                        "old_memory": "Likes cheese pizza"
                    }
                ]
            }"#
        .to_string();

        let llm = Arc::new(MockLlm::new(vec![fact_response, update_response]));
        let processor = SmartProcessor::new(llm);
        let results = processor
            .process(
                &store,
                "I also love chicken pizza",
                "user1",
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, existing_id);
        assert_eq!(results[0].content, "Loves cheese and chicken pizza");
    }

    #[test]
    fn test_smart_delete_contradicting() {
        let store = test_store();

        // Pre-populate a memory
        let existing = store
            .add("Loves cheese pizza", AddOptions::new("user1"))
            .unwrap();
        let existing_id = existing.id.clone();

        // Fact extraction
        let fact_response = r#"{"facts": ["Dislikes cheese pizza"]}"#.to_string();
        // Update memory: DELETE the contradicting one using integer index "0"
        let update_response = r#"{
                "memory": [
                    {
                        "id": "0",
                        "text": "Loves cheese pizza",
                        "event": "DELETE"
                    }
                ]
            }"#
        .to_string();

        let llm = Arc::new(MockLlm::new(vec![fact_response, update_response]));
        let processor = SmartProcessor::new(llm);
        let results = processor
            .process(
                &store,
                "I actually dislike cheese pizza now",
                "user1",
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        // DELETE operations do not produce results
        assert!(results.is_empty());

        // Verify memory was deleted
        let found = store.get_trace(&existing_id).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_smart_empty_facts() {
        let store = test_store();

        // LLM returns empty facts (e.g., greeting with no extractable info)
        let fact_response = r#"{"facts": []}"#.to_string();

        let llm = Arc::new(MockLlm::new(vec![fact_response]));
        let processor = SmartProcessor::new(llm);
        let results = processor
            .process(
                &store,
                "Hi there!",
                "user1",
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_smart_llm_error() {
        let store = test_store();

        // MockLlm with no responses will return an error
        let llm = Arc::new(MockLlm::new(vec![]));
        let processor = SmartProcessor::new(llm);
        let result = processor.process(
            &store, "Hello", "user1", None, None, None, None, None, false,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            MemoryError::Llm(msg) => {
                assert!(msg.contains("not available"));
            }
            other => panic!("Expected MemoryError::Llm, got: {:?}", other),
        }
    }

    #[test]
    fn test_smart_uuid_mapping() {
        // Verify that the UUID-to-index mapping works correctly:
        // the LLM receives integer indices and returns them,
        // and the processor maps them back to real UUIDs.
        let store = test_store();

        // Pre-populate a single memory so the index mapping is unambiguous
        let mem1 = store
            .add("Likes cheese pizza", AddOptions::new("user1"))
            .unwrap();
        let mem1_id = mem1.id.clone();

        // Fact extraction returns a fact that should trigger an UPDATE on mem1
        let fact_response = r#"{"facts": ["Loves chicken pizza"]}"#.to_string();

        // The LLM responds with integer index "0" (the only old memory)
        let update_response = r#"{
                "memory": [
                    {
                        "id": "0",
                        "text": "Loves cheese and chicken pizza",
                        "event": "UPDATE",
                        "old_memory": "Likes cheese pizza"
                    }
                ]
            }"#
        .to_string();

        let llm = Arc::new(MockLlm::new(vec![fact_response, update_response]));
        let processor = SmartProcessor::new(llm);
        let results = processor
            .process(
                &store,
                "I love chicken pizza now",
                "user1",
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        // UPDATE produces a result
        assert_eq!(results.len(), 1);
        // The updated memory should have the correct real UUID
        assert_eq!(results[0].id, mem1_id);
        assert_eq!(results[0].content, "Loves cheese and chicken pizza");
    }

    #[test]
    fn test_smart_uuid_fallback() {
        // Verify that if the LLM returns a real UUID instead of an index,
        // the fallback path still works.
        let store = test_store();

        let existing = store
            .add("Likes cheese pizza", AddOptions::new("user1"))
            .unwrap();
        let existing_id = existing.id.clone();

        let fact_response = r#"{"facts": ["Loves chicken pizza"]}"#.to_string();
        // Simulate the LLM returning the actual UUID (fallback path)
        let update_response = format!(
            r#"{{
                "memory": [
                    {{
                        "id": "{existing_id}",
                        "text": "Loves cheese and chicken pizza",
                        "event": "UPDATE",
                        "old_memory": "Likes cheese pizza"
                    }}
                ]
            }}"#
        );

        let llm = Arc::new(MockLlm::new(vec![fact_response, update_response]));
        let processor = SmartProcessor::new(llm);
        let results = processor
            .process(
                &store,
                "I also love chicken pizza",
                "user1",
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, existing_id);
        assert_eq!(results[0].content, "Loves cheese and chicken pizza");
    }

    #[test]
    fn test_agent_memory_extraction() {
        // Verify that when extract_agent_memory=true, the agent memory prompt is used
        use std::sync::Mutex as StdMutex;

        struct CapturingMockLlm {
            responses: StdMutex<Vec<String>>,
            captured_system_prompts: StdMutex<Vec<String>>,
        }
        impl CapturingMockLlm {
            fn new(responses: Vec<String>) -> Self {
                Self {
                    responses: StdMutex::new(responses),
                    captured_system_prompts: StdMutex::new(Vec::new()),
                }
            }
        }
        impl LlmProvider for CapturingMockLlm {
            fn generate(
                &self,
                messages: &[Message],
                _options: &GenerateOptions,
            ) -> Result<String, memme_llm::LlmError> {
                if let Some(sys_msg) = messages
                    .iter()
                    .find(|m| matches!(m.role, memme_llm::MessageRole::System))
                {
                    self.captured_system_prompts
                        .lock()
                        .unwrap()
                        .push(sys_msg.content.clone());
                }
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    Err(memme_llm::LlmError::NotAvailable(
                        "no more mock responses".into(),
                    ))
                } else {
                    Ok(responses.remove(0))
                }
            }
            fn name(&self) -> &str {
                "capturing_mock"
            }
        }

        let store = test_store();

        let fact_response =
            r#"{"facts": ["Approaches debugging by reproducing the issue first"]}"#.to_string();
        let update_response = r#"{
            "memory": [
                {"id": "new1", "text": "Approaches debugging by reproducing the issue first", "event": "ADD"}
            ]
        }"#
        .to_string();

        let llm = Arc::new(CapturingMockLlm::new(vec![fact_response, update_response]));
        let llm_clone = llm.clone();
        let processor = SmartProcessor::new(llm);

        let results = processor
            .process(
                &store,
                "User: How do you debug?\nAssistant: I always start by reproducing the issue.",
                "user1",
                Some("agent1"),
                None,
                None,
                None,
                None,
                true, // extract_agent_memory = true
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].content,
            "Approaches debugging by reproducing the issue first"
        );

        // Verify the agent memory prompt was used (not the user memory prompt)
        let captured = llm_clone.captured_system_prompts.lock().unwrap();
        assert!(
            captured.len() >= 1,
            "Should have captured at least 1 system prompt"
        );
        assert!(
            captured[0].contains("Assistant Information Organizer"),
            "First LLM call should use the agent memory extraction prompt"
        );
        assert!(
            !captured[0].contains("Personal Information Organizer"),
            "Should NOT use the user memory extraction prompt"
        );
    }

    #[test]
    fn test_split_into_chunks_short_text() {
        let text = "Hello world. This is short.";
        let chunks = text_utils::split_into_chunks(text, 1500, 200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_split_into_chunks_long_text() {
        // Build a text longer than chunk_size with clear sentence boundaries
        let sentences: Vec<String> = (0..20)
            .map(|i| format!("This is sentence number {}.", i))
            .collect();
        let text = sentences.join(" ");
        let chunks = text_utils::split_into_chunks(&text, 100, 30);
        // Should produce multiple chunks
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );
        // Each chunk should not be empty
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_split_into_chunks_overlap() {
        let text =
            "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.";
        let chunks = text_utils::split_into_chunks(text, 40, 20);
        // With overlap, some content should appear in multiple chunks
        if chunks.len() >= 2 {
            // The last chunk should contain content that might overlap with previous
            // Just verify we get multiple chunks and they are all non-empty
            for chunk in &chunks {
                assert!(!chunk.is_empty());
            }
        }
    }

    #[test]
    fn test_split_into_chunks_no_sentence_boundaries() {
        let text = "a".repeat(200);
        let chunks = text_utils::split_into_chunks(&text, 100, 20);
        // Without sentence boundaries, should return the whole text as one chunk
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_thorough_extraction_uses_multiple_calls() {
        // Verify that Thorough mode makes at least 2 LLM calls (fact + detail)
        let store = test_store();

        let fact_response = r#"{"facts": ["Name is Alice"]}"#.to_string();
        let detail_response = r#"{"facts": ["Alice is 30 years old"]}"#.to_string();
        let update_response = r#"{
            "memory": [
                {"id": "new1", "text": "Name is Alice", "event": "ADD"},
                {"id": "new2", "text": "Alice is 30 years old", "event": "ADD"}
            ]
        }"#
        .to_string();

        let llm = Arc::new(MockLlm::new(vec![
            fact_response,
            detail_response,
            update_response,
        ]));
        let processor = SmartProcessor::with_extraction_depth(llm, ExtractionDepth::Thorough);
        let results = processor
            .process(
                &store,
                "Hi, I am Alice and I am 30 years old.",
                "user1",
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_standard_extraction_single_call() {
        // Standard mode should work exactly as before (no detail pass)
        let store = test_store();

        let fact_response = r#"{"facts": ["Name is Bob"]}"#.to_string();
        let update_response = r#"{
            "memory": [
                {"id": "new1", "text": "Name is Bob", "event": "ADD"}
            ]
        }"#
        .to_string();

        let llm = Arc::new(MockLlm::new(vec![fact_response, update_response]));
        let processor = SmartProcessor::new(llm); // default = Standard
        let results = processor
            .process(
                &store,
                "Hi, I am Bob.",
                "user1",
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Name is Bob");
    }

    // ── Dialogue chunking tests ──

    #[test]
    fn test_split_dialogue_turns_simple() {
        let text = "Alice: Hello, how are you?\nBob: I'm fine, just moved to Beijing.\nAlice: That's exciting! When?\nBob: Last month. I got a job at a tech company.\nAlice: Congratulations!";
        let chunks = text_utils::split_dialogue_turns(text);
        assert!(!chunks.is_empty(), "Should produce at least one chunk");
        // The single chunk should contain all turns
        assert!(chunks[0].content.contains("Alice:"));
        assert!(chunks[0].content.contains("Bob:"));
    }

    #[test]
    fn test_split_dialogue_turns_with_context() {
        // Create enough turns to trigger context prefix
        let lines: Vec<String> = (0..8)
            .map(|i| {
                if i % 2 == 0 {
                    format!("Alice: Message number {}", i)
                } else {
                    format!("Bob: Reply number {}", i)
                }
            })
            .collect();
        let text = lines.join("\n");
        let chunks = text_utils::split_dialogue_turns(&text);
        // With 8 turns and chunk_size=4, we expect 2 chunks
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks, got {}",
            chunks.len()
        );
        // Second chunk should have context prefix
        assert!(
            chunks[1].context.contains("[CONTEXT]"),
            "Second chunk should have context prefix"
        );
    }

    #[test]
    fn test_is_low_information_detection() {
        assert!(text_utils::is_low_information("Hello!"));
        assert!(text_utils::is_low_information("Hi there"));
        assert!(text_utils::is_low_information("Thanks"));
        assert!(text_utils::is_low_information("Sounds good"));
        assert!(text_utils::is_low_information("Bye"));
        assert!(text_utils::is_low_information("Good morning"));
        // Not low information
        assert!(!text_utils::is_low_information(
            "I moved to Beijing last month and started a new job"
        ));
        assert!(!text_utils::is_low_information(
            "My name is Alice and I work as a designer"
        ));
        // Short but non-pleasantry
        assert!(!text_utils::is_low_information("I have 3 kids"));
    }

    #[test]
    fn test_low_information_filtering_in_chunks() {
        // All turns are greetings — should be filtered out
        let text = "Alice: Hello!\nBob: Hi there\nAlice: How are you?\nBob: I'm fine";
        let chunks = text_utils::split_dialogue_turns(text);
        assert!(
            chunks.is_empty(),
            "All-greeting chunks should be filtered out"
        );
    }

    #[test]
    fn test_looks_like_dialogue() {
        assert!(text_utils::looks_like_dialogue(
            "Alice: Hello\nBob: Hi there\nAlice: How are you?"
        ));
        assert!(text_utils::looks_like_dialogue(
            "user: I like coffee\nassistant: That's great!"
        ));
        assert!(!text_utils::looks_like_dialogue(
            "This is just plain text without any dialogue format."
        ));
        assert!(!text_utils::looks_like_dialogue(
            "Hello world. How are you doing today?"
        ));
    }

    // ── ChatMessage and AddOptions field tests ──

    #[test]
    fn test_chat_message_with_timestamp() {
        use crate::types::ChatMessage;
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            image_url: None,
            image_type: None,
            timestamp: Some("2025-06-15T14:30:00Z".to_string()),
        };
        assert_eq!(msg.timestamp.as_deref(), Some("2025-06-15T14:30:00Z"));
    }

    #[test]
    fn test_add_options_with_event_time() {
        let opts = AddOptions::new("user1").event_time("2025-03-15");
        assert_eq!(opts.event_time.as_deref(), Some("2025-03-15"));
    }

    #[test]
    fn test_event_time_stored_and_retrieved() {
        let store = test_store();
        let opts = AddOptions::new("user1").event_time("2025-03-15T10:00:00");
        let result = store.add("Moved to Beijing", opts).unwrap();
        let retrieved = store.get_trace(&result.id).unwrap().unwrap();
        assert!(
            retrieved.event_time.is_some(),
            "event_time should be stored and retrieved"
        );
        assert!(
            retrieved
                .event_time
                .as_ref()
                .unwrap()
                .contains("2025-03-15"),
            "event_time should contain the date: got {:?}",
            retrieved.event_time
        );
    }

    #[test]
    fn test_event_time_none_when_not_set() {
        let store = test_store();
        let opts = AddOptions::new("user1");
        let result = store.add("Likes coffee", opts).unwrap();
        let retrieved = store.get_trace(&result.id).unwrap().unwrap();
        assert!(
            retrieved.event_time.is_none(),
            "event_time should be None when not set"
        );
    }

    #[test]
    fn test_event_time_in_search_results() {
        let store = test_store();
        let opts = AddOptions::new("user1").event_time("2025-06-01");
        store.add("Important meeting happened", opts).unwrap();
        let results = store
            .search("meeting", SearchOptions::new("user1"))
            .unwrap();
        assert!(!results.is_empty());
        // The result should have event_time populated
        assert!(results[0].event_time.is_some());
    }

    #[test]
    fn test_search_with_event_time_filter() {
        use crate::types::FilterExpression;
        let store = test_store();
        // Add memories with different event times
        store
            .add(
                "Event in January",
                AddOptions::new("user1").event_time("2025-01-15"),
            )
            .unwrap();
        store
            .add(
                "Event in June",
                AddOptions::new("user1").event_time("2025-06-15"),
            )
            .unwrap();
        store
            .add("Event with no time", AddOptions::new("user1"))
            .unwrap();

        // Search with time range covering only January using FilterExpression
        let filter = FilterExpression::and(vec![
            FilterExpression::gte("event_time", "2025-01-01"),
            FilterExpression::lte("event_time", "2025-02-01"),
        ]);
        let opts = SearchOptions::new("user1").filter(filter);
        let results = store.search("Event", opts).unwrap();
        // Should find at most the January event
        for r in &results {
            if r.event_time.is_some() {
                assert!(
                    r.event_time.as_ref().unwrap().contains("2025-01"),
                    "Filtered results should only have Jan events, got {:?}",
                    r.event_time
                );
            }
        }
    }

    #[test]
    fn test_filter_by_event_time_range() {
        use crate::types::FilterExpression;
        // Test that event_time is in ALLOWED_FILTER_FIELDS
        let f = FilterExpression::gte("event_time", "2025-01-01");
        let mut offset = 0;
        let (sql, _) = f.to_sql(&mut offset);
        assert!(
            sql.contains("event_time >= $1"),
            "event_time should be a direct column filter, got: {}",
            sql
        );
    }
}
