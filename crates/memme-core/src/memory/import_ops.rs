use uuid::Uuid;

use crate::error::Result;
use crate::import::{ImportConversationsResult, ImportedConversation};
use crate::types::ChatMessage;

impl super::MemoryStore {
    /// Import external conversations into MemMe.
    ///
    /// Each conversation becomes a session with events ingested via
    /// [`append_events`](Self::append_events). After importing, the caller
    /// can run `compact()` and `meditate()` on the created sessions to
    /// extract memories.
    ///
    /// The `user_id` parameter assigns ownership of all imported data.
    pub fn import_conversations(
        &self,
        conversations: &[ImportedConversation],
        user_id: &str,
    ) -> Result<ImportConversationsResult> {
        let mut total_sessions = 0u64;
        let mut total_events = 0u64;

        for conv in conversations {
            if conv.messages.is_empty() {
                continue;
            }

            let session_id = format!("import-{}-{}", conv.source, Uuid::new_v4());

            let messages: Vec<ChatMessage> = conv
                .messages
                .iter()
                .map(|(role, content)| ChatMessage {
                    role: role.clone(),
                    content: content.clone(),
                    image_url: None,
                    image_type: None,
                    timestamp: None,
                })
                .collect();

            let result = self.append_events(&session_id, &messages, user_id, None)?;

            total_sessions += 1;
            total_events += result.events_appended as u64;
        }

        Ok(ImportConversationsResult {
            sessions_created: total_sessions,
            events_created: total_events,
        })
    }
}
