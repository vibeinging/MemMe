use crate::{Message, MessageRole};

use super::helpers::today_string;

// ---------------------------------------------------------------------------
// Agent Memory Extraction Prompt
// ---------------------------------------------------------------------------

/// Build the complete message list for extracting facts about the AI assistant
/// (agent memory mode). This extracts preferences, capabilities, and
/// characteristics from assistant messages.
pub fn get_agent_memory_messages(text: &str) -> Vec<Message> {
    let today = today_string();
    let system = format!(
        r#"You are an Assistant Information Organizer, specialized in accurately storing facts, preferences, and characteristics about the AI assistant. Your primary role is to extract relevant pieces of information about the AI assistant from conversations and organize them into distinct, manageable facts.

Types of Information to Remember about the Assistant:

1. Assistant's Preferences and Personality Traits: Keep track of the assistant's communication style, tone preferences, and personality characteristics.
2. Assistant's Capabilities and Knowledge Areas: Remember what the assistant is good at, what tools or skills it has demonstrated.
3. Assistant's Approach to Tasks: Note how the assistant handles different types of requests, its problem-solving strategies.
4. Any Unique Characteristics: Record any distinctive behaviors, catchphrases, or patterns the assistant displays.

Here are some few shot examples:

Input: User: How do you approach debugging?\nAssistant: I always start by reproducing the issue, then I systematically narrow down the cause using binary search.
Output: {{"facts" : ["Approaches debugging by first reproducing the issue", "Uses binary search strategy to narrow down bug causes"]}}

Input: User: Hello\nAssistant: Hi there!
Output: {{"facts" : []}}

Input: User: Can you help with Python?\nAssistant: Absolutely! Python is one of my strongest areas. I particularly enjoy working with pandas and data analysis.
Output: {{"facts" : ["Python is one of the strongest areas", "Particularly enjoys working with pandas and data analysis"]}}

Return the facts and preferences in a json format as shown above.

Remember the following:
- Today's date is {today}.
- ONLY extract facts from ASSISTANT messages, ignore user messages.
- Do not return anything from the custom few shot example prompts provided above.
- If you do not find anything relevant in the below conversation, you can return an empty list corresponding to the "facts" key.
- Make sure to return the response in the format mentioned in the examples. The response should be in json with a key as "facts" and corresponding value will be a list of strings.
- You should detect the language of the input and record the facts in the same language.
- Each fact should be atomic — one piece of information per fact.

Following is a conversation between the user and the assistant. You have to extract the relevant facts and preferences about the AI assistant from the assistant's messages and return them in the json format as shown above."#,
        today = today
    );

    vec![
        Message {
            role: MessageRole::System,
            content: system,
        },
        Message {
            role: MessageRole::User,
            content: format!(
                "Extract facts about the assistant from the following conversation:\n\n{}",
                text
            ),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MessageRole;

    #[test]
    fn test_agent_memory_messages_format() {
        let msgs = get_agent_memory_messages(
            "User: How do you debug?\nAssistant: I always start by reproducing the issue.",
        );
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0].role, MessageRole::System));
        assert!(
            msgs[0].content.contains("Assistant Information Organizer"),
            "System prompt should identify as Assistant Information Organizer"
        );
        assert!(
            msgs[0]
                .content
                .contains("ONLY extract facts from ASSISTANT messages"),
            "System prompt should instruct to only extract from assistant messages"
        );
        assert!(
            msgs[1].content.contains("reproducing the issue"),
            "User message should contain the conversation text"
        );
    }
}
