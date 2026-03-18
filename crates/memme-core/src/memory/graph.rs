use crate::error::Result;
use crate::types::GraphSearchResult;

impl super::MemoryStore {
    /// Add knowledge graph entries from text using LLM extraction.
    ///
    /// Extracts entities and relationships from the text using the LLM,
    /// then stores them in the knowledge graph tables.
    pub fn add_graph(
        &self,
        text: &str,
        user_id: &str,
        llm: std::sync::Arc<dyn memme_llm::LlmProvider>,
    ) -> Result<GraphSearchResult> {
        let processor = crate::graph::GraphProcessor::new(llm);
        processor.process(&self.storage, text, user_id)
    }

    /// Search the knowledge graph using SQL LIKE search and neighborhood traversal.
    ///
    /// This is a pure SQL operation — no LLM required.
    pub fn search_graph(
        &self,
        query: &str,
        user_id: &str,
        depth: usize,
    ) -> Result<GraphSearchResult> {
        crate::graph::search_graph(&self.storage, query, user_id, depth)
    }
}
