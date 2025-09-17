# Task 016: Consolidate Cue Handling Patterns

## Status: Pending
## Priority: P1 - Architecture Quality
## Estimated Effort: 1 day
## Dependencies: Tasks 014 (error handling), 015 (function refactoring)

## Objective
Eliminate code duplication by creating a centralized cue handling strategy that replaces the 7 identical pattern-matching blocks scattered across different modules.

## Current State Analysis
- **Critical Issue**: Same cue pattern-matching logic duplicated in 7 locations
- **Files Affected**: `memory.rs`, `query/evidence.rs`, `store.rs`, `completion/` modules
- **Problem**: Changes require updates in multiple places, inconsistent behavior

## Implementation Plan

### 1. Create Centralized Cue Dispatcher (engram-core/src/cue/dispatcher.rs)

```rust
use crate::{Cue, CueType, Episode, Confidence, CueError};
use std::collections::HashMap;

/// Centralized cue handling with strategy pattern
pub struct CueDispatcher {
    handlers: HashMap<String, Box<dyn CueHandler>>,
}

/// Strategy interface for different cue types
pub trait CueHandler: Send + Sync {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError>;
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool;
    
    fn handler_name(&self) -> &'static str;
}

/// Context interface for cue handling operations
pub trait CueContext: Send + Sync {
    fn get_episodes(&self) -> &HashMap<String, Episode>;
    fn get_hnsw_index(&self) -> &dyn HnswIndex;
    fn compute_similarity(&self, a: &[f32; 768], b: &[f32; 768]) -> f32;
    fn apply_decay(&self, episode: &Episode, confidence: Confidence) -> Confidence;
}

impl CueDispatcher {
    pub fn new() -> Self {
        let mut dispatcher = Self {
            handlers: HashMap::new(),
        };
        
        // Register default handlers
        dispatcher.register_handler(Box::new(EmbeddingCueHandler));
        dispatcher.register_handler(Box::new(ContextCueHandler));
        dispatcher.register_handler(Box::new(SemanticCueHandler));
        dispatcher.register_handler(Box::new(TemporalCueHandler));
        
        dispatcher
    }
    
    pub fn register_handler(&mut self, handler: Box<dyn CueHandler>) {
        self.handlers.insert(handler.handler_name().to_string(), handler);
    }
    
    pub fn handle_cue(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        // Find appropriate handler for this cue type
        for handler in self.handlers.values() {
            if handler.supports_cue_type(&cue.cue_type) {
                return handler.handle(cue, context);
            }
        }
        
        Err(CueError::UnsupportedCueType {
            cue_type: format!("{:?}", cue.cue_type),
            operation: "cue_dispatch".to_string(),
        })
    }
}
```

### 2. Implement Specific Cue Handlers (engram-core/src/cue/handlers.rs)

```rust
use super::{CueHandler, CueContext};
use crate::{Cue, CueType, Episode, Confidence, CueError};

/// Handler for embedding-based cues
pub struct EmbeddingCueHandler;

impl CueHandler for EmbeddingCueHandler {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        match &cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                let mut results = Vec::new();
                
                // Use HNSW for efficient similarity search
                let candidates = context.get_hnsw_index().search_with_confidence(
                    vector,
                    cue.max_results * 2, // Over-fetch for better results
                    Confidence::from_raw(*threshold),
                );
                
                for (episode_id, hnsw_confidence) in candidates {
                    if let Some(episode) = context.get_episodes().get(&episode_id) {
                        let similarity = context.compute_similarity(vector, &episode.embedding);
                        
                        if similarity >= *threshold {
                            let confidence = Confidence::exact(similarity);
                            let decayed_confidence = context.apply_decay(episode, confidence);
                            results.push((episode.clone(), decayed_confidence));
                        }
                    }
                }
                
                Ok(results)
            }
            _ => Err(CueError::UnsupportedCueType {
                cue_type: "non-embedding".to_string(),
                operation: "embedding_handler".to_string(),
            })
        }
    }
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool {
        matches!(cue_type, CueType::Embedding { .. })
    }
    
    fn handler_name(&self) -> &'static str {
        "embedding"
    }
}

/// Handler for context-based cues
pub struct ContextCueHandler;

impl CueHandler for ContextCueHandler {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        match &cue.cue_type {
            CueType::Context { context: cue_context, threshold } => {
                let mut results = Vec::new();
                
                for episode in context.get_episodes().values() {
                    let mut match_confidence = 0.0f32;
                    
                    // Check location context
                    if let Some(location) = &episode.where_location {
                        let location_similarity = self.compute_text_similarity(cue_context, location);
                        match_confidence = match_confidence.max(location_similarity);
                    }
                    
                    // Check people context
                    if let Some(who_list) = &episode.who {
                        for person in who_list {
                            let person_similarity = self.compute_text_similarity(cue_context, person);
                            match_confidence = match_confidence.max(person_similarity);
                        }
                    }
                    
                    if match_confidence >= *threshold {
                        let confidence = Confidence::exact(match_confidence);
                        let decayed_confidence = context.apply_decay(episode, confidence);
                        results.push((episode.clone(), decayed_confidence));
                    }
                }
                
                Ok(results)
            }
            _ => Err(CueError::UnsupportedCueType {
                cue_type: "non-context".to_string(),
                operation: "context_handler".to_string(),
            })
        }
    }
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool {
        matches!(cue_type, CueType::Context { .. })
    }
    
    fn handler_name(&self) -> &'static str {
        "context"
    }
    
    fn compute_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simple text similarity (can be enhanced later)
        let text1_lower = text1.to_lowercase();
        let text2_lower = text2.to_lowercase();
        
        if text1_lower == text2_lower {
            1.0
        } else if text1_lower.contains(&text2_lower) || text2_lower.contains(&text1_lower) {
            0.8
        } else {
            // Jaccard similarity for basic text matching
            let words1: std::collections::HashSet<_> = text1_lower.split_whitespace().collect();
            let words2: std::collections::HashSet<_> = text2_lower.split_whitespace().collect();
            
            let intersection_size = words1.intersection(&words2).count();
            let union_size = words1.union(&words2).count();
            
            if union_size == 0 {
                0.0
            } else {
                intersection_size as f32 / union_size as f32
            }
        }
    }
}

/// Handler for semantic query cues
pub struct SemanticCueHandler;

impl CueHandler for SemanticCueHandler {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        match &cue.cue_type {
            CueType::Semantic { query, threshold } => {
                let mut results = Vec::new();
                
                for episode in context.get_episodes().values() {
                    let similarity = self.compute_semantic_similarity(query, &episode.what);
                    
                    if similarity >= *threshold {
                        let confidence = Confidence::exact(similarity);
                        let decayed_confidence = context.apply_decay(episode, confidence);
                        results.push((episode.clone(), decayed_confidence));
                    }
                }
                
                Ok(results)
            }
            _ => Err(CueError::UnsupportedCueType {
                cue_type: "non-semantic".to_string(),
                operation: "semantic_handler".to_string(),
            })
        }
    }
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool {
        matches!(cue_type, CueType::Semantic { .. })
    }
    
    fn handler_name(&self) -> &'static str {
        "semantic"
    }
    
    fn compute_semantic_similarity(&self, query: &str, episode_what: &str) -> f32 {
        // Enhanced text similarity for semantic matching
        let query_lower = query.to_lowercase();
        let what_lower = episode_what.to_lowercase();
        
        // Exact match
        if query_lower == what_lower {
            return 1.0;
        }
        
        // Substring match
        if what_lower.contains(&query_lower) {
            return 0.9;
        }
        
        // Word-level similarity with stemming consideration
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let what_words: Vec<&str> = what_lower.split_whitespace().collect();
        
        let mut matching_words = 0;
        for query_word in &query_words {
            for what_word in &what_words {
                if query_word == what_word || 
                   query_word.starts_with(what_word) || 
                   what_word.starts_with(query_word) {
                    matching_words += 1;
                    break;
                }
            }
        }
        
        if query_words.is_empty() {
            0.0
        } else {
            matching_words as f32 / query_words.len() as f32
        }
    }
}

/// Handler for temporal cues
pub struct TemporalCueHandler;

impl CueHandler for TemporalCueHandler {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        match &cue.cue_type {
            CueType::Temporal { when, threshold: _ } => {
                let mut results = Vec::new();
                
                for episode in context.get_episodes().values() {
                    let time_diff = (episode.when - *when).num_seconds().abs() as f32;
                    let hours_diff = time_diff / 3600.0;
                    
                    // Exponential decay: closer times have higher confidence
                    let temporal_confidence = (-hours_diff / 24.0).exp(); // 24-hour half-life
                    
                    if temporal_confidence > 0.1 {
                        let confidence = Confidence::exact(temporal_confidence);
                        let decayed_confidence = context.apply_decay(episode, confidence);
                        results.push((episode.clone(), decayed_confidence));
                    }
                }
                
                Ok(results)
            }
            _ => Err(CueError::UnsupportedCueType {
                cue_type: "non-temporal".to_string(),
                operation: "temporal_handler".to_string(),
            })
        }
    }
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool {
        matches!(cue_type, CueType::Temporal { .. })
    }
    
    fn handler_name(&self) -> &'static str {
        "temporal"
    }
}
```

### 3. Update MemoryStore to Use Dispatcher (engram-core/src/store.rs)

```rust
use crate::cue::{CueDispatcher, CueContext};

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            // ... existing fields ...
            cue_dispatcher: CueDispatcher::new(),
        }
    }
    
    // Simplified recall_in_memory using dispatcher
    fn recall_in_memory(&self, cue: Cue) -> Result<Vec<(Episode, Confidence)>, CueError> {
        // Use centralized dispatcher instead of manual pattern matching
        let results = self.cue_dispatcher.handle_cue(&cue, self)?;
        
        // Apply spreading activation if requested
        let activated_results = if cue.spread_activation {
            let pressure = self.get_memory_pressure();
            self.hnsw_index.apply_spreading_activation(results, &cue, pressure)
        } else {
            results
        };
        
        // Sort and limit results
        let mut final_results = activated_results;
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        final_results.truncate(cue.max_results);
        
        Ok(final_results)
    }
}

// Implement CueContext for MemoryStore
impl CueContext for MemoryStore {
    fn get_episodes(&self) -> &HashMap<String, Episode> {
        // Note: This requires changing RwLock to appropriate access pattern
        // For now, we'll need to adjust the interface
        unimplemented!("Requires interface adjustment for RwLock access")
    }
    
    fn get_hnsw_index(&self) -> &dyn HnswIndex {
        &*self.hnsw_index
    }
    
    fn compute_similarity(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        crate::compute::cosine_similarity_768(a, b)
    }
    
    fn apply_decay(&self, episode: &Episode, confidence: Confidence) -> Confidence {
        // Use existing decay logic
        self.decay_processor.apply_decay(episode, confidence)
    }
}
```

### 4. Update Module Structure (engram-core/src/cue/mod.rs)

```rust
//! Centralized cue handling with strategy pattern

pub mod dispatcher;
pub mod handlers;

pub use dispatcher::{CueDispatcher, CueHandler, CueContext};
pub use handlers::{
    EmbeddingCueHandler,
    ContextCueHandler, 
    SemanticCueHandler,
    TemporalCueHandler,
};
```

### 5. Update lib.rs Exports (engram-core/src/lib.rs)

Add around line 20:
```rust
pub mod cue;
pub use cue::{CueDispatcher, CueHandler, CueContext};
```

## Testing Strategy

### Unit Tests for Handlers
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cue, CueType, Episode, Confidence};
    
    struct MockCueContext {
        episodes: HashMap<String, Episode>,
    }
    
    impl CueContext for MockCueContext {
        fn get_episodes(&self) -> &HashMap<String, Episode> {
            &self.episodes
        }
        
        fn get_hnsw_index(&self) -> &dyn HnswIndex {
            unimplemented!() // Mock as needed
        }
        
        fn compute_similarity(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
            // Simple dot product for testing
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
        
        fn apply_decay(&self, _episode: &Episode, confidence: Confidence) -> Confidence {
            confidence // No decay for testing
        }
    }
    
    #[test]
    fn test_embedding_handler() {
        let handler = EmbeddingCueHandler;
        let context = MockCueContext {
            episodes: create_test_episodes(),
        };
        
        let cue = Cue {
            cue_type: CueType::Embedding {
                vector: [0.5; 768],
                threshold: 0.8,
            },
            max_results: 10,
            spread_activation: false,
        };
        
        let results = handler.handle(&cue, &context).unwrap();
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_dispatcher_routing() {
        let dispatcher = CueDispatcher::new();
        let context = MockCueContext {
            episodes: create_test_episodes(),
        };
        
        let embedding_cue = Cue::embedding([0.7; 768], 0.6);
        let context_cue = Cue::context("office", 0.5);
        
        // Test that dispatcher routes to correct handlers
        assert!(dispatcher.handle_cue(&embedding_cue, &context).is_ok());
        assert!(dispatcher.handle_cue(&context_cue, &context).is_ok());
    }
}
```

## Acceptance Criteria
- [ ] All 7 duplicate pattern-matching blocks replaced with dispatcher calls
- [ ] Single source of truth for cue handling logic
- [ ] Easy to add new cue types without modifying existing code
- [ ] Consistent behavior across all modules
- [ ] No performance degradation from abstraction
- [ ] Comprehensive test coverage for all handlers

## Performance Impact
- **Minimal overhead**: Dynamic dispatch adds <10ns per cue operation
- **Benefit**: Better compiler optimization opportunities
- **Benefit**: Reduced code size and improved instruction cache locality

## Risk Mitigation
- Gradual migration: Update one module at a time
- Maintain existing API compatibility
- Add benchmarks to ensure performance is maintained
- Comprehensive testing of all cue types and edge cases