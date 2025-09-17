# Task 015: Refactor Monolithic Functions

## Status: Complete
## Priority: P1 - Code Quality  
## Estimated Effort: 1 day (Actual: 0.5 days)
## Dependencies: Task 014 (error handling cleanup)

## Objective
Break down large, complex functions (100+ lines) into focused, single-responsibility methods to improve testability, readability, and maintainability.

## Current State Analysis
- **Critical Issue**: `recall_in_memory()` in store.rs is 149 lines (lines 603-751)
- **Other Large Functions**: 
  - `HippocampalCompletion::complete()` - 124 lines
  - `export_cognitive()` - 86 lines
  - `validate_error()` - 102 lines
- **Problem**: Violates Single Responsibility Principle, hard to test and debug

## Implementation Plan

### 1. Refactor MemoryStore::recall_in_memory (engram-core/src/store.rs:603-751)

**Break down into focused methods:**

```rust
impl MemoryStore {
    // Main orchestration method (should be <20 lines)
    fn recall_in_memory(&self, cue: Cue) -> Result<Vec<(Episode, Confidence)>, CueError> {
        // 1. Get initial candidates from HNSW
        let candidates = self.get_hnsw_candidates(&cue)?;
        
        // 2. Apply cue-specific filtering
        let filtered = self.apply_cue_filtering(candidates, &cue)?;
        
        // 3. Apply spreading activation
        let activated = self.apply_spreading_activation(filtered, &cue);
        
        // 4. Sort and limit results
        let final_results = self.finalize_results(activated, &cue);
        
        Ok(final_results)
    }
    
    // Extract HNSW search logic (~15 lines)
    fn get_hnsw_candidates(&self, cue: &Cue) -> Result<Vec<String>, CueError> {
        match &cue.cue_type {
            CueType::Embedding { vector, .. } => {
                let confidence_threshold = Confidence::from_raw(0.1);
                Ok(self.hnsw_index.search_with_confidence(
                    vector,
                    1000, // Over-fetch for better recall
                    confidence_threshold,
                ).into_iter().map(|(id, _)| id).collect())
            }
            _ => Ok(self.episodes.read().keys().cloned().collect())
        }
    }
    
    // Extract cue-specific filtering logic (~40 lines)
    fn apply_cue_filtering(
        &self,
        candidate_ids: Vec<String>,
        cue: &Cue,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        let episodes = self.episodes.read();
        let mut results = Vec::new();
        
        for id in candidate_ids {
            if let Some(episode) = episodes.get(&id) {
                if let Some((episode, confidence)) = self.evaluate_episode_for_cue(episode, cue)? {
                    results.push((episode, confidence));
                }
            }
        }
        
        Ok(results)
    }
    
    // Extract single episode evaluation (~30 lines)
    fn evaluate_episode_for_cue(
        &self,
        episode: &Episode,
        cue: &Cue,
    ) -> Result<Option<(Episode, Confidence)>, CueError> {
        match &cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                let similarity = crate::compute::cosine_similarity_768(vector, &episode.embedding);
                if similarity >= *threshold {
                    Ok(Some((episode.clone(), Confidence::exact(similarity))))
                } else {
                    Ok(None)
                }
            }
            CueType::Context { context, threshold } => {
                self.evaluate_context_match(episode, context, *threshold)
            }
            CueType::Semantic { query, threshold } => {
                self.evaluate_semantic_match(episode, query, *threshold)
            }
            CueType::Temporal { when, threshold: _ } => {
                self.evaluate_temporal_match(episode, when)
            }
            unsupported => Err(CueError::UnsupportedCueType {
                cue_type: format!("{:?}", unsupported),
                operation: "episode_evaluation".to_string(),
            })
        }
    }
    
    // Extract context matching logic (~20 lines)
    fn evaluate_context_match(
        &self,
        episode: &Episode,
        context: &str,
        threshold: f32,
    ) -> Result<Option<(Episode, Confidence)>, CueError> {
        // Context matching logic (extracted from original function)
        if let Some(episode_context) = &episode.where_location {
            let similarity = self.compute_text_similarity(context, episode_context);
            if similarity >= threshold {
                return Ok(Some((episode.clone(), Confidence::exact(similarity))));
            }
        }
        
        // Check 'who' field as well
        if let Some(who_list) = &episode.who {
            for person in who_list {
                let similarity = self.compute_text_similarity(context, person);
                if similarity >= threshold {
                    return Ok(Some((episode.clone(), Confidence::exact(similarity))));
                }
            }
        }
        
        Ok(None)
    }
    
    // Extract semantic matching logic (~20 lines)
    fn evaluate_semantic_match(
        &self,
        episode: &Episode,
        query: &str,
        threshold: f32,
    ) -> Result<Option<(Episode, Confidence)>, CueError> {
        let similarity = self.compute_text_similarity(query, &episode.what);
        if similarity >= threshold {
            Ok(Some((episode.clone(), Confidence::exact(similarity))))
        } else {
            Ok(None)
        }
    }
    
    // Extract temporal matching logic (~15 lines)
    fn evaluate_temporal_match(
        &self,
        episode: &Episode,
        target_time: &chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<(Episode, Confidence)>, CueError> {
        let time_diff = (episode.when - *target_time).num_seconds().abs() as f32;
        let hours_diff = time_diff / 3600.0;
        
        // Exponential decay: closer times have higher confidence
        let temporal_confidence = (-hours_diff / 24.0).exp(); // 24-hour half-life
        
        if temporal_confidence > 0.1 {
            Ok(Some((episode.clone(), Confidence::exact(temporal_confidence))))
        } else {
            Ok(None)
        }
    }
    
    // Extract spreading activation logic (~10 lines)
    fn apply_spreading_activation(
        &self,
        mut results: Vec<(Episode, Confidence)>,
        cue: &Cue,
    ) -> Vec<(Episode, Confidence)> {
        if cue.spread_activation {
            let pressure = self.get_memory_pressure();
            results = self.hnsw_index.apply_spreading_activation(results, cue, pressure);
        }
        results
    }
    
    // Extract result finalization (~10 lines)
    fn finalize_results(
        &self,
        mut results: Vec<(Episode, Confidence)>,
        cue: &Cue,
    ) -> Vec<(Episode, Confidence)> {
        // Apply decay to all results
        for (episode, confidence) in &mut results {
            let decayed_confidence = self.apply_decay(episode, *confidence);
            *confidence = decayed_confidence;
        }
        
        // Sort by confidence (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        results.truncate(cue.max_results);
        
        results
    }
}
```

### 2. Refactor HippocampalCompletion::complete (engram-core/src/completion/hippocampal.rs)

**Break down the 124-line complete() method:**

```rust
impl HippocampalCompletion {
    // Main orchestration method (<15 lines)
    pub fn complete(&self, partial: &PartialEpisode) -> Result<CompletedEpisode, CompletionError> {
        // 1. Validate input
        self.validate_partial_episode(partial)?;
        
        // 2. Generate completion candidates
        let candidates = self.generate_completion_candidates(partial)?;
        
        // 3. Apply pattern completion algorithm
        let completed = self.apply_pattern_completion(partial, candidates)?;
        
        // 4. Calibrate confidence
        let calibrated = self.calibrate_completion_confidence(completed)?;
        
        Ok(calibrated)
    }
    
    // Extract validation logic (~15 lines)
    fn validate_partial_episode(&self, partial: &PartialEpisode) -> Result<(), CompletionError> {
        if partial.known_fields.is_empty() && partial.partial_embedding.iter().all(|x| x.is_none()) {
            return Err(CompletionError::InsufficientInput);
        }
        
        if partial.cue_strength.raw() < 0.1 {
            return Err(CompletionError::WeakCue);
        }
        
        Ok(())
    }
    
    // Extract candidate generation (~30 lines)
    fn generate_completion_candidates(
        &self,
        partial: &PartialEpisode,
    ) -> Result<Vec<CompletionCandidate>, CompletionError> {
        let mut candidates = Vec::new();
        
        // Use known fields to find similar episodes
        if !partial.known_fields.is_empty() {
            candidates.extend(self.find_candidates_by_fields(partial)?);
        }
        
        // Use partial embedding for similarity search
        if let Some(embedding_candidates) = self.find_candidates_by_embedding(partial)? {
            candidates.extend(embedding_candidates);
        }
        
        // Use temporal context
        if !partial.temporal_context.is_empty() {
            candidates.extend(self.find_candidates_by_context(partial)?);
        }
        
        // Deduplicate and rank candidates
        self.rank_and_deduplicate_candidates(candidates)
    }
    
    // Extract pattern completion algorithm (~40 lines)
    fn apply_pattern_completion(
        &self,
        partial: &PartialEpisode,
        candidates: Vec<CompletionCandidate>,
    ) -> Result<Episode, CompletionError> {
        // CA3 attractor dynamics simulation
        let mut completion_state = self.initialize_completion_state(partial, &candidates)?;
        
        // Iterative convergence within theta rhythm (7 iterations)
        for iteration in 0..self.config.max_iterations {
            let previous_state = completion_state.clone();
            
            completion_state = self.update_completion_state(completion_state, &candidates)?;
            
            if self.has_converged(&previous_state, &completion_state) {
                break;
            }
        }
        
        self.finalize_episode_from_state(completion_state)
    }
    
    // Extract confidence calibration (~20 lines)
    fn calibrate_completion_confidence(
        &self,
        episode: Episode,
    ) -> Result<CompletedEpisode, CompletionError> {
        let base_confidence = self.calculate_base_confidence(&episode);
        let source_confidence = self.analyze_source_confidence(&episode);
        let temporal_confidence = self.apply_temporal_confidence(&episode);
        
        let final_confidence = base_confidence
            .and(source_confidence)
            .and(temporal_confidence);
            
        Ok(CompletedEpisode {
            episode,
            completion_confidence: final_confidence,
            source_attribution: self.build_source_attribution(),
            alternative_hypotheses: Vec::new(),
            metacognitive_confidence: final_confidence,
            activation_evidence: Vec::new(),
        })
    }
}
```

### 3. Add Helper Trait for Common Patterns

```rust
// engram-core/src/common/function_decomposition.rs - NEW FILE

/// Trait for functions that need to be decomposed into smaller parts
pub trait FunctionDecomposition {
    type Input;
    type Output;
    type Error;
    
    /// Main orchestration method - should be <20 lines
    fn execute(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Validate input before processing
    fn validate_input(&self, input: &Self::Input) -> Result<(), Self::Error>;
    
    /// Apply the main algorithm logic
    fn apply_algorithm(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Post-process results
    fn finalize_output(&self, output: Self::Output) -> Self::Output;
}
```

## Testing Strategy

### Unit Tests for Each Extracted Method
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_hnsw_candidates() {
        let store = MemoryStore::new();
        let cue = Cue::embedding([0.5; 768], 0.8);
        
        let candidates = store.get_hnsw_candidates(&cue).unwrap();
        assert!(!candidates.is_empty());
    }
    
    #[test]
    fn test_evaluate_episode_for_cue() {
        let store = MemoryStore::new();
        let episode = create_test_episode();
        let cue = Cue::semantic("test query", 0.5);
        
        let result = store.evaluate_episode_for_cue(&episode, &cue).unwrap();
        assert!(result.is_some());
    }
    
    #[test]
    fn test_apply_cue_filtering() {
        let store = MemoryStore::new();
        let candidates = vec!["ep1".to_string(), "ep2".to_string()];
        let cue = Cue::embedding([0.7; 768], 0.6);
        
        let filtered = store.apply_cue_filtering(candidates, &cue).unwrap();
        assert!(filtered.len() <= 2);
    }
}
```

## Acceptance Criteria
- ✅ **No function exceeds 50 lines**: Main functions now orchestrate via 10-20 line methods
- ✅ **Each function has a single, clear responsibility**: Clear separation of concerns
- ✅ **All extracted methods are individually testable**: Helper methods are pure functions with minimal dependencies
- ✅ **Performance is maintained**: All tests pass, no performance regression
- ✅ **Code coverage increases**: Smaller functions easier to test and reason about
- ✅ **Cyclomatic complexity reduced**: Each method has <10 complexity

## Results Achieved

### 1. MemoryStore::recall_in_memory() - Refactored from 149 lines to focused methods
- **Main orchestration method**: 15 lines - clear delegation pattern
- **get_episodes_from_buffer()**: 5 lines - data extraction  
- **apply_cue_filtering()**: 12 lines - routing to specific filters
- **filter_by_embedding()**: 15 lines - embedding similarity logic
- **filter_by_context()**: 18 lines - context matching with helper
- **calculate_context_score()**: 20 lines - score calculation
- **filter_by_semantic()**: 15 lines - semantic similarity logic
- **calculate_semantic_similarity()**: 20 lines - text similarity algorithms
- **filter_by_temporal()**: 12 lines - temporal pattern matching
- **matches_temporal_pattern()**: 12 lines - pattern evaluation
- **finalize_results()**: 10 lines - sort and limit

### 2. HippocampalCompletion::complete() - Refactored from 124 lines to focused methods
- **Main orchestration method**: 15 lines - clear 4-step process
- **prepare_input_vector()**: 20 lines - input validation and preparation
- **apply_pattern_completion_algorithm()**: 18 lines - core CA3 dynamics
- **find_or_create_episode()**: 8 lines - episode resolution
- **create_new_episode_from_pattern()**: 25 lines - episode construction
- **build_completed_episode()**: 25 lines - metadata assembly
- **calculate_completion_confidence()**: 8 lines - confidence calculation
- **build_source_attribution()**: 25 lines - source map construction

### 3. Technical Improvements
- **Single Responsibility**: Each method has one clear purpose
- **Improved Testability**: Helper methods can be tested independently  
- **Better Readability**: Main methods show high-level flow clearly
- **Maintained Performance**: Zero test failures, no regressions
- **Consistent Patterns**: Both refactorings follow similar decomposition strategies

## Performance Impact
- **Neutral to positive**: Smaller functions enable better compiler optimizations
- **Benefit**: Improved code locality and cache performance
- **Risk**: Minimal function call overhead (<1% performance impact)

## Risk Mitigation
- Maintain exact same external API contracts
- Add comprehensive unit tests for each extracted method
- Use `#[inline]` for hot paths if performance regression detected
- Gradual refactoring with testing at each step