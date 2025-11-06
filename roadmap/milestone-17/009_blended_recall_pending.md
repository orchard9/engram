# Task 009: Blended Recall

## Objective
Implement recall that blends episodic and semantic sources, using concepts for pattern completion and generalization.

## Background
Blended recall combines specific episodic memories with semantic knowledge from concepts, enabling richer and more robust memory retrieval.

## Requirements
1. Design recall strategy that queries both episodes and concepts
2. Implement result blending with appropriate weights
3. Use concepts for pattern completion of partial cues
4. Track provenance of recalled information
5. Support configurable blend ratios

## Technical Specification

### Files to Create
- `engram-core/src/activation/blended_recall.rs` - Blending logic

### Files to Modify
- `engram-core/src/activation/recall.rs` - Integration point

### Blended Recall Architecture
```rust
pub struct BlendedRecall {
    episodic_weight: f32,  // Default 0.7
    semantic_weight: f32,  // Default 0.3
    use_completion: bool,
}

impl BlendedRecall {
    pub async fn recall_blended(
        &self,
        cue: &Cue,
        context: RecallContext,
    ) -> RecallResult {
        // Parallel recall from both sources
        let (episodic_results, semantic_results) = tokio::join!(
            self.recall_episodic(cue, &context),
            self.recall_semantic(cue, &context)
        );
        
        // Blend results
        let blended = self.blend_results(
            episodic_results?,
            semantic_results?,
            &context
        );
        
        // Optional pattern completion
        if self.use_completion && blended.confidence < context.min_confidence {
            self.complete_from_concepts(&blended, &context).await
        } else {
            Ok(blended)
        }
    }
    
    fn recall_semantic(&self, cue: &Cue, context: &RecallContext) -> RecallResult {
        // Find relevant concepts
        let concepts = self.graph.find_concepts_by_embedding(&cue.embedding, 10)?;
        
        // Reconstruct from concepts
        let mut episodes = Vec::new();
        for (concept, similarity) in concepts {
            let bindings = self.graph.get_bindings_from_concept(&concept.id);
            
            // Weight by concept similarity and binding strength
            for binding in bindings {
                let weight = similarity * binding.strength.load(Ordering::Relaxed);
                episodes.push((binding.episode_id, weight));
            }
        }
        
        RecallResult {
            episodes,
            source: RecallSource::Semantic,
            confidence: self.calculate_semantic_confidence(&concepts),
        }
    }
}
```

### Result Blending
```rust
impl BlendedRecall {
    fn blend_results(
        &self,
        episodic: RecallResult,
        semantic: RecallResult,
        context: &RecallContext,
    ) -> BlendedResult {
        let mut combined_scores = HashMap::new();
        let mut provenance = HashMap::new();
        
        // Add episodic results
        for (episode_id, score) in episodic.episodes {
            let weighted = score * self.episodic_weight;
            combined_scores.insert(episode_id.clone(), weighted);
            provenance.insert(episode_id, RecallSource::Episodic);
        }
        
        // Blend in semantic results
        for (episode_id, score) in semantic.episodes {
            let weighted = score * self.semantic_weight;
            combined_scores.entry(episode_id.clone())
                .and_modify(|s| *s = (*s + weighted) / 2.0)
                .or_insert(weighted);
            
            provenance.entry(episode_id)
                .and_modify(|source| *source = RecallSource::Blended)
                .or_insert(RecallSource::Semantic);
        }
        
        BlendedResult {
            episodes: combined_scores,
            provenance,
            confidence: self.calculate_blended_confidence(&episodic, &semantic),
        }
    }
}
```

## Implementation Notes
- Use async for parallel recall
- Cache concept lookups for performance
- Consider memory budget in large result sets
- Track detailed provenance for explainability

## Testing Approach
1. Unit tests for blending logic
2. Integration tests with mock data
3. Quality comparison vs pure episodic
4. Performance benchmarks

## Acceptance Criteria
- [ ] Blended recall returns richer results
- [ ] Provenance tracking works correctly
- [ ] Pattern completion improves recall
- [ ] Performance within 20% of episodic-only
- [ ] Configurable blend weights

## Dependencies
- Task 007 (Fan Effect Spreading)
- Task 008 (Hierarchical Spreading)
- Existing recall system

## Estimated Time
3 days