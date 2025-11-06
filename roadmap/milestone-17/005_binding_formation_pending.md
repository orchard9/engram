# Task 005: Binding Formation

## Objective
Create bidirectional bindings between episodes and concepts with appropriate strength weights and decay properties.

## Background
Bindings connect specific episodes to their generalized concepts, enabling both bottom-up (episode to concept) and top-down (concept to episode) processing.

## Requirements
1. Design binding data structure with strength and metadata
2. Implement bidirectional edge creation
3. Calculate initial binding strengths
4. Support binding decay and strengthening
5. Enable efficient traversal in both directions

## Technical Specification

### Files to Create
- `engram-core/src/memory/bindings.rs` - Binding types
- `engram-core/src/graph/binding_index.rs` - Efficient lookup

### Binding Structure
```rust
#[derive(Debug, Clone)]
pub struct ConceptBinding {
    pub episode_id: NodeId,
    pub concept_id: NodeId,
    pub strength: AtomicF32,
    pub contribution: f32, // How much episode contributed to concept
    pub created_at: DateTime<Utc>,
    pub last_activated: AtomicCell<DateTime<Utc>>,
}

pub struct BindingIndex {
    // Episode -> Concepts
    episode_to_concepts: DashMap<NodeId, Vec<Arc<ConceptBinding>>>,
    // Concept -> Episodes  
    concept_to_episodes: DashMap<NodeId, Vec<Arc<ConceptBinding>>>,
}
```

### Binding Formation
```rust
impl ConceptFormationEngine {
    pub fn create_bindings(
        &self,
        concept: &DualMemoryNode,
        episodes: &[Episode],
    ) -> Vec<ConceptBinding> {
        episodes.iter().map(|episode| {
            ConceptBinding {
                episode_id: episode.id.clone(),
                concept_id: concept.id.clone(),
                strength: AtomicF32::new(self.calculate_strength(episode, concept)),
                contribution: self.calculate_contribution(episode, episodes),
                created_at: Utc::now(),
                last_activated: AtomicCell::new(Utc::now()),
            }
        }).collect()
    }
    
    fn calculate_strength(&self, episode: &Episode, concept: &DualMemoryNode) -> f32 {
        // Based on similarity to concept centroid
        cosine_similarity(&episode.embedding, &concept.embedding)
    }
}
```

## Implementation Notes
- Use Arc for shared ownership across indices
- Atomic types for concurrent activation updates
- Consider memory overhead of bidirectional indices
- Implement lazy cleanup of weak bindings

## Testing Approach
1. Unit tests for binding creation
2. Concurrent access to binding strengths
3. Traversal performance tests
4. Memory overhead measurements

## Acceptance Criteria
- [ ] Bidirectional traversal in O(1) average case
- [ ] Atomic strength updates without locks
- [ ] Memory overhead <20% of node storage
- [ ] Binding creation <1ms per concept
- [ ] Concurrent access maintains consistency

## Dependencies
- Task 001 (Dual Memory Types)
- Task 004 (Concept Formation)

## Estimated Time
2 days