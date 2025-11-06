# Task 010: Confidence Propagation

## Objective
Implement proper confidence propagation through the dual memory architecture, ensuring uncertainty is tracked through concept formation and blended recall.

## Background
Confidence scores must propagate correctly through all dual memory operations to maintain the probabilistic nature of the system.

## Requirements
1. Define confidence rules for concept formation
2. Implement confidence decay through bindings
3. Calculate blended confidence scores
4. Track confidence through spreading paths
5. Ensure mathematical consistency

## Technical Specification

### Files to Create
- `engram-core/src/confidence/dual_memory.rs` - Confidence rules

### Confidence Rules
```rust
pub struct DualMemoryConfidence {
    concept_formation_penalty: f32, // Default 0.9
    binding_confidence_decay: f32,  // Default 0.95
    blend_confidence_bonus: f32,    // Default 1.1
}

impl DualMemoryConfidence {
    /// Calculate concept confidence from clustered episodes
    pub fn concept_confidence(
        &self,
        episodes: &[Episode],
        coherence: f32,
    ) -> Confidence {
        // Average episode confidence weighted by coherence
        let avg_confidence = episodes.iter()
            .map(|e| e.confidence.value())
            .sum::<f32>() / episodes.len() as f32;
        
        // Apply formation penalty and coherence weighting
        let concept_conf = avg_confidence * self.concept_formation_penalty * coherence;
        
        Confidence::new(concept_conf)
    }
    
    /// Propagate confidence through bindings
    pub fn propagate_through_binding(
        &self,
        source_confidence: Confidence,
        binding_strength: f32,
    ) -> Confidence {
        let propagated = source_confidence.value() 
            * binding_strength 
            * self.binding_confidence_decay;
            
        Confidence::new(propagated)
    }
    
    /// Calculate blended recall confidence
    pub fn blend_confidence(
        &self,
        episodic_conf: Confidence,
        semantic_conf: Confidence,
        episodic_weight: f32,
        semantic_weight: f32,
    ) -> Confidence {
        // Weighted average with blend bonus
        let base = (episodic_conf.value() * episodic_weight 
                   + semantic_conf.value() * semantic_weight)
                   / (episodic_weight + semantic_weight);
        
        // Apply blend bonus for convergent evidence
        let blended = if episodic_conf.value() > 0.5 && semantic_conf.value() > 0.5 {
            (base * self.blend_confidence_bonus).min(1.0)
        } else {
            base
        };
        
        Confidence::new(blended)
    }
}
```

### Integration with Spreading
```rust
impl SpreadingWithConfidence {
    pub fn spread_with_confidence(
        &self,
        activations: Vec<(NodeId, f32, Confidence)>,
    ) -> Vec<(NodeId, f32, Confidence)> {
        let mut results = Vec::new();
        
        for (node_id, activation, confidence) in activations {
            let spread_targets = self.get_spread_targets(&node_id);
            
            for (target_id, spread_strength) in spread_targets {
                // Propagate both activation and confidence
                let spread_activation = activation * spread_strength;
                let spread_confidence = self.confidence_rules
                    .propagate_through_binding(confidence, spread_strength);
                
                results.push((target_id, spread_activation, spread_confidence));
            }
        }
        
        results
    }
}
```

## Implementation Notes
- Ensure confidence values stay in [0, 1] range
- Use type system to enforce confidence presence
- Consider computational cost of propagation
- Log confidence degradation for debugging

## Testing Approach
1. Property tests for value ranges
2. Propagation chain tests
3. Mathematical consistency validation
4. Integration with existing confidence tests

## Acceptance Criteria
- [ ] All confidence values in valid range
- [ ] Propagation preserves uncertainty monotonicity
- [ ] Blended confidence reflects convergent evidence
- [ ] No confidence inflation through cycles
- [ ] Performance overhead <5%

## Dependencies
- Task 009 (Blended Recall)
- Existing confidence system

## Estimated Time
2 days