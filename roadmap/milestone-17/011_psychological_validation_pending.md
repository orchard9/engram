# Task 011: Psychological Validation

## Objective
Validate dual memory implementation against established psychological research on semantic memory, fan effects, and category formation.

## Background
Our dual memory architecture must align with cognitive science findings to ensure biological plausibility and predictable behavior.

## Requirements
1. Implement test scenarios from key psychology papers
2. Measure fan effect correlation with Anderson (1974)
3. Validate category formation against Rosch (1975)
4. Test semantic priming effects
5. Document deviations and rationale

## Technical Specification

### Files to Create
- `engram-core/tests/psychological_validation.rs` - Test suite
- `engram-core/benches/cognitive_benchmarks.rs` - Performance tests

### Key Validation Tests
```rust
#[cfg(test)]
mod psychological_validation {
    use super::*;
    
    /// Anderson (1974) fan effect validation
    #[test]
    fn test_fan_effect_correlation() {
        // Setup: Create concepts with varying fan-out
        let graph = setup_fan_effect_graph();
        
        // Test retrieval times for different fan levels
        let mut results = Vec::new();
        for fan_size in [1, 2, 3, 4, 6, 8] {
            let concept = create_concept_with_fan(fan_size);
            let start = Instant::now();
            let recalled = graph.recall_from_concept(&concept);
            let duration = start.elapsed();
            
            results.push((fan_size, duration.as_millis()));
        }
        
        // Validate logarithmic relationship
        let correlation = calculate_log_correlation(&results);
        assert!(correlation > 0.8, "Fan effect correlation too low: {}", correlation);
    }
    
    /// Rosch (1975) prototype effects
    #[test]
    fn test_prototype_category_formation() {
        let graph = DualMemoryGraph::new();
        
        // Add bird examples with varying typicality
        let robin = create_episode("robin", bird_features_typical());
        let penguin = create_episode("penguin", bird_features_atypical());
        let ostrich = create_episode("ostrich", bird_features_atypical());
        let sparrow = create_episode("sparrow", bird_features_typical());
        
        // Form concept
        let bird_concept = graph.form_concept_from_episodes(&[
            robin, penguin, ostrich, sparrow
        ]);
        
        // Typical examples should have stronger bindings
        let robin_binding = graph.get_binding(&robin.id, &bird_concept.id);
        let penguin_binding = graph.get_binding(&penguin.id, &bird_concept.id);
        
        assert!(
            robin_binding.strength > penguin_binding.strength,
            "Prototype effect not observed"
        );
    }
    
    /// Semantic priming validation
    #[test]
    fn test_semantic_priming_effects() {
        let graph = setup_semantic_network();
        
        // Prime with "doctor"
        let doctor_activation = 1.0;
        graph.activate_concept("doctor", doctor_activation);
        
        // Measure activation spread to related concepts
        thread::sleep(Duration::from_millis(50)); // SOA
        
        let nurse_activation = graph.get_activation("nurse");
        let bread_activation = graph.get_activation("bread");
        
        // Related concept should have higher activation
        assert!(
            nurse_activation > bread_activation * 2.0,
            "Semantic priming effect not observed"
        );
    }
}
```

### Benchmark Suite
```rust
#[bench]
fn bench_fan_effect_scaling(b: &mut Bencher) {
    let graph = setup_large_graph();
    
    b.iter(|| {
        // Measure retrieval time vs fan-out
        for concept in graph.iter_concepts() {
            let fan_out = graph.get_binding_count(&concept.id);
            let _results = graph.recall_from_concept(&concept);
        }
    });
}
```

## Implementation Notes
- Use statistical tests for correlations
- Control for confounding variables
- Document any simplifications made
- Consider cultural biases in category formation

## Testing Approach
1. Replicate classic experiments
2. Statistical validation of effects
3. Cross-cultural category tests
4. Performance scaling validation

## Acceptance Criteria
- [ ] Fan effect correlation r > 0.8
- [ ] Prototype effects observed in categories
- [ ] Semantic priming shows facilitation
- [ ] Performance scales as predicted
- [ ] Results documented and explained

## Dependencies
- All previous dual memory tasks
- Psychology literature access

## Estimated Time
3 days