# Task 014: Integration Testing

## Objective
Create comprehensive integration tests for the dual memory system, covering migration paths, feature flag transitions, and end-to-end scenarios.

## Background
Integration testing ensures all dual memory components work together correctly and maintain compatibility.

## Requirements
1. Test migration from single to dual memory
2. Validate feature flag combinations
3. Test A/B comparison scenarios
4. Verify backwards compatibility
5. Load test with realistic workloads

## Technical Specification

### Files to Create
- `engram-core/tests/dual_memory_integration.rs` - Main test suite
- `engram-core/tests/dual_memory_migration.rs` - Migration tests
- `tools/dual_memory_load_test.rs` - Load testing tool

### Integration Test Suite
```rust
#[cfg(test)]
mod dual_memory_integration {
    use super::*;
    
    /// End-to-end test with gradual rollout
    #[tokio::test]
    async fn test_gradual_rollout_scenario() {
        // Start with dual memory disabled
        let config = Config {
            dual_memory_enabled: false,
            concept_sample_rate: 0.0,
            ..Default::default()
        };
        
        let graph = MemoryGraph::new(config);
        
        // Add episodes
        for i in 0..1000 {
            let episode = create_test_episode(i);
            graph.add_memory(episode).await.unwrap();
        }
        
        // Enable dual memory in shadow mode
        graph.update_config(|c| {
            c.dual_memory_enabled = true;
            c.concept_sample_rate = 0.1;
        });
        
        // Run consolidation - should create concepts
        let stats = graph.consolidate().await.unwrap();
        assert!(stats.concepts_formed > 0);
        assert!(stats.concepts_formed < 50); // Sample rate limiting
        
        // Verify episodes still work
        let recall_result = graph.recall(&test_cue()).await.unwrap();
        assert!(!recall_result.episodes.is_empty());
        
        // Enable blended recall
        graph.update_config(|c| {
            c.use_blended_recall = true;
            c.semantic_weight = 0.3;
        });
        
        // Test blended recall
        let blended_result = graph.recall(&test_cue()).await.unwrap();
        assert!(blended_result.sources.contains(&RecallSource::Blended));
        assert!(blended_result.confidence > recall_result.confidence);
    }
    
    /// Test migration with data verification
    #[tokio::test]
    async fn test_migration_correctness() {
        // Create graph with legacy data
        let legacy_graph = create_legacy_graph();
        let episodes = collect_all_episodes(&legacy_graph);
        
        // Migrate to dual memory
        let migrator = DualMemoryMigrator::new();
        let dual_graph = migrator.migrate(&legacy_graph).await.unwrap();
        
        // Verify all episodes preserved
        for episode in &episodes {
            let node = dual_graph.get_node(&episode.id).unwrap();
            assert!(matches!(node.node_type, MemoryNodeType::Episode { .. }));
            assert_eq!(node.embedding, episode.embedding);
        }
        
        // Verify recall produces same results
        for test_cue in test_cues() {
            let legacy_results = legacy_graph.recall(&test_cue).await.unwrap();
            let dual_results = dual_graph.recall(&test_cue).await.unwrap();
            
            assert_similar_results(&legacy_results, &dual_results, 0.95);
        }
    }
}
```

### Feature Flag Testing
```rust
#[cfg(test)]
mod feature_flag_tests {
    use super::*;
    
    #[test_case(false, false, false; "all disabled")]
    #[test_case(true, false, false; "types only")]
    #[test_case(true, true, false; "types and formation")]
    #[test_case(true, true, true; "full dual memory")]
    fn test_feature_combinations(
        dual_types: bool,
        concept_formation: bool,
        blended_recall: bool,
    ) {
        let config = Config {
            features: Features {
                dual_memory_types: dual_types,
                concept_formation,
                blended_recall,
            },
            ..Default::default()
        };
        
        let graph = MemoryGraph::new(config);
        
        // Operations should work regardless of flags
        let episode = create_test_episode(1);
        graph.add_memory(episode).unwrap();
        
        let result = graph.recall(&test_cue()).unwrap();
        assert!(!result.episodes.is_empty());
        
        // Feature-specific behavior
        if concept_formation {
            graph.consolidate().unwrap();
            let stats = graph.get_stats();
            assert!(stats.concept_count >= 0);
        }
    }
}
```

### Load Testing
```rust
/// Realistic workload simulation
pub async fn load_test_dual_memory(config: LoadTestConfig) {
    let graph = create_dual_memory_graph();
    let mut stats = LoadTestStats::default();
    
    // Spawn concurrent workers
    let mut handles = Vec::new();
    
    // Memory addition workers
    for i in 0..config.write_workers {
        let graph_clone = graph.clone();
        handles.push(tokio::spawn(async move {
            for j in 0..config.episodes_per_worker {
                let episode = generate_realistic_episode(i, j);
                graph_clone.add_memory(episode).await.unwrap();
            }
        }));
    }
    
    // Recall workers
    for i in 0..config.read_workers {
        let graph_clone = graph.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..config.recalls_per_worker {
                let cue = generate_random_cue();
                let start = Instant::now();
                let result = graph_clone.recall(&cue).await.unwrap();
                let duration = start.elapsed();
                
                // Record latency
                RECALL_LATENCY.observe(duration.as_secs_f64());
            }
        }));
    }
    
    // Consolidation worker
    let graph_clone = graph.clone();
    handles.push(tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            let stats = graph_clone.consolidate().await.unwrap();
            info!("Consolidation: {} concepts formed", stats.concepts_formed);
        }
    }));
    
    // Wait for completion
    futures::future::join_all(handles).await;
}
```

## Implementation Notes
- Use property-based testing for invariants
- Test with various data distributions
- Include chaos testing scenarios
- Monitor resource usage during tests

## Testing Approach
1. Unit test each integration point
2. Scenario-based integration tests
3. Load tests with production-like data
4. Soak tests for memory leaks

## Acceptance Criteria
- [ ] All feature flag combinations tested
- [ ] Migration preserves data integrity
- [ ] Load tests show stable performance
- [ ] No memory leaks in 24-hour test
- [ ] Backwards compatibility verified

## Dependencies
- All previous dual memory tasks
- Test infrastructure

## Estimated Time
3 days