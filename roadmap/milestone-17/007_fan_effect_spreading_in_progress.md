# Task 007: Fan Effect Spreading

## Objective
Implement Anderson's ACT-R fan effect in spreading activation where concepts with more bindings spread activation more weakly to each connected episode, accurately modeling human memory interference patterns.

## Background
The fan effect (Anderson, 1974) is a fundamental cognitive phenomenon where retrieval speed decreases linearly with the number of associations to a concept. Key empirical findings:
- Fan 1 (baseline): 1159ms ± 22ms
- Fan 2: 1236ms ± 25ms (+77ms)
- Fan 3: 1305ms ± 28ms (+69ms average from fan 2)
- Average slope: ~70ms per additional association

This is a **retrieval-stage** phenomenon affecting activation spreading, not encoding or consolidation. Unlike proactive/retroactive interference, fan effect doesn't reduce accuracy - it only affects retrieval latency.

## Cognitive Architecture

### ACT-R Activation Equation
The fan effect in ACT-R is modeled through activation spreading:

```
A_i = B_i + Σ_j W_j * S_ji

Where:
- A_i = activation of memory i
- B_i = base-level activation
- W_j = attentional weight of source j
- S_ji = associative strength from j to i

Associative strength with fan effect:
S_ji = S - ln(fan_j)

Where:
- S = maximum associative strength
- fan_j = number of associations from source j
```

### Spreading Activation Formula
For our implementation, we adapt this to:

```
activation_per_edge = base_activation / divisor(fan)

Where divisor(fan) can be:
- Linear: fan (default, matches psychological data)
- Sqrt: sqrt(fan) (softer falloff for large fans)
```

## Requirements
1. Calculate fan-out penalty based on binding count
2. Integrate with existing FanEffectDetector from cognitive interference module
3. Support bidirectional spreading (episode↔concept) with different characteristics
4. Implement configurable fan effect strength (0.5-0.8 typical)
5. Maintain spreading determinism and performance
6. Achieve correlation r > 0.8 with Anderson (1974) data

## Technical Specification

### Files to Modify
- `engram-core/src/activation/parallel.rs` - Integrate fan effect into spreading
- `engram-core/src/activation/mod.rs` - Add FanEffectConfig
- `engram-core/src/cognitive/interference/fan_effect.rs` - Reuse existing detector
- `engram-core/src/memory/bindings.rs` - Add fan counting methods

### Fan Effect Configuration
```rust
#[derive(Debug, Clone)]
pub struct FanEffectConfig {
    /// Enable fan effect during spreading (default: true)
    pub enabled: bool,
    
    /// Base retrieval time in ms (default: 1150.0 per Anderson 1974)
    pub base_retrieval_time_ms: f32,
    
    /// Time penalty per association in ms (default: 70.0)
    pub time_per_association_ms: f32,
    
    /// Divisor mode: Linear (default) or Sqrt
    pub use_sqrt_divisor: bool,
    
    /// Asymmetry factor for episode→concept spreading (default: 1.2)
    /// Higher values mean episodes spread more strongly to concepts
    pub upward_spreading_boost: f32,
    
    /// Minimum fan value to prevent division by zero (default: 1)
    pub min_fan: usize,
}

impl Default for FanEffectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_retrieval_time_ms: 1150.0,
            time_per_association_ms: 70.0,
            use_sqrt_divisor: false,
            upward_spreading_boost: 1.2,
            min_fan: 1,
        }
    }
}
```

### Integration with Parallel Spreading Engine
```rust
// In parallel.rs WorkerContext
struct WorkerContext {
    // ... existing fields ...
    fan_detector: Arc<FanEffectDetector>,
    binding_index: Arc<BindingIndex>,
}

impl WorkerContext {
    fn apply_fan_effect_spreading(
        &self,
        task: &ActivationTask,
        neighbors: Vec<WeightedEdge>,
    ) -> Vec<WeightedEdge> {
        let config = self.config();
        if !config.fan_effect_config.enabled {
            return neighbors;
        }
        
        // Determine node type from graph
        let node_type = self.memory_graph.get_node_type(&task.target_node);
        
        match node_type {
            Some(MemoryNodeType::Concept { .. }) => {
                // Concept → Episodes: Apply fan effect
                let fan = self.binding_index.get_episode_count(&task.target_node);
                let fan_result = self.fan_detector.detect_fan_effect_for_count(fan);
                
                // Adjust each neighbor's weight by fan divisor
                neighbors.into_iter().map(|mut edge| {
                    edge.weight /= fan_result.activation_divisor;
                    edge
                }).collect()
            }
            Some(MemoryNodeType::Episode { .. }) => {
                // Episode → Concepts: Apply upward boost, no fan penalty
                neighbors.into_iter().map(|mut edge| {
                    edge.weight *= config.fan_effect_config.upward_spreading_boost;
                    edge
                }).collect()
            }
            None => neighbors, // Unknown type, no adjustment
        }
    }
}
```

### Efficient Fan Computation with Caching
```rust
// Extension to BindingIndex for O(1) fan lookups
impl BindingIndex {
    /// Get episode count for a concept (its fan)
    pub fn get_episode_count(&self, concept_id: &NodeId) -> usize {
        self.concept_to_episodes
            .get(concept_id)
            .map_or(1, |bindings| bindings.len().max(1))
    }
    
    /// Get concept count for an episode (rarely high)
    pub fn get_concept_count(&self, episode_id: &NodeId) -> usize {
        self.episode_to_concepts
            .get(episode_id)
            .map_or(1, |bindings| bindings.len().max(1))
    }
    
    /// Precompute fan counts for a set of nodes (optimization)
    pub fn precompute_fans(&self, node_ids: &[NodeId]) -> DashMap<NodeId, usize> {
        let fans = DashMap::new();
        for node_id in node_ids {
            // Check both indices to determine type and fan
            if let Some(bindings) = self.concept_to_episodes.get(node_id) {
                fans.insert(node_id.clone(), bindings.len().max(1));
            } else if let Some(bindings) = self.episode_to_concepts.get(node_id) {
                // Episodes typically have low fan, but check anyway
                fans.insert(node_id.clone(), bindings.len().max(1));
            }
        }
        fans
    }
}

// Optimized batch processing in parallel spreading
impl WorkerContext {
    fn process_batch_with_fan_effect(
        &self,
        tasks: Vec<ActivationTask>,
    ) -> Vec<ActivationTask> {
        // Precompute fans for all nodes in batch
        let node_ids: Vec<_> = tasks.iter().map(|t| t.target_node.clone()).collect();
        let fan_cache = self.binding_index.precompute_fans(&node_ids);
        
        // Process tasks with cached fan values
        let mut output_tasks = Vec::new();
        for task in tasks {
            let neighbors = self.memory_graph.get_neighbors(&task.target_node)
                .unwrap_or_default();
                
            let adjusted_neighbors = if let Some(&fan) = fan_cache.get(&task.target_node) {
                self.apply_fan_adjustment(neighbors, fan)
            } else {
                neighbors
            };
            
            // Generate spreading tasks with fan-adjusted weights
            for edge in adjusted_neighbors {
                if task.should_continue() {
                    let contribution = task.source_activation * edge.weight * task.decay_factor;
                    if contribution > self.config().threshold {
                        output_tasks.push(ActivationTask::new(
                            edge.target,
                            contribution,
                            edge.weight,
                            task.decay_factor * self.config().decay_function.apply(1),
                            task.depth + 1,
                            task.max_depth,
                        ));
                    }
                }
            }
        }
        output_tasks
    }
}

## Implementation Notes

### Mathematical Precision
- Default to linear divisor (fan) based on Anderson's data
- Sqrt divisor available for networks with very high fan (>10)
- Activation divisor ensures total spreading is conserved
- Asymmetric spreading models cognitive reality: episodes activate concepts more strongly

### Performance Optimizations
1. **Fan Caching**: Precompute fans for batch processing
2. **Hot Path**: Store fan counts in CacheOptimizedNode for frequent access
3. **Batch SIMD**: Apply fan penalties in SIMD when processing 8 neighbors
4. **Memory Budget**: Cap spreading when fan > threshold to prevent explosion

### Integration with Existing Systems
1. **Cognitive Interference**: Reuse FanEffectDetector for consistency
2. **Metrics**: Record fan distribution and high-fan nodes
3. **Storage Tiers**: Higher fan nodes likely in Warm/Cold tiers
4. **Determinism**: Fan lookups are deterministic, maintain spreading order

## Testing Approach

### 1. Empirical Validation Tests
```rust
#[test]
fn test_fan_effect_matches_anderson_1974() {
    // Create memory graph with controlled fan structure
    // Node A: fan=1, Node B: fan=2, Node C: fan=3
    
    // Measure simulated retrieval times
    // Assert correlation > 0.8 with Anderson's data
}
```

### 2. Classic Fan Effect Experiments
```rust
#[test]
fn test_person_location_fan_effect() {
    // Implement Anderson's person-location experiment:
    // "The doctor is in the park" (low fan)
    // "The lawyer is in the church/park/store" (high fan)
    
    // Verify retrieval times increase with fan
}
```

### 3. Spreading Conservation Tests
```rust
#[test]
fn test_fan_effect_preserves_total_activation() {
    // Total activation leaving a node should equal input
    // Σ(activation_per_edge * fan) = input_activation
}
```

### 4. Performance Benchmarks
```rust
#[bench]
fn bench_high_fan_spreading() {
    // Create hub nodes with fan=50, fan=100, fan=500
    // Measure spreading performance degradation
    // Ensure <15% overhead vs. no fan effect
}
```

### 5. Integration Tests
- Fan effect + cycle detection interaction
- Fan effect + storage tiers (hot/warm/cold)
- Fan effect + SIMD batch spreading
- Fan effect + adaptive batching

## Acceptance Criteria
- [ ] Fan effect reduces activation by configurable divisor (linear or sqrt)
- [ ] Correlation with Anderson (1974) data r > 0.8
- [ ] Retrieval times match 70ms/association slope (±20ms tolerance)
- [ ] Asymmetric spreading: episodes→concepts stronger than concepts→episodes
- [ ] Performance degradation <15% vs. no fan effect
- [ ] Deterministic spreading order maintained
- [ ] Integration with existing cognitive interference metrics
- [ ] Batch processing optimizations for high-throughput
- [ ] Memory budget enforcement for high-fan nodes (>50 associations)
- [ ] Configuration through ParallelSpreadingConfig

## Dependencies
- Task 001 (Dual Memory Types) - for MemoryNodeType enum
- Task 002 (Graph Storage Adaptation) - for type-aware node queries
- Task 005 (Binding Formation) - for BindingIndex and fan counting
- Existing cognitive/interference/fan_effect.rs module
- Existing parallel spreading engine

## Estimated Time
3 days (increased due to comprehensive testing requirements)

## References
- Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. Cognitive Psychology, 6(4), 451-474.
- Anderson, J. R., & Reder, L. M. (1999). The fan effect: New results and new theories. Journal of Experimental Psychology: General, 128(2), 186-197.
- ACT-R: The Adaptive Control of Thought—Rational theory

## Comprehensive Test Specification

### Cognitive Test Cases

#### 1. Person-Location Experiment (Anderson, 1974)
```rust
#[test]
fn test_anderson_person_location_paradigm() {
    // Setup: Create a dual memory graph with controlled associations
    let graph = create_dual_memory_graph();
    
    // Low fan: "The doctor is in the park" (doctor→1 location, park→1 person)
    let doctor = create_person_concept("doctor");
    let park = create_location_concept("park");
    let episode1 = create_episode("doctor-in-park", &doctor, &park);
    
    // High fan: "The lawyer is in the church/park/store" (lawyer→3 locations)
    let lawyer = create_person_concept("lawyer");
    let church = create_location_concept("church");
    let store = create_location_concept("store");
    let episode2 = create_episode("lawyer-in-church", &lawyer, &church);
    let episode3 = create_episode("lawyer-in-park", &lawyer, &park);
    let episode4 = create_episode("lawyer-in-store", &lawyer, &store);
    
    // Test spreading from concepts
    let doctor_activation = spread_from_concept(&doctor, 1.0);
    let lawyer_activation = spread_from_concept(&lawyer, 1.0);
    
    // Lawyer should spread ~1/3 activation per episode due to fan=3
    assert!((doctor_activation[0].1 - 1.0).abs() < 0.01); // fan=1
    assert!((lawyer_activation[0].1 - 0.33).abs() < 0.05); // fan=3
}
```

#### 2. Retrieval Time Simulation
```rust
#[test]
fn test_retrieval_time_simulation() {
    let detector = FanEffectDetector::default();
    let mut retrieval_times = Vec::new();
    
    for fan in 1..=5 {
        let rt = detector.compute_retrieval_time_ms(fan);
        retrieval_times.push((fan, rt));
    }
    
    // Verify linear relationship with ~70ms slope
    for i in 1..retrieval_times.len() {
        let slope = (retrieval_times[i].1 - retrieval_times[i-1].1) 
                   / (retrieval_times[i].0 - retrieval_times[i-1].0) as f32;
        assert!((slope - 70.0).abs() < 5.0, "Slope should be ~70ms");
    }
}
```

#### 3. Asymmetric Spreading Test
```rust
#[test]
fn test_asymmetric_episode_concept_spreading() {
    let config = FanEffectConfig {
        upward_spreading_boost: 1.2,
        ..Default::default()
    };
    
    // Episode → Concept should be stronger
    let episode_to_concept = spread_from_episode(&episode, 1.0, &config);
    let concept_to_episode = spread_from_concept(&concept, 1.0, &config);
    
    assert!(episode_to_concept[0].1 > concept_to_episode[0].1);
    assert!((episode_to_concept[0].1 / concept_to_episode[0].1 - 1.2).abs() < 0.01);
}
```

### Performance Test Cases

#### 4. High Fan Stress Test
```rust
#[bench]
fn bench_extreme_fan_spreading(b: &mut Bencher) {
    // Create a "super concept" with 100+ episode bindings
    let super_concept = create_concept_with_episodes(100);
    
    b.iter(|| {
        let tasks = vec![
            ActivationTask::new(super_concept.id.clone(), 1.0, 1.0, 0.8, 0, 4)
        ];
        process_batch_with_fan_effect(tasks)
    });
}
```

#### 5. Cache Effectiveness Test
```rust
#[test]
fn test_fan_cache_hit_rate() {
    let mut metrics = SpreadingMetrics::default();
    
    // Process same nodes multiple times
    for _ in 0..10 {
        let tasks = create_test_tasks(&["A", "B", "C"]);
        process_with_fan_caching(tasks, &mut metrics);
    }
    
    // Should have high cache hit rate after first iteration
    assert!(metrics.cache_hit_rate() > 0.8);
}
```

### Integration Test Cases

#### 6. Fan Effect with Storage Tiers
```rust
#[test]
fn test_fan_effect_across_storage_tiers() {
    // High-fan concepts likely in warm/cold tier
    let hot_concept = create_concept_in_tier("hot", StorageTier::Hot, fan=2);
    let warm_concept = create_concept_in_tier("warm", StorageTier::Warm, fan=10);
    let cold_concept = create_concept_in_tier("cold", StorageTier::Cold, fan=50);
    
    // Verify tier-appropriate thresholds and timeouts
    let hot_result = spread_with_tier_awareness(&hot_concept);
    let warm_result = spread_with_tier_awareness(&warm_concept);
    let cold_result = spread_with_tier_awareness(&cold_concept);
    
    assert!(hot_result.latency < Duration::from_micros(100));
    assert!(warm_result.latency < Duration::from_millis(1));
    assert!(cold_result.latency < Duration::from_millis(10));
}
```

#### 7. Determinism Verification
```rust
#[test]
fn test_fan_effect_deterministic_spreading() {
    let config = ParallelSpreadingConfig::deterministic(42);
    let graph = create_complex_graph_with_varied_fans();
    
    // Run spreading multiple times
    let results: Vec<_> = (0..10).map(|_| {
        let engine = ParallelSpreadingEngine::new(config.clone(), graph.clone())?;
        engine.spread_activation(initial_seeds.clone())
    }).collect();
    
    // All runs should produce identical results
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "Run {} differs from run 0", i);
    }
}