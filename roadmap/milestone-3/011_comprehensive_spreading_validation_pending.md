# Task 011: Comprehensive Spreading Validation

## Objective
Create comprehensive test suite validating all spreading behaviors, performance targets, and cognitive correctness.

## Priority
P0 (Critical Path)

## Effort Estimate
1.5 days

## Dependencies
- Task 010: Spreading Performance Optimization

## Technical Approach

### Implementation Details
- Create spreading correctness tests with known graph topologies
- Implement performance regression tests ensuring <10ms P95 latency
- Add cognitive validation tests comparing against psychological research
- Create stress tests with large graphs (1M+ nodes) and high concurrency

### Files to Create/Modify
- `engram-core/tests/spreading_validation.rs` - Comprehensive spreading tests
- `engram-core/tests/performance_regression.rs` - Performance validation
- `engram-core/tests/cognitive_correctness.rs` - Psychology-based validation
- `engram-core/benches/spreading_benchmarks.rs` - Performance benchmarks

### Integration Points
- Tests all spreading components from Tasks 001-010
- Validates against psychological research standards
- Integrates with existing test infrastructure
- Uses performance monitoring from Task 010

## Implementation Details

### Spreading Correctness Tests
```rust
#[cfg(test)]
mod spreading_correctness {
    use super::*;

    #[tokio::test]
    async fn test_linear_chain_spreading() {
        // Create chain: A -> B -> C -> D -> E
        let graph = create_linear_chain(5);
        let spreading_engine = create_test_engine();

        let results = spreading_engine
            .spread_from_source("A", max_hops: 4)
            .await?;

        // Validate activation decreases with distance
        assert!(results["A"].activation > results["B"].activation);
        assert!(results["B"].activation > results["C"].activation);
        assert!(results["C"].activation > results["D"].activation);
        assert!(results["D"].activation > results["E"].activation);
    }

    #[tokio::test]
    async fn test_branching_tree_spreading() {
        // Create tree with known activation patterns
        let graph = create_binary_tree(depth: 4);
        let spreading_engine = create_test_engine();

        let results = spreading_engine
            .spread_from_source("root", max_hops: 4)
            .await?;

        // Validate symmetric spreading in balanced tree
        assert_eq!(
            results["left_child"].activation,
            results["right_child"].activation
        );
    }

    #[tokio::test]
    async fn test_cyclic_graph_termination() {
        // Create cycle: A -> B -> C -> A
        let graph = create_cycle(3);
        let spreading_engine = create_test_engine();

        let start_time = Instant::now();
        let results = spreading_engine
            .spread_from_source("A", max_hops: 10)
            .await?;
        let elapsed = start_time.elapsed();

        // Must terminate in bounded time
        assert!(elapsed < Duration::from_millis(100));
        // Must detect and handle cycle
        assert!(results.cycle_detected);
    }
}
```

### Performance Regression Tests
```rust
#[cfg(test)]
mod performance_regression {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn benchmark_single_hop_spreading(c: &mut Criterion) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (graph, engine) = runtime.block_on(setup_benchmark_graph(1000));

        c.bench_function("single_hop_1k_nodes", |b| {
            b.to_async(&runtime).iter(|| async {
                let results = engine
                    .spread_from_source(black_box("source"), max_hops: 1)
                    .await
                    .unwrap();
                black_box(results)
            })
        });
    }

    fn benchmark_multi_hop_spreading(c: &mut Criterion) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (graph, engine) = runtime.block_on(setup_benchmark_graph(10000));

        c.bench_function("multi_hop_10k_nodes", |b| {
            b.to_async(&runtime).iter(|| async {
                let results = engine
                    .spread_from_source(black_box("source"), max_hops: 5)
                    .await
                    .unwrap();
                black_box(results)
            })
        });
    }

    criterion_group!(
        spreading_benchmarks,
        benchmark_single_hop_spreading,
        benchmark_multi_hop_spreading
    );
    criterion_main!(spreading_benchmarks);
}
```

### Cognitive Correctness Tests
```rust
#[cfg(test)]
mod cognitive_correctness {
    use super::*;

    #[tokio::test]
    async fn test_semantic_priming_effect() {
        // Based on Meyer & Schvaneveldt (1971) semantic priming research
        let graph = create_semantic_network();
        let spreading_engine = create_test_engine();

        // Prime with "DOCTOR"
        let primed_results = spreading_engine
            .spread_from_source("DOCTOR", max_hops: 2)
            .await?;

        // "NURSE" should be more activated than unrelated "BREAD"
        assert!(
            primed_results["NURSE"].activation > primed_results["BREAD"].activation,
            "Semantic priming effect not observed"
        );
    }

    #[tokio::test]
    async fn test_fan_effect() {
        // Based on Anderson (1974) fan effect research
        let graph = create_fan_effect_network();
        let spreading_engine = create_test_engine();

        // High fan node (many connections)
        let high_fan_results = spreading_engine
            .spread_from_source("high_fan_concept", max_hops: 1)
            .await?;

        // Low fan node (few connections)
        let low_fan_results = spreading_engine
            .spread_from_source("low_fan_concept", max_hops: 1)
            .await?;

        // Low fan should spread more activation to each neighbor
        assert!(
            low_fan_results.average_neighbor_activation() >
            high_fan_results.average_neighbor_activation(),
            "Fan effect not observed in spreading activation"
        );
    }

    #[tokio::test]
    async fn test_decay_function_plausibility() {
        // Validate against psychological decay curves
        let graph = create_temporal_test_network();
        let spreading_engine = create_test_engine();

        let mut decay_measurements = Vec::new();
        for hop_count in 1..=5 {
            let results = spreading_engine
                .spread_from_source("source", max_hops: hop_count)
                .await?;

            decay_measurements.push((hop_count, results.target_activation("target")));
        }

        // Validate exponential decay pattern
        let correlation = calculate_exponential_correlation(&decay_measurements);
        assert!(
            correlation > 0.95,
            "Decay function does not follow expected exponential pattern: r = {}",
            correlation
        );
    }
}
```

### Stress Tests
```rust
#[cfg(test)]
mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_large_graph_performance() {
        // Test with 1M+ nodes
        let graph = create_scale_free_graph(nodes: 1_000_000, edges: 5_000_000);
        let spreading_engine = create_test_engine();

        let start_time = Instant::now();
        let results = spreading_engine
            .spread_from_source("random_source", max_hops: 3)
            .await?;
        let elapsed = start_time.elapsed();

        // Must complete within latency budget
        assert!(
            elapsed < Duration::from_millis(10),
            "Large graph spreading exceeded 10ms: {}ms",
            elapsed.as_millis()
        );
    }

    #[tokio::test]
    async fn test_concurrent_spreading() {
        let graph = create_test_graph(nodes: 10_000);
        let spreading_engine = Arc::new(create_test_engine());

        // Launch 100 concurrent spreading operations
        let mut handles = Vec::new();
        for i in 0..100 {
            let engine = Arc::clone(&spreading_engine);
            let source = format!("source_{}", i);

            handles.push(tokio::spawn(async move {
                engine.spread_from_source(&source, max_hops: 3).await
            }));
        }

        // All must complete successfully
        let results: Result<Vec<_>, _> = futures::future::join_all(handles)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .collect();

        assert!(results.is_ok(), "Concurrent spreading failed");
    }

    #[tokio::test]
    async fn test_memory_pressure() {
        let graph = create_test_graph(nodes: 100_000);
        let spreading_engine = create_test_engine();

        let initial_memory = get_memory_usage();

        // Run many spreading operations to test memory management
        for _ in 0..1000 {
            let _results = spreading_engine
                .spread_from_source("random_source", max_hops: 2)
                .await?;
        }

        let final_memory = get_memory_usage();
        let memory_growth = final_memory - initial_memory;

        // Memory growth should be bounded
        assert!(
            memory_growth < 100_000_000, // 100MB limit
            "Excessive memory growth: {} bytes",
            memory_growth
        );
    }
}
```

## Acceptance Criteria
- [ ] All spreading components pass correctness tests
- [ ] Performance meets <10ms P95 latency for realistic workloads
- [ ] Cognitive behavior matches psychological research patterns
- [ ] Large graph tests (1M+ nodes) complete within latency budget
- [ ] Concurrent spreading operations execute safely
- [ ] Memory usage remains bounded under stress
- [ ] Regression test suite prevents performance degradation

## Testing Approach
- **Unit Tests**: Individual component correctness
- **Integration Tests**: End-to-end spreading pipeline
- **Performance Tests**: Latency and throughput validation
- **Cognitive Tests**: Psychology research replication
- **Stress Tests**: Large-scale and concurrent operation validation
- **Regression Tests**: Prevent performance degradation over time

## Risk Mitigation
- **Risk**: Tests fail to catch real-world performance issues
- **Mitigation**: Use realistic graph topologies and workload patterns
- **Monitoring**: Production performance comparison with test results

- **Risk**: Cognitive tests based on outdated psychological research
- **Mitigation**: Review current literature, validate against multiple studies
- **Testing**: Replicate multiple classic experiments, not just one

- **Risk**: Stress tests don't reveal memory leaks or race conditions
- **Mitigation**: Long-running tests, memory profiling, thread sanitizers
- **Validation**: Run tests under valgrind and AddressSanitizer

## Implementation Strategy

### Phase 1: Correctness Validation
- Implement basic spreading correctness tests
- Add cycle detection and termination validation
- Mathematical correctness of confidence aggregation

### Phase 2: Performance Validation
- Implement latency regression tests
- Add throughput and scalability validation
- Memory usage and allocation pattern tests

### Phase 3: Cognitive Validation
- Replicate psychological experiments
- Validate against cognitive research patterns
- Add stress tests for production readiness

## Test Data Sets
- **Small Graphs**: 10-1000 nodes for detailed validation
- **Semantic Networks**: Psychology-based concept networks
- **Scale-Free Graphs**: Realistic social/knowledge network topologies
- **Random Graphs**: Stress testing with various connectivity patterns
- **Real-World Data**: Wikipedia link graphs, citation networks

## Notes
This task validates that Engram successfully implements cognitive spreading activation that is both computationally efficient and cognitively plausible. The tests here serve as acceptance criteria for the entire Milestone 3, ensuring that the transformation from vector database to cognitive database actually works correctly and performantly.