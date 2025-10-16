# Task 006: Comprehensive Temporal Dynamics Testing

## Objective
Create comprehensive test suite covering temporal decay integration, edge cases, performance, and memory safety under long-running decay operations.

## Priority
P1 (critical - ensures correctness and stability)

## Effort Estimate
1.5 days

## Dependencies
- Task 002: Last Access Tracking
- Task 003: Lazy Decay Integration
- Task 004: Decay Configuration API
- Task 005: Forgetting Curve Validation

## Technical Approach

### Files to Create
- `engram-core/tests/temporal_integration_test.rs` - Integration tests
- `engram-core/tests/temporal_edge_cases_test.rs` - Edge case tests
- `engram-core/benches/temporal_performance.rs` - Performance benchmarks

### Test Categories

**1. Integration Tests** - End-to-end decay behavior:
```rust
// engram-core/tests/temporal_integration_test.rs

#[tokio::test]
async fn test_temporal_decay_reduces_recall_confidence() {
    let store = MemoryStore::new_temp();
    let decay_system = BiologicalDecaySystem::default();

    let recall = CognitiveRecallBuilder::new()
        .vector_seeder(seeder)
        .spreading_engine(engine)
        .decay_system(Arc::new(decay_system))
        .build()
        .unwrap();

    // Store memory with high confidence
    let episode = Episode::new("important meeting")
        .with_confidence(Confidence::from_raw(0.9));
    store.insert_episode(episode.clone());

    // Immediate recall - high confidence
    let results_immediate = recall.recall(&cue, &store).unwrap();
    assert_eq!(results_immediate[0].confidence.raw(), 0.9);

    // Simulate 7 days passing
    std::thread::sleep(Duration::from_millis(100)); // Fast simulation
    // In real test, mock time or adjust last_access

    let results_delayed = recall.recall(&cue, &store).unwrap();
    assert!(
        results_delayed[0].confidence.raw() < 0.9,
        "Confidence should decay over time"
    );
}

#[tokio::test]
async fn test_frequently_accessed_memories_decay_slower() {
    // Store two identical memories
    let episode1 = Episode::new("content").with_id("rarely-accessed");
    let episode2 = Episode::new("content").with_id("frequently-accessed");

    store.insert_episode(episode1);
    store.insert_episode(episode2);

    // Access episode2 multiple times
    for _ in 0..5 {
        let _ = recall.recall(&cue_for("frequently-accessed"), &store);
        std::thread::sleep(Duration::from_millis(10));
    }

    // Access episode1 only once
    let _ = recall.recall(&cue_for("rarely-accessed"), &store);

    // Wait for decay
    std::thread::sleep(Duration::from_secs(1));

    // Frequently accessed should have higher confidence
    let rarely = store.get_episode("rarely-accessed").unwrap();
    let frequently = store.get_episode("frequently-accessed").unwrap();

    assert!(
        frequently.access_count > rarely.access_count,
        "Frequent access should be tracked"
    );
}

#[tokio::test]
async fn test_decay_respects_configured_function() {
    let exponential_system = BiologicalDecaySystem::new(
        DecayConfig::default()
            .with_function(DecayFunction::Exponential { rate: 0.1 })
    );

    let power_law_system = BiologicalDecaySystem::new(
        DecayConfig::default()
            .with_function(DecayFunction::PowerLaw { exponent: 0.3 })
    );

    // Same elapsed time, different decay functions should produce different results
    let exp_decay = exponential_system.compute_decayed_confidence(
        Confidence::from_raw(1.0),
        Duration::from_secs(100),
        0.1,
        1,
    );

    let pow_decay = power_law_system.compute_decayed_confidence(
        Confidence::from_raw(1.0),
        Duration::from_secs(100),
        0.1,
        1,
    );

    assert_ne!(exp_decay.raw(), pow_decay.raw(), "Different decay functions should produce different results");
}
```

**2. Edge Case Tests**:
```rust
// engram-core/tests/temporal_edge_cases_test.rs

#[test]
fn test_zero_elapsed_time_no_decay() {
    let decay_func = DecayFunction::Exponential { rate: 0.1 };
    let retention = decay_func.compute_decay(Duration::from_secs(0), 0);
    assert_eq!(retention, 1.0, "Zero time should mean no decay");
}

#[test]
fn test_very_long_elapsed_time_bounded() {
    let decay_func = DecayFunction::Exponential { rate: 0.1 };
    let retention = decay_func.compute_decay(Duration::from_secs(u64::MAX), 0);
    assert!(retention >= 0.0 && retention <= 1.0, "Retention must stay in [0, 1]");
}

#[test]
fn test_negative_decay_rate_rejected() {
    // Should panic or return error
    // Negative decay rate would cause exponential growth
}

#[test]
fn test_decay_with_very_high_access_count() {
    let decay_func = DecayFunction::TwoComponent {
        hippocampal_rate: 0.1,
        neocortical_rate: 0.01,
        consolidation_threshold: 0.7,
    };

    let retention = decay_func.compute_decay(Duration::from_secs(100), u64::MAX);
    assert!(retention >= 0.0 && retention <= 1.0);
}

#[test]
fn test_concurrent_decay_computation_thread_safe() {
    use std::sync::Arc;
    use std::thread;

    let decay_system = Arc::new(BiologicalDecaySystem::default());
    let mut handles = vec![];

    // Spawn 100 threads computing decay simultaneously
    for _ in 0..100 {
        let system = Arc::clone(&decay_system);
        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                let _ = system.compute_decayed_confidence(
                    Confidence::from_raw(0.8),
                    Duration::from_secs(100),
                    0.1,
                    5,
                );
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_decay_disabled_no_effect() {
    let config = DecayConfig {
        enabled: false,
        ..Default::default()
    };

    let recall = CognitiveRecallBuilder::new()
        .decay_system(Arc::new(BiologicalDecaySystem::new(config)))
        .build()
        .unwrap();

    // Confidence should not decay when disabled
    let results = recall.recall(&cue, &store).unwrap();
    assert_eq!(results[0].confidence.raw(), original_confidence);
}
```

**3. Performance Benchmarks**:
```rust
// engram-core/benches/temporal_performance.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engram_core::decay::*;
use std::time::Duration;

fn bench_exponential_decay(c: &mut Criterion) {
    let decay_func = DecayFunction::Exponential { rate: 0.1 };

    c.bench_function("exponential_decay", |b| {
        b.iter(|| {
            decay_func.compute_decay(
                black_box(Duration::from_secs(1000)),
                black_box(5),
            )
        })
    });
}

fn bench_power_law_decay(c: &mut Criterion) {
    let decay_func = DecayFunction::PowerLaw { exponent: 0.3 };

    c.bench_function("power_law_decay", |b| {
        b.iter(|| {
            decay_func.compute_decay(
                black_box(Duration::from_secs(1000)),
                black_box(5),
            )
        })
    });
}

fn bench_batch_decay_100_memories(c: &mut Criterion) {
    let decay_system = BiologicalDecaySystem::default();
    let confidences: Vec<Confidence> = (0..100)
        .map(|i| Confidence::from_raw(0.5 + (i as f32) * 0.005))
        .collect();

    c.bench_function("batch_decay_100", |b| {
        b.iter(|| {
            for conf in &confidences {
                decay_system.compute_decayed_confidence(
                    *conf,
                    Duration::from_secs(1000),
                    0.1,
                    3,
                );
            }
        })
    });
}

criterion_group!(benches, bench_exponential_decay, bench_power_law_decay, bench_batch_decay_100_memories);
criterion_main!(benches);
```

**4. Memory Safety Tests**:
```rust
#[test]
#[ignore] // Long-running test
fn test_no_memory_leaks_during_continuous_decay() {
    use std::time::Duration;

    let store = MemoryStore::new_temp();
    let recall = CognitiveRecallBuilder::new()
        .decay_system(Arc::new(BiologicalDecaySystem::default()))
        .build()
        .unwrap();

    // Insert 10K memories
    for i in 0..10_000 {
        store.insert_episode(Episode::new(&format!("memory_{}", i)));
    }

    let initial_memory = get_process_memory_usage();

    // Perform 10K recalls with decay
    for i in 0..10_000 {
        let cue = Cue::new(&format!("query_{}", i % 1000));
        let _ = recall.recall(&cue, &store);
    }

    let final_memory = get_process_memory_usage();
    let memory_increase = final_memory - initial_memory;

    // Memory increase should be bounded (< 100MB)
    assert!(
        memory_increase < 100 * 1024 * 1024,
        "Memory leak detected: {}MB increase",
        memory_increase / (1024 * 1024)
    );
}
```

## Acceptance Criteria

- [ ] All integration tests pass (decay reduces confidence, access patterns affect decay)
- [ ] All edge case tests pass (boundary conditions, thread safety)
- [ ] Performance benchmarks show <1ms p95 for decay computation
- [ ] Memory safety tests show no leaks during long-running operations
- [ ] Test coverage ≥90% for decay-related code
- [ ] Benchmarks run in CI to detect regressions
- [ ] Documentation includes test examples

## Testing Approach

**CI Integration**:
- Run fast tests (<1s) on every PR
- Run comprehensive suite nightly
- Run memory leak tests weekly (long-running)

**Performance Regression Detection**:
- Store benchmark results
- Alert if performance degrades >10%
- Track decay computation latency distribution

## Risk Mitigation

**Risk**: Tests don't catch real-world edge cases
**Mitigation**: Add property-based testing with quickcheck/proptest. Fuzz decay inputs.

**Risk**: Performance benchmarks not representative of production
**Mitigation**: Add integration benchmarks with realistic recall patterns, not just isolated decay computation.

**Risk**: Memory leak tests take too long
**Mitigation**: Use sampling - test 1K operations thoroughly rather than 1M minimally.

## Notes

This task ensures temporal dynamics are thoroughly tested for correctness, performance, and safety. The combination of unit, integration, edge case, performance, and safety tests provides comprehensive coverage.

**Testing Philosophy**: Fast feedback (unit tests) + comprehensive validation (integration tests) + regression prevention (benchmarks).

**Performance Target**: <1ms p95 for decay computation to keep recall latency low.
