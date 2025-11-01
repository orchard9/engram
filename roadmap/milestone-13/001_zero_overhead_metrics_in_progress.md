# Task 001: Zero-Overhead Metrics Infrastructure

**Status:** Pending
**Priority:** P0 (Foundation)
**Estimated Effort:** 2 days
**Dependencies:** None

## Objective

Implement conditional compilation infrastructure for cognitive pattern metrics with provably zero overhead when disabled and <1% overhead when enabled. Create lock-free atomic collection with cache-line alignment.

## Research Foundation

Traditional observability tools add 5-15% latency overhead even when metrics aren't being collected. For a memory system processing 10K recalls/sec with sub-millisecond latency requirements, this is unacceptable. We need instrumentation that costs literally zero when disabled and <1% when enabled.

**Conditional compilation solution:**
Unlike runtime feature flags that branch at every instrumentation point, `#[cfg(feature = "monitoring")]` removes code entirely during compilation. When metrics disabled, compiler optimizes away not just metric recording calls, but entire surrounding code path - assembly output identical to un-instrumented code.

**Lock-free atomic metrics (Vyukov 2007):**
Each thread maintains own histogram buckets using atomic integers. During spreading activation, threads update local buckets with atomic fetch-add (15-20ns on modern x86_64). Background thread periodically aggregates per-thread buckets into global metrics without blocking workers. This achieves constant-time metric recording with minimal cache coherence traffic.

**Performance budget:**
- Metric recording: <50ns per event (vs 2-5μs for traditional metrics libraries)
- Zero overhead when disabled: 0ns (proven via assembly inspection)
- <1% overhead when enabled: measured via microbenchmarks
- Lock-free guarantee: no contention, scales linearly with thread count

**Statistical validation requirements:**
- Large sample sizes: N > 1000 trials per condition
- Proper randomization of study lists
- Controlled retention intervals
- Automatic statistical tests (t-tests, ANOVAs) for effect detection
- For DRM: 55-65% false recall within 10% tolerance (Roediger & McDermott 1995)

## Integration Points

**Extends:**
- `/engram-core/src/metrics/mod.rs` - Add cognitive pattern exports
- `/engram-core/src/metrics/lockfree.rs` - Reuse lock-free infrastructure

**Creates:**
- `/engram-core/src/metrics/cognitive_patterns.rs` - New cognitive metrics types
- `/engram-core/benches/metrics_overhead.rs` - Overhead validation benchmarks
- `/engram-core/tests/metrics/zero_overhead_tests.rs` - Conditional compilation tests

## Detailed Specification

### 1. Conditional Compilation Strategy

```rust
// /engram-core/src/metrics/cognitive_patterns.rs

/// Zero-overhead cognitive pattern metrics
///
/// When `monitoring` feature is disabled, this compiles to zero-sized type
/// with all methods optimized away entirely.
pub struct CognitivePatternMetrics {
    #[cfg(feature = "monitoring")]
    inner: Arc<CognitivePatternMetricsInner>,
}

#[cfg(feature = "monitoring")]
struct CognitivePatternMetricsInner {
    // Priming metrics
    priming_events_total: CachePadded<AtomicU64>,
    priming_strength_histogram: LockFreeHistogram,
    priming_type_counters: [CachePadded<AtomicU64>; 3], // semantic, associative, repetition

    // Interference metrics
    interference_detections_total: CachePadded<AtomicU64>,
    proactive_interference_strength: LockFreeHistogram,
    retroactive_interference_strength: LockFreeHistogram,
    fan_effect_magnitude: LockFreeHistogram,

    // Reconsolidation metrics
    reconsolidation_events_total: CachePadded<AtomicU64>,
    reconsolidation_modifications: CachePadded<AtomicU64>,
    reconsolidation_window_hits: CachePadded<AtomicU64>,
    reconsolidation_window_misses: CachePadded<AtomicU64>,

    // False memory metrics
    false_memory_generations: CachePadded<AtomicU64>,
    drm_critical_lure_recalls: CachePadded<AtomicU64>,
    drm_list_item_recalls: CachePadded<AtomicU64>,

    // Spacing effect metrics
    massed_practice_events: CachePadded<AtomicU64>,
    distributed_practice_events: CachePadded<AtomicU64>,
    retention_improvement_histogram: LockFreeHistogram,
}

impl CognitivePatternMetrics {
    /// Record priming event with zero overhead when monitoring disabled
    #[inline(always)]
    pub fn record_priming(&self, priming_type: PrimingType, strength: f32) {
        #[cfg(feature = "monitoring")]
        {
            self.inner.priming_events_total.fetch_add(1, Ordering::Relaxed);
            self.inner.priming_strength_histogram.record(strength as f64);

            let idx = priming_type as usize;
            self.inner.priming_type_counters[idx].fetch_add(1, Ordering::Relaxed);
        }

        #[cfg(not(feature = "monitoring"))]
        {
            let _ = (priming_type, strength); // Suppress unused warnings
        }
    }

    #[inline(always)]
    pub fn record_interference(&self,
        interference_type: InterferenceType,
        magnitude: f32
    ) {
        #[cfg(feature = "monitoring")]
        {
            self.inner.interference_detections_total.fetch_add(1, Ordering::Relaxed);

            match interference_type {
                InterferenceType::Proactive => {
                    self.inner.proactive_interference_strength.record(magnitude as f64);
                }
                InterferenceType::Retroactive => {
                    self.inner.retroactive_interference_strength.record(magnitude as f64);
                }
                InterferenceType::Fan => {
                    self.inner.fan_effect_magnitude.record(magnitude as f64);
                }
            }
        }

        #[cfg(not(feature = "monitoring"))]
        {
            let _ = (interference_type, magnitude);
        }
    }

    #[inline(always)]
    pub fn record_reconsolidation(&self, window_position: f32) {
        #[cfg(feature = "monitoring")]
        {
            self.inner.reconsolidation_events_total.fetch_add(1, Ordering::Relaxed);

            if window_position >= 0.0 && window_position <= 1.0 {
                self.inner.reconsolidation_window_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.inner.reconsolidation_window_misses.fetch_add(1, Ordering::Relaxed);
            }
        }

        #[cfg(not(feature = "monitoring"))]
        {
            let _ = window_position;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PrimingType {
    Semantic = 0,
    Associative = 1,
    Repetition = 2,
}

#[derive(Debug, Clone, Copy)]
pub enum InterferenceType {
    Proactive,
    Retroactive,
    Fan,
}
```

### 2. Overhead Validation Benchmark

```rust
// /engram-core/benches/metrics_overhead.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use engram_core::metrics::CognitivePatternMetrics;

fn benchmark_priming_recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("cognitive_metrics_overhead");

    let metrics = CognitivePatternMetrics::new();

    group.bench_function("record_priming", |b| {
        b.iter(|| {
            black_box(metrics.record_priming(
                PrimingType::Semantic,
                0.75
            ));
        });
    });

    group.finish();
}

fn benchmark_recall_with_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_overhead");

    // Baseline: realistic recall operation without metrics
    group.bench_function("baseline_no_metrics", |b| {
        b.iter(|| {
            black_box(perform_recall_operation());
        });
    });

    // With metrics enabled
    #[cfg(feature = "monitoring")]
    group.bench_function("with_metrics", |b| {
        b.iter(|| {
            black_box(perform_recall_with_metrics());
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_priming_recording, benchmark_recall_with_metrics);
criterion_main!(benches);
```

### 3. Zero-Cost Verification Test

```rust
// /engram-core/tests/metrics/zero_overhead_tests.rs

/// Verify that with monitoring disabled, metrics compile to zero code
#[cfg(not(feature = "monitoring"))]
#[test]
fn verify_zero_size_when_disabled() {
    use std::mem::size_of;
    use engram_core::metrics::CognitivePatternMetrics;

    // Should be zero-sized or pointer-sized when feature disabled
    assert!(
        size_of::<CognitivePatternMetrics>() <= size_of::<usize>(),
        "CognitivePatternMetrics should be zero-sized when monitoring disabled"
    );
}

/// Verify overhead is <1% when enabled
#[cfg(all(feature = "monitoring", test))]
#[test]
fn verify_overhead_under_one_percent() {
    use std::time::Instant;

    const ITERATIONS: usize = 1_000_000;

    // Baseline: operation without metrics
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        perform_baseline_operation();
    }
    let baseline_duration = start.elapsed();

    // With metrics
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        perform_instrumented_operation();
    }
    let instrumented_duration = start.elapsed();

    let overhead = (instrumented_duration.as_nanos() as f64
        - baseline_duration.as_nanos() as f64)
        / baseline_duration.as_nanos() as f64;

    assert!(
        overhead < 0.01,
        "Metrics overhead {:.2}% exceeds 1% threshold",
        overhead * 100.0
    );
}
```

## Acceptance Criteria

1. **Zero-cost when disabled:**
   - `cargo build --release --no-default-features` produces no cognitive_patterns symbols
   - Assembly inspection shows no metrics code in hot paths
   - `size_of::<CognitivePatternMetrics>()` ≤ pointer size

2. **<1% overhead when enabled:**
   - Criterion benchmark shows <1% regression on recall operations
   - P50, P95, P99 latencies within 1% of baseline
   - Verified on production-scale workload (10K ops/sec)

3. **Lock-free correctness:**
   - All atomic operations use appropriate ordering
   - No false sharing (cache-line padding verified)
   - Loom tests pass for concurrent access

4. **API completeness:**
   - Record methods for: priming, interference, reconsolidation, false memory
   - Query methods for: totals, histograms, rates
   - Integration with existing MetricsRegistry

## Testing Strategy

```bash
# Verify zero-cost elimination
cargo build --release --no-default-features
objdump -d target/release/engram-core | grep -c "cognitive_patterns"
# Expected: 0

# Run overhead benchmarks
cargo bench --bench metrics_overhead -- --save-baseline baseline
cargo bench --bench metrics_overhead --features monitoring -- --baseline instrumented
critcmp baseline instrumented

# Run correctness tests
cargo test --features monitoring metrics::cognitive_patterns
```

## Performance Budgets

- Recording operation: <50ns (atomic increment + cache-aligned access)
- Query operation: <100ns (atomic load)
- Memory footprint: <512 bytes per CognitivePatternMetrics instance

## Follow-ups

- Task 002: Semantic Priming (uses record_priming)
- Task 004: Proactive Interference (uses record_interference)
- Task 006: Reconsolidation (uses record_reconsolidation)
