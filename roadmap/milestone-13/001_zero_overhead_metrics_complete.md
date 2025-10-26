# Task 001: Zero-Overhead Metrics Infrastructure (CORRECTED)

**Status:** Pending (Requires architectural fixes before implementation)
**Priority:** P0 (Foundation)
**Estimated Effort:** 3 days (increased from 2 days due to architectural corrections)
**Dependencies:** None

## CRITICAL CORRECTIONS APPLIED

This is the architecturally-corrected version of Task 001 based on systems architecture review. See `SYSTEMS_ARCHITECTURE_REVIEW.md` for detailed rationale.

**Key changes from original spec:**
1. Removed Arc wrapper (eliminates pointer indirection overhead)
2. Fixed histogram sum calculation (was broken, now uses atomic_float)
3. Corrected zero-cost abstraction approach (proper conditional compilation)
4. Added loom tests for lock-free correctness
5. Fixed assembly verification methodology
6. Added cache behavior assumptions and NUMA considerations

---

## Objective

Implement conditional compilation infrastructure for cognitive pattern metrics with provably zero overhead when disabled and <1% overhead when enabled. Create lock-free atomic collection with cache-line alignment.

---

## Research Foundation

Traditional observability tools add 5-15% latency overhead even when metrics aren't being collected. For a memory system processing 10K recalls/sec with sub-millisecond latency requirements, this is unacceptable. We need instrumentation that costs literally zero when disabled and <1% when enabled.

**Conditional compilation solution:**
Unlike runtime feature flags that branch at every instrumentation point, `#[cfg(feature = "monitoring")]` removes code entirely during compilation. When metrics disabled, compiler optimizes away not just metric recording calls, but entire surrounding code path - assembly output identical to un-instrumented code.

**Lock-free atomic metrics (Vyukov 2007):**
Each thread updates shared atomic counters with cache-line padding to prevent false sharing. Atomic fetch-add operations (15-20ns on modern x86_64) provide lock-free progress guarantee. Background thread periodically reads aggregated values without blocking workers.

---

## Performance Budget (CORRECTED)

**Hot path (L1 cached):**
- Counter increment: <25ns (atomic LOCK ADD instruction)
- Histogram record: <80ns (includes binary search + atomic update)

**Warm path (L3 cached):**
- Counter increment: <80ns
- Histogram record: <200ns

**Cold path (main memory):**
- Counter increment: <250ns (acceptable for infrequent operations)
- Histogram record: <500ns

**Overhead target:**
- When monitoring DISABLED: 0ns (compiler eliminates all code)
- When monitoring ENABLED: <1% on production workload (10K ops/sec)

**Assumptions:**
- Metrics struct is hot in L1/L2 cache (>95% hit rate)
- Atomic operations are uncontended (no lock escalation)
- Cache line size: 64 bytes (x86-64) or 128 bytes (Apple Silicon)

---

## Integration Points

**Extends:**
- `/engram-core/src/metrics/mod.rs` - Add cognitive pattern exports
- `/engram-core/src/metrics/lockfree.rs` - Fix histogram sum bug, add atomic_float support

**Creates:**
- `/engram-core/src/metrics/cognitive_patterns.rs` - New cognitive metrics types
- `/engram-core/benches/metrics_overhead.rs` - Overhead validation benchmarks
- `/engram-core/tests/metrics/zero_overhead_tests.rs` - Conditional compilation tests
- `/engram-core/tests/metrics/loom_tests.rs` - Concurrency correctness verification

---

## Detailed Specification

### 1. Conditional Compilation Strategy (CORRECTED)

**File:** `/engram-core/src/metrics/cognitive_patterns.rs`

```rust
//! Cognitive pattern metrics with zero-overhead when monitoring disabled

use crate::metrics::lockfree::{LockFreeCounter, LockFreeHistogram};
use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicU64, Ordering};

/// Zero-overhead cognitive pattern metrics
///
/// When `monitoring` feature is disabled, this struct is zero-sized and all
/// methods are no-ops that compile away entirely.
///
/// When `monitoring` feature is enabled, provides lock-free atomic metrics
/// with <1% overhead.
#[cfg(feature = "monitoring")]
pub struct CognitivePatternMetrics {
    // Direct struct fields - NO Arc wrapper
    // This eliminates pointer indirection overhead (critical fix #1)

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

/// When monitoring disabled, use zero-sized type with PhantomData marker
#[cfg(not(feature = "monitoring"))]
pub struct CognitivePatternMetrics {
    _phantom: core::marker::PhantomData<()>,
}

impl CognitivePatternMetrics {
    /// Create new metrics instance
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn new() -> Self {
        Self {
            priming_events_total: CachePadded::new(AtomicU64::new(0)),
            priming_strength_histogram: LockFreeHistogram::new(),
            priming_type_counters: [
                CachePadded::new(AtomicU64::new(0)),
                CachePadded::new(AtomicU64::new(0)),
                CachePadded::new(AtomicU64::new(0)),
            ],
            interference_detections_total: CachePadded::new(AtomicU64::new(0)),
            proactive_interference_strength: LockFreeHistogram::new(),
            retroactive_interference_strength: LockFreeHistogram::new(),
            fan_effect_magnitude: LockFreeHistogram::new(),
            reconsolidation_events_total: CachePadded::new(AtomicU64::new(0)),
            reconsolidation_modifications: CachePadded::new(AtomicU64::new(0)),
            reconsolidation_window_hits: CachePadded::new(AtomicU64::new(0)),
            reconsolidation_window_misses: CachePadded::new(AtomicU64::new(0)),
            false_memory_generations: CachePadded::new(AtomicU64::new(0)),
            drm_critical_lure_recalls: CachePadded::new(AtomicU64::new(0)),
            drm_list_item_recalls: CachePadded::new(AtomicU64::new(0)),
            massed_practice_events: CachePadded::new(AtomicU64::new(0)),
            distributed_practice_events: CachePadded::new(AtomicU64::new(0)),
            retention_improvement_histogram: LockFreeHistogram::new(),
        }
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }

    /// Record priming event with zero overhead when monitoring disabled
    ///
    /// # Performance
    /// - Monitoring disabled: 0ns (function is empty, completely optimized away)
    /// - Monitoring enabled: ~25ns (hot path, L1 cached)
    ///
    /// # Implementation note
    /// Using separate #[cfg] blocks instead of single block with unused variable
    /// suppression. This is more verbose but ensures compiler can optimize each
    /// variant independently.
    #[inline(always)]
    pub fn record_priming(&self, priming_type: PrimingType, strength: f32) {
        #[cfg(feature = "monitoring")]
        {
            self.priming_events_total.fetch_add(1, Ordering::Relaxed);
            self.priming_strength_histogram.record(f64::from(strength));

            let idx = priming_type as usize;
            self.priming_type_counters[idx].fetch_add(1, Ordering::Relaxed);
        }

        // When monitoring disabled, function body is empty - compiles to nothing
        #[cfg(not(feature = "monitoring"))]
        {
            let _ = (priming_type, strength); // Suppress unused warnings
        }
    }

    #[inline(always)]
    pub fn record_interference(
        &self,
        interference_type: InterferenceType,
        magnitude: f32
    ) {
        #[cfg(feature = "monitoring")]
        {
            self.interference_detections_total.fetch_add(1, Ordering::Relaxed);

            match interference_type {
                InterferenceType::Proactive => {
                    self.proactive_interference_strength.record(f64::from(magnitude));
                }
                InterferenceType::Retroactive => {
                    self.retroactive_interference_strength.record(f64::from(magnitude));
                }
                InterferenceType::Fan => {
                    self.fan_effect_magnitude.record(f64::from(magnitude));
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
            self.reconsolidation_events_total.fetch_add(1, Ordering::Relaxed);

            if (0.0..=1.0).contains(&window_position) {
                self.reconsolidation_window_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.reconsolidation_window_misses.fetch_add(1, Ordering::Relaxed);
            }
        }

        #[cfg(not(feature = "monitoring"))]
        {
            let _ = window_position;
        }
    }

    /// Get priming event total
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn priming_events_total(&self) -> u64 {
        // CORRECTED: Use Relaxed ordering for counter reads
        // Acquire is unnecessarily strong - we just want latest visible value
        self.priming_events_total.load(Ordering::Relaxed)
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn priming_events_total(&self) -> u64 {
        0
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
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

#[cfg(feature = "monitoring")]
impl Default for CognitivePatternMetrics {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 2. Lock-Free Histogram Fix (CRITICAL)

**File:** `/engram-core/src/metrics/lockfree.rs`

**Problem:** Current implementation adds f64 bit representations, producing garbage values.

**Fix:** Use `atomic_float::AtomicF64` for sum tracking (crate already in dependencies).

```rust
// Add to imports
use atomic_float::AtomicF64;

pub struct LockFreeHistogram {
    /// Exponential bucket boundaries
    buckets: Vec<f64>,
    /// Atomic counters for each bucket
    counts: Vec<CachePadded<AtomicU64>>,
    /// Total count
    total_count: CachePadded<AtomicU64>,
    /// Sum of all values (for mean calculation)
    /// CORRECTED: Use AtomicF64 instead of storing bit representation in AtomicU64
    sum: CachePadded<AtomicF64>,
}

impl LockFreeHistogram {
    #[must_use]
    pub fn new() -> Self {
        Self::with_buckets(Self::default_exponential_buckets())
    }

    #[must_use]
    pub fn with_buckets(buckets: Vec<f64>) -> Self {
        let counts = (0..=buckets.len())
            .map(|_| CachePadded::new(AtomicU64::new(0)))
            .collect();

        Self {
            buckets,
            counts,
            total_count: CachePadded::new(AtomicU64::new(0)),
            // CORRECTED: Initialize with AtomicF64, not bit-packed u64
            sum: CachePadded::new(AtomicF64::new(0.0)),
        }
    }

    /// Record a value with <100ns overhead
    pub fn record(&self, value: f64) {
        // Find the appropriate bucket using binary search
        let bucket_idx = match self.buckets.binary_search_by(|boundary| {
            boundary
                .partial_cmp(&value)
                .unwrap_or(std::cmp::Ordering::Greater)
        }) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };

        // Increment the bucket counter
        self.counts[bucket_idx].fetch_add(1, Ordering::Relaxed);
        self.total_count.fetch_add(1, Ordering::Relaxed);

        // CORRECTED: Use atomic f64 add, not bit representation
        self.sum.fetch_add(value, Ordering::Relaxed);
    }

    /// Get the mean value
    #[must_use]
    pub fn mean(&self) -> f64 {
        let count = self.total_count.load(Ordering::Acquire);
        if count == 0 {
            return 0.0;
        }

        // CORRECTED: Load actual f64 value
        let sum = self.sum.load(Ordering::Acquire);
        sum / (count as f64)
    }

    /// Reset all counters
    pub fn reset(&self) {
        for count in &self.counts {
            count.store(0, Ordering::Release);
        }
        self.total_count.store(0, Ordering::Release);
        // CORRECTED: Reset f64 sum properly
        self.sum.store(0.0, Ordering::Release);
    }
}
```

---

### 3. Loom Concurrency Tests (NEW - REQUIRED)

**File:** `/engram-core/tests/metrics/loom_tests.rs`

```rust
//! Loom-based verification of lock-free correctness
//!
//! These tests use the loom library to explore all possible thread interleavings
//! and verify that our lock-free data structures maintain correctness under
//! concurrent access.

#![cfg(all(test, loom))]

use loom::sync::Arc;
use loom::thread;
use engram_core::metrics::lockfree::LockFreeCounter;

#[test]
fn loom_concurrent_counter_increments() {
    loom::model(|| {
        let counter = Arc::new(LockFreeCounter::new());

        let handles: Vec<_> = (0..2).map(|_| {
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                counter.increment(1);
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }

        // After 2 threads each increment by 1, total must be 2
        assert_eq!(counter.get(), 2);
    });
}

#[test]
fn loom_concurrent_histogram_records() {
    loom::model(|| {
        use engram_core::metrics::lockfree::LockFreeHistogram;

        let histogram = Arc::new(LockFreeHistogram::new());

        let handles: Vec<_> = (0..2).map(|i| {
            let histogram = Arc::clone(&histogram);
            thread::spawn(move || {
                histogram.record(i as f64 + 1.0);
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }

        // After 2 threads each record 1 value, count must be 2
        assert_eq!(histogram.count(), 2);

        // Mean should be (1.0 + 2.0) / 2 = 1.5
        let mean = histogram.mean();
        assert!((mean - 1.5).abs() < 0.01, "mean = {mean}, expected ~1.5");
    });
}

#[test]
fn loom_concurrent_priming_type_counters() {
    loom::model(|| {
        use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};

        let metrics = Arc::new(CognitivePatternMetrics::new());

        let handles: Vec<_> = (0..2).map(|_| {
            let metrics = Arc::clone(&metrics);
            thread::spawn(move || {
                metrics.record_priming(PrimingType::Semantic, 0.5);
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }

        // After 2 threads each record 1 event, total must be 2
        assert_eq!(metrics.priming_events_total(), 2);
    });
}
```

---

### 4. Overhead Validation Benchmark (CORRECTED)

**File:** `/engram-core/benches/metrics_overhead.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};
use std::time::Duration;

/// Baseline: operation without any metrics
fn baseline_operation() -> f32 {
    // Simulate a typical spreading activation step
    let mut sum = 0.0f32;
    for i in 0..100 {
        sum += (i as f32).sin();
    }
    sum
}

/// Operation with metrics recording
#[cfg(feature = "monitoring")]
fn instrumented_operation(metrics: &CognitivePatternMetrics) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..100 {
        sum += (i as f32).sin();

        // Record priming event every 10 iterations (realistic pattern)
        if i % 10 == 0 {
            metrics.record_priming(PrimingType::Semantic, sum / 100.0);
        }
    }
    sum
}

fn benchmark_metrics_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_overhead");
    group.sample_size(1000);
    group.warm_up_time(Duration::from_secs(3));

    // Baseline without metrics
    group.bench_function("baseline_no_metrics", |b| {
        b.iter(|| {
            black_box(baseline_operation());
        });
    });

    // With metrics enabled (only compiles if monitoring feature enabled)
    #[cfg(feature = "monitoring")]
    {
        let metrics = CognitivePatternMetrics::new();

        group.bench_function("with_metrics", |b| {
            b.iter(|| {
                black_box(instrumented_operation(&metrics));
            });
        });
    }

    group.finish();
}

fn benchmark_single_record_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_record_latency");
    group.sample_size(10000);

    #[cfg(feature = "monitoring")]
    {
        let metrics = CognitivePatternMetrics::new();

        group.bench_function("record_priming", |b| {
            b.iter(|| {
                metrics.record_priming(
                    black_box(PrimingType::Semantic),
                    black_box(0.75)
                );
            });
        });
    }

    group.finish();
}

fn benchmark_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");

    #[cfg(feature = "monitoring")]
    {
        let metrics = CognitivePatternMetrics::new();

        for count in [100, 1000, 10000, 100000] {
            group.throughput(Throughput::Elements(count));
            group.bench_with_input(
                BenchmarkId::new("record_priming", count),
                &count,
                |b, &count| {
                    b.iter(|| {
                        for i in 0..count {
                            metrics.record_priming(
                                PrimingType::Semantic,
                                (i as f32) / (count as f32)
                            );
                        }
                    });
                }
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_metrics_overhead,
    benchmark_single_record_latency,
    benchmark_throughput_scaling
);
criterion_main!(benches);
```

---

### 5. Zero-Cost Verification Test (CORRECTED)

**File:** `/engram-core/tests/metrics/zero_overhead_tests.rs`

```rust
//! Verification that metrics have truly zero overhead when disabled

#[cfg(not(feature = "monitoring"))]
#[test]
fn verify_zero_size_when_disabled() {
    use std::mem::size_of;
    use engram_core::metrics::cognitive_patterns::CognitivePatternMetrics;

    // When monitoring disabled, struct should be zero-sized (just PhantomData)
    assert_eq!(
        size_of::<CognitivePatternMetrics>(),
        0,
        "CognitivePatternMetrics should be zero-sized when monitoring disabled"
    );
}

#[cfg(not(feature = "monitoring"))]
#[test]
fn verify_methods_are_noops() {
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};

    // Methods should compile and do nothing
    let metrics = CognitivePatternMetrics::new();
    metrics.record_priming(PrimingType::Semantic, 0.5);

    // If this compiles and runs, conditional compilation works
    // No assertions needed - successful compilation is the test
}

#[cfg(all(feature = "monitoring", test))]
#[test]
fn verify_overhead_under_one_percent() {
    use std::time::Instant;
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};

    const ITERATIONS: usize = 1_000_000;

    // Baseline: operation without metrics
    fn baseline_op() -> f32 {
        let mut sum = 0.0f32;
        for i in 0..10 {
            sum += (i as f32).sin();
        }
        sum
    }

    // Baseline measurement
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(baseline_op());
    }
    let baseline_duration = start.elapsed();

    // With metrics
    let metrics = CognitivePatternMetrics::new();
    let start = Instant::now();
    for i in 0..ITERATIONS {
        std::hint::black_box(baseline_op());
        metrics.record_priming(PrimingType::Semantic, (i % 100) as f32 / 100.0);
    }
    let instrumented_duration = start.elapsed();

    let overhead = (instrumented_duration.as_nanos() as f64
        - baseline_duration.as_nanos() as f64)
        / baseline_duration.as_nanos() as f64;

    println!("Baseline: {:?}", baseline_duration);
    println!("Instrumented: {:?}", instrumented_duration);
    println!("Overhead: {:.2}%", overhead * 100.0);

    assert!(
        overhead < 0.01,
        "Metrics overhead {:.2}% exceeds 1% threshold",
        overhead * 100.0
    );
}
```

---

## Acceptance Criteria (CORRECTED)

### 1. Zero-cost when disabled

- [ ] `cargo build --release --no-default-features` compiles successfully
- [ ] `size_of::<CognitivePatternMetrics>() == 0` when monitoring disabled
- [ ] Zero-overhead test passes (verify_zero_size_when_disabled)
- [ ] Methods compile but are no-ops when monitoring disabled

### 2. <1% overhead when enabled

- [ ] Criterion benchmark shows <1% regression on realistic workload
- [ ] Single record latency: P50 <25ns, P99 <100ns
- [ ] Throughput scales linearly with operation count
- [ ] Overhead test passes (verify_overhead_under_one_percent)

### 3. Lock-free correctness

- [ ] All atomic operations use documented memory ordering
- [ ] Loom tests pass for concurrent increments (loom_concurrent_counter_increments)
- [ ] Loom tests pass for histogram records (loom_concurrent_histogram_records)
- [ ] No false sharing (verified via cache-line padding)

### 4. Histogram correctness

- [ ] Mean calculation produces correct values (CORRECTED: uses AtomicF64)
- [ ] Quantiles are monotonically increasing
- [ ] Reset operation clears all state
- [ ] Concurrent records produce correct totals

### 5. API completeness

- [ ] Record methods for: priming, interference, reconsolidation, false memory
- [ ] Query methods for: totals, histograms, rates
- [ ] Integration with existing MetricsRegistry
- [ ] Documentation for all public methods

---

## Testing Strategy (CORRECTED)

```bash
# 1. Verify zero-cost elimination
cargo test --lib --no-default-features metrics::zero_overhead
# Expected: All zero-overhead tests pass

# 2. Run loom concurrency tests
RUSTFLAGS="--cfg loom" cargo test --lib --release metrics::loom_tests
# Expected: All loom models pass (may take several minutes)

# 3. Run overhead benchmarks
cargo bench --bench metrics_overhead -- --save-baseline without_monitoring
cargo bench --bench metrics_overhead --features monitoring -- --save-baseline with_monitoring

# 4. Compare baselines (requires critcmp tool)
critcmp without_monitoring with_monitoring

# 5. Verify histogram mean calculation fix
cargo test --lib --features monitoring lockfree::histogram
```

---

## Performance Budgets (CORRECTED)

**Per-operation costs (monitoring enabled):**
- Counter increment (hot path): <25ns (P99)
- Histogram record (hot path): <80ns (P99)
- Counter read: <10ns (Relaxed load)
- Histogram quantile query: <1μs (all buckets)

**Memory footprint:**
- CognitivePatternMetrics struct: ~2KB (cache-aligned atomics + histograms)
- Per-histogram: ~512 bytes (64 buckets × 8 bytes/counter)
- Zero when monitoring disabled: 0 bytes

**Cache behavior assumptions:**
- L1 hit rate: >95% for hot metrics
- L3 hit rate: >99.9% for warm metrics
- False sharing: <1% (verified via CachePadded alignment)

---

## Follow-ups

After Task 001 completes:
- Task 002: Semantic Priming (uses record_priming)
- Task 004: Proactive Interference (uses record_interference)
- Task 006: Reconsolidation (uses record_reconsolidation)
- Task 011: Cognitive Tracing Infrastructure (depends on metrics foundation)

---

## References

1. Dmitry Vyukov (2007). "Bounded MPMC queue". https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
2. Herlihy & Shavit (2012). "The Art of Multiprocessor Programming". Chapter 18: Linearizability.
3. Crossbeam documentation: https://docs.rs/crossbeam-utils/latest/crossbeam_utils/struct.CachePadded.html
4. atomic_float crate: https://docs.rs/atomic_float/latest/atomic_float/

---

**IMPORTANT:** Do not begin implementation until architectural review is approved and this corrected specification is accepted.
