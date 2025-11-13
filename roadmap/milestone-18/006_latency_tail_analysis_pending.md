# Task 006: Latency Tail Analysis

**Status**: Pending
**Estimated Duration**: 3-4 days
**Priority**: High - Production SLAs depend on P99.9

## Objective

Characterize P99.9 and P99.99 latency distribution at scale. Identify outlier causes (GC pauses, page faults, scheduler preemption) and establish percentile budgets for SLA planning.

## Problem Analysis

P99 hiding critical outliers:
- **P99 = 10ms**: 99% of requests <10ms, but 1% could be 1000ms
- **P99.9 = 50ms**: Acceptable for background, not for interactive
- **P99.99 = 500ms**: Unacceptable timeout territory

Need to measure:
- Full latency distribution (P50, P95, P99, P99.9, P99.99, P99.999)
- Outlier attribution (what causes 500ms+ requests?)
- Temporal patterns (do outliers cluster?)

## Implementation

```rust
pub struct TailAnalyzer {
    /// High-resolution histogram (HDR histogram)
    hdr_histogram: HdrHistogram,

    /// Outlier trace capture (>P99.9)
    outlier_traces: Vec<OutlierTrace>,
}

pub struct OutlierTrace {
    pub latency_ms: f64,
    pub operation: OpType,
    pub timestamp: Instant,
    pub stack_trace: Option<Vec<String>>, // Sampled
    pub system_context: SystemContext,
}

pub struct SystemContext {
    pub cpu_usage: f64,
    pub available_memory_mb: f64,
    pub disk_queue_depth: usize,
    pub network_latency_ms: f64,
}
```

## Key Metrics

- **P99.9 Target**: <50ms (100x slower than P50 acceptable)
- **P99.99 Target**: <200ms (must stay below timeout)
- **Max observed**: <1000ms (circuit breaker threshold)

## Success Criteria

- **P99.9 Measurement**: Accurate to ±5% with HDR histogram
- **Outlier Attribution**: Identify cause for >80% of P99.9+ requests
- **Temporal Clustering**: Detect if outliers cluster (GC pauses)
- **SLA Recommendation**: Propose P99.9 SLA based on measurements

## Files

- `tools/loadtest/src/tail/analyzer.rs` (420 lines)
- `tools/loadtest/src/tail/outlier_tracer.rs` (310 lines)
- `scripts/analyze_tail_latency.py` (280 lines)
