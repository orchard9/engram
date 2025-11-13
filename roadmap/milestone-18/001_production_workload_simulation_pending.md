# Task 001: Realistic Production Workload Simulation

**Status**: Pending
**Estimated Duration**: 3-4 days
**Priority**: High - Foundational for M18

## Objective

Build comprehensive production workload simulator that generates realistic traffic patterns (daily cycles, burst patterns, gradual ramps) with configurable operation mixes and temporal correlations. Validate system behavior under real-world conditions beyond synthetic uniform load.

## Problem Analysis

Current loadtest tool (M17) generates constant-rate uniform traffic with fixed operation weights. Production systems experience:
- **Diurnal cycles**: Peak hours (9am-5pm), off-peak, weekend patterns
- **Burst traffic**: Sudden spikes (2-10x baseline) from viral content, alerts, batch jobs
- **Temporal correlation**: Related operations cluster (store → immediate recall)
- **Operation sequences**: Multi-step workflows (search → recall → update)

Synthetic constant-rate tests miss critical failure modes: memory pressure during sustained peaks, cache thrashing during bursts, contention during correlated writes.

## Architecture

### Workload Pattern Generator

```rust
// File: tools/loadtest/src/patterns/production.rs

pub enum WorkloadPattern {
    /// Constant rate (existing M17 behavior)
    Constant { ops_per_sec: f64 },

    /// Diurnal cycle: sine wave with configurable amplitude
    Diurnal {
        baseline_ops: f64,
        peak_multiplier: f64,  // 2.0 = 2x at peak
        peak_hour: u8,         // 0-23, local time
        trough_hour: u8,
    },

    /// Poisson bursts: random spikes following Poisson process
    Bursts {
        baseline_ops: f64,
        burst_multiplier: f64, // 5.0 = 5x during burst
        burst_lambda: f64,     // Average bursts per minute
        burst_duration: Duration,
    },

    /// Ramp: gradual increase/decrease
    Ramp {
        start_ops: f64,
        end_ops: f64,
        duration: Duration,
    },

    /// Composite: combine multiple patterns
    Composite {
        patterns: Vec<(WorkloadPattern, f64)>, // (pattern, weight)
    },
}

impl WorkloadPattern {
    /// Calculate target ops/sec at given elapsed time
    pub fn ops_at_time(&self, elapsed: Duration, rng: &mut impl Rng) -> f64 {
        match self {
            Self::Constant { ops_per_sec } => *ops_per_sec,
            Self::Diurnal { baseline_ops, peak_multiplier, peak_hour, trough_hour } => {
                let hour = (elapsed.as_secs() / 3600) % 24;
                let phase = (hour as f64 - *peak_hour as f64) * PI / 12.0;
                baseline_ops * (1.0 + (peak_multiplier - 1.0) * phase.cos())
            },
            Self::Bursts { baseline_ops, burst_multiplier, burst_lambda, burst_duration } => {
                // Poisson process: P(burst in dt) = lambda * dt
                let burst_prob = burst_lambda * elapsed.as_secs_f64() / 60.0;
                if rng.gen_bool(burst_prob) {
                    baseline_ops * burst_multiplier
                } else {
                    *baseline_ops
                }
            },
            // ... other patterns
        }
    }
}
```

### Temporal Correlation Engine

```rust
// File: tools/loadtest/src/patterns/correlation.rs

pub struct TemporalCorrelation {
    /// Probability that next operation is correlated with previous
    correlation_strength: f64,  // 0.0-1.0

    /// Operation transition matrix: P(op_j | op_i)
    transition_matrix: [[f64; 4]; 4],  // 4x4 for store/recall/search/complete

    /// Time delay distribution for correlated operations
    delay_distribution: DelayDistribution,
}

pub enum DelayDistribution {
    /// Immediate (same batch)
    Immediate,

    /// Exponential delay with mean lambda
    Exponential { lambda: f64 },

    /// Fixed delay
    Fixed { duration: Duration },
}

impl TemporalCorrelation {
    /// Default production correlation: store → recall with 200ms delay
    pub fn default_production() -> Self {
        let mut matrix = [[0.25; 4]; 4]; // Uniform fallback
        matrix[0][1] = 0.6; // store → recall (60%)
        matrix[1][2] = 0.4; // recall → search (40%)
        matrix[2][1] = 0.5; // search → recall (50%)

        Self {
            correlation_strength: 0.3,
            transition_matrix: matrix,
            delay_distribution: DelayDistribution::Exponential { lambda: 0.2 },
        }
    }

    /// Sample next operation based on current operation
    pub fn next_operation(&self, current: OpType, rng: &mut impl Rng) -> (OpType, Duration) {
        if rng.gen_bool(self.correlation_strength) {
            // Correlated operation
            let next_op = self.sample_from_row(current as usize, rng);
            let delay = self.delay_distribution.sample(rng);
            (next_op, delay)
        } else {
            // Uncorrelated operation (uniform)
            (rng.gen(), Duration::ZERO)
        }
    }
}
```

### Production Scenario Definitions

```toml
# File: scenarios/production/daily_cycle.toml
name = "Daily Cycle Production Workload"
description = "24-hour simulation with diurnal pattern and realistic operation mix"

[duration]
total_seconds = 86400  # 24 hours

[arrival]
pattern = "diurnal"
baseline_ops = 500.0
peak_multiplier = 3.0
peak_hour = 14  # 2pm
trough_hour = 3  # 3am

[operations]
# Morning: heavy reads
diurnal_schedule = [
    { hours = [0, 6], store_weight = 0.2, recall_weight = 0.5, search_weight = 0.3 },
    { hours = [6, 12], store_weight = 0.4, recall_weight = 0.4, search_weight = 0.2 },
    { hours = [12, 18], store_weight = 0.3, recall_weight = 0.4, search_weight = 0.3 },
    { hours = [18, 24], store_weight = 0.2, recall_weight = 0.5, search_weight = 0.3 },
]

[correlation]
enabled = true
correlation_strength = 0.3
store_to_recall_prob = 0.6
delay_distribution = { type = "exponential", lambda_sec = 0.2 }

[data]
num_nodes = 100_000
embedding_dim = 768
memory_spaces = 4

[validation]
expected_p99_latency_ms = 10.0
expected_throughput_ops_sec = 400.0  # Average over 24h
max_error_rate = 0.01
```

```toml
# File: scenarios/production/burst_traffic.toml
name = "Burst Traffic Stress Test"
description = "Sudden 10x traffic spike to validate burst handling"

[duration]
total_seconds = 600  # 10 minutes

[arrival]
pattern = "composite"
patterns = [
    { type = "constant", ops_per_sec = 500.0, weight = 0.8 },
    { type = "bursts", baseline = 500.0, multiplier = 10.0, lambda = 0.5, duration_sec = 30, weight = 0.2 },
]

[operations]
store_weight = 0.3
recall_weight = 0.4
embedding_search_weight = 0.3

[data]
num_nodes = 100_000
embedding_dim = 768

[validation]
expected_p99_latency_ms = 50.0  # Higher tolerance during bursts
burst_recovery_time_sec = 5.0   # Time to return to baseline
max_error_rate = 0.05           # 5% errors acceptable during burst
```

```toml
# File: scenarios/production/gradual_ramp.toml
name = "Gradual Traffic Ramp"
description = "Slow increase from 100 to 2000 ops/s to find capacity limits"

[duration]
total_seconds = 3600  # 1 hour

[arrival]
pattern = "ramp"
start_ops = 100.0
end_ops = 2000.0

[operations]
store_weight = 0.35
recall_weight = 0.35
embedding_search_weight = 0.3

[data]
num_nodes = 100_000
embedding_dim = 768

[validation]
# No fixed targets - find breaking point
capacity_threshold_p99_ms = 100.0  # Fail if P99 exceeds this
throughput_degradation_pct = 20.0  # Fail if throughput drops >20%
```

## Implementation Plan

### 1. Extend Loadtest Arrival Pattern Engine

**Files to modify:**
- `tools/loadtest/src/workload_generator.rs`
- `tools/loadtest/src/distribution.rs` (add diurnal/burst distributions)

**Changes:**
```rust
// Add to WorkloadConfig
pub struct ArrivalConfig {
    pub pattern: ArrivalPattern,
    // ... existing fields
}

pub enum ArrivalPattern {
    Constant { rate: f64 },
    Diurnal { baseline: f64, peak_multiplier: f64, peak_hour: u8, trough_hour: u8 },
    Bursts { baseline: f64, multiplier: f64, lambda: f64, duration: Duration },
    Ramp { start: f64, end: f64 },
    Composite { patterns: Vec<(Box<ArrivalPattern>, f64)> },
}
```

**Test strategy:**
- Unit tests: Verify each pattern generates expected distribution
- Property tests: Ensure ops/sec stays within bounds
- Visualization: Plot generated patterns to validate shape

### 2. Implement Temporal Correlation

**New file:** `tools/loadtest/src/patterns/correlation.rs`

**Key functions:**
- `TemporalCorrelation::next_operation()` - Sample correlated op
- `CorrelationTracker::record_operation()` - Track for analysis
- `CorrelationMetrics::measure()` - Validate correlation strength

**Test strategy:**
- Statistical tests: Chi-squared for transition matrix
- Correlation coefficient: Measure realized vs target correlation

### 3. Create Production Scenario Suite

**New files:**
- `scenarios/production/daily_cycle.toml`
- `scenarios/production/burst_traffic.toml`
- `scenarios/production/gradual_ramp.toml`
- `scenarios/production/README.md`

**Validation:**
- Dry-run mode: Generate workload without executing
- Visualization: Export operation timeline to CSV for plotting

### 4. Build Analysis Tools

**New file:** `tools/loadtest/src/analysis/production_metrics.rs`

**Metrics to track:**
- **Burst recovery time**: Time from burst end to baseline P99
- **Capacity headroom**: Max sustained ops/s before degradation
- **Diurnal stability**: Coefficient of variation across day
- **Correlation effectiveness**: Realized vs target correlation

**Implementation:**
```rust
pub struct BurstMetrics {
    pub burst_start: Instant,
    pub burst_end: Instant,
    pub peak_latency: Duration,
    pub recovery_time: Duration,  // Time to baseline P99
    pub error_spike: f64,          // Peak error rate
}

pub struct CapacityMetrics {
    pub max_sustained_ops: f64,
    pub breaking_point_ops: f64,   // When P99 > threshold
    pub degradation_curve: Vec<(f64, Duration)>, // (ops/s, P99)
}
```

### 5. Integration Testing

**Test scenarios:**
1. **Diurnal pattern**: 1-hour accelerated cycle, verify peak/trough
2. **Burst handling**: 10x spike, verify recovery <5s
3. **Ramp test**: Find capacity limit on current hardware
4. **Correlation**: Verify store→recall sequences happen

**Acceptance criteria:**
- Diurnal: Peak ops/s = baseline * peak_multiplier ± 10%
- Burst: Recovery time <5s after burst end
- Ramp: Identify breaking point with <10% variance across runs
- Correlation: Realized correlation ≥0.8 * target correlation

## File Changes

### New Files
- `tools/loadtest/src/patterns/production.rs` (350 lines)
- `tools/loadtest/src/patterns/correlation.rs` (280 lines)
- `tools/loadtest/src/analysis/production_metrics.rs` (420 lines)
- `scenarios/production/daily_cycle.toml` (60 lines)
- `scenarios/production/burst_traffic.toml` (45 lines)
- `scenarios/production/gradual_ramp.toml` (40 lines)
- `scenarios/production/README.md` (150 lines)
- `tools/loadtest/tests/production_pattern_tests.rs` (380 lines)

### Modified Files
- `tools/loadtest/src/workload_generator.rs`: Add pattern support
- `tools/loadtest/src/distribution.rs`: Add diurnal/burst distributions
- `tools/loadtest/src/main.rs`: Add --pattern CLI flag
- `tools/loadtest/src/report.rs`: Add production metrics section

**Total new code**: ~1,700 lines
**Total modified code**: ~200 lines

## Testing Strategy

### Unit Tests
- Pattern generators produce expected distributions
- Correlation engine follows transition probabilities
- Delay distributions match specified parameters

### Integration Tests
- End-to-end workload generation
- Metrics collection during production patterns
- Analysis tools produce correct statistics

### Validation Tests
- Visual inspection: Plot generated patterns
- Statistical tests: Chi-squared for distributions
- Correlation analysis: Pearson coefficient for temporal correlation

## Performance Validation

**Baseline (before changes):**
```bash
./scripts/m17_performance_check.sh 001 before
```

**After implementation:**
```bash
./scripts/m17_performance_check.sh 001 after
./scripts/compare_m17_performance.sh 001
```

**Expected impact:**
- Internal regression: <2% (pattern generation is lightweight)
- New capability: Production workload simulation
- Analysis time: <5s for metric computation

## Success Criteria

1. **Pattern Generation**: All patterns (diurnal, burst, ramp) generate expected distributions
2. **Temporal Correlation**: Realized correlation ≥80% of target
3. **Burst Handling**: System recovers to baseline P99 within 5s of burst end
4. **Capacity Discovery**: Ramp test identifies breaking point with <10% variance
5. **Zero Regression**: <2% impact on M17 baseline performance
6. **Documentation**: README with usage examples and pattern descriptions

## Risk Mitigation

1. **Complexity**: Start with simple patterns (constant, ramp), add sophisticated patterns incrementally
2. **Determinism**: Use fixed seeds for all RNG operations
3. **Test duration**: Short scenarios (10min) for CI, long scenarios (24h) for nightly runs
4. **Resource usage**: Monitor memory during pattern generation to avoid allocation spikes

## Dependencies

- M17 loadtest tool (existing)
- rand crate for statistical distributions
- plotters crate (optional) for visualization

## Next Steps After Completion

1. Run 24-hour diurnal cycle on production hardware
2. Characterize burst recovery time across dataset sizes
3. Document capacity limits for current hardware tier
4. Create production deployment guide with recommended workload patterns
