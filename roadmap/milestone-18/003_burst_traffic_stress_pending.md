# Task 003: Burst Traffic Stress Testing

**Status**: Pending
**Estimated Duration**: 2-3 days
**Priority**: Medium - Validates production resilience

## Objective

Validate system behavior under sudden traffic spikes (2-10x baseline) with automated recovery time measurement and resource pressure monitoring. Ensure graceful degradation rather than catastrophic failure.

## Problem Analysis

Production systems face sudden load spikes from:
- Viral content triggers (social media spikes)
- Scheduled batch jobs (nightly consolidation + morning queries)
- Cascade failures (retry storms from dependent services)
- DDoS attacks (legitimate but overwhelming traffic)

Need to measure:
- **Burst response**: Latency during 10x spike
- **Recovery time**: Time to return to baseline P99 after burst ends
- **Resource pressure**: Memory/CPU spikes, not just averages
- **Error behavior**: Graceful errors (503) vs crashes

## Implementation Highlights

### Burst Pattern Generator (extends Task 001)

```toml
# scenarios/stress/burst_10x_recovery.toml
[arrival]
pattern = "burst_sequence"
baseline_ops = 500.0
bursts = [
    { multiplier = 2.0, duration_sec = 30, delay_sec = 300 },
    { multiplier = 5.0, duration_sec = 60, delay_sec = 300 },
    { multiplier = 10.0, duration_sec = 30, delay_sec = 300 },
]

[validation]
max_burst_p99_ms = 100.0       # 10x during burst OK
recovery_time_target_sec = 5.0  # Must recover within 5s
max_error_rate_burst = 0.10     # 10% errors OK during burst
max_error_rate_steady = 0.01    # 1% after recovery
```

### Recovery Time Measurement

```rust
pub struct BurstRecoveryAnalyzer {
    baseline_p99: f64,
    burst_start: Option<Instant>,
    recovery_threshold: f64, // baseline_p99 * 1.1
}

impl BurstRecoveryAnalyzer {
    pub fn analyze_checkpoint(&mut self, metrics: &Metrics) -> Option<Duration> {
        if metrics.current_rate > self.baseline_rate * 1.5 {
            // Burst detected
            self.burst_start = Some(Instant::now());
        } else if let Some(start) = self.burst_start {
            // Check if recovered
            if metrics.p99_latency < self.recovery_threshold {
                let recovery_time = start.elapsed();
                self.burst_start = None;
                return Some(recovery_time);
            }
        }
        None
    }
}
```

## Key Scenarios

1. **Gradual Burst Sequence**: 2x → 5x → 10x with recovery time between
2. **Sustained Burst**: 5x for 5 minutes (tests resource exhaustion)
3. **Rapid Oscillation**: Alternate baseline/10x every 30s (tests adaptive capacity)

## Success Criteria

- **Recovery Time**: <5s to baseline P99 after burst end (10x spike)
- **Error Rate**: <10% during burst, <1% after recovery
- **No Crashes**: System stays up through all burst scenarios
- **Resource Bounds**: RSS stays <2x baseline during burst

## Files

- `tools/loadtest/src/burst/recovery_analyzer.rs` (280 lines)
- `scenarios/stress/burst_10x_recovery.toml`
- `scenarios/stress/sustained_burst.toml`
- `scenarios/stress/oscillating_load.toml`
- `tools/loadtest/tests/burst_tests.rs` (250 lines)
