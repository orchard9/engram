# Task 007: Thread Scalability Benchmarking

**Status**: Pending
**Estimated Duration**: 4-5 days
**Priority**: High - Validates lock-free design

## Objective

Measure throughput/latency scaling from 1 → 128 concurrent threads. Validate lock-free DashMap achieves >80% parallel efficiency to 32 cores. Identify contention hotspots using lock_stat and perf.

## Target Metrics

| Threads | Expected Speedup | Efficiency |
|---------|-----------------|------------|
| 1 | 1.0x (baseline) | 100% |
| 4 | 3.8x | 95% |
| 8 | 7.2x | 90% |
| 16 | 14.4x | 90% |
| 32 | 25.6x | 80% |
| 64 | 40x | 62% |
| 128 | 60x | 47% |

**Parallel efficiency** = (Speedup / Threads) * 100%

## Contention Analysis

```rust
pub struct ContentionAnalyzer {
    /// Per-thread performance counters
    thread_counters: Vec<ThreadCounters>,

    /// Lock contention events
    contention_events: Vec<ContentionEvent>,
}

pub struct ContentionEvent {
    pub lock_name: String,
    pub wait_time_ns: u64,
    pub thread_id: usize,
    pub stack_trace: Vec<String>,
}

impl ContentionAnalyzer {
    pub fn parallel_efficiency(&self) -> f64 {
        let total_cpu_time: u64 = self.thread_counters.iter()
            .map(|c| c.cpu_time_ns)
            .sum();

        let wall_clock_time = self.thread_counters[0].wall_clock_ns;
        let num_threads = self.thread_counters.len() as f64;

        (total_cpu_time as f64 / (wall_clock_time as f64 * num_threads)) * 100.0
    }

    pub fn identify_hotspots(&self) -> Vec<(String, f64)> {
        // Group contention events by lock, report top 5
        // ...
    }
}
```

## Test Scenarios

1. **Read-Heavy**: 90% recall, 10% store (minimal contention)
2. **Write-Heavy**: 90% store, 10% recall (maximum contention)
3. **Mixed**: 50/50 (realistic production)

## Success Criteria

- **8 cores**: >90% efficiency (14.4x speedup)
- **32 cores**: >80% efficiency (25.6x speedup)
- **Contention**: <1% time spent in lock contention
- **Hotspots**: No single lock >5% of contention time

## Files

- `tools/loadtest/src/concurrency/thread_scaler.rs` (450 lines)
- `scenarios/concurrency/thread_scaling.toml`
- `scripts/analyze_thread_scaling.sh` (150 lines)
