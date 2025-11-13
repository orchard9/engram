# Task 005: Throughput Scaling Tests

**Status**: Pending
**Estimated Duration**: 3-4 days
**Priority**: High - Find breaking points

## Objective

Find system breaking point by gradually increasing QPS from 100 → 5000 ops/s. Identify bottlenecks (CPU, memory bandwidth, lock contention) at each inflection point using automated profiling.

## Key Approach

**Ramp Test**: Increase load 100 ops/s every 60s until P99 exceeds 100ms threshold or error rate >5%.

**Bottleneck Identification**:
1. **CPU-bound**: perf shows hot functions consuming >80% CPU
2. **Memory-bound**: Cache misses >10% of instructions
3. **Lock-bound**: Lock contention >1% of wall-clock time
4. **I/O-bound**: iowait >20% or disk queue depth >4

## Implementation

```rust
pub struct CapacityTest {
    start_ops: f64,       // 100 ops/s
    end_ops: f64,         // 5000 ops/s
    ramp_step: f64,       // 100 ops/s
    step_duration: Duration, // 60s
    breaking_point: Option<(f64, BottleneckType)>,
}

pub enum BottleneckType {
    Cpu { hot_functions: Vec<String> },
    Memory { cache_miss_rate: f64 },
    LockContention { contended_locks: Vec<String> },
    DiskIo { queue_depth: f64 },
}
```

## Success Criteria

- **Baseline Capacity**: >1000 ops/s at P99 <10ms
- **Breaking Point**: Identify bottleneck type automatically
- **Headroom**: 2x capacity beyond expected production load
- **Profiling**: Auto-capture flamegraph at breaking point

## Files

- `tools/loadtest/src/capacity/ramp_tester.rs` (380 lines)
- `scenarios/capacity/ramp_to_breaking_point.toml`
- `scripts/analyze_capacity.sh` (120 lines)
