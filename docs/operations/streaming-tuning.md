# Streaming Performance Tuning Guide

Advanced guide for tuning Engram's streaming infrastructure for specific workload profiles and performance targets.

## Table of Contents

- [Workload Profiles](#workload-profiles)
- [Configuration Parameters](#configuration-parameters)
- [Tuning Methodology](#tuning-methodology)
- [Trade-offs Analysis](#trade-offs-analysis)
- [Benchmarking](#benchmarking)
- [Case Studies](#case-studies)

## Workload Profiles

### Low-Latency Interactive

**Use Cases**: Real-time recommendations, interactive search, conversational AI

**Requirements**:
- P99 latency < 10ms
- Throughput: 1K-10K obs/sec
- Recall latency < 5ms

**Configuration**:
```rust
WorkerPoolConfig {
    num_workers: 8,              // Maximum parallelism
    min_batch_size: 10,          // Small batches
    max_batch_size: 50,          // Limit batch growth
    idle_sleep_ms: 0,            // Busy-wait (no sleep)
    queue_config: QueueConfig {
        capacity: 10_000,
        high_priority_ratio: 0.2, // 20% for recalls
    },
    steal_threshold: 100,         // Aggressive work stealing
}
```

**Rationale**:
- **8 workers**: Maximize parallelism for lowest latency
- **Small batches**: Process observations immediately (10-50)
- **No sleep**: Busy-wait eliminates scheduling latency
- **Aggressive stealing**: Redistribute work quickly

**Expected Performance**:
- P50 latency: 2-3ms
- P99 latency: 5-8ms
- Throughput: 8K-12K obs/sec
- CPU utilization: 90-100% (busy-wait)

**Trade-offs**:
- Higher CPU usage (busy-wait)
- Lower throughput (small batches)
- Best for latency-critical applications

---

### High-Throughput Batch

**Use Cases**: Log ingestion, batch ETL, offline indexing

**Requirements**:
- Throughput: 100K+ obs/sec
- Latency: P99 < 500ms acceptable
- Batch processing preferred

**Configuration**:
```rust
WorkerPoolConfig {
    num_workers: 4,              // Fewer workers, larger batches
    min_batch_size: 100,         // Large minimum batch
    max_batch_size: 1000,        // Very large maximum batch
    idle_sleep_ms: 5,            // Sleep when idle (save CPU)
    queue_config: QueueConfig {
        capacity: 100_000,        // Large buffer for bursts
        high_priority_ratio: 0.05, // 5% for recalls (rare)
    },
    steal_threshold: 5000,        // Conservative stealing
}
```

**Rationale**:
- **4 workers**: Fewer workers with larger batches
- **Large batches**: Amortize overhead (100-1000 observations)
- **Large queue**: Absorb burst traffic (100K capacity)
- **Conservative stealing**: Avoid thrashing

**Expected Performance**:
- P50 latency: 50-100ms
- P99 latency: 200-400ms
- Throughput: 100K-150K obs/sec
- CPU utilization: 70-80% (efficient)

**Trade-offs**:
- Higher latency (large batches)
- Better throughput (amortized overhead)
- Best for offline/batch workloads

---

### Balanced Production

**Use Cases**: General-purpose deployments, mixed workloads

**Requirements**:
- Throughput: 50K obs/sec
- Latency: P99 < 100ms
- Moderate resource usage

**Configuration**:
```rust
WorkerPoolConfig::default()  // Use defaults

// Equivalent to:
WorkerPoolConfig {
    num_workers: 4,
    min_batch_size: 10,
    max_batch_size: 500,
    idle_sleep_ms: 1,
    queue_config: QueueConfig {
        capacity: 10_000,
        high_priority_ratio: 0.1,
    },
    steal_threshold: 1000,
}
```

**Rationale**:
- **Adaptive batching**: Small batches when lightly loaded, large when busy
- **Moderate workers**: Balance parallelism and overhead
- **Conservative resources**: Reasonable CPU/memory usage

**Expected Performance**:
- P50 latency: 10-20ms
- P99 latency: 50-80ms
- Throughput: 40K-60K obs/sec
- CPU utilization: 60-70% (balanced)

**Trade-offs**:
- Good balance of latency and throughput
- Works well for most use cases
- Recommended starting point

---

### Memory-Constrained

**Use Cases**: Edge devices, containerized deployments with memory limits

**Requirements**:
- Memory: < 512 MB
- Throughput: 1K-5K obs/sec
- Minimize allocations

**Configuration**:
```rust
WorkerPoolConfig {
    num_workers: 2,              // Fewer workers
    min_batch_size: 10,
    max_batch_size: 100,         // Smaller max batch
    idle_sleep_ms: 10,           // More aggressive sleep
    queue_config: QueueConfig {
        capacity: 1_000,          // Small queue
        high_priority_ratio: 0.1,
    },
    steal_threshold: 200,
}

// Also configure HNSW for memory efficiency
SpaceIsolatedHnswConfig {
    max_connections: 8,           // Reduce M (default 16)
    ef_construction: 100,         // Reduce ef (default 200)
    ..Default::default()
}
```

**Rationale**:
- **2 workers**: Minimal thread overhead
- **Small queue**: Limit memory footprint
- **Small HNSW**: Reduce graph memory
- **Aggressive sleep**: Save CPU when idle

**Expected Performance**:
- P50 latency: 20-50ms
- P99 latency: 100-200ms
- Throughput: 2K-4K obs/sec
- Memory: 256-512 MB RSS

**Trade-offs**:
- Lower throughput (fewer workers)
- Higher latency (smaller batches)
- Best for resource-constrained environments

---

### Multi-Tenant SaaS

**Use Cases**: Multi-tenant platforms, per-customer isolation

**Requirements**:
- Strong isolation between memory spaces
- Fair scheduling across tenants
- Per-tenant rate limiting

**Configuration**:
```rust
WorkerPoolConfig {
    num_workers: 8,              // High worker count
    min_batch_size: 10,
    max_batch_size: 100,         // Medium batches
    idle_sleep_ms: 1,
    queue_config: QueueConfig {
        capacity: 50_000,         // Large shared queue
        high_priority_ratio: 0.15,
    },
    steal_threshold: 500,         // Balanced stealing
}

// Per-tenant rate limiting (application layer)
struct TenantRateLimiter {
    max_rate_per_tenant: u32,    // e.g., 1000 obs/sec
    per_space_quotas: HashMap<MemorySpaceId, u32>,
}
```

**Rationale**:
- **8 workers**: Serve many spaces concurrently
- **Space sharding**: Natural isolation via consistent hashing
- **Medium batches**: Balance fairness and throughput
- **Rate limiting**: Prevent noisy neighbors

**Expected Performance**:
- P50 latency: 10-30ms
- P99 latency: 50-100ms
- Throughput: 80K+ obs/sec (aggregate)
- Per-tenant: 1K-5K obs/sec

**Trade-offs**:
- Requires application-level rate limiting
- More complex capacity planning
- Best for multi-tenant SaaS platforms

## Configuration Parameters

### Worker Count (`num_workers`)

**Range**: 1-16 (typically 4-8)

**Impact**:
- **Throughput**: Linear scaling up to core count
- **Latency**: Diminishing returns beyond 8 workers
- **Memory**: ~10 MB per worker (stack + state)
- **CPU**: More workers = higher CPU usage

**Tuning**:
```rust
// Match physical cores (not hyperthreads)
num_workers = num_cpus::get_physical()

// Low latency: Use all cores
num_workers = num_cpus::get()

// High throughput: Fewer workers, larger batches
num_workers = num_cpus::get_physical() / 2
```

**Guidelines**:
- Start with 4 workers (balanced)
- Increase for latency-critical workloads
- Decrease for memory-constrained environments
- Never exceed physical core count + 2

---

### Batch Size (`min_batch_size`, `max_batch_size`)

**Range**: min [1-100], max [10-1000]

**Impact**:
- **Latency**: Smaller batches = lower latency
- **Throughput**: Larger batches = higher throughput
- **CPU**: Larger batches amortize overhead

**Tuning**:
```rust
// Low latency
min_batch_size: 10
max_batch_size: 50

// High throughput
min_batch_size: 100
max_batch_size: 1000

// Balanced (default)
min_batch_size: 10
max_batch_size: 500
```

**Adaptive Batching Algorithm**:
```rust
fn compute_batch_size(queue_depth: usize, capacity: usize, config: &Config) -> usize {
    let ratio = queue_depth as f32 / capacity as f32;

    if ratio < 0.1 {
        config.min_batch_size  // Low load: small batches
    } else if ratio > 0.8 {
        config.max_batch_size  // High load: large batches
    } else {
        // Linear interpolation
        let range = config.max_batch_size - config.min_batch_size;
        config.min_batch_size + (ratio * range as f32) as usize
    }
}
```

**Guidelines**:
- Start with defaults (10-500)
- Profile actual latency distribution
- Adjust based on P99 target

---

### Queue Capacity (`capacity`)

**Range**: 1K-1M observations

**Impact**:
- **Memory**: ~3.5 KB per observation
- **Burst absorption**: Larger queue = better burst handling
- **Backpressure**: Smaller queue = earlier backpressure

**Tuning**:
```rust
// Formula: capacity = throughput × buffer_duration
capacity = target_throughput × buffer_seconds

// Examples:
// 100K obs/sec with 0.5s buffer
capacity: 50_000

// 10K obs/sec with 2s buffer
capacity: 20_000
```

**Guidelines**:
- Start with 10K (0.1s at 100K/sec)
- Increase for bursty workloads (2-5x)
- Monitor queue depth metric
- Alert when sustained > 80%

---

### Idle Sleep (`idle_sleep_ms`)

**Range**: 0-100 milliseconds

**Impact**:
- **Latency**: 0ms = busy-wait (lowest latency)
- **CPU**: 0ms = 100% CPU (even when idle)
- **Power**: Higher sleep = lower power consumption

**Tuning**:
```rust
// Low latency (busy-wait)
idle_sleep_ms: 0

// High throughput (save CPU)
idle_sleep_ms: 5

// Balanced (default)
idle_sleep_ms: 1

// Power-saving
idle_sleep_ms: 10
```

**Guidelines**:
- Use 0ms only for latency-critical workloads
- Use 5-10ms for batch workloads
- Default 1ms is good for most cases

---

### Work Stealing Threshold (`steal_threshold`)

**Range**: 100-10000 observations

**Impact**:
- **Load balancing**: Lower threshold = more stealing
- **Overhead**: More stealing = higher synchronization cost
- **Fairness**: Stealing improves work distribution

**Tuning**:
```rust
// Aggressive (low latency)
steal_threshold: 100

// Balanced (default)
steal_threshold: 1000

// Conservative (high throughput)
steal_threshold: 5000
```

**Guidelines**:
- Lower threshold for uneven workloads
- Higher threshold for uniform workloads
- Monitor `stolen_batches` metric
- Target: 5-10% of batches stolen

## Tuning Methodology

### Step 1: Establish Baseline

1. **Deploy default configuration**:
   ```rust
   WorkerPoolConfig::default()
   ```

2. **Run representative workload**:
   - Actual traffic patterns
   - Realistic data distributions
   - Expected throughput

3. **Collect baseline metrics**:
   ```bash
   # Observation rate
   rate(engram_streaming_observations_total[1m])

   # Latency distribution
   histogram_quantile(0.5, engram_streaming_observation_latency_seconds_bucket)
   histogram_quantile(0.99, engram_streaming_observation_latency_seconds_bucket)

   # Queue depth
   engram_streaming_queue_depth

   # Worker utilization
   engram_streaming_worker_utilization_ratio
   ```

4. **Identify bottleneck**:
   - High latency → Reduce batch size
   - Low throughput → Increase workers or batch size
   - High queue depth → Increase workers
   - Uneven worker utilization → Tune work stealing

### Step 2: Single-Parameter Tuning

Change **one parameter at a time**, measure impact:

**Example: Tuning batch size**
```rust
// Experiment 1: Reduce max_batch_size
WorkerPoolConfig {
    max_batch_size: 250,  // Baseline: 500
    ..Default::default()
}

// Measure: Did P99 latency improve?
// Measure: Did throughput decrease?

// Experiment 2: Reduce further
WorkerPoolConfig {
    max_batch_size: 100,  // Further reduction
    ..Default::default()
}

// Compare: P99 vs throughput trade-off
```

**Record results**:
| Config | P50 (ms) | P99 (ms) | Throughput (obs/sec) | CPU % |
|--------|----------|----------|----------------------|-------|
| Baseline (500) | 15 | 80 | 50K | 65% |
| Medium (250) | 12 | 60 | 45K | 70% |
| Small (100) | 8 | 40 | 35K | 75% |

### Step 3: Multi-Parameter Optimization

After identifying sensitive parameters, tune multiple together:

**Example: Latency optimization**
```rust
// Experiment: Reduce batch + increase workers
WorkerPoolConfig {
    num_workers: 8,         // Baseline: 4
    max_batch_size: 100,    // Baseline: 500
    idle_sleep_ms: 0,       // Baseline: 1
    ..Default::default()
}
```

**Measure combined impact**:
- Did P99 meet target?
- What's the CPU cost?
- Is throughput acceptable?

### Step 4: Stress Testing

Validate configuration under extreme conditions:

1. **Burst traffic**: 10x normal rate for 30 seconds
2. **Sustained high load**: 2x normal rate for 10 minutes
3. **Worker failures**: Kill 50% of workers mid-stream
4. **Memory pressure**: Limit RAM to 50% of normal

**Pass criteria**:
- No observations lost
- Graceful degradation
- Recovery within 60 seconds

### Step 5: Production Rollout

1. **Canary deployment**: 5% of traffic
2. **Monitor for 24 hours**
3. **Gradual rollout**: 25% → 50% → 100%
4. **Rollback plan**: Keep old configuration ready

## Trade-offs Analysis

### Latency vs Throughput

**Fundamental trade-off**: Small batches = low latency, large batches = high throughput

**Graph**:
```
Throughput
    ^
    |         ┌────────────── (large batches)
    |       ┌─┘
    |     ┌─┘
    |   ┌─┘
    | ┌─┘ (small batches)
    └─────────────────────> Latency
  High                    Low
```

**Decision matrix**:
| Use Case | Optimize For | Config |
|----------|-------------|---------|
| Interactive | Latency | Small batches, more workers |
| Batch ETL | Throughput | Large batches, fewer workers |
| Balanced | Both | Adaptive batching (default) |

---

### CPU vs Latency

**Fundamental trade-off**: Busy-wait = low latency but high CPU

**Graph**:
```
CPU Usage
    ^
    |     ┌─────────────────── (idle_sleep_ms = 0)
100%|     │
    |     │
 50%|     │        ┌────────── (idle_sleep_ms = 5)
    |     │      ┌─┘
  0%└─────┴──────┴───────────> Latency
       Low    Medium    High
```

**Decision matrix**:
| Scenario | idle_sleep_ms | CPU | Latency |
|----------|---------------|-----|---------|
| Low latency | 0 | 100% | 2-5ms |
| Balanced | 1 | 60-70% | 10-20ms |
| Efficient | 5-10 | 40-50% | 20-50ms |

---

### Memory vs Burst Capacity

**Fundamental trade-off**: Large queue = high memory but better burst handling

**Sizing**:
```
memory_mb = (capacity × 3.5 KB) / 1024

Examples:
- 10K capacity = 34 MB
- 50K capacity = 172 MB
- 100K capacity = 343 MB
```

**Decision matrix**:
| Workload | Capacity | Memory | Burst Duration |
|----------|----------|--------|----------------|
| Steady | 10K | 34 MB | 0.1s @ 100K/sec |
| Bursty | 50K | 172 MB | 0.5s @ 100K/sec |
| Very bursty | 100K | 343 MB | 1.0s @ 100K/sec |

## Benchmarking

### Microbenchmarks

Use Criterion for component-level benchmarks:

```bash
# Worker pool throughput
cargo bench --bench streaming_throughput

# Latency distribution
cargo bench --bench streaming_latency

# Work stealing overhead
cargo bench --bench worker_stealing
```

### Integration Benchmarks

End-to-end performance testing:

```rust
// File: benches/streaming_integration.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_e2e");

    for rate in [10_000, 50_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(rate), &rate, |b, &rate| {
            b.iter(|| {
                // Setup
                let worker_pool = WorkerPool::new(/* ... */);

                // Stream observations
                for i in 0..rate {
                    worker_pool.enqueue(/* ... */);
                }

                // Wait for processing
                worker_pool.wait_idle();
            });
        });
    }
}

criterion_group!(benches, bench_end_to_end);
criterion_main!(benches);
```

### Load Testing

Use external load generator:

```python
# File: scripts/load_test.py
import asyncio
import time
from engram_client import StreamingClient

async def load_test(rate: int, duration: int):
    client = StreamingClient("localhost:50051")
    await client.initialize()

    interval = 1.0 / rate
    start = time.time()

    count = 0
    while time.time() - start < duration:
        await client.send_observation(create_observation(count))
        count += 1
        await asyncio.sleep(interval)

    print(f"Sent {count} observations in {duration}s")
    print(f"Average rate: {count / duration:.0f} obs/sec")

asyncio.run(load_test(rate=100_000, duration=60))
```

## Case Studies

### Case Study 1: Conversational AI Platform

**Requirements**:
- 50K concurrent users
- P99 recall latency < 10ms
- 5K observations/sec per user (conversational turns)

**Initial Configuration** (default):
```rust
WorkerPoolConfig::default()  // 4 workers, batch 10-500
```

**Results**:
- P99 latency: 85ms (missed target)
- Throughput: 40K obs/sec (adequate)
- CPU: 60%

**Tuning**:
```rust
WorkerPoolConfig {
    num_workers: 8,         // Double workers
    max_batch_size: 50,     // Reduce max batch
    idle_sleep_ms: 0,       // Busy-wait
    ..Default::default()
}
```

**Final Results**:
- P99 latency: 8ms (met target)
- Throughput: 35K obs/sec (adequate for 50K users @ 5K/sec = aggregate rate)
- CPU: 90% (acceptable trade-off)

**Lessons**:
- Conversational AI needs low latency for responsiveness
- CPU cost acceptable for user experience
- Monitoring revealed batch size as bottleneck

---

### Case Study 2: Log Aggregation Pipeline

**Requirements**:
- 200K logs/sec ingestion
- Latency not critical (batch processing)
- Memory constrained (512 MB limit)

**Initial Configuration** (default):
```rust
WorkerPoolConfig::default()
```

**Results**:
- Throughput: 50K obs/sec (missed target)
- Memory: 300 MB (ok)
- CPU: 70%

**Tuning**:
```rust
WorkerPoolConfig {
    num_workers: 4,
    min_batch_size: 200,     // Large batches
    max_batch_size: 1000,
    idle_sleep_ms: 5,
    queue_config: QueueConfig {
        capacity: 50_000,     // Larger queue
        ..Default::default()
    },
    ..Default::default()
}
```

**Final Results**:
- Throughput: 180K obs/sec (met target)
- Memory: 480 MB (under limit)
- CPU: 75%

**Lessons**:
- Batch workloads benefit from large batches
- Queue capacity critical for burst absorption
- Memory trade-off acceptable

---

**See Also**:
- [Streaming Operations Guide](streaming.md)
- [Benchmarking Suite](../../benches/streaming_throughput.rs)
- [Performance FAQ](../faq/performance.md)
