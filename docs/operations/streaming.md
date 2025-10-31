# Streaming Operations Guide

Comprehensive guide for operating Engram's high-performance streaming infrastructure in production.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Configuration Guide](#configuration-guide)
- [Monitoring Guide](#monitoring-guide)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [Capacity Planning](#capacity-planning)
- [Best Practices](#best-practices)

## Architecture Overview

The Engram streaming infrastructure provides high-throughput observation ingestion with eventual consistency guarantees and bounded staleness (P99 < 100ms).

### Components

```
Client (gRPC/WebSocket)
        ↓
SessionManager (session lifecycle)
        ↓
ObservationQueue (lock-free MPMC, 4M+ ops/sec)
        ↓
WorkerPool (4-8 workers, space-partitioned)
        ↓
SpaceIsolatedHnsw (per-space HNSW indices)
        ↓
Persistent Storage (memory-mapped)
```

#### SessionManager

- **Purpose**: Tracks client sessions, validates monotonic sequence numbers
- **State**: DashMap-based, lock-free concurrent access
- **Capacity**: 10K+ concurrent sessions per instance
- **Cleanup**: Idle sessions evicted after 5 minutes

#### ObservationQueue

- **Implementation**: Lock-free SegQueue (crossbeam)
- **Throughput**: 4M+ enqueue/dequeue operations per second
- **Capacity**: Configurable (default: 10K observations = 0.1s buffer at 100K/sec)
- **Backpressure**: Triggered at 80% capacity

#### WorkerPool

- **Workers**: 4-8 threads (typically 1 per core)
- **Sharding**: Space-based consistent hashing
- **Work stealing**: Automatic load balancing when queue depth > threshold
- **Batch processing**: Adaptive batch size (10-1000) based on queue depth

#### SpaceIsolatedHnsw

- **Isolation**: Per-space HNSW indices (zero cross-space contention)
- **Concurrency**: Space-partitioned sharding ensures workers don't compete
- **Performance**: 1,600 insertions/sec per worker (single-threaded HNSW)
- **Scaling**: Linear with number of active spaces × workers

#### BackpressureMonitor

- **Threshold**: 80% queue capacity
- **Strategy**: Adaptive rejection (reject new observations when overloaded)
- **Recovery**: Automatic when queue depth < 50%
- **Metrics**: Tracks activation count, rejection rate

## Configuration Guide

### Worker Pool Configuration

```rust
WorkerPoolConfig {
    num_workers: 4,              // 1-8 workers, tune based on CPU cores
    queue_config: QueueConfig {
        capacity: 10_000,        // Queue capacity (observations)
        high_priority_ratio: 0.1, // 10% reserved for high-priority
    },
    steal_threshold: 1000,       // Queue depth for work stealing
    min_batch_size: 10,          // Minimum batch (low latency)
    max_batch_size: 500,         // Maximum batch (high throughput)
    idle_sleep_ms: 1,            // Worker sleep when idle
}
```

#### Tuning Recommendations

**Low Latency Profile** (P99 < 10ms)
```rust
WorkerPoolConfig {
    num_workers: 8,
    min_batch_size: 10,
    max_batch_size: 50,
    idle_sleep_ms: 0,  // Busy-wait for lowest latency
    ..Default::default()
}
```

**High Throughput Profile** (>100K obs/sec)
```rust
WorkerPoolConfig {
    num_workers: 4,
    min_batch_size: 100,
    max_batch_size: 1000,
    idle_sleep_ms: 5,
    queue_config: QueueConfig {
        capacity: 50_000,  // Larger buffer
        ..Default::default()
    },
    ..Default::default()
}
```

**Balanced Profile** (recommended default)
```rust
WorkerPoolConfig::default()  // 4 workers, batch_size 10-500
```

### Queue Capacity

Default: 10,000 observations

**Sizing formula**:
```
capacity = target_throughput × buffer_duration_seconds
```

Examples:
- 100K obs/sec with 0.5s buffer = 50,000 capacity
- 10K obs/sec with 1s buffer = 10,000 capacity

**Recommendations**:
- Increase for bursty workloads (2-5x average rate)
- Monitor queue depth metric (should stay < 80%)
- Alert when sustained > 80% capacity

### Session Management

```rust
SessionManager {
    idle_timeout: Duration::from_secs(300),  // 5 minutes
    cleanup_interval: Duration::from_secs(60), // Check every 60s
    max_sessions: 10_000,  // Per instance
}
```

## Monitoring Guide

### Key Metrics

#### Observation Rate
```
engram_streaming_observations_total
```
- **Type**: Counter
- **Labels**: `memory_space`, `priority`
- **Expected**: Should match client input rate
- **Alert**: Rate drops below expected (potential processing bottleneck)

#### Queue Depth
```
engram_streaming_queue_depth
```
- **Type**: Gauge
- **Expected**: < 80% capacity under normal load
- **Alert**: Sustained > 80% (risk of backpressure)

#### Worker Utilization
```
engram_streaming_worker_utilization_ratio
```
- **Type**: Gauge
- **Labels**: `worker_id`
- **Expected**: 70-90% (balanced load)
- **Alert**: < 50% (underutilized) or > 95% (overloaded)

#### Observation Latency
```
engram_streaming_observation_latency_seconds
```
- **Type**: Histogram
- **Buckets**: [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
- **Expected P99**: < 100ms
- **Alert**: P99 > 200ms

#### Backpressure Activations
```
engram_streaming_backpressure_activations_total
```
- **Type**: Counter
- **Expected**: 0 under normal load
- **Alert**: Rate > 0 (capacity exceeded)

#### Recall Latency
```
engram_streaming_recall_latency_seconds
```
- **Type**: Histogram
- **Expected P99**: < 20ms
- **Alert**: P99 > 50ms

### Grafana Dashboard

See example dashboard: `deployments/grafana/dashboards/streaming_dashboard.json`

**Key Panels**:
1. **Observation Rate** (time series): Track input vs processed rate
2. **Queue Depth** (gauge + time series): Monitor capacity utilization
3. **Worker Utilization** (heatmap): Identify load imbalance
4. **Latency Distribution** (histogram): P50/P99/P99.9 tracking
5. **Backpressure Events** (counter): Alert on capacity issues
6. **Rejection Rate** (time series): Track dropped observations
7. **Active Sessions** (gauge): Monitor session count
8. **Memory Usage** (gauge): Track RSS and heap

### Prometheus Alerts

See example alerts: `deployments/prometheus/alerts/streaming.yml`

**Critical Alerts**:
- `HighQueueDepth`: Queue > 80% for 2 minutes
- `WorkerCrashed`: Worker crash rate > 0
- `HighLatency`: P99 > 200ms for 5 minutes
- `BackpressureActive`: Backpressure activations > 10/min
- `SessionLeakSuspected`: Session count continuously growing

## Troubleshooting

### High Queue Depth

**Symptoms**:
```
engram_streaming_queue_depth > 8000 (80% of default 10K)
```

**Causes**:
1. Worker capacity exceeded (throughput > 1,600 × num_workers)
2. Slow HNSW insertions (large graphs, low memory)
3. Disk I/O bottleneck (memory-mapped pages swapped out)
4. Work stealing not activating (threshold too high)

**Resolution**:

1. Check worker utilization:
   ```bash
   curl http://localhost:9090/metrics | grep engram_streaming_worker_utilization
   ```
   - If < 70%: Reduce idle_sleep_ms or increase batch size
   - If > 95%: Increase worker count

2. Check HNSW performance:
   ```bash
   # Look for slow insertion latency
   curl http://localhost:9090/metrics | grep engram_hnsw_insert_duration
   ```
   - If P99 > 10ms: Check memory pressure, consider index tuning

3. Check disk I/O:
   ```bash
   iostat -x 1 10
   ```
   - If %util > 80%: Memory-mapped pages swapping, increase RAM

4. Adjust work stealing:
   ```rust
   WorkerPoolConfig {
       steal_threshold: 500,  // Lower threshold (default 1000)
       ..Default::default()
   }
   ```

### High Latency

**Symptoms**:
```
histogram_quantile(0.99, engram_streaming_observation_latency_seconds_bucket) > 0.1
```

**Causes**:
1. Large batch sizes (high throughput mode)
2. GC pauses (heap pressure)
3. Resource contention (CPU/memory)
4. Network issues (gRPC latency)

**Resolution**:

1. Profile with flamegraph:
   ```bash
   cargo flamegraph --bin engram-cli -- serve
   ```

2. Reduce batch size for lower latency:
   ```rust
   WorkerPoolConfig {
       max_batch_size: 50,  // Lower bound (default 500)
       ..Default::default()
   }
   ```

3. Check memory allocator:
   ```bash
   # Engram uses mimalloc by default
   MIMALLOC_SHOW_STATS=1 ./engram-cli serve
   ```

4. Monitor GC (Rust doesn't have GC, but check allocator metrics)

### Backpressure Activation

**Symptoms**:
```
rate(engram_streaming_backpressure_activations_total[1m]) > 0
```

**Causes**:
1. Client rate exceeds system capacity
2. Queue capacity too small for burst traffic
3. Worker pool undersized for sustained load

**Resolution**:

1. Increase queue capacity:
   ```rust
   QueueConfig {
       capacity: 50_000,  // Larger buffer (default 10K)
       ..Default::default()
   }
   ```

2. Scale worker pool:
   ```rust
   WorkerPoolConfig {
       num_workers: 8,  // More workers (default 4)
       ..Default::default()
   }
   ```

3. Implement client-side rate limiting:
   ```rust
   // Respect 429 Retry-After header
   if response.status == 429 {
       let retry_after = response.headers["Retry-After"];
       tokio::time::sleep(Duration::from_secs(retry_after)).await;
   }
   ```

### Worker Crash/Panic

**Symptoms**:
```
Worker thread panicked: ...
```

**Causes**:
1. Corrupted observation data
2. HNSW index corruption
3. Out-of-memory (OOM)
4. Invariant violation (bug)

**Resolution**:

1. Check logs for panic message:
   ```bash
   grep "panicked" /var/log/engram/engram-cli.log
   ```

2. Verify observation format:
   ```rust
   // Validate embeddings are normalized
   assert!(embedding.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 0.01);
   ```

3. Check memory limits:
   ```bash
   # Increase memory limit
   ulimit -v unlimited
   ```

4. Enable debug logging:
   ```bash
   RUST_LOG=engram_core::streaming::worker_pool=debug ./engram-cli serve
   ```

5. Report bug with reproducible case

## Performance Tuning

### Throughput Optimization

**Target**: 100K+ observations/sec

**Strategy**:
1. Maximize batch size (500-1000)
2. Minimize worker idle time (idle_sleep_ms = 0)
3. Increase worker count to match cores
4. Increase queue capacity for burst absorption

**Configuration**:
```rust
WorkerPoolConfig {
    num_workers: 8,
    min_batch_size: 100,
    max_batch_size: 1000,
    idle_sleep_ms: 0,
    queue_config: QueueConfig {
        capacity: 50_000,
        ..Default::default()
    },
    ..Default::default()
}
```

### Latency Optimization

**Target**: P99 < 10ms

**Strategy**:
1. Minimize batch size (10-50)
2. Increase worker count for parallelism
3. Use busy-wait instead of sleep (idle_sleep_ms = 0)
4. Pin workers to CPU cores (NUMA awareness)

**Configuration**:
```rust
WorkerPoolConfig {
    num_workers: 8,
    min_batch_size: 10,
    max_batch_size: 50,
    idle_sleep_ms: 0,
    ..Default::default()
}
```

### Memory Optimization

**Target**: < 2GB RSS for 1M observations

**Strategy**:
1. Use memory-mapped HNSW indices
2. Enable compression for cold storage
3. Tune queue capacity to minimum viable
4. Aggressive session cleanup

**Configuration**:
```rust
// Reduce memory footprint
QueueConfig {
    capacity: 5_000,  // Smaller buffer
    ..Default::default()
}

SessionManager {
    idle_timeout: Duration::from_secs(60),  // Aggressive cleanup
    ..Default::default()
}
```

## Capacity Planning

### Per-Worker Capacity

**HNSW Insertion Rate**: ~1,600 observations/sec per worker (single-threaded)

**Formula**:
```
max_throughput = num_workers × 1,600 obs/sec × num_active_spaces
```

**Examples**:
- 4 workers × 20 active spaces = 128,000 obs/sec
- 8 workers × 10 active spaces = 128,000 obs/sec
- 4 workers × 1 active space = 6,400 obs/sec

**Note**: Scaling is linear with number of active spaces due to space-partitioned sharding.

### Memory Requirements

**Per-Observation**:
- Episode metadata: ~256 bytes
- Embedding (768 × f32): 3,072 bytes
- HNSW node: ~128 bytes
- Total: ~3.5 KB per observation

**Formula**:
```
memory_gb = (observations × 3.5 KB) / (1024^3)
```

**Examples**:
- 1M observations = 3.5 GB
- 10M observations = 35 GB
- 100M observations = 350 GB

**Recommendations**:
- Allocate 2x memory for headroom
- Use memory-mapped storage for large deployments
- Monitor RSS and swap usage

### Scaling Guidelines

**Vertical Scaling** (single instance):
- Up to 8 cores: Linear throughput scaling
- Up to 64 GB RAM: 15M+ observations in memory
- NVMe SSD: Essential for memory-mapped indices

**Horizontal Scaling** (multiple instances):
- Shard by memory_space_id (consistent hashing)
- Load balancer routes by space_id
- No cross-instance communication needed

**Example 3-Instance Deployment**:
- Instance 1: Spaces A-J (40K obs/sec)
- Instance 2: Spaces K-T (40K obs/sec)
- Instance 3: Spaces U-Z (40K obs/sec)
- Total: 120K obs/sec

## Best Practices

### Production Deployment

1. **Use dedicated hardware**
   - Engram is CPU and memory intensive
   - Avoid co-locating with other workloads

2. **Enable metrics and monitoring**
   - Export Prometheus metrics
   - Set up Grafana dashboards
   - Configure alerts for critical thresholds

3. **Tune for workload**
   - Profile actual traffic patterns
   - Adjust worker pool configuration
   - Benchmark before production rollout

4. **Implement graceful degradation**
   - Respect backpressure signals
   - Implement client-side retry with exponential backoff
   - Use priority queues for critical observations

5. **Regular maintenance**
   - Monitor memory growth
   - Clean up idle sessions
   - Compact HNSW indices periodically

### Development and Testing

1. **Use realistic test data**
   - Actual embedding distributions
   - Representative memory space cardinality
   - Burst traffic patterns

2. **Load testing**
   - Gradually ramp up rate
   - Measure P50/P99/P99.9 latency
   - Identify bottlenecks with profiling

3. **Chaos testing**
   - Worker crashes
   - Network partitions
   - Backpressure scenarios

4. **Integration testing**
   - End-to-end workflows
   - Multi-client concurrent access
   - Space isolation validation

---

**See Also**:
- [Performance Tuning Guide](streaming-tuning.md)
- [Monitoring Setup](streaming-monitoring.md)
- [API Reference](../reference/streaming-api.md)
