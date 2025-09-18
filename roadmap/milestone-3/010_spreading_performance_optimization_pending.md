# Task 010: Spreading Performance Optimization

## Objective
Optimize spreading performance to meet latency targets with cache-aware algorithms and memory pool management.

## Priority
P1 (Performance Critical)

## Effort Estimate
1.5 days

## Dependencies
- Task 008: Integrated Recall Implementation

## Technical Approach

### Implementation Details
- Implement lock-free memory pool for activation records
- Add cache-aware node layout minimizing cache misses during traversal
- Create adaptive batching based on CPU topology and memory bandwidth
- Implement spreading latency prediction for timeout management

### Files to Create/Modify
- `engram-core/src/activation/performance.rs` - New file for performance optimizations
- `engram-core/src/activation/memory_pool.rs` - Memory pool implementation
- `engram-core/src/activation/cache_optimization.rs` - Cache-aware algorithms

### Integration Points
- Optimizes integrated recall from Task 008
- Uses SIMD operations from Task 007
- Connects to tier-aware scheduling from Task 003
- Integrates with existing memory management

## Implementation Details

### Lock-Free Memory Pool
```rust
pub struct ActivationRecordPool {
    free_list: lockfree::stack::Stack<*mut ActivationRecord>,
    chunk_allocator: ChunkAllocator,
    pool_size: AtomicUsize,
    high_water_mark: usize,
}

impl ActivationRecordPool {
    pub fn acquire(&self) -> Option<Box<ActivationRecord>> {
        // Fast path: pop from free list
        if let Some(ptr) = self.free_list.pop() {
            return Some(unsafe { Box::from_raw(ptr) });
        }

        // Slow path: allocate new record
        self.allocate_new()
    }

    pub fn release(&self, record: Box<ActivationRecord>) {
        record.reset();
        let ptr = Box::into_raw(record);
        self.free_list.push(ptr);
    }
}
```

### Cache-Aware Node Layout
```rust
// Optimize for cache line size (64 bytes)
#[repr(C, align(64))]
pub struct CacheOptimizedNode {
    // Hot data: frequently accessed during spreading
    memory_id: MemoryId,           // 8 bytes
    activation: AtomicF32,         // 4 bytes
    confidence: f32,               // 4 bytes
    hop_count: u16,               // 2 bytes
    storage_tier: StorageTier,     // 1 byte
    flags: u8,                    // 1 byte

    // Warm data: accessed during result collection
    embedding_ptr: *const [f32; 768],  // 8 bytes
    adjacency_list: *const [MemoryId], // 8 bytes

    // Cold data: accessed less frequently
    metadata: NodeMetadata,        // 32 bytes
    _padding: [u8; 0],            // Align to 64 bytes
}

pub struct NodeMetadata {
    last_access: f64,
    creation_time: f64,
    access_count: u32,
    tier_migration_history: u32,
}
```

### Adaptive Batching Strategy
```rust
pub struct AdaptiveBatcher {
    cpu_topology: CPUTopology,
    memory_bandwidth: f64,        // GB/s
    cache_sizes: CacheSizes,
    optimal_batch_size: AtomicUsize,
}

impl AdaptiveBatcher {
    pub fn compute_optimal_batch_size(&self, workload_characteristics: &WorkloadStats) -> usize {
        // Consider L1/L2/L3 cache sizes
        let cache_friendly_size = self.cache_sizes.l2_size / size_of::<CacheOptimizedNode>();

        // Consider memory bandwidth
        let bandwidth_optimal = (self.memory_bandwidth * 1_000_000.0) /
                               (size_of::<[f32; 768]>() as f64 * 2.0); // Read + write

        // Consider CPU parallelism
        let cpu_optimal = self.cpu_topology.logical_cores * 8; // 8 vectors per core

        // Take geometric mean for balanced optimization
        let optimal = ((cache_friendly_size * bandwidth_optimal * cpu_optimal) as f64).powf(1.0/3.0);

        optimal.round() as usize
    }
}
```

### Latency Prediction Model
```rust
pub struct LatencyPredictor {
    historical_latencies: RingBuffer<LatencySample>,
    regression_model: SimpleLinearRegression,
}

struct LatencySample {
    batch_size: usize,
    hop_count: u16,
    tier_distribution: [f32; 3], // Hot, warm, cold percentages
    actual_latency: Duration,
}

impl LatencyPredictor {
    pub fn predict_latency(&self, spreading_parameters: &SpreadingParams) -> Duration {
        // Simple model: latency = base + (batch_size * vector_cost) + (hop_count * traversal_cost)
        let base_latency = Duration::from_micros(100);
        let vector_cost = Duration::from_nanos(50);   // Per vector similarity
        let traversal_cost = Duration::from_micros(200); // Per hop

        base_latency
            + vector_cost * spreading_parameters.batch_size as u32
            + traversal_cost * spreading_parameters.max_hops as u32
    }

    pub fn update_model(&mut self, actual: LatencySample) {
        self.historical_latencies.push(actual);
        if self.historical_latencies.len() >= 100 {
            self.retrain_model();
        }
    }
}
```

## Acceptance Criteria
- [ ] Memory pool reduces allocation overhead by >50%
- [ ] Cache-aware layout improves traversal performance by >20%
- [ ] Adaptive batching automatically tunes for CPU topology
- [ ] Latency prediction accurate within 20% for typical workloads
- [ ] Overall spreading performance meets <10ms P95 latency target
- [ ] Memory usage remains bounded under high load
- [ ] Performance improvements validated across different CPU architectures

## Testing Approach
- Performance benchmarks comparing optimized vs baseline implementations
- Cache performance analysis using hardware performance counters
- Stress tests with high allocation rates and memory pressure
- Latency prediction accuracy tests across various workload patterns
- Cross-platform testing on different CPU architectures

## Risk Mitigation
- **Risk**: Memory pool fragmentation leading to memory leaks
- **Mitigation**: Implement pool compaction, monitor pool health
- **Testing**: Long-running stress tests with allocation pattern analysis

- **Risk**: Cache optimizations beneficial only on specific architectures
- **Mitigation**: Runtime detection of cache sizes, adaptive layouts
- **Validation**: Test on ARM, Intel, AMD architectures

- **Risk**: Adaptive batching oscillates or converges slowly
- **Mitigation**: Implement damping factors, minimum convergence windows
- **Testing**: Simulate various workload patterns and measure convergence

## Implementation Strategy

### Phase 1: Memory Pool Implementation
- Implement lock-free activation record pool
- Add pool metrics and monitoring
- Basic performance validation

### Phase 2: Cache Optimization
- Design cache-aware node layout
- Implement prefetching strategies
- Cache performance measurement

### Phase 3: Adaptive Systems
- Implement adaptive batching
- Add latency prediction model
- End-to-end performance optimization

## Performance Targets
- **Allocation Overhead**: <10ns per activation record from pool
- **Cache Miss Rate**: <5% L2 cache misses during node traversal
- **Batch Efficiency**: >90% of theoretical peak SIMD performance
- **Latency Prediction**: ±20% accuracy for 95% of predictions
- **Memory Overhead**: <10% additional memory for optimization structures

## Monitoring and Metrics
```rust
pub struct SpreadingPerformanceMetrics {
    // Memory pool metrics
    pool_hit_rate: Histogram,
    pool_high_water_mark: Gauge,

    // Cache performance
    cache_miss_rate: Histogram,
    prefetch_effectiveness: Counter,

    // Latency tracking
    spreading_latency_p95: Histogram,
    latency_prediction_error: Histogram,

    // Batch optimization
    optimal_batch_size: Gauge,
    batch_utilization: Histogram,
}
```

## Notes
This task transforms the cognitive database from functional to performant. Unlike traditional databases that optimize for ACID compliance, cognitive systems must optimize for low-latency exploration of large memory networks. The optimizations here determine whether Engram can achieve real-time cognitive performance or remains a research prototype.