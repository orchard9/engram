# Tier-Aware Spreading Scheduler: Four Architectural Perspectives

## Cognitive-Architecture Perspective

### Biological Memory Prioritization

Human memory operates with clear hierarchical priorities that optimize for both immediate responsiveness and long-term storage efficiency. When you hear your name called, your brain doesn't search through decades of stored experiences linearly - it prioritizes active working memory, then recent episodic memory, before diving into long-term semantic storage.

**Working Memory (Hot Tier)**
- Capacity: ~7 items in active maintenance
- Access time: 50-100ms for conscious retrieval
- Priority: Immediate, preempts all other processing
- Biological basis: Prefrontal cortex sustained activity

**Episodic Buffer (Warm Tier)**
- Capacity: Recent experiences and context
- Access time: 200ms-1s for contextual recall
- Priority: High, but yields to working memory
- Biological basis: Hippocampal binding and replay

**Semantic Memory (Cold Tier)**
- Capacity: Lifetime of consolidated knowledge
- Access time: 1-10s for complex retrieval
- Priority: Background, can be interrupted
- Biological basis: Neocortical distributed networks

### Cognitive Scheduling Principles

The tier-aware scheduler mirrors three key cognitive mechanisms:

1. **Attention Capture**: Hot tier activations can interrupt ongoing processing, just as urgent stimuli capture attention
2. **Contextual Priming**: Warm tier maintains activation of recently accessed memories for faster retrieval
3. **Consolidation Processing**: Cold tier runs background consolidation, strengthening important memories during idle periods

### Implications for Activation Spreading

Cognitive research suggests that memory activation follows predictable priority patterns:
- Recent memories receive activation boosts (recency effect)
- Frequently accessed memories maintain higher baseline activation
- Emotionally significant memories get priority processing
- Interference occurs when multiple activation sources compete

The scheduler must respect these patterns to produce cognitively plausible recall behaviors.

## Memory-Systems Perspective

### Hierarchical Memory Consolidation

Memory systems research reveals that consolidation occurs at multiple timescales, with different neural circuits handling different consolidation phases. This directly informs our tier architecture.

**Fast Consolidation (Hot → Warm)**
- Timescale: Minutes to hours
- Mechanism: Hippocampal binding of cortical patterns
- Trigger: Working memory capacity limits
- Implementation: LRU-based migration with frequency weighting

**Slow Consolidation (Warm → Cold)**
- Timescale: Days to years
- Mechanism: Neocortical reorganization and strengthening
- Trigger: Repeated reactivation and importance
- Implementation: Confidence-based promotion with decay modeling

**Reconsolidation (Cold → Warm/Hot)**
- Timescale: Immediate upon reactivation
- Mechanism: Memory becomes labile when recalled
- Trigger: Activation above threshold during recall
- Implementation: Dynamic tier promotion during spreading

### Memory Replay Patterns

Neuroscience research on memory replay provides scheduling insights:

**Sharp-Wave Ripples (Online Consolidation)**
- High-frequency bursts during awake rest periods
- Rapid replay of recent experiences
- Strengthens important memory sequences
- Maps to: Burst processing of warm tier activations

**Slow-Wave Sleep Replay (Offline Consolidation)**
- Slower, systematic replay during deep sleep
- Transfers memories from hippocampus to cortex
- Optimizes memory organization
- Maps to: Background cold tier processing

**Theta Sequences (Active Recall)**
- Forward and reverse sequence replay during navigation
- Supports planning and spatial memory
- Maps to: Hot tier priority processing during active queries

### Interference and Competition

Memory systems must handle interference between competing activations:

- **Proactive Interference**: Old memories interfere with new ones
- **Retroactive Interference**: New memories interfere with old ones
- **Output Interference**: Retrieval of one memory blocks others

The scheduler addresses interference through:
- Priority-based processing (hot tier preempts others)
- Separate queues preventing cross-tier blocking
- Time-budget allocation limiting interference duration

## Rust-Graph-Engine Perspective

### Lock-Free Concurrent Scheduling

High-performance graph traversal requires careful coordination between threads accessing different memory tiers. Traditional locking approaches create bottlenecks that violate our latency requirements.

**Queue Architecture**
```rust
// Per-tier lock-free queues using crossbeam
struct TierQueues {
    hot: crossbeam_queue::SegQueue<ActivationTask>,
    warm: crossbeam_queue::SegQueue<ActivationTask>,
    cold: crossbeam_queue::SegQueue<ActivationTask>,
}

// Work-stealing within tiers
struct TierWorkerPool {
    hot_workers: Vec<crossbeam_deque::Worker<ActivationTask>>,
    warm_workers: Vec<crossbeam_deque::Worker<ActivationTask>>,
    cold_workers: Vec<crossbeam_deque::Worker<ActivationTask>>,
}
```

**Memory Ordering Guarantees**
- `Acquire-Release` ordering for queue operations
- `SeqCst` ordering for priority preemption flags
- Memory barriers ensuring consistent view across threads

**ABA Problem Prevention**
- Epoch-based memory reclamation for queue nodes
- Hazard pointers protecting in-flight activations
- Generation counters preventing use-after-free

### Cache-Optimal Graph Traversal

Graph processing exhibits poor spatial locality, making cache optimization critical:

**Hot Tier Optimization**
- Keep recently activated nodes in L1/L2 cache
- Use SIMD instructions for batch activation processing
- Minimize cache line bouncing between worker threads

**Warm Tier Strategy**
- Leverage memory-mapped file locality
- Sequential prefetching for edge traversal
- NUMA-aware thread placement

**Cold Tier Approach**
- Columnar storage for cache-friendly embedding operations
- Vectorized distance computations using AVX-512
- Asynchronous I/O to hide storage latency

### Scalability Considerations

The scheduler must scale efficiently across multiple CPU cores:

**Thread Pool Management**
```rust
// Separate thread pools per tier prevent priority inversion
async fn schedule_activation_spreading() {
    let hot_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get() / 4)
        .thread_name("hot-tier")
        .build()?;

    let warm_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get() / 2)
        .thread_name("warm-tier")
        .build()?;

    let cold_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .thread_name("cold-tier")
        .build()?;
}
```

**Load Balancing Strategy**
- Work-stealing within tiers maintains utilization
- No cross-tier stealing prevents priority violations
- Dynamic thread allocation based on queue depths

## Systems-Architecture Perspective

### Multi-Tier Storage Scheduling

Modern storage systems require sophisticated scheduling to handle devices with vastly different performance characteristics. Our three-tier system amplifies these challenges.

**Storage Device Characteristics**
- **Hot Tier (DRAM)**: 50ns access, 100GB/s bandwidth, volatile
- **Warm Tier (NVMe SSD)**: 100μs access, 3GB/s bandwidth, persistent
- **Cold Tier (Network Storage)**: 10ms access, 100MB/s bandwidth, archival

**Queue Depth Management**
Each tier requires different queue depth optimization:
- Hot tier: Shallow queues (1-4) to minimize latency
- Warm tier: Medium queues (16-32) for throughput
- Cold tier: Deep queues (64-128) to hide network latency

### Latency Optimization Strategies

**Priority Preemption Implementation**
```rust
struct PriorityScheduler {
    hot_flag: AtomicBool,      // Signals hot tier needs immediate attention
    preemption_count: AtomicU64, // Tracks preemption frequency
    tier_budgets: [AtomicU64; 3], // Remaining time budget per tier
}

impl PriorityScheduler {
    fn should_preempt(&self, current_tier: TierType) -> bool {
        match current_tier {
            TierType::Hot => false, // Hot tier never preempted
            TierType::Warm | TierType::Cold => {
                self.hot_flag.load(Ordering::Acquire) ||
                self.tier_budgets[current_tier as usize].load(Ordering::Relaxed) == 0
            }
        }
    }
}
```

**Time-Budget Allocation**
- Hot tier: 100μs budget, strict deadline scheduling
- Warm tier: 1ms budget, fair queuing with hot preemption
- Cold tier: 10ms budget, best-effort with cancellation

**Bypass Mechanisms**
When time budgets are exceeded:
- Hot tier: Never bypassed, system invariant
- Warm tier: Graceful degradation to hot-only results
- Cold tier: Immediate bypass, schedule for background processing

### Performance Monitoring and Adaptation

**Real-Time Metrics Collection**
```rust
struct TierMetrics {
    processing_latency: Histogram,    // P50, P95, P99 latencies
    queue_depth: Gauge,              // Current queue occupancy
    preemption_rate: Counter,        // Hot tier preemptions per second
    bypass_rate: Counter,            // Cold tier bypasses per second
    cache_hit_rate: Gauge,           // Memory locality effectiveness
}
```

**Adaptive Scheduling Parameters**
- Monitor P95 latency and adjust time budgets dynamically
- Track cache miss rates and adjust prefetching strategies
- Observe preemption patterns and rebalance thread allocation
- Use feedback control to maintain latency SLAs

**Failure Mode Detection**
- Hot tier latency spikes indicate memory pressure
- Warm tier timeouts suggest storage device issues
- Cold tier failures require graceful degradation
- Cross-tier correlation analysis identifies systemic problems

### Integration with Storage Tiers

The scheduler interfaces with existing storage infrastructure:

**Hot Tier Integration (DashMap)**
- Direct memory access with atomic operations
- Lock-free concurrent hash table access
- Memory barriers for consistency across cores

**Warm Tier Integration (Memory-Mapped Files)**
- Page fault handling and prefetching
- Operating system buffer cache coordination
- NUMA-aware memory allocation

**Cold Tier Integration (Columnar Storage)**
- Asynchronous I/O with completion queues
- Batch processing for vectorized operations
- Connection pooling for network storage

Each perspective reveals different optimization opportunities and constraints, but all converge on the same architectural principles: priority-based scheduling, lock-free concurrency, time-budget enforcement, and graceful degradation under load.