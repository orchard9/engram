# Spreading Performance Optimization Research

## Research Topics for Milestone 3 Task 010: Spreading Performance Optimization

### 1. Memory Pooling and Allocation Strategies
- Lock-free free lists (Treiber stack, Michael-Scott queue)
- Slab allocators for fixed-size activation records
- Cache-friendly object pools with ABA mitigation
- NUMA-aware pool partitioning
- Metrics for memory pool health

### 2. Cache-Aware Data Layouts
- Structure-of-Arrays vs Array-of-Structures trade-offs
- Cache line alignment and false sharing avoidance
- Software prefetching (`_mm_prefetch`, `prefetcht0`)
- Hot/warm/cold data segmentation
- Impact of adjacency ordering on traversal locality

### 3. Adaptive Batching and Auto-Tuning
- Roofline modeling to balance compute and bandwidth
- Feedback controllers for batch size selection
- CPU topology detection (hwloc, `libc::sysctl`)
- Preventing oscillations in adaptive systems
- Benchmarking to calibrate heuristics

### 4. Latency Prediction and Budgeting
- Online regression for systems performance
- Time series smoothing (EWMA) for latency trends
- Deadline-aware scheduling and admission control
- Confidence intervals for latency predictions
- Telemetry-driven feedback loops

### 5. Observability and Performance Monitoring
- Hardware counter sampling with `perf_event_open`
- Histograms vs summary metrics for latency
- Alerting on performance regressions
- Linking activation metrics to recall SLAs
- Visualization patterns for cache and pool metrics

## Research Findings

### Memory Pool Design
Lock-free stacks (Treiber, 1986) offer high throughput but suffer from ABA issues. Michael and Scott's hazard pointers or epoch-based reclamation solve safe reclamation in lock-free allocators (Michael, 2004). For fixed-size activation records, slab allocators with per-core caches minimize contention (Berger et al., 2000). We can combine a lock-free global pool with per-thread caches (`SmallVec` of 32 entries) to reduce cross-core traffic.

### Cache Optimization Techniques
Splitting data into hot and cold regions reduces cache pollution (Kogan & Petrank, 2015). During spreading, we frequently access activation, confidence, and tier metadata; adjacency lists and metadata can be pointer-referenced. Aligning `CacheOptimizedNode` to 64 bytes prevents false sharing and maximizes hardware prefetcher effectiveness. Precomputing adjacency order to bring hot-tier edges together improves spatial locality, echoing techniques in graph processing frameworks like Ligra and GraphIt (Shun & Blelloch, 2013; Zhang et al., 2018).

### Adaptive Batching
Roofline analysis helps determine when computation becomes bandwidth-bound (Williams et al., 2009). For Engram, we measure effective GFLOPS of SIMD kernels and memory bandwidth. Adaptive batching can use a geometric mean of cache capacity, bandwidth, and CPU parallelism. To prevent oscillation, apply an exponential moving average with damping factor 0.5 and only adjust batch size when deviation exceeds 10%. Systems like TensorFlow's autotuner follow similar principles (Abadi et al., 2016).

### Latency Prediction
Online linear regression or gradient boosting can predict latency based on batch size, hop count, and tier distribution. Simpler models—base + per-vector + per-hop cost—are easier to maintain and often sufficient (Dean & Barroso, 2013). Incorporating confidence intervals (±20%) helps scheduler decisions: if predicted latency exceeds budget, reduce hop count or shrink batches.

### Observability
High-quality performance engineering requires visibility. Linux `perf` counters track cache misses (`LLC-load-misses`), while custom metrics monitor pool hit rate and prediction error. Histograms for latency allow precise percentile tracking; Prometheus `Summary` objects or HDR histograms capture P95 accurately (Gil Tene, 2014).

## Key Citations
- Michael, M. M. "Hazard pointers: Safe memory reclamation for lock-free objects." *IEEE TPDS* (2004).
- Treiber, R. K. "Systems programming: Coping with parallelism." *IBM Research Report RJ 5118* (1986).
- Berger, E. D., McKinley, K. S., Blumofe, R. D., & Wilson, P. R. "Hoard: A scalable memory allocator for multithreaded applications." *ASPLOS* (2000).
- Shun, J., & Blelloch, G. E. "Ligra: A lightweight graph processing framework for shared memory." *PPoPP* (2013).
- Zhang, Y., et al. "GraphIt: A high-performance graph DSL." *OOPSLA* (2018).
- Williams, S., Waterman, A., & Patterson, D. "Roofline: An insightful visual performance model." *Communications of the ACM* (2009).
- Abadi, M., et al. "TensorFlow: Large-scale machine learning on heterogeneous systems." *OSDI* (2016).
- Dean, J., & Barroso, L. A. *The Tail at Scale.* (2013).
- Tene, G. "Understanding latency and the jitter of garbage collection." *JavaOne* (2014).
