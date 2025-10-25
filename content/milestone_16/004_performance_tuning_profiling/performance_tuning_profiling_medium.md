# From 20ms to 3ms: How We 7x'd Graph Database Performance

The first production deployment of Engram had terrible performance. P50 latency was 20ms. P99 was over 100ms. For a cognitive architecture designed to mimic instant pattern recognition, this was unacceptable.

Six weeks of systematic profiling and optimization brought P50 down to 3ms and P99 to 7ms. Here's how we did it, with the profiling methodology you can apply to any performance problem.

## The Performance Debugging Loop

Guessing doesn't work. Measurement does.

**The Loop:**
1. Profile: Identify the hottest code path
2. Hypothesize: Why is it slow?
3. Optimize: Change one thing
4. Measure: Did it actually help?
5. Repeat

**Tools We Used:**

- **perf:** Linux profiling for CPU-bound issues
- **flamegraph:** Visualization of CPU time
- **valgrind cachegrind:** Cache miss analysis
- **tokio-console:** Async runtime inspection
- **Custom instrumentation:** Application-level metrics

## Iteration 1: The Serialization Disaster

**Baseline:** P50=20ms, P99=105ms

**Profile:**
```bash
perf record -F 99 -g ./target/release/engram benchmark
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

**Result:** 40% of CPU time in bincode::serialize().

Every activation result was being serialized to send over gRPC. Even for internal activations that never left the process.

**Root cause:** Premature abstraction. The API layer serialized everything, even when unnecessary.

**Fix:**
```rust
// Before: Always serialize
async fn activate(&self, query: &Query) -> Result<Vec<u8>> {
    let result = self.graph.activate_internal(query).await?;
    Ok(bincode::serialize(&result)?)  // Wasteful for internal calls
}

// After: Separate internal and external paths
async fn activate_internal(&self, query: &Query) -> Result<ActivationResult> {
    self.graph.activate(query).await
}

async fn activate_grpc(&self, query: &Query) -> Result<Vec<u8>> {
    let result = self.activate_internal(query).await?;
    Ok(bincode::serialize(&result)?)  // Only serialize for gRPC
}
```

**Result:** P50=12ms, P99=45ms. 40% improvement from one line change.

**Lesson:** Profile before optimizing. The bottleneck wasn't where we thought.

## Iteration 2: The Cache Miss Cascade

**Baseline:** P50=12ms, P99=45ms

**Profile:**
```bash
valgrind --tool=cachegrind ./target/release/engram benchmark
cg_annotate cachegrind.out.12345
```

**Result:**
- L1 cache miss rate: 15%
- L3 cache miss rate: 8%
- Main memory accesses: 200K per activation

**Root cause:** Graph nodes stored in HashMap with random memory layout. Each edge traversal = cache miss.

**Fix:** Array-of-structs to struct-of-arrays transformation.

```rust
// Before: Array of structures (bad cache locality)
struct Node {
    id: Uuid,
    embedding: [f32; 768],
    edges: Vec<EdgeId>,
    strength: f32,
}

let nodes: HashMap<Uuid, Node> = ...;

// Traversing edges = jumping through random memory
for edge in node.edges {
    let target = nodes.get(&edge.target)?;  // Cache miss
    // ...
}

// After: Structure of arrays (good cache locality)
struct NodeStore {
    ids: Vec<Uuid>,
    embeddings: Vec<[f32; 768]>,
    edges: Vec<Vec<EdgeId>>,
    strengths: Vec<f32>,
}

// Edges stored contiguously
let target_strength = self.strengths[edge.target_idx];  // Cache hit
```

**Result:** P50=7ms, P99=20ms. Another 40% improvement.

**Lesson:** Data structure layout matters more than algorithm complexity for modern CPUs.

## Iteration 3: The Allocation Storm

**Baseline:** P50=7ms, P99=20ms

**Profile:**
```bash
MALLOC_CONF=prof:true ./target/release/engram benchmark
jeprof --svg engram jeprof.heap > allocations.svg
```

**Result:** 3 million allocations per second. 200MB/sec allocation rate.

**Root cause:** Creating Vec for every activation result, even when activation set was small.

**Fix:** Object pooling with pre-allocated capacity.

```rust
// Before: Allocate every time
async fn activate(&self, query: &Query) -> Result<Vec<ActivatedNode>> {
    let mut result = Vec::new();  // Fresh allocation
    // ... populate result
    Ok(result)
}

// After: Reuse pooled allocation
use object_pool::Pool;

lazy_static! {
    static ref RESULT_POOL: Pool<Vec<ActivatedNode>> =
        Pool::new(100, || Vec::with_capacity(1000));
}

async fn activate(&self, query: &Query) -> Result<Vec<ActivatedNode>> {
    let mut result = RESULT_POOL.pull()?;
    result.clear();  // Reuse existing capacity
    // ... populate result
    Ok(result)
}
```

**Result:** P50=5ms, P99=12ms. 30% improvement.

**Lesson:** Allocations are not free. Pool large or frequently allocated objects.

## Iteration 4: The Lock Contention

**Baseline:** P50=5ms, P99=12ms

**Profile:**
```bash
perf record -e sched:sched_switch -g ./target/release/engram benchmark
perf script | stackcollapse-perf.pl | flamegraph.pl > contention.svg
```

**Result:** 20% of time spent waiting on RwLock for node access.

**Root cause:** Coarse-grained locking. Single RwLock protected entire node store.

**Fix:** Fine-grained locking with sharded storage.

```rust
// Before: Single lock
struct GraphStore {
    nodes: RwLock<HashMap<Uuid, Node>>,
}

// High contention: All readers and writers contend on one lock

// After: Sharded locks
struct GraphStore {
    shards: Vec<RwLock<HashMap<Uuid, Node>>>,
}

impl GraphStore {
    fn shard_for(&self, id: &Uuid) -> usize {
        let hash = id.as_u128();
        (hash % self.shards.len() as u128) as usize
    }

    async fn get(&self, id: &Uuid) -> Option<Node> {
        let shard = self.shard_for(id);
        self.shards[shard].read().await.get(id).cloned()
    }
}
```

**Result:** P50=4ms, P99=9ms. 20% improvement.

**Lesson:** Lock granularity matters. Shard hot data structures.

## Iteration 5: The Async Runtime Overhead

**Baseline:** P50=4ms, P99=9ms

**Profile:**
```bash
TOKIO_CONSOLE_BIND=0.0.0.0:6669 ./target/release/engram benchmark

# In another terminal
tokio-console http://localhost:6669
```

**Result:** 60% of tasks spent <100us of actual work. Async overhead dominated.

**Root cause:** Over-asyncification. Making everything async, even fast operations.

**Fix:** Hybrid sync/async approach.

```rust
// Before: Async for simple lookups
async fn get_node(&self, id: &Uuid) -> Option<Node> {
    self.nodes.read().await.get(id).cloned()
}

// Read lock held for <1us. Async overhead is 10x the work.

// After: Synchronous for fast paths
fn get_node_sync(&self, id: &Uuid) -> Option<Node> {
    self.nodes.blocking_read().get(id).cloned()
}

// Use async only for genuinely slow operations
async fn activate(&self, query: &Query) -> Result<ActivationResult> {
    // This is actually slow (multi-hop traversal), keep async
}
```

**Result:** P50=3.5ms, P99=8ms. 12% improvement.

**Lesson:** Async has overhead. Use it for I/O and slow operations, not everything.

## Iteration 6: The Final Mile - SIMD

**Baseline:** P50=3.5ms, P99=8ms

**Profile:**
```bash
perf record -e cycles,instructions ./target/release/engram benchmark
perf report
```

**Result:** Vector similarity calculation (dot product) was 30% of remaining CPU time.

**Fix:** SIMD vectorization for embedding operations.

```rust
// Before: Scalar dot product
fn dot_product(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// After: SIMD dot product
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
fn dot_product_simd(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    unsafe {
        let mut sum = _mm256_setzero_ps();
        for i in (0..768).step_by(8) {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        horizontal_sum_avx(sum)
    }
}
```

**Result:** P50=3ms, P99=7ms. Final 15% improvement.

**Lesson:** For hot loops over numeric data, SIMD is the last optimization lever.

## The Configuration That Actually Matters

After all optimizations, we found only 4 configuration parameters significantly impact performance:

**1. Shard Count (Lock Granularity)**
```toml
[graph]
shard_count = 256  # Must be power of 2
```

Too low: Lock contention
Too high: Memory overhead
Sweet spot: num_cpus * 32

**2. Fast Tier Size (Cache Size)**
```toml
[memory]
fast_tier_size = "4GB"
```

Too low: Cache thrashing
Too high: Wasted RAM
Sweet spot: 2x working set size

**3. Object Pool Size**
```toml
[performance]
result_pool_size = 100
result_pool_capacity = 1000
```

Too low: Frequent allocations
Too high: Memory overhead
Sweet spot: concurrent_requests * 2

**4. Async Worker Threads**
```toml
[runtime]
worker_threads = 8
```

Too low: CPU underutilization
Too high: Context switch overhead
Sweet spot: num_cpus (not more)

**Everything else has <1% impact on performance.**

## The Performance Tuning Methodology

For any performance problem:

**1. Establish Baseline**
```bash
engram benchmark --duration=60s --output=baseline.json
```

**2. Profile Under Load**
```bash
perf record -F 99 -g -- engram benchmark --duration=60s
perf script | stackcollapse-perf.pl | flamegraph.pl > before.svg
```

**3. Identify Hotspot**
- Look for wide bars in flamegraph (high CPU time)
- Focus on user code, not system libraries
- Verify with perf report percentages

**4. Hypothesize Cause**
- Cache misses? Run valgrind cachegrind
- Lock contention? Check perf sched:sched_switch events
- Allocations? Run with malloc profiler
- Async overhead? Use tokio-console

**5. Make ONE Change**
- Don't change multiple things simultaneously
- Git commit before changing anything
- Document hypothesis in commit message

**6. Measure Impact**
```bash
engram benchmark --duration=60s --output=after.json
engram compare-benchmarks baseline.json after.json
```

**7. Keep or Revert**
- If improvement >5%: Keep and profile again
- If improvement <5%: Revert and try different hypothesis
- If regression: Immediately revert

**8. Repeat Until Target Met**

## Common Performance Anti-Patterns

**Anti-Pattern 1: Premature SIMD**

Don't start with SIMD. Start with better algorithms and data structures. SIMD is the last 10-20%, not the first 80%.

**Anti-Pattern 2: Micro-Optimizations**

Shaving 5ns off a function that runs 100 times per second saves 0.5us/sec total. Not worth the code complexity. Profile first.

**Anti-Pattern 3: Async Everything**

Async has overhead. Use it for I/O and slow operations. Don't make simple getters async just because "async is faster."

**Anti-Pattern 4: Zero-Copy Obsession**

Copying 1KB takes ~100ns on modern CPUs. Zero-copy adds complexity. Only optimize copies if profiler shows it matters.

**Anti-Pattern 5: Ignoring Cache Locality**

The fastest code is code that doesn't run. The second fastest is code that hits L1 cache. Data structure layout > algorithm complexity.

## The Results

After 6 iterations:

**Before:**
- P50 latency: 20ms
- P99 latency: 105ms
- Throughput: 1,500 ops/sec
- CPU usage: 80%

**After:**
- P50 latency: 3ms (7x faster)
- P99 latency: 7ms (15x faster)
- Throughput: 12,000 ops/sec (8x higher)
- CPU usage: 40% (2x more efficient)

**Time invested:** 6 weeks of profiling and optimization

**Code changes:** ~500 lines modified, mostly data structure layouts

**Key insight:** The bottleneck is never where you think it is. Profile, measure, optimize, repeat.

Performance optimization is not magic. It's systematic application of measurement, hypothesis testing, and iteration. The tools are free. The methodology is simple. The results are dramatic.

Profile your code. You might be surprised what you find.
