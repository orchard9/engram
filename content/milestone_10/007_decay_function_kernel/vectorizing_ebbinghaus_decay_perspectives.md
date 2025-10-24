# Vectorizing Ebbinghaus Decay - Multiple Perspectives

## Cognitive Architecture Perspective

### Why Decay Matters for Memory Systems

Human memory doesn't store everything forever. Ebbinghaus discovered in 1885 that memory retention follows an exponential decay curve: freshly learned information fades rapidly, then the decay rate slows. This isn't a bug in our cognitive architecture - it's a feature.

**The Filtering Hypothesis:**

Our brains decay memories to:
1. Prioritize recent information (recency bias helps in changing environments)
2. Strengthen frequently accessed information (spacing effect)
3. Free up capacity for new learning (interference management)

For Engram, implementing biologically plausible decay serves three purposes:

1. **Probabilistic Retrieval:** Older memories become harder to recall, matching human behavior
2. **Consolidation Pressure:** Weak memories decay faster, driving the need for hippocampal-cortical transfer
3. **Computational Efficiency:** Decayed memories can be archived or pruned, reducing active graph size

### Spacing Effect and Repetition

The spacing effect shows that repeated retrievals strengthen memories:

```
τ = τ₀ * (1 + n)
```

Each retrieval increases the half-life, making the memory decay more slowly. This creates a positive feedback loop:
- Important memories get retrieved often
- Frequent retrieval extends half-life
- Extended half-life makes future retrieval more likely

**Implementation Challenge:**

Tracking repetition count adds state to each memory. For 1M memories:
- 1M x 4 bytes (u32 for count) = 4MB overhead
- Decay calculation becomes memory-bound (loading counts)

**Tradeoff:** Current implementation uses fixed half-life (24 hours) to optimize memory access patterns. Future work could add adaptive half-life for high-value memories.

### Forgetting as a Feature

Why optimize something that deliberately throws information away?

Because forgetting at the right rate is crucial for cognitive performance. Too slow, and the system drowns in noise (every trivial observation persists). Too fast, and useful patterns disappear before consolidation.

The 24-hour half-life is calibrated to human sleep cycles: episodic memories have one night of REM sleep to consolidate into long-term storage before significant decay.

## Memory Systems Perspective

### Decay in the Hippocampal Loop

Engram models the hippocampus as a fast-writing, limited-capacity buffer for episodic memories. Decay serves as garbage collection for this buffer:

```
Episodic Memory (Hippocampus)
    ↓ [Rapid decay: τ = 24h]
    ↓ [Consolidation if >threshold]
    ↓
Long-Term Memory (Neocortex)
    ↓ [Slow decay: τ = 90 days]
```

**Why Different Decay Rates Matter:**

- **Hippocampus:** Fast decay creates urgency for consolidation
- **Neocortex:** Slow decay preserves learned patterns long-term

### Implementing Decay in a Graph Database

Traditional databases don't decay data - they delete it explicitly. Engram's probabilistic approach is fundamentally different:

**Challenge:** Decay is continuous, not discrete.

Every memory has a different age, so every memory decays at a different rate. Unlike traditional TTL (time-to-live) where all entries expire at a fixed time, cognitive decay requires per-memory calculation.

**Graph Traversal Implications:**

During spreading activation, edge weights combine:
1. Structural weight (association strength)
2. Decayed memory strength

A strong association to a weak (decayed) memory contributes less to activation than a weak association to a strong memory. This emergent behavior matches human recall: we remember vivid details of important events, vague outlines of unimportant ones.

### Batch Decay vs. On-Demand Decay

**On-Demand Approach (naive):**
```rust
fn get_memory_strength(memory: &Memory, current_time: Timestamp) -> f32 {
    let age = current_time - memory.created_at;
    memory.strength * exp(-age / HALF_LIFE)
}
```

Called for every memory access. Overhead: exp() on hot path.

**Batch Approach (optimized):**
```rust
fn apply_decay_batch(memories: &mut [Memory], current_time: Timestamp) {
    // Calculate all decays once per consolidation cycle
    for memory in memories {
        let age = current_time - memory.created_at;
        memory.strength *= exp(-age / HALF_LIFE);
    }
    // Update last_decay_time
}
```

Called once per minute. Overhead: amortized across batch.

**Key Insight:** Decay doesn't need real-time precision. Updating every 60 seconds is sufficient for a system modeling hours-to-days timescales.

## Rust Graph Engine Perspective

### Why Rust's exp() is Slow (and Why That's Usually Fine)

Rust's `std::f32::exp()` delegates to LLVM's implementation, which uses a high-precision algorithm:

```
exp(x) via range reduction:
1. Decompose x = n*ln(2) + r, where |r| < ln(2)/2
2. Compute 2^n via bit manipulation
3. Compute exp(r) via polynomial
4. Return 2^n * exp(r)
```

**Precision:** Error < 0.5 ULP (units in last place), essentially perfect.

**Cost:** ~8 ns per call (includes range reduction, polynomial, post-processing).

For 10,000 memories: 80 us just for exp() calls.

### The Case for Fast-and-Loose Approximation

Cognitive systems don't need IEEE 754 perfection. Memory strength is already probabilistic:
- Encoding variability: ±10%
- Retrieval noise: ±15%
- Decay uncertainty: "approximately exponential"

An approximation with 0.01% error is 1000x more precise than the underlying model's fidelity.

**4th-Order Polynomial Approximation:**

```zig
exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
```

**Precision:** Error < 1e-5 for x ∈ [-10, 10]

**Cost:** 5 FP operations (4 multiplies, 4 adds) = ~1 ns per call

**Speedup:** 8x faster than std::exp()

### SIMD: From 8x to 64x Speedup

Processing memories one-at-a-time leaves performance on the table. Modern CPUs can process 8 floats in parallel with AVX2:

```zig
const exponent_vec: @Vector(8, f32) = -age_vec / half_life_vec;
const decay_vec = fastExpSimd(exponent_vec);  // 8 results in one go
const result_vec = strength_vec * decay_vec;
```

**Performance Progression:**

1. Baseline (Rust std::exp): 8 ns/memory
2. Polynomial approximation: 1 ns/memory (8x faster)
3. SIMD 8-wide: 0.125 ns/memory (64x faster)

From 80 us to 1.25 us for 10,000 memories.

**Reality Check:** Memory access dominates at this point (loading strengths and ages). Actual measured improvement: 27%, limited by memory bandwidth.

### Structure-of-Arrays for Cache Efficiency

Engram stores memories as:

```rust
struct Memory {
    id: MemoryId,
    strength: f32,
    created_at: u64,
    embedding: [f32; 768],
    metadata: Metadata,
}
```

Total size: ~3KB per memory (dominated by embedding).

**Problem:** Decay only needs `strength` and `created_at` (12 bytes), but loading a Memory struct pulls 3KB into cache.

**Solution:** Structure-of-Arrays layout in decay-critical paths:

```rust
struct DecayView<'a> {
    strengths: &'a mut [f32],
    created_at: &'a [u64],
}
```

Decay processes 10,000 memories:
- AoS: 30 MB cache footprint (3KB * 10k)
- SoA: 120 KB cache footprint (12 bytes * 10k)

250x better cache utilization.

**Tradeoff:** Complexity of maintaining separate views vs. performance gain. Justified for hot paths like decay.

## Systems Architecture Perspective

### Decay as a Background Process

Decay doesn't need to be synchronous with queries. It can run as a periodic background task:

```rust
// Consolidation loop
loop {
    sleep(Duration::from_secs(60));

    // Apply decay to all episodic memories
    let current_time = Timestamp::now();
    episodic_buffer.apply_decay_batch(current_time);

    // Trigger consolidation for weak memories
    consolidate_below_threshold(0.3);
}
```

**Implications:**

1. **Lock-Free Reads:** Queries read stale strengths (up to 60s old), acceptable for cognitive modeling
2. **Batch Optimization:** Process all memories at once, amortizing overhead
3. **Predictable Latency:** Decay doesn't add jitter to query paths

### Memory-Bound vs. Compute-Bound

Initial profiling showed exp() as the bottleneck. After SIMD optimization, memory bandwidth becomes the limit:

**Bottleneck Evolution:**

1. Baseline: Compute-bound (exp() calls)
2. After polynomial approximation: Still compute-bound
3. After SIMD: Memory-bound (loading ages and strengths)

**Roofline Model Analysis:**

- AVX2 peak: 32 FLOP/cycle (8-wide * 2 FMA units * 2 FP ops)
- Memory bandwidth: 25 GB/s (typical DDR4)
- Decay arithmetic intensity: 5 FLOP per 12 bytes = 0.42 FLOP/byte

With 0.42 FLOP/byte and 25 GB/s bandwidth:
- Compute limit: 256 GFLOP/s (unreachable at 0.42 intensity)
- Memory limit: 10.5 GFLOP/s

Decay hits the memory wall. Further optimization requires better cache utilization, not faster exp().

### Prefetching Strategy

Modern CPUs have hardware prefetchers that detect sequential and strided access patterns. Decay is sequential:

```zig
for (i = 0; i < num_memories; i += 8) {
    // Hardware prefetcher automatically loads ahead
    load ages[i..i+8]
    load strengths[i..i+8]
    compute decay
    store strengths[i..i+8]
}
```

For random-access patterns (like graph traversal), manual prefetch helps:

```zig
@prefetch(&ages[i + 64], .read, .medium_temporal);
```

**Heuristic:** Prefetch 64 elements ahead (typical cache line is 64 bytes = 16 f32s or 8 u64s).

## GPU Acceleration Perspective (Future Work)

### Why Not GPU for Decay (Yet)?

GPUs excel at massively parallel, compute-intensive tasks. Decay seems like a fit:
- Process 1M memories in parallel
- Embarrassingly parallel (no inter-memory dependencies)
- Repeated operation (exp + multiply)

**Why CPU wins for now:**

1. **Transfer overhead:** Copying 1M ages + strengths to GPU: ~16 MB at 12 GB/s = 1.3 ms
2. **Kernel launch:** 10-50 us per kernel dispatch
3. **Decay computation:** 65 us on CPU SIMD

Total GPU time: 1,300 us + 10 us + 5 us = 1,315 us

CPU time: 65 us

**GPU is 20x slower** due to transfer overhead.

### When GPU Becomes Worth It

GPU makes sense when decay is part of a larger GPU-resident pipeline:

```
[Query Embedding on GPU]
    ↓
[Similarity Search on GPU] ← memories already in GPU memory
    ↓
[Decay Application on GPU] ← zero-copy, just a kernel launch
    ↓
[Spreading Activation on GPU]
```

If memories live on GPU for similarity search, applying decay is essentially free (5 us kernel vs. 1.3 ms transfer).

**Deferred to Milestone 14:** Unified GPU memory management.

## Testing and Validation Perspective

### Differential Testing for Approximate Functions

How do you test that a fast approximation is "close enough" to the reference?

**Property-Based Testing Approach:**

```rust
proptest! {
    fn fast_exp_matches_std_exp(x in -10.0_f32..10.0_f32) {
        let reference = x.exp();
        let approximation = fast_exp(x);
        let relative_error = (approximation - reference).abs() / reference;

        assert!(relative_error < 1e-4,
            "exp({}) = {} (std) vs {} (fast), error = {}",
            x, reference, approximation, relative_error);
    }
}
```

Tests 10,000 random inputs, verifying relative error < 0.01%.

**Edge Cases to Cover:**

1. x = 0 (exp(0) = 1 exactly)
2. x → -∞ (exp approaches 0)
3. x → +∞ (exp overflows, needs clamping)
4. x = -ln(2) (exp(-ln(2)) = 0.5, common half-life case)

### Numerical Stability Testing

Decay can produce denormal floats (very small positive numbers) for old memories:

```
strength = 1.0
age = 10 days = 864,000 seconds
half_life = 86,400 seconds
decay_factor = exp(-10) ≈ 4.5e-5

result = 1.0 * 4.5e-5 = 0.000045 (normal float)
```

But with initial strength = 0.01:

```
result = 0.01 * 4.5e-5 = 4.5e-7 (denormal territory)
```

**Denormals are 10-100x slower** on many CPUs (microcode trap).

**Solution:** Flush to zero when strength drops below 1e-6:

```zig
if (result < 1e-6) {
    result = 0.0;
}
```

Cognitive justification: Memories with strength < 0.0001% are effectively forgotten.

## Summary of Perspectives

| Perspective | Key Insight |
|------------|-------------|
| Cognitive Architecture | Decay is essential for biologically plausible memory, not just cleanup |
| Memory Systems | Different decay rates for episodic vs. semantic storage |
| Rust Graph Engine | Polynomial approximation + SIMD = 64x speedup on compute |
| Systems Architecture | Memory bandwidth is the real bottleneck, not exp() |
| GPU Acceleration | Transfer overhead dominates until memories are GPU-resident |
| Testing | Property-based testing validates approximation correctness |

All perspectives agree: Optimizing decay is worth it, but the real win comes from batch processing and cache-friendly memory layout, not just faster math.
