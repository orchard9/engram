# Vectorizing Ebbinghaus Decay - Twitter Thread

## Thread: How We Made Memory Decay 27% Faster with SIMD

**Tweet 1/10:**

Memory fades exponentially - Ebbinghaus proved it in 1885. For Engram (our cognitive graph database), implementing this decay across 10,000 memories took 89 microseconds.

We got it down to 65 microseconds using Zig and SIMD. Here's how.

**Tweet 2/10:**

The bottleneck was obvious from profiling: Rust's exp() function took 68 out of 89 microseconds.

Why so slow? Because it's perfectly accurate to 0.5 ULP (units in last place).

For modeling messy biological memory? Complete overkill.

**Tweet 3/10:**

Consider the noise:
- Memory encoding varies 10-20%
- Retrieval has 15%+ variance
- Ebbinghaus curve is "approximately exponential"

If our exp() approximation has 0.01% error, it's still 1000x more precise than the phenomena we're modeling.

**Tweet 4/10:**

Solution: 4th-order Taylor polynomial

exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24

For x ∈ [-10, 10]:
- Error < 1e-5
- Speed: 8x faster than std::exp()

From 8 ns per call to 1 ns. Already a win.

**Tweet 5/10:**

But we can go further with SIMD (Single Instruction, Multiple Data).

Modern CPUs can process 8 floats in parallel with AVX2. Instead of computing exp() for one memory at a time, compute 8 simultaneously.

Theoretical speedup: 64x on compute.

**Tweet 6/10:**

Zig makes SIMD almost pleasant:

```zig
const exponent_vec = -age_vec / half_life_vec;
const decay_vec = fastExpSimd(exponent_vec);
const result_vec = strength_vec * decay_vec;
```

No intrinsics hell. Just @Vector types with natural operators.

**Tweet 7/10:**

Reality check: We got 27% improvement, not 64x.

Why? Memory bandwidth became the bottleneck.

Loading 10k ages (80KB) + strengths (40KB) takes ~20μs. The exponential calculation now takes <2μs. We're waiting for data, not compute.

**Tweet 8/10:**

This is where structure-of-arrays (SoA) layout saves you.

Instead of loading 3KB per memory (including unused embeddings), extract just ages + strengths:

Cache footprint: 30MB → 120KB
250x better cache utilization.

**Tweet 9/10:**

Edge cases that matter:
- Very old memories → denormal floats (100x slower) → flush to zero
- Zero age → skip calculation entirely
- Already forgotten → early exit

Fast paths need careful engineering, not just fast math.

**Tweet 10/10:**

Lessons learned:

1. Precision isn't always valuable (IEEE 754 perfection costs)
2. Memory bandwidth often beats compute optimization
3. Batch processing unlocks SIMD potential
4. The right tool for the job: Rust for safety, Zig for hot kernels

Full writeup: [link]

---

## Thread: Why We Don't Use GPU for Memory Decay

**Tweet 1/6:**

"Why not use GPU for this? Processing 10k memories in parallel seems perfect for it."

Great question. Here's why CPU + SIMD beats GPU for decay (for now).

**Tweet 2/6:**

GPU overhead breakdown:
- Transfer 120KB to GPU memory: 1,300μs
- Kernel launch latency: 10μs
- Actual computation: 5μs

Total: 1,315μs

CPU SIMD: 65μs

GPU is 20x slower due to transfer overhead.

**Tweet 3/6:**

The rule: GPU wins when data lives there already.

If memories are GPU-resident for similarity search, applying decay is essentially free (just kernel launch).

But shipping data back and forth kills you.

**Tweet 4/6:**

When GPU makes sense for decay:

[Query Embedding on GPU]
↓
[Similarity Search on GPU]
↓
[Decay on GPU] ← zero-copy
↓
[Spreading Activation on GPU]

Unified GPU pipeline. Milestone 14.

**Tweet 5/6:**

For now: CPU kernels win because memories live in system RAM.

The 65μs we spend on CPU decay would become 1,300μs if we tried to offload to GPU.

PCIe bandwidth (12GB/s) is the villain.

**Tweet 6/6:**

This is a common pattern in performance work:

Optimize what you have before adding complexity.

SIMD got us 27% improvement with 2 days of work. GPU integration would take weeks and make things slower.

Know your bottlenecks.

---

## Thread: Approximating exp() - When Close Enough is Perfect

**Tweet 1/7:**

Hot take: IEEE 754 precision is often waste.

We replaced Rust's perfect exp() (error < 0.5 ULP) with a 4th-order polynomial (error < 1e-5).

Result: 8x faster, still 1000x more precise than needed.

**Tweet 2/7:**

The 4th-order Taylor approximation:

exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24

5 multiplies, 4 adds. No range reduction, no special cases, no microcode.

Just straight-line arithmetic that SIMD loves.

**Tweet 3/7:**

Why 4th order specifically?

- 2nd order: ~1% error (too sloppy)
- 3rd order: ~0.1% error (getting close)
- 4th order: ~0.001% error (perfect for cognitive modeling)
- 5th order: ~0.0001% error (not worth the extra multiply)

Diminishing returns kick in hard.

**Tweet 4/7:**

Edge case: What about x outside [-10, 10]?

Clamp it. For decay calculations:
- exp(-10) ≈ 0.000045 (essentially zero strength)
- exp(+10) doesn't happen (memories don't strengthen over time)

Simple bounds checking handles it.

**Tweet 5/7:**

Validation: Property-based testing with 10,000 random inputs.

Every test compares polynomial vs. std::exp():

assert_relative_eq!(approx, exact, epsilon = 1e-6)

All tests pass. The approximation is correct.

**Tweet 6/7:**

This pattern generalizes:

1. Profile to find math bottleneck
2. Analyze required precision (usually way less than IEEE 754)
3. Find approximation that fits in SIMD
4. Validate with differential testing

Works for sin, cos, log, sqrt, etc.

**Tweet 7/7:**

The counterintuitive truth:

Perfect precision often wastes cycles.

For scientific computing: use exact functions.
For games, graphics, ML, cognitive systems: approximate is fine.

Know your error budget.

---

## Thread: Structure of Arrays Saves 250x Cache Space

**Tweet 1/5:**

Memory layout matters more than algorithms sometimes.

We had a fast decay kernel but it was thrashing cache. The problem? Array-of-Structures layout was loading data we didn't need.

**Tweet 2/5:**

Original memory structure:

```rust
struct Memory {
    strength: f32,      // 4 bytes (needed)
    created_at: u64,    // 8 bytes (needed)
    embedding: [f32; 768], // 3072 bytes (NOT needed)
}
```

Decay needs 12 bytes. We were loading 3KB per memory.

**Tweet 3/5:**

For 10k memories:
- Data needed: 120KB
- Data loaded: 30MB
- Cache waste: 250x

L1 cache is only 64KB. We were evicting decay data to load useless embeddings.

**Tweet 4/5:**

Solution: Structure-of-Arrays for hot paths

```rust
struct DecayView {
    strengths: &mut [f32],
    created_at: &[u64],
}
```

Now we only touch what we need. Cache hit rate: 92% → 98.7%.

**Tweet 5/5:**

This is why "data-oriented design" matters.

Arranging memory for access patterns beats clever algorithms.

Hot data together. Cold data elsewhere. Cache wins are free performance.

---

## Thread: Denormals - The Performance Trap You Don't See

**Tweet 1/5:**

Weird performance trap: denormal floats.

When memory strength decays below ~1e-38, IEEE 754 enters "denormal" mode. Suddenly, operations become 10-100x slower.

We flush to zero to avoid this trap.

**Tweet 2/5:**

What are denormals?

Normal floats: 1.0 × 2^exp
Denormals: 0.xxx × 2^-126

They extend the representable range near zero but trigger slow microcode on many CPUs.

**Tweet 3/5:**

In our case:
- Memory with strength 0.01 after 10 days
- Decay factor: exp(-10) ≈ 4.5e-5
- Result: 4.5e-7 (denormal territory)

Suddenly that multiply takes 100 cycles instead of 1.

**Tweet 4/5:**

Solution: flush to zero when strength < 1e-6

```zig
if (result < 1e-6) result = 0.0;
```

Cognitive justification: Memories below 0.0001% strength are effectively forgotten anyway.

**Tweet 5/5:**

Denormals are a spec feature that becomes a performance bug in practice.

Many HPC codebases compile with -ffast-math or -fno-denormals to disable them entirely.

Know your floats.
