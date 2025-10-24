# Twitter Thread: Sub-100μs Parse Times

**Tweet 1/6**

Parser performance matters more than you think.

Slow parser (500μs): 10k qps = 5 sec CPU (50% overhead)
Fast parser (50μs): 10k qps = 0.5 sec CPU (5% overhead)

10x faster parsing = 10x system throughput.

Here's how we got from 450μs to 90μs.

---

**Tweet 2/6**

Optimization 1: Zero-copy strings

Before: Clone every identifier
```rust
let id = token.text.clone();  // Allocates
```

After: Slice input buffer
```rust
let id = &input[token.start..token.end];  // Zero-copy
```

Result: 40% faster (450μs → 280μs)

Biggest win. String handling dominates parser performance.

---

**Tweet 3/6**

Optimization 2: Arena allocation

Before: Each AST node allocated separately (20+ allocations)
After: Entire AST in one arena allocation

Contiguous memory, cache-friendly, bulk free.

Result: 30% faster (280μs → 196μs)

Allocation is the enemy of performance.

---

**Tweet 4/6**

Optimization 3: Keyword hash map

Before: Linear search O(n)
After: Hash map O(1)

Even for just 5 keywords, checked thousands of times per parse.

Result: 15% faster (196μs → 167μs)

Big-O matters even for small n when called frequently.

---

**Tweet 5/6**

Optimization 4: SIMD float parsing

Large embeddings (1536 floats) were bottleneck.

Scalar: Parse one float at a time
SIMD (AVX2): Parse 8 floats in parallel

Result: 4x faster (800μs → 200μs)

Critical for real-time query processing.

---

**Tweet 6/6**

Final results:

Simple: 45μs (4.4x faster)
Complex: 90μs (5x faster)
Large embedding: 180μs (6.7x faster)

Regression testing in CI: Fail if >10% slower.

Caught 3 regressions before production.

Performance is a feature, not an afterthought.

Code: https://github.com/engram-memory/engram

---

**Hashtags**: #Rust #Performance #Optimization #SystemsDesign #Benchmarking
