# Performance Regression Framework - Twitter Thread

**Tweet 1/7:**

We spent 3 weeks making Zig kernels 35% faster.

Then someone added debug logging to the hot path and erased half the gains.

It passed all tests. Code review missed it. It shipped.

This is why we automated performance regression detection.

**Tweet 2/7:**

Performance bugs are silent. They don't crash. They don't fail tests. They just make everything slower.

Automated regression detection treats performance like correctness:

Regression >5%? Build fails. Same as a broken test.

**Tweet 3/7:**

Regression benchmark pattern:

```rust
let baseline = load_baseline(); // 1.7μs
let current = benchmark_kernel();

if current > baseline * 1.05 {
    eprintln!("REGRESSION: 5% slower");
    exit(1);
}
```

Run on every PR. Catch regressions pre-merge.

**Tweet 4/7:**

Why 5% threshold?

CI runner variance: ±2%
5% = 2.5x noise floor

<5%: probably noise
5-10%: investigate
>10%: clear regression

Signal vs noise matters.

**Tweet 5/7:**

Real regressions we caught:

1. Debug assertions in release: 8% slower
2. Hidden clone calls: 12% slower
3. Lock contention: 40% slower (32 threads)

All passed functional tests.
All failed regression benchmarks.

**Tweet 6/7:**

Store baselines as version-controlled JSON:

```json
{
  "vector_similarity_768d": {
    "mean_ns": 1700000,
    "std_dev_ns": 50000
  }
}
```

Performance changes become explicit and reviewable.

**Tweet 7/7:**

Performance regression frameworks prevent death by a thousand cuts.

Individual PRs rarely tank performance. But small regressions accumulate.

Automated checks keep optimizations optimized.
