# Preventing Regressions: Automated Benchmarking in CI

We spent three weeks optimizing Zig kernels. Vector similarity got 25% faster. Spreading activation got 35% faster. Decay calculations got 27% faster.

Then someone added debug logging to the hot path and erased half the gains.

The logging PR passed all tests. Code review didn't catch it. It shipped to production. Performance degraded silently.

This is why performance regression frameworks exist.

## The Problem: Performance Bugs are Silent

Functional bugs fail tests. Performance bugs pass tests and fail users.

A developer adds a convenience feature:
```rust
pub fn batch_cosine_similarity(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    log::debug!("Computing similarity for {} candidates", candidates.len());  // Added
    // ... actual computation
}
```

Innocent change. Adds observability. Passes all tests.

But log::debug!() checks if logging is enabled on every call. For 1000-candidate batches called 10,000 times/sec, that's 10M checks/sec - pure overhead when logging is disabled.

Measured impact: 6% slower.

Without automated detection, this regression ships.

## The Solution: Regression Benchmarks in CI

Treat performance as a correctness property:

```rust
fn regression_benchmark(c: &mut Criterion) {
    let baselines = Baselines::load(); // Previous performance
    let query = generate_embedding(768);
    let candidates: Vec<_> = (0..1000).map(|_| generate_embedding(768)).collect();

    c.bench_function("vector_similarity", |b| {
        b.iter(|| batch_cosine_similarity(&query, &candidates));
    });

    let current = extract_benchmark_result("vector_similarity");
    if current.mean_ns > baselines["vector_similarity"].mean_ns * 1.05 {
        eprintln!("REGRESSION: 5% slower than baseline");
        std::process::exit(1);  // Fail the build
    }
}
```

CI runs this on every PR. If performance regresses >5%, build fails.

## Baseline Storage: Know Your Numbers

Store baseline performance as version-controlled JSON:

```json
{
  "version": "1.0",
  "platform": "x86_64-linux",
  "baselines": {
    "vector_similarity_768d_1000c": {
      "mean_ns": 1700000,
      "std_dev_ns": 50000
    }
  }
}
```

When baseline changes (intentionally), update via:
```bash
UPDATE_BASELINES=1 cargo bench --bench regression
git commit benches/baselines.json -m "chore: update baselines after kernel optimization"
```

This makes performance changes explicit and reviewable.

## The 5% Threshold: Signal vs Noise

Why 5% regression threshold?

- <5%: Likely measurement noise or acceptable tradeoff
- 5-10%: Investigate if expected (e.g., added feature)
- >10%: Clear regression, block merge

Measured variance on dedicated CI runners: ±2%. So 5% threshold is 2.5x noise floor - confident signal.

## Lessons from Real Regressions

Three regressions we caught in CI:

**Regression 1: Debug Assertions in Release**
- Change: Added bounds checking to vector indexing
- Impact: 8% slower (bounds checks on hot path)
- Fix: Use unsafe indexing after length validation
- Verdict: Caught pre-merge

**Regression 2: Unnecessary Cloning**
- Change: Refactored to take owned values instead of references
- Impact: 12% slower (hidden clone calls)
- Fix: Change signatures to take references
- Verdict: Caught pre-merge

**Regression 3: Lock Contention**
- Change: Added telemetry with shared counter
- Impact: 40% slower under 32-thread load
- Fix: Use thread-local counters, aggregate periodically
- Verdict: Caught pre-merge

All three passed functional tests. All three failed regression benchmarks.

## CI Integration: Fast Feedback

GitHub Actions workflow:

```yaml
performance-regression:
  runs-on: ubuntu-latest
  steps:
    - name: Run regression benchmarks
      run: cargo bench --features zig-kernels --bench regression

    - name: Check results
      run: |
        if [ $? -ne 0 ]; then
          echo "Performance regression detected"
          exit 1
        fi
```

Results post as PR comments:
> Performance check: FAILED
> - vector_similarity: 1.82μs (baseline: 1.70μs, +7.1% REGRESSION)
> - spreading_activation: 96μs (baseline: 95μs, +1.1% OK)
> - decay: 65μs (baseline: 65μs, 0% OK)

Developer sees immediate feedback. Regression doesn't merge.

## Beyond Pass/Fail: Performance History

Tracking performance over time reveals trends:

- Are we getting faster or slower overall?
- Which optimizations had lasting impact?
- Where should we invest optimization effort?

Store benchmark results in time-series database (Prometheus, InfluxDB) for dashboards.

## Conclusion

Performance regression frameworks prevent death by a thousand cuts.

Individual PRs rarely tank performance. But accumulated small regressions compound. Without automated checks, systems slowly degrade.

With automated checks, performance becomes a release gate. Just like tests. Just like lints.

The result: Zig kernel optimizations stay optimized.
