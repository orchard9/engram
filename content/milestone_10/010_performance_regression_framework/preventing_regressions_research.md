# Preventing Regressions: Automated Benchmarking in CI - Research

## Key Concepts

### 1. Performance Regression Detection

**Definition:** Performance regression occurs when code changes degrade system performance below acceptable thresholds.

**Detection Strategy:**
- Establish baseline performance metrics
- Run benchmarks on every commit/PR
- Compare current performance against baseline
- Fail build if regression exceeds threshold (typically 5-10%)

**Citations:**
- Daly, D., et al. (2020). "The Use of Change Point Detection to Identify Software Performance Regressions in a Continuous Integration System", ICPE
- Foo, K. C., et al. (2015). "Mining Performance Regression Inducing Code Changes in Evolving Software", MSR

### 2. Criterion.rs Benchmarking Framework

Criterion provides:
- Statistical analysis (mean, median, std dev, outliers)
- Comparison against previous runs
- HTML reports with visualizations
- Integration with CI systems

**Usage:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_kernel(c: &mut Criterion) {
    c.bench_function("vector_similarity", |b| {
        b.iter(|| vector_similarity(black_box(&query), black_box(&candidates)));
    });
}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);
```

**Citation:** Criterion.rs documentation, https://bheisler.github.io/criterion.rs/

### 3. Baseline Storage and Comparison

Store performance baselines as JSON:
```json
{
  "vector_similarity_768d": {
    "mean_ns": 1700000,
    "std_dev_ns": 50000,
    "sample_size": 100
  }
}
```

Compare current run against baseline:
- Regression: current > baseline * 1.05 (5% slower)
- Improvement: current < baseline * 0.95 (5% faster)
- Stable: within ±5% of baseline

**Citation:** Google (2016). "Site Reliability Engineering: Measuring and Operating Services"

### 4. CI Integration Patterns

**GitHub Actions Workflow:**
```yaml
- name: Run performance benchmarks
  run: cargo bench --features zig-kernels --bench regression

- name: Check for regressions
  run: |
    if grep "REGRESSION" bench_results.txt; then
      exit 1
    fi
```

**Considerations:**
- Use dedicated CI runners for consistent results
- Pin CPU frequency (avoid turbo boost variability)
- Run multiple iterations to reduce noise
- Store historical data for trend analysis

**Citation:** Bulej, L., et al. (2017). "Unit Testing Performance in Java Projects: Are We There Yet?", ASE
