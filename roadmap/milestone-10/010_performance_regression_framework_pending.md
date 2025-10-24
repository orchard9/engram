# Task 010: Performance Regression Framework

**Duration:** 2 days
**Status:** Pending
**Dependencies:** 009 (Integration Testing)

## Objectives

Establish automated performance regression testing framework that prevents performance degradation in future changes. The framework runs benchmarks in CI, compares against baseline performance, and fails the build if regressions exceed acceptable thresholds.

1. **Baseline establishment** - Record baseline performance for all kernels
2. **Automated benchmarking** - Run benchmarks in CI on every commit
3. **Regression detection** - Fail build if performance regresses >5%
4. **Performance history** - Track performance trends over time

## Dependencies

- Task 009 (Integration Testing) - All kernels validated in integration tests

## Deliverables

### Files to Create

1. `/benches/regression/mod.rs` - Regression benchmark suite
   - Standardized benchmark harness
   - Baseline comparison logic
   - Threshold-based pass/fail

2. `/benches/regression/baselines.json` - Performance baselines
   - Recorded baseline times for each benchmark
   - Platform-specific baselines (x86_64, ARM64)
   - Confidence intervals

3. `/scripts/benchmark_regression.sh` - CI benchmark script
   - Run regression benchmarks
   - Compare against baselines
   - Generate report and exit code

4. `/.github/workflows/performance.yml` - CI workflow
   - Trigger on pull requests and main commits
   - Run regression benchmarks
   - Post results as PR comment

5. `/docs/internal/performance_regression_guide.md` - Documentation
   - How to update baselines
   - Interpreting regression reports
   - Debugging performance issues

### Files to Modify

1. `/Cargo.toml` - Add regression benchmark configuration
   ```toml
   [[bench]]
   name = "regression"
   harness = false
   required-features = ["zig-kernels"]
   ```

2. `/.gitignore` - Ignore benchmark artifacts
   ```
   /target/criterion/
   /benches/regression/results/
   ```

## Acceptance Criteria

1. Regression benchmarks run successfully in CI
2. Baselines established for all three kernels on supported platforms
3. Build fails if any benchmark regresses >5% from baseline
4. Performance history tracked and visualized
5. Documentation explains baseline update process

## Implementation Guidance

### Baseline Storage Format

```json
{
  "version": "1.0",
  "platform": "x86_64-apple-darwin",
  "cpu": "Apple M1 Pro",
  "timestamp": "2025-10-23T12:00:00Z",
  "baselines": {
    "vector_similarity_768d_1000c": {
      "mean_ns": 1700000,
      "std_dev_ns": 50000,
      "sample_size": 100
    },
    "spreading_activation_1000n_100i": {
      "mean_ns": 95000000,
      "std_dev_ns": 2000000,
      "sample_size": 100
    },
    "decay_calculation_10000m": {
      "mean_ns": 65000000,
      "std_dev_ns": 1500000,
      "sample_size": 100
    }
  }
}
```

### Regression Benchmark Harness

```rust
// benches/regression/mod.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct Baseline {
    mean_ns: u64,
    std_dev_ns: u64,
    sample_size: usize,
}

#[derive(Serialize, Deserialize)]
struct Baselines {
    version: String,
    platform: String,
    cpu: String,
    timestamp: String,
    baselines: std::collections::HashMap<String, Baseline>,
}

impl Baselines {
    fn load() -> Self {
        let path = "benches/regression/baselines.json";
        if Path::new(path).exists() {
            let json = fs::read_to_string(path).expect("Failed to read baselines");
            serde_json::from_str(&json).expect("Failed to parse baselines")
        } else {
            Self {
                version: "1.0".to_string(),
                platform: std::env::consts::ARCH.to_string(),
                cpu: "unknown".to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                baselines: Default::default(),
            }
        }
    }

    fn save(&self) {
        let json = serde_json::to_string_pretty(self).unwrap();
        fs::write("benches/regression/baselines.json", json).unwrap();
    }

    fn check_regression(&self, name: &str, mean_ns: u64) -> Result<(), String> {
        if let Some(baseline) = self.baselines.get(name) {
            let baseline_mean = baseline.mean_ns as f64;
            let current_mean = mean_ns as f64;
            let regression_pct = ((current_mean - baseline_mean) / baseline_mean) * 100.0;

            if regression_pct > 5.0 {
                return Err(format!(
                    "REGRESSION: {} is {:.2}% slower than baseline ({} ns vs {} ns)",
                    name, regression_pct, current_mean as u64, baseline.mean_ns
                ));
            } else if regression_pct < -5.0 {
                println!(
                    "IMPROVEMENT: {} is {:.2}% faster than baseline",
                    name, -regression_pct
                );
            }
        } else {
            println!("No baseline for {}, recording current performance", name);
        }
        Ok(())
    }

    fn update_baseline(&mut self, name: String, mean_ns: u64, std_dev_ns: u64, sample_size: usize) {
        self.baselines.insert(
            name,
            Baseline {
                mean_ns,
                std_dev_ns,
                sample_size,
            },
        );
        self.timestamp = chrono::Utc::now().to_rfc3339();
    }
}

fn regression_benchmarks(c: &mut Criterion) {
    let mut baselines = Baselines::load();
    let update_mode = std::env::var("UPDATE_BASELINES").is_ok();

    // Vector similarity regression benchmark
    {
        let name = "vector_similarity_768d_1000c";
        let query = generate_embedding(768);
        let candidates: Vec<_> = (0..1000).map(|_| generate_embedding(768)).collect();

        let mut group = c.benchmark_group("regression");
        group.bench_function(name, |b| {
            b.iter(|| {
                let scores = batch_cosine_similarity(&query, &candidates);
                criterion::black_box(scores);
            });
        });
        group.finish();

        // Extract timing from Criterion
        let result = extract_benchmark_result(name);
        if update_mode {
            baselines.update_baseline(
                name.to_string(),
                result.mean_ns,
                result.std_dev_ns,
                result.sample_size,
            );
        } else {
            if let Err(msg) = baselines.check_regression(name, result.mean_ns) {
                eprintln!("{}", msg);
                std::process::exit(1);
            }
        }
    }

    // Spreading activation regression benchmark
    {
        let name = "spreading_activation_1000n_100i";
        let graph = generate_random_graph(1000, 5000);
        let source = graph.random_node();

        let mut group = c.benchmark_group("regression");
        group.bench_function(name, |b| {
            b.iter(|| {
                let result = spread_activation(&graph, source, 1.0, 100);
                criterion::black_box(result);
            });
        });
        group.finish();

        let result = extract_benchmark_result(name);
        if update_mode {
            baselines.update_baseline(
                name.to_string(),
                result.mean_ns,
                result.std_dev_ns,
                result.sample_size,
            );
        } else {
            if let Err(msg) = baselines.check_regression(name, result.mean_ns) {
                eprintln!("{}", msg);
                std::process::exit(1);
            }
        }
    }

    // Decay calculation regression benchmark
    {
        let name = "decay_calculation_10000m";
        let mut strengths: Vec<f32> = (0..10_000).map(|_| rand::random()).collect();
        let ages: Vec<u64> = (0..10_000).map(|_| rand::random::<u64>() % 1_000_000).collect();

        let mut group = c.benchmark_group("regression");
        group.bench_function(name, |b| {
            b.iter(|| {
                apply_decay(&mut strengths, &ages);
                criterion::black_box(&strengths);
            });
        });
        group.finish();

        let result = extract_benchmark_result(name);
        if update_mode {
            baselines.update_baseline(
                name.to_string(),
                result.mean_ns,
                result.std_dev_ns,
                result.sample_size,
            );
        } else {
            if let Err(msg) = baselines.check_regression(name, result.mean_ns) {
                eprintln!("{}", msg);
                std::process::exit(1);
            }
        }
    }

    if update_mode {
        baselines.save();
        println!("Baselines updated and saved to benches/regression/baselines.json");
    }
}

criterion_group!(benches, regression_benchmarks);
criterion_main!(benches);
```

### CI Benchmark Script

```bash
#!/bin/bash
# scripts/benchmark_regression.sh
set -euo pipefail

echo "Running performance regression benchmarks..."

# Build with Zig kernels
cargo build --release --features zig-kernels

# Run regression benchmarks
cargo bench --features zig-kernels --bench regression -- --noplot

# Exit code will be non-zero if regressions detected
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Performance regression checks passed"
    exit 0
else
    echo ""
    echo "✗ Performance regressions detected"
    exit 1
fi
```

### Update Baselines Script

```bash
#!/bin/bash
# scripts/update_baselines.sh
set -euo pipefail

echo "Updating performance baselines..."

# Ensure clean build
cargo clean
cargo build --release --features zig-kernels

# Run benchmarks in update mode
UPDATE_BASELINES=1 cargo bench --features zig-kernels --bench regression -- --noplot

echo ""
echo "Baselines updated. Review changes and commit:"
echo "  git diff benches/regression/baselines.json"
echo "  git add benches/regression/baselines.json"
echo "  git commit -m 'chore: update performance baselines'"
```

### CI Workflow

```yaml
# .github/workflows/performance.yml
name: Performance Regression

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          profile: minimal

      - name: Install Zig
        uses: goto-bus-stop/setup-zig@v2
        with:
          version: 0.13.0

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Run regression benchmarks
        run: ./scripts/benchmark_regression.sh

      - name: Post results
        if: always()
        uses: actions/github-script@v6
        with:
          script: |
            // Post benchmark results as PR comment
            // (implementation details omitted)
```

### Performance Visualization

```rust
// scripts/visualize_performance.py (optional)
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Load baseline history
with open('benches/regression/baselines.json') as f:
    baselines = json.load(f)

# Plot performance over time
# (implementation for tracking historical trends)
```

## Testing Approach

1. **Baseline validation**
   - Run benchmarks multiple times
   - Verify baseline stability (variance <5%)
   - Test on different platforms

2. **Regression detection**
   - Artificially slow down kernel
   - Verify build fails
   - Check error messages are clear

3. **Update workflow**
   - Test baseline update script
   - Verify changes are reviewable
   - Ensure update process is documented

## Integration Points

- **Task 001 (Profiling)** - Baselines use same benchmarks
- **Task 011 (Documentation)** - Document regression framework usage
- **Task 012 (Final Validation)** - Regression tests part of sign-off

## Notes

- Use dedicated CI runners for consistent performance measurements
- Consider using codspeed.io or similar for automated performance tracking
- Store historical baselines for trend analysis
- Document acceptable variance based on CI runner variability
- Add performance badges to README showing benchmark status
