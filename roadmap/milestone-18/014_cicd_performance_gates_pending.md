# Task 014: CI/CD Performance Gate Integration

**Status**: Pending
**Estimated Duration**: 4-5 days
**Priority**: Critical - Prevents regressions in production

## Objective

Integrate M17 performance framework into CI/CD pipeline to block merges on >5% internal regression or >10% competitive regression. Run automated performance tests on every main branch merge with <15min test duration.

## Architecture

```yaml
# .github/workflows/performance_gate.yml (example - not using GitHub Actions)
name: Performance Gate

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  performance_check:
    runs-on: [self-hosted, performance-tier-1]
    timeout-minutes: 30

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Build release
        run: cargo build --release

      - name: Run baseline test
        run: |
          ./scripts/m17_performance_check.sh ci_baseline before

      - name: Run regression test
        run: |
          ./scripts/m17_performance_check.sh ci_baseline after

      - name: Compare results
        run: |
          ./scripts/compare_m17_performance.sh ci_baseline
          if [ $? -ne 0 ]; then
            echo "::error::Performance regression detected"
            exit 1
          fi
```

## Fast Performance Tests (CI-Optimized)

```toml
# scenarios/ci/fast_regression_check.toml
name = "CI Fast Regression Check"
description = "5-minute smoke test for CI performance gates"

[duration]
total_seconds = 300  # 5 minutes (not 60s for statistical power)

[arrival]
pattern = "constant"
rate = 1000.0

[operations]
store_weight = 0.35
recall_weight = 0.35
embedding_search_weight = 0.3

[data]
num_nodes = 10_000  # Smaller dataset for CI speed
embedding_dim = 768

[validation]
expected_p99_latency_ms = 10.0
expected_throughput_ops_sec = 800.0
max_error_rate = 0.01
```

## Git Hook Integration

```bash
# .git/hooks/pre-push
#!/bin/bash
# Run quick performance check before pushing to main

if [[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]]; then
    echo "Running performance gate..."

    cargo build --release --quiet
    ./scripts/m17_performance_check.sh pre_push after --quick

    if [ $? -ne 0 ]; then
        echo "ERROR: Performance regression detected"
        echo "Run 'make performance-baseline' to update baseline"
        exit 1
    fi
fi
```

## Success Criteria

- **Regression Detection**: Block merges on >5% P99 increase
- **Fast Feedback**: <15min from push to result
- **Low False Positives**: <2% false alarm rate (statistical tests)
- **Automatic Baseline Update**: Weekly baseline refresh

## Files

- `scripts/ci_performance_gate.sh` (250 lines)
- `scenarios/ci/fast_regression_check.toml`
- `.git/hooks/pre-push` (80 lines)
- `docs/development/performance_gates.md` (200 lines)
