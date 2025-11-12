# Task 006: Initial Baseline Measurement and Validation (Enhanced)

**Status**: Pending
**Complexity**: Moderate
**Dependencies**: Task 005 (requires workflow), Task 004 (requires report generator), Task 002 (requires baseline documentation), Task 001 (requires scenarios)
**Estimated Effort**: 5 hours

## Objective

Execute the first competitive baseline measurement with rigorous validation methodology, perform statistical analysis to verify result integrity, document Engram's current competitive positioning with confidence intervals, and create data-driven optimization tasks based on quantified performance gaps.

## Enhanced Specifications

### 1. Pre-Measurement Validation (System Readiness)

**System Requirements Verification**:
```bash
# Check hardware meets minimum specifications
python3 <<EOF
import psutil
import os

# Verify RAM (minimum 16GB for 1M scenarios)
total_ram_gb = psutil.virtual_memory().total / (1024**3)
available_ram_gb = psutil.virtual_memory().available / (1024**3)

if total_ram_gb < 16:
    print(f"ERROR: Insufficient RAM ({total_ram_gb:.1f}GB < 16GB required)")
    exit(1)

if available_ram_gb < 12:
    print(f"WARNING: Low available RAM ({available_ram_gb:.1f}GB < 12GB recommended)")
    print("Close other applications before running 1M scenarios")

# Verify CPU cores (minimum 4, recommended 8+)
cpu_count = os.cpu_count()
if cpu_count < 4:
    print(f"ERROR: Insufficient CPU cores ({cpu_count} < 4 required)")
    exit(1)

if cpu_count < 8:
    print(f"WARNING: Low CPU count ({cpu_count} < 8 recommended)")
    print("Performance may be lower than documented baselines")

# Check disk space (minimum 10GB free)
disk_free_gb = psutil.disk_usage('/').free / (1024**3)
if disk_free_gb < 10:
    print(f"ERROR: Insufficient disk space ({disk_free_gb:.1f}GB < 10GB required)")
    exit(1)

print(f"PASS: System meets requirements ({total_ram_gb:.1f}GB RAM, {cpu_count} cores, {disk_free_gb:.1f}GB free)")
EOF
```

**Code Quality Verification**:
```bash
# Verify main branch is at stable commit (no uncommitted changes)
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: Working directory has uncommitted changes"
    echo "Commit or stash changes before running baseline measurement"
    exit 1
fi

# Verify we're on main branch or stable milestone branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" && "$CURRENT_BRANCH" != "milestone-17"* ]]; then
    echo "WARNING: Not on main or milestone-17 branch (currently on $CURRENT_BRANCH)"
    echo "Results may not be representative of production baseline"
fi

# Run make quality to ensure zero clippy warnings
echo "Running code quality checks..."
make quality
if [ $? -ne 0 ]; then
    echo "ERROR: Code quality checks failed"
    echo "Fix all clippy warnings before running baseline measurement"
    exit 1
fi

# Build loadtest in release mode with optimizations
echo "Building loadtest in release mode..."
cargo build --release --bin loadtest
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build loadtest binary"
    exit 1
fi

# Verify binary was built in release mode (check for optimization markers)
if ! cargo build --release --bin loadtest 2>&1 | grep -q "opt-level"; then
    echo "WARNING: Cannot verify release mode optimization level"
fi

echo "PASS: Code quality verified, release binary built"
```

**Baseline Thermal Stability**:
```bash
# Check CPU temperature and thermal throttling state
# (macOS-specific, adapt for Linux if needed)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check for thermal pressure (macOS)
    if pmset -g thermlog | grep -q "CPU_Scheduler_Limit"; then
        echo "WARNING: System under thermal pressure"
        echo "Allow system to cool before running baseline measurement"
        echo "Wait 5 minutes, close resource-intensive applications"
    fi
fi

# Clear system caches to ensure cold-start measurement
if [[ "$OSTYPE" == "darwin"* ]]; then
    sudo purge  # macOS cache clear
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
fi

echo "PASS: System thermal state acceptable, caches cleared"
```

### 2. Execute Baseline Measurement with Monitoring

**Run Competitive Benchmark Suite**:
```bash
# Execute with comprehensive monitoring
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
echo "Starting competitive baseline measurement at $TIMESTAMP"

# Run benchmark suite with monitoring
./scripts/quarterly_competitive_review.sh 2>&1 | tee tmp/competitive_benchmarks/${TIMESTAMP}_execution.log

# Capture exit code
SUITE_EXIT_CODE=$?

if [ $SUITE_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Benchmark suite failed with exit code $SUITE_EXIT_CODE"
    echo "Check tmp/competitive_benchmarks/${TIMESTAMP}_execution.log for details"
    exit 1
fi

# Wait for all background processes to finish
sleep 5

echo "PASS: Benchmark suite completed successfully"
```

**Monitor Execution Health**:
- Detect OOM (Out of Memory) kills by checking dmesg or system logs
- Detect excessive swapping (>1GB swap used indicates memory pressure)
- Detect thermal throttling (CPU frequency drops >20% during test)
- Detect disk I/O bottlenecks (queue depth >16, latency >100ms)

### 3. Result Validation (Data Integrity)

**Statistical Sanity Checks**:

Create validation script: `scripts/validate_baseline_results.py`

```python
#!/usr/bin/env python3
"""
Validate baseline measurement results for data integrity.

Checks for measurement errors, outliers, and statistical anomalies that
indicate invalid results requiring re-measurement.
"""

import json
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import statistics

@dataclass
class ValidationResult:
    """Result of validation with detailed error context."""
    scenario: str
    metric: str
    value: Optional[float]
    valid: bool
    reason: str
    severity: str  # "ERROR", "WARNING", "INFO"

def validate_latency_sanity(p99_ms: float, scenario_name: str) -> ValidationResult:
    """
    Validate P99 latency is in reasonable range.

    Sanity bounds based on operation complexity:
    - Pure ANN search (1M vectors): 5ms - 200ms (HNSW traversal time)
    - Graph traversal (100K nodes): 2ms - 100ms (cache lookup + edge traversal)
    - Hybrid workload: 10ms - 500ms (combined operations)
    - Pattern completion: 50ms - 2000ms (iterative refinement)

    Below minimum: Measurement error (timer not started, empty dataset)
    Above maximum: Severe performance issue (misconfiguration, resource contention)
    """
    scenario_bounds = {
        "qdrant_ann_1m_768d": (5.0, 200.0),
        "neo4j_traversal_100k": (2.0, 100.0),
        "hybrid_production_100k": (10.0, 500.0),
        "milvus_ann_10m_768d": (20.0, 2000.0),
    }

    min_ms, max_ms = scenario_bounds.get(scenario_name, (1.0, 10000.0))

    if p99_ms is None or p99_ms <= 0:
        return ValidationResult(
            scenario=scenario_name,
            metric="P99 Latency",
            value=p99_ms,
            valid=False,
            reason=f"P99 latency is zero or negative ({p99_ms}ms) - timer measurement error",
            severity="ERROR"
        )

    if p99_ms < min_ms:
        return ValidationResult(
            scenario=scenario_name,
            metric="P99 Latency",
            value=p99_ms,
            valid=False,
            reason=f"P99 latency too low ({p99_ms}ms < {min_ms}ms) - possible empty dataset or measurement error",
            severity="ERROR"
        )

    if p99_ms > max_ms:
        return ValidationResult(
            scenario=scenario_name,
            metric="P99 Latency",
            value=p99_ms,
            valid=False,
            reason=f"P99 latency too high ({p99_ms}ms > {max_ms}ms) - severe performance issue",
            severity="ERROR"
        )

    # Check for suspiciously round numbers (indicates averaging artifact)
    if p99_ms == round(p99_ms) and p99_ms > 10:
        return ValidationResult(
            scenario=scenario_name,
            metric="P99 Latency",
            value=p99_ms,
            valid=True,
            reason=f"P99 latency is exact integer ({p99_ms}ms) - may indicate measurement quantization",
            severity="WARNING"
        )

    return ValidationResult(
        scenario=scenario_name,
        metric="P99 Latency",
        value=p99_ms,
        valid=True,
        reason=f"P99 latency within expected range ({min_ms}ms - {max_ms}ms)",
        severity="INFO"
    )

def validate_throughput_sanity(qps: float, scenario_name: str, duration_s: int) -> ValidationResult:
    """
    Validate throughput is in reasonable range.

    Sanity bounds based on operation type and hardware:
    - Single-threaded client: 100-2000 QPS (limited by client overhead)
    - Multi-threaded client: 500-50,000 QPS (limited by server capacity)

    Zero QPS: Scenario failed to execute any operations
    >100K QPS: Measurement error (timer resolution, counter overflow)
    """
    min_qps = 10  # Absolute minimum (1 operation per 100ms)
    max_qps = 100000  # Absolute maximum (single machine limit)

    # Expected QPS based on P99 latency bounds (inverse relationship)
    scenario_expected_qps = {
        "qdrant_ann_1m_768d": (500, 10000),
        "neo4j_traversal_100k": (1000, 20000),
        "hybrid_production_100k": (200, 5000),
        "milvus_ann_10m_768d": (50, 2000),
    }

    expected_min, expected_max = scenario_expected_qps.get(scenario_name, (min_qps, max_qps))

    if qps is None or qps <= 0:
        return ValidationResult(
            scenario=scenario_name,
            metric="Throughput",
            value=qps,
            valid=False,
            reason=f"Throughput is zero ({qps} QPS) - no operations completed",
            severity="ERROR"
        )

    if qps < expected_min:
        return ValidationResult(
            scenario=scenario_name,
            metric="Throughput",
            value=qps,
            valid=True,
            reason=f"Throughput lower than expected ({qps} QPS < {expected_min} QPS) - performance concern",
            severity="WARNING"
        )

    if qps > max_qps:
        return ValidationResult(
            scenario=scenario_name,
            metric="Throughput",
            value=qps,
            valid=False,
            reason=f"Throughput impossibly high ({qps} QPS > {max_qps} QPS) - measurement error",
            severity="ERROR"
        )

    # Validate total operations matches duration
    expected_total_ops = qps * duration_s
    # Allow ±20% variance due to startup/shutdown overhead
    min_total_ops = expected_total_ops * 0.8
    max_total_ops = expected_total_ops * 1.2

    return ValidationResult(
        scenario=scenario_name,
        metric="Throughput",
        value=qps,
        valid=True,
        reason=f"Throughput within expected range ({expected_min}-{expected_max} QPS)",
        severity="INFO"
    )

def validate_distribution_consistency(p50: float, p95: float, p99: float, scenario: str) -> ValidationResult:
    """
    Validate latency percentiles form a monotonic distribution.

    Required invariants:
    - P50 <= P95 <= P99 (strict monotonicity)
    - P99/P50 < 20 (tail latency not excessive)
    - P95/P50 < 5 (reasonable variance)

    Violations indicate measurement errors or severe bimodal distributions.
    """
    if not (p50 <= p95 <= p99):
        return ValidationResult(
            scenario=scenario,
            metric="Percentile Distribution",
            value=None,
            valid=False,
            reason=f"Percentiles not monotonic (P50={p50}ms, P95={p95}ms, P99={p99}ms)",
            severity="ERROR"
        )

    # Check for excessive tail latency (P99 >> P50)
    tail_ratio = p99 / p50 if p50 > 0 else float('inf')
    if tail_ratio > 20:
        return ValidationResult(
            scenario=scenario,
            metric="Percentile Distribution",
            value=tail_ratio,
            valid=True,
            reason=f"Excessive tail latency (P99/P50 = {tail_ratio:.1f}x) - investigate outliers",
            severity="WARNING"
        )

    # Check for suspiciously low variance (all percentiles equal)
    if p50 == p95 == p99:
        return ValidationResult(
            scenario=scenario,
            metric="Percentile Distribution",
            value=None,
            valid=True,
            reason="All percentiles identical - may indicate insufficient samples",
            severity="WARNING"
        )

    return ValidationResult(
        scenario=scenario,
        metric="Percentile Distribution",
        value=tail_ratio,
        valid=True,
        reason=f"Latency distribution consistent (P99/P50 = {tail_ratio:.1f}x)",
        severity="INFO"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate baseline measurement results")
    parser.add_argument("--timestamp", required=True, help="Benchmark timestamp (e.g., 2025-11-08_14-30-00)")
    args = parser.parse_args()

    results_dir = Path(f"tmp/competitive_benchmarks/{args.timestamp}")
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(2)

    # Validation results accumulator
    validations: List[ValidationResult] = []

    # Find all scenario result files
    scenario_files = list(results_dir.glob("*_loadtest.txt"))
    if len(scenario_files) == 0:
        print(f"ERROR: No scenario result files found in {results_dir}")
        sys.exit(2)

    print(f"\n=== Validating {len(scenario_files)} Scenario Results ===\n")

    for scenario_file in scenario_files:
        scenario_name = scenario_file.stem.replace("_loadtest", "")
        print(f"Validating {scenario_name}...")

        # Parse results (simplified - real implementation would parse actual format)
        # This is a placeholder showing the validation logic
        try:
            with open(scenario_file) as f:
                content = f.read()

                # Extract metrics (actual parsing would be more robust)
                import re
                p50_match = re.search(r"P50.*?(\d+\.?\d*)ms", content)
                p95_match = re.search(r"P95.*?(\d+\.?\d*)ms", content)
                p99_match = re.search(r"P99.*?(\d+\.?\d*)ms", content)
                qps_match = re.search(r"Throughput.*?(\d+\.?\d*)\s*(?:ops/s|QPS)", content)
                duration_match = re.search(r"Duration.*?(\d+)s", content)

                if not (p50_match and p95_match and p99_match and qps_match):
                    validations.append(ValidationResult(
                        scenario=scenario_name,
                        metric="Parsing",
                        value=None,
                        valid=False,
                        reason="Failed to parse required metrics from result file",
                        severity="ERROR"
                    ))
                    continue

                p50 = float(p50_match.group(1))
                p95 = float(p95_match.group(1))
                p99 = float(p99_match.group(1))
                qps = float(qps_match.group(1))
                duration = int(duration_match.group(1)) if duration_match else 60

                # Run validations
                validations.append(validate_latency_sanity(p99, scenario_name))
                validations.append(validate_throughput_sanity(qps, scenario_name, duration))
                validations.append(validate_distribution_consistency(p50, p95, p99, scenario_name))

        except Exception as e:
            validations.append(ValidationResult(
                scenario=scenario_name,
                metric="Parsing",
                value=None,
                valid=False,
                reason=f"Exception during validation: {str(e)}",
                severity="ERROR"
            ))

    # Summarize validation results
    print("\n=== Validation Summary ===\n")

    errors = [v for v in validations if v.severity == "ERROR"]
    warnings = [v for v in validations if v.severity == "WARNING"]

    for v in validations:
        icon = "✗" if v.severity == "ERROR" else "⚠" if v.severity == "WARNING" else "✓"
        print(f"{icon} {v.scenario} - {v.metric}: {v.reason}")

    print(f"\nTotal: {len(validations)} checks, {len(errors)} errors, {len(warnings)} warnings")

    if errors:
        print("\nVALIDATION FAILED: Errors detected in measurement results")
        print("Review errors above and re-run baseline measurement")
        sys.exit(1)

    if warnings:
        print("\nVALIDATION PASSED WITH WARNINGS: Results usable but investigate warnings")
        sys.exit(0)

    print("\nVALIDATION PASSED: All results within expected ranges")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

**Spot-Check Data Integrity**:
```bash
# Extract latest benchmark timestamp
TIMESTAMP=$(ls -t tmp/competitive_benchmarks/*_metadata.txt | head -1 | xargs basename | sed 's/_metadata.txt//')

# Run validation script
python3 scripts/validate_baseline_results.py --timestamp "$TIMESTAMP"

# Capture validation exit code
VALIDATION_EXIT_CODE=$?

if [ $VALIDATION_EXIT_CODE -eq 1 ]; then
    echo "ERROR: Baseline validation failed"
    echo "Results are invalid and should not be used for competitive comparison"
    exit 1
elif [ $VALIDATION_EXIT_CODE -eq 0 ] && [ -n "$(grep WARNING tmp/competitive_benchmarks/${TIMESTAMP}_validation.log)" ]; then
    echo "WARNING: Baseline validation passed with warnings"
    echo "Review warnings before accepting results"
fi

echo "PASS: Baseline validation successful"
```

### 4. Statistical Confidence Analysis

**Multi-Run Variance Assessment**:

For critical positioning claims (e.g., "Engram is faster than Neo4j"), validate statistical significance through repeated measurement:

```bash
# Run each critical scenario 3 times to assess variance
CRITICAL_SCENARIOS=("neo4j_traversal_100k" "qdrant_ann_1m_768d")

for scenario in "${CRITICAL_SCENARIOS[@]}"; do
    echo "Running variance assessment for $scenario (3 runs)..."

    for run in {1..3}; do
        echo "  Run $run/3..."
        cargo run --release --bin loadtest -- run \
            --scenario "scenarios/competitive/${scenario}.toml" \
            --duration 60 \
            --output "tmp/competitive_benchmarks/${TIMESTAMP}_${scenario}_run${run}.json"

        # Cool-down between runs to prevent thermal throttling
        sleep 30
    done

    # Compute coefficient of variation (CV) for P99 latency
    python3 <<EOF
import json
import statistics

runs = []
for i in range(1, 4):
    with open("tmp/competitive_benchmarks/${TIMESTAMP}_${scenario}_run{i}.json") as f:
        data = json.load(f)
        runs.append(data["p99_latency_ms"])

mean = statistics.mean(runs)
stdev = statistics.stdev(runs)
cv = (stdev / mean) * 100  # Coefficient of variation (%)

print(f"\\n{scenario} Variance Analysis:")
print(f"  P99 Latency: {runs}")
print(f"  Mean: {mean:.2f}ms")
print(f"  Std Dev: {stdev:.2f}ms")
print(f"  CV: {cv:.1f}%")

if cv > 5.0:
    print(f"  WARNING: High variance (CV={cv:.1f}% > 5%) - results may be unstable")
    print(f"  Recommendation: Investigate thermal throttling, background processes")
elif cv > 2.0:
    print(f"  ACCEPTABLE: Moderate variance (CV={cv:.1f}%)")
else:
    print(f"  EXCELLENT: Low variance (CV={cv:.1f}% < 2%)")
EOF
done
```

**Confidence Interval Calculation**:

For each metric reported, compute 95% confidence interval to quantify measurement uncertainty:

```python
def calculate_confidence_interval(samples: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for performance metric.

    Uses t-distribution for small samples (n < 30).

    Returns:
        (lower_bound, upper_bound) for the true population mean
    """
    from scipy import stats
    import statistics

    if len(samples) < 2:
        return (samples[0], samples[0])

    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples)
    n = len(samples)

    # Standard error of the mean
    sem = stdev / (n ** 0.5)

    # Critical value from t-distribution
    df = n - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)

    # Margin of error
    margin = t_crit * sem

    return (mean - margin, mean + margin)

# Example usage for Neo4j traversal comparison:
# Engram: [15.1ms, 15.3ms, 14.9ms] (3 runs)
# CI = (14.85ms, 15.35ms) at 95% confidence
# Neo4j baseline: 27.96ms
# Conclusion: Engram is 46.8% faster (95% CI: 45.1% - 48.3%)
```

### 5. Positioning Analysis with Statistical Rigor

**Classification Criteria**:

Use statistical hypothesis testing to classify positioning:

```python
def classify_competitive_position(
    engram_samples: List[float],
    competitor_baseline: float,
    metric_name: str,
    lower_is_better: bool = True,
    alpha: float = 0.05
) -> Tuple[str, bool, float]:
    """
    Classify competitive positioning with statistical confidence.

    Args:
        engram_samples: List of Engram measurements (3+ samples)
        competitor_baseline: Published competitor baseline
        metric_name: Name of metric for logging
        lower_is_better: True for latency, False for throughput
        alpha: Significance level (0.05 = 95% confidence)

    Returns:
        (status, is_significant, p_value)
        status: "Better", "Comparable", "Worse"
        is_significant: Whether difference is statistically significant
        p_value: Probability that observed difference is due to chance
    """
    from scipy import stats
    import statistics

    engram_mean = statistics.mean(engram_samples)
    engram_std = statistics.stdev(engram_samples) if len(engram_samples) > 1 else 0
    n = len(engram_samples)

    # One-sample t-test: H0: engram_mean == competitor_baseline
    if engram_std == 0:
        # Zero variance: use point estimate comparison
        delta_pct = ((engram_mean - competitor_baseline) / competitor_baseline) * 100
        if lower_is_better:
            delta_pct = -delta_pct

        if delta_pct > 10:
            return ("Better", True, 0.0)
        elif delta_pct < -10:
            return ("Worse", True, 0.0)
        else:
            return ("Comparable", False, 1.0)

    # Compute t-statistic
    t_stat = (engram_mean - competitor_baseline) / (engram_std / (n ** 0.5))

    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

    # Calculate percentage difference
    delta_pct = ((engram_mean - competitor_baseline) / competitor_baseline) * 100
    if lower_is_better:
        delta_pct = -delta_pct  # Invert for "lower is better" metrics

    # Classify based on delta magnitude and significance
    is_significant = p_value < alpha

    if delta_pct > 10 and is_significant:
        status = "Better"
    elif delta_pct < -10 and is_significant:
        status = "Worse"
    else:
        status = "Comparable"

    return (status, is_significant, p_value)

# Example: Neo4j Graph Traversal
# Engram P99: [15.1ms, 15.3ms, 14.9ms]
# Neo4j P99: 27.96ms
# Result: ("Better", True, 0.001) - 95%+ confidence Engram is faster
```

**Documentation Format**:

Update `docs/reference/competitive_baselines.md` with statistical metadata:

```markdown
## Engram Baseline (M17.1 - November 2025)

**Measurement Date**: 2025-11-08
**Commit**: a1b2c3d4e5f6 (milestone-17.1/competitive-baseline)
**Hardware**: M1 Max (10-core CPU, 32GB RAM), macOS 14.6
**Methodology**: 3 runs per scenario, 95% confidence intervals

| Workload | Engram P99 (95% CI) | Competitor P99 | Delta | Status | Significance |
|----------|---------------------|----------------|-------|--------|--------------|
| Graph Traversal (100K) | 15.1ms (14.9-15.4ms) | 27.96ms (Neo4j) | -45.9% | Better | p < 0.001 |
| ANN Search (1M 768d) | 26.2ms (25.8-26.6ms) | 23.5ms (Qdrant) | +11.5% | Worse | p = 0.032 |
| Hybrid Workload (100K) | 18.4ms (18.0-18.8ms) | N/A | N/A | Unique | N/A |
| ANN Search (10M 768d) | 142ms (138-146ms) | 708ms (Milvus) | -79.9% | Better | p < 0.001 |

**Legend**:
- **Better**: >10% improvement with p < 0.05 (95% confidence)
- **Comparable**: Within ±10% or not statistically significant
- **Worse**: >10% regression with p < 0.05 (95% confidence)
- **Unique**: No direct competitor equivalent

**Key Findings**:
1. Graph traversal is production-ready and best-in-class (46% faster than Neo4j)
2. Large-scale ANN search significantly outperforms Milvus (80% faster)
3. Small-scale ANN search lags Qdrant by 11.5% - optimization target for M18
4. Hybrid workload demonstrates competitive advantage (no comparable baseline)
```

### 6. Optimization Task Creation Framework

**Gap Analysis**:

Create optimization tasks when performance gaps meet threshold criteria:

```python
def should_create_optimization_task(
    status: str,
    delta_pct: float,
    is_significant: bool,
    metric_importance: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
) -> bool:
    """
    Determine if performance gap warrants optimization task creation.

    Thresholds:
    - CRITICAL metrics: Create task if >5% worse and significant
    - HIGH metrics: Create task if >15% worse and significant
    - MEDIUM metrics: Create task if >25% worse and significant
    - LOW metrics: Create task if >50% worse and significant

    Always create task if >100% worse (regardless of importance).
    """
    if status != "Worse":
        return False  # Only create tasks for regressions

    if not is_significant:
        return False  # Don't optimize based on statistical noise

    # Absolute delta (remove sign)
    abs_delta = abs(delta_pct)

    if abs_delta > 100:
        return True  # Severe regression always requires attention

    thresholds = {
        "CRITICAL": 5,
        "HIGH": 15,
        "MEDIUM": 25,
        "LOW": 50,
    }

    threshold = thresholds.get(metric_importance, 50)
    return abs_delta > threshold
```

**Task Specification Template**:

```markdown
# Task XXX: Optimize {Workload} {Metric}

**Status**: Pending
**Complexity**: {Simple|Moderate|Complex}
**Dependencies**: M17.1 Task 006 (competitive baseline)
**Estimated Effort**: {X} hours

## Motivation

M17.1 competitive baseline measurement identified a {X}% performance gap vs {Competitor}:

- **Engram {Metric}**: {value}ms (95% CI: {lower}-{upper}ms)
- **{Competitor} {Metric}**: {value}ms
- **Gap**: +{X}% slower (p < {p_value})
- **Market Impact**: {HIGH|MEDIUM|LOW}

This task aims to close the gap and achieve parity or better performance.

## Target

- **Baseline**: {current_value}ms (M17.1 measurement)
- **Target**: <{target_value}ms ({competitor baseline} × 0.95 for 5% buffer)
- **Stretch Target**: <{stretch_value}ms (10% better than competitor)

## Hypothesized Bottlenecks

Based on profiling data from baseline measurement:

1. **{Bottleneck 1}**: {description} (estimated {X}% overhead)
2. **{Bottleneck 2}**: {description} (estimated {Y}% overhead)
3. **{Bottleneck 3}**: {description} (estimated {Z}% overhead)

## Optimization Approach

### Phase 1: Profiling and Root Cause Analysis
- Run flamegraph on {scenario}: `cargo flamegraph --bin engram -- run-scenario {scenario}`
- Identify hot functions (>10% CPU time)
- Measure cache miss rates with `perf stat`
- Analyze memory allocation patterns with `heaptrack`

### Phase 2: Targeted Optimizations
- Optimize {specific_function} (identified in profiling)
- Consider {specific_technique} (e.g., SIMD, cache-friendly layout, lock-free structures)
- Validate each optimization in isolation with micro-benchmarks

### Phase 3: Validation
- Re-run competitive baseline measurement
- Verify {metric} improved by ≥{X}%
- Ensure no regression in other metrics (P50, P95, throughput)
- Confirm statistical significance (p < 0.05)

## Acceptance Criteria

1. {Metric} improves by ≥{X}% (measured with 3+ runs)
2. Improvement is statistically significant (p < 0.05)
3. No regression >5% in other metrics
4. Zero clippy warnings (`make quality` passes)
5. Updated baseline documentation reflects new performance

## References

- M17.1 baseline report: `tmp/competitive_benchmarks/{timestamp}_report.md`
- {Competitor} benchmark source: {URL}
- Profiling data: `tmp/competitive_benchmarks/{timestamp}_{scenario}_flamegraph.svg`
```

**Example Optimization Task Creation**:

```bash
# After baseline measurement, generate optimization tasks automatically
python3 <<EOF
import json

# Parse baseline comparison results
results = {
    "qdrant_ann_1m_768d": {
        "engram_p99": 26.2,
        "competitor_p99": 23.5,
        "status": "Worse",
        "delta_pct": 11.5,
        "p_value": 0.032,
        "importance": "HIGH"  # Core vector search capability
    },
    "neo4j_traversal_100k": {
        "engram_p99": 15.1,
        "competitor_p99": 27.96,
        "status": "Better",
        "delta_pct": -45.9,
        "p_value": 0.001,
        "importance": "HIGH"
    }
}

task_number = 1
for scenario, data in results.items():
    if data["status"] == "Worse" and abs(data["delta_pct"]) > 10 and data["p_value"] < 0.05:
        print(f"Creating optimization task for {scenario}:")
        print(f"  Gap: {data['delta_pct']:+.1f}%")
        print(f"  Target: <{data['competitor_p99']:.1f}ms (parity)")
        print(f"  File: roadmap/milestone-18/{task_number:03d}_optimize_{scenario}_pending.md")

        # Create task file (simplified - real implementation would use template)
        # with open(f"roadmap/milestone-18/{task_number:03d}_optimize_{scenario}_pending.md", "w") as f:
        #     f.write(generate_task_from_template(scenario, data))

        task_number += 1
EOF
```

### 7. Documentation Update Standards

**Required Metadata for Reproducibility**:

Every baseline measurement must capture:

```yaml
# tmp/competitive_benchmarks/{timestamp}_metadata.yml

measurement:
  timestamp: 2025-11-08T14:30:00Z
  duration_minutes: 12
  operator: jordan@engram.dev

environment:
  os: macOS 14.6.1 (Darwin 23.6.0)
  cpu: Apple M1 Max (10-core, 3.2GHz)
  ram: 32GB LPDDR5
  disk: NVMe SSD (2TB, 3.5GB/s read)
  thermal_state: nominal  # no throttling detected

software:
  engram_version: 0.1.0
  git_commit: a1b2c3d4e5f6
  git_branch: milestone-17.1/competitive-baseline
  git_dirty: false
  rust_version: 1.83.0
  loadtest_version: 0.1.0
  optimization_level: release

configuration:
  hot_tier_size_gb: 4
  warm_tier_size_gb: 12
  simd: AVX2
  thread_pools:
    recall_workers: 16
    store_workers: 8

scenarios_executed:
  - name: qdrant_ann_1m_768d
    file: scenarios/competitive/qdrant_ann_1m_768d.toml
    sha256: <file_hash>
    duration_s: 60
    seed: 42
    exit_code: 0
  - name: neo4j_traversal_100k
    file: scenarios/competitive/neo4j_traversal_100k.toml
    sha256: <file_hash>
    duration_s: 60
    seed: 43
    exit_code: 0
  # ... (repeat for all scenarios)

validation:
  script: scripts/validate_baseline_results.py
  exit_code: 0
  errors: 0
  warnings: 1
  warnings_detail:
    - "Neo4j scenario: Excessive tail latency (P99/P50 = 12.3x)"

statistical_confidence:
  runs_per_scenario: 3
  confidence_level: 0.95
  max_coefficient_of_variation: 5.0
```

**Commit Message Format**:

```
feat(m17.1): Complete Task 006 - Initial Competitive Baseline Measurement

Executed first competitive baseline with rigorous statistical validation:

Results Summary:
- Neo4j Graph Traversal: 15.1ms P99 vs 27.96ms (-45.9%, p<0.001) ✓ BETTER
- Qdrant ANN Search: 26.2ms P99 vs 23.5ms (+11.5%, p=0.032) ✗ WORSE
- Hybrid Workload: 18.4ms P99 (no competitor baseline) ⓘ UNIQUE
- Milvus Large ANN: 142ms P99 vs 708ms (-79.9%, p<0.001) ✓ BETTER

Optimization Tasks Created:
- M18/001: Optimize ANN search to match Qdrant (<23ms target)

Validation Status: PASSED (0 errors, 1 warning)
Statistical Confidence: 95% (3 runs per scenario)
Measurement Timestamp: 2025-11-08_14-30-00

Files updated:
- docs/reference/competitive_baselines.md (added Engram M17.1 section)
- roadmap/milestone-18/001_optimize_qdrant_ann_pending.md (created)
```

## File Paths

```
scripts/validate_baseline_results.py              # Created (validation script)
tmp/competitive_benchmarks/<timestamp>/           # Generated (results directory)
tmp/competitive_benchmarks/<timestamp>_metadata.yml  # Generated (run metadata)
tmp/competitive_benchmarks/<timestamp>_report.md  # Generated (comparison report)
docs/reference/competitive_baselines.md           # Updated (Engram baseline section)
roadmap/milestone-18/0XX_optimize_*_pending.md    # Created (if gaps found)
```

## Enhanced Acceptance Criteria

1. **Execution Success**:
   - Baseline measurement completes without errors (exit code 0)
   - All 4 scenarios produce valid results files
   - System monitoring detects no resource exhaustion (OOM, thermal throttling)

2. **Data Integrity**:
   - All P99 latencies within reasonable bounds (validation script passes)
   - Throughput values > 0 and < 100K QPS (sanity bounds)
   - Percentile distributions are monotonic (P50 ≤ P95 ≤ P99)
   - Total operations match expected count (±20% tolerance)

3. **Statistical Confidence**:
   - Multi-run variance assessment shows CV < 5% for critical scenarios
   - 95% confidence intervals computed and documented
   - Positioning claims backed by hypothesis testing (p-values reported)

4. **Competitive Positioning**:
   - At least 1 scenario classified as "Better" with statistical significance
   - No more than 1 scenario classified as "Worse" by >50%
   - Report clearly identifies Engram's strengths and optimization priorities

5. **Documentation Quality**:
   - Baseline section added to `competitive_baselines.md` with full metadata
   - All measurements include 95% confidence intervals
   - Reproducibility metadata captured (commit hash, system specs, configuration)
   - Statistical significance documented for all positioning claims

6. **Optimization Task Creation**:
   - Optimization tasks created for all gaps >15% (HIGH importance metrics)
   - Tasks include target metrics, profiling data references, and acceptance criteria
   - Task complexity estimated based on gap magnitude and bottleneck analysis

## Testing Approach

### Phase 1: Pre-Flight Validation

```bash
# System requirements check
python3 scripts/validate_baseline_results.py --pre-flight

# Code quality verification
make quality

# Binary verification
cargo build --release --bin loadtest
ls -lh target/release/loadtest  # Should show optimized binary size
```

### Phase 2: Execute Baseline Measurement

```bash
# Run full competitive benchmark suite
./scripts/quarterly_competitive_review.sh

# Extract timestamp
TIMESTAMP=$(ls -t tmp/competitive_benchmarks/*_metadata.txt | head -1 | xargs basename | sed 's/_metadata.txt//')

# Verify all scenarios completed
ls tmp/competitive_benchmarks/${TIMESTAMP}_*.txt | wc -l
# Should be 13 files (4 scenarios × 3 outputs + 1 metadata)
```

### Phase 3: Validate Results

```bash
# Run validation script
python3 scripts/validate_baseline_results.py --timestamp "$TIMESTAMP"

# Check validation exit code
if [ $? -eq 0 ]; then
    echo "✓ Validation passed"
else
    echo "✗ Validation failed - review errors and re-run"
    exit 1
fi
```

### Phase 4: Multi-Run Variance Assessment (Critical Scenarios Only)

```bash
# Run Neo4j and Qdrant scenarios 3 times each
for scenario in neo4j_traversal_100k qdrant_ann_1m_768d; do
    for run in {1..3}; do
        cargo run --release --bin loadtest -- run \
            --scenario "scenarios/competitive/${scenario}.toml" \
            --duration 60 \
            --output "tmp/competitive_benchmarks/${TIMESTAMP}_${scenario}_run${run}.json"
        sleep 30  # Cool-down
    done
done

# Compute variance statistics
python3 scripts/analyze_variance.py --timestamp "$TIMESTAMP"
```

### Phase 5: Generate Comparison Report

```bash
# Generate statistical comparison report
python3 scripts/generate_competitive_report.py \
    --input "tmp/competitive_benchmarks/${TIMESTAMP}" \
    --output "tmp/competitive_benchmarks/${TIMESTAMP}_report.md" \
    --confidence 0.95 \
    --verbose

# Review report
less "tmp/competitive_benchmarks/${TIMESTAMP}_report.md"
```

### Phase 6: Update Documentation

```bash
# Extract baseline data from report
python3 scripts/extract_baseline_for_docs.py \
    --report "tmp/competitive_benchmarks/${TIMESTAMP}_report.md" \
    --output-section docs/reference/competitive_baselines_engram_section.md

# Manually merge into competitive_baselines.md
# (Or use automated merge if section marker exists)

# Verify documentation is valid markdown
npx markdownlint-cli2 docs/reference/competitive_baselines.md
```

### Phase 7: Create Optimization Tasks (If Needed)

```bash
# Identify gaps requiring optimization
python3 scripts/create_optimization_tasks.py \
    --report "tmp/competitive_benchmarks/${TIMESTAMP}_report.md" \
    --threshold 15 \
    --output-dir roadmap/milestone-18

# Review generated tasks
ls -la roadmap/milestone-18/*_optimize_*_pending.md
```

## Integration Points

- **Executes**: Task 005 quarterly review workflow script
- **Validates with**: Task 004 report generator (statistical analysis)
- **Documents in**: Task 002 baseline documentation (Engram section)
- **Uses scenarios from**: Task 001 competitive scenario suite
- **May trigger**: M18 optimization task creation (if gaps found)

## Success Criteria Summary

Task is complete when:

1. Baseline measurement executes successfully (0 errors, 0-1 warnings)
2. Validation script confirms data integrity (all sanity checks pass)
3. Multi-run variance assessment shows CV < 5% (statistical stability)
4. Comparison report generated with 95% confidence intervals
5. Competitive positioning documented with p-values (statistical rigor)
6. At least 1 "Better" classification with significance (p < 0.05)
7. Optimization tasks created for all significant gaps >15%
8. Documentation updated with reproducibility metadata
9. Commit message includes performance summary and statistical confidence

## References

- Statistical Hypothesis Testing: Welch's t-test, Student's t-test
- Confidence Intervals: Student's t-distribution for small samples
- Effect Size: Cohen's d for practical significance
- Multiple Comparisons: Bonferroni correction (if testing >5 scenarios)
- Coefficient of Variation: https://en.wikipedia.org/wiki/Coefficient_of_variation
- Existing validation patterns: `tools/loadtest/src/hypothesis_testing.rs`
