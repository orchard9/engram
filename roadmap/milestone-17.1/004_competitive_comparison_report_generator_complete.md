# Task 004: Competitive Comparison Report Generator (Enhanced)

**Status**: Pending
**Complexity**: Complex
**Dependencies**: Task 003 (requires benchmark results), Task 002 (requires baseline documentation)
**Estimated Effort**: 8 hours

## Objective

Create a production-hardened Python script that parses benchmark results from the competitive suite runner, performs statistical analysis comparing against documented competitor baselines, and generates actionable markdown reports with clear performance positioning and optimization priorities.

## Enhanced Specifications

Create `scripts/generate_competitive_report.py` with the following behavior:

### 1. Robust Input Parsing

**Loadtest Results Parsing**:
- Read all `tmp/competitive_benchmarks/<timestamp>_*.txt` files for a given timestamp
- Extract metrics with comprehensive error handling:
  - P50/P95/P99 latency (milliseconds)
  - Throughput (operations per second)
  - Error rate (percentage)
  - Total operations executed
  - Test duration
- **Graceful Degradation**:
  - Handle partially written files (interrupted runs)
  - Handle missing fields (use `None` as sentinel, not `0`)
  - Handle malformed JSON (skip with warning, don't crash)
  - Handle unexpected units (detect ns vs ms, auto-convert)
  - Handle negative values (detect measurement errors, flag as invalid)
  - Handle division by zero in delta calculations
  - Handle missing scenario results (mark as "NOT_RUN" in report)

**Metadata Parsing**:
- Read `tmp/competitive_benchmarks/<timestamp>_metadata.txt`
- Extract with validation:
  - Git commit hash (verify 40-char hex or abbreviated 7-char)
  - Branch name (sanitize for markdown)
  - System information (OS, CPU, RAM)
  - Engram version (parse from Cargo.toml metadata)
  - Benchmark timestamp (parse ISO 8601)
- **Error Handling**:
  - Missing metadata file: Use fallback values ("UNKNOWN" commit, current timestamp)
  - Malformed fields: Log warning, use sentinel values
  - Git dirty state: Include warning in report

**Baseline Documentation Parsing**:
- Parse markdown table from `docs/reference/competitive_baselines.md`
- Extract competitor baselines with validation:
  - System name, workload type
  - P99 latency, throughput, recall rate
  - Dataset size, source URL
- **Error Handling**:
  - Missing baseline file: Fatal error (cannot compare)
  - Malformed table: Attempt fuzzy parsing, warn on failures
  - Missing competitors: Mark as "NO_BASELINE" in comparisons
  - Invalid numeric values: Skip row with warning

**Data Validation Layer**:
```python
@dataclass
class ValidationResult:
    """Result of data validation with detailed error context."""
    valid: bool
    warnings: List[str]
    errors: List[str]
    sanitized_value: Optional[Any]

def validate_latency(value: Any, metric_name: str) -> ValidationResult:
    """Validate latency metric (must be positive, reasonable magnitude)."""
    # Handle None, negative, zero, infinity, NaN, unreasonable values (>10s)
    # Return sanitized value or None with detailed error messages

def validate_throughput(value: Any, metric_name: str) -> ValidationResult:
    """Validate throughput metric (must be positive, non-zero)."""
    # Similar validation logic for QPS

def validate_percentage(value: Any, metric_name: str) -> ValidationResult:
    """Validate percentage metric (must be 0-100 or 0.0-1.0)."""
    # Detect range and normalize to 0-100
```

### 2. Statistical Analysis with Rigor

**Percentage Delta Calculation**:
```python
def calculate_delta(engram_value: float, competitor_value: float,
                     metric_name: str) -> Optional[float]:
    """
    Calculate percentage delta with proper error handling.

    Args:
        engram_value: Engram measurement
        competitor_value: Competitor baseline
        metric_name: Metric name for error messages

    Returns:
        Delta as percentage (-100 to +inf), or None if invalid

    Examples:
        calculate_delta(15.12, 27.96, "P99") -> -45.99%  # 46% faster
        calculate_delta(27.96, 15.12, "P99") -> +84.92%  # 85% slower
        calculate_delta(10.0, 0.0, "P99") -> None  # Division by zero
    """
    if competitor_value is None or engram_value is None:
        return None

    if competitor_value == 0:
        # Handle division by zero gracefully
        logger.warning(f"Cannot calculate delta for {metric_name}: "
                       f"competitor baseline is zero")
        return None

    # For latency: negative delta is improvement (lower is better)
    # For throughput: positive delta is improvement (higher is better)
    delta = ((engram_value - competitor_value) / competitor_value) * 100
    return delta
```

**Statistical Significance Testing**:
```python
def calculate_confidence_interval(mean: float, std: float, n: int,
                                   confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for performance metrics.

    Uses t-distribution for small samples (n < 30), normal distribution otherwise.
    Handles insufficient sample size (n < 2) gracefully.

    Returns:
        (lower_bound, upper_bound) or (None, None) if cannot calculate
    """
    from scipy import stats

    if n < 2:
        logger.warning(f"Cannot calculate CI: insufficient samples (n={n})")
        return (None, None)

    if std == 0:
        # Zero variance: CI is point estimate
        return (mean, mean)

    # Standard error of the mean
    sem = std / np.sqrt(n)

    # Choose distribution
    if n < 30:
        # Use t-distribution for small samples
        df = n - 1
        critical_value = stats.t.ppf((1 + confidence) / 2, df)
    else:
        # Use normal distribution for large samples
        critical_value = stats.norm.ppf((1 + confidence) / 2)

    margin = critical_value * sem
    return (mean - margin, mean + margin)

def is_statistically_significant(engram_mean: float, engram_std: float, engram_n: int,
                                  competitor_mean: float, alpha: float = 0.05) -> bool:
    """
    Test if difference from competitor is statistically significant.

    Uses one-sample t-test comparing Engram samples to competitor baseline mean.
    Null hypothesis: Engram mean equals competitor mean.

    Returns:
        True if p-value < alpha (reject null hypothesis)
    """
    from scipy import stats

    if engram_n < 2:
        return False  # Cannot test with insufficient samples

    # One-sample t-test
    t_stat = (engram_mean - competitor_mean) / (engram_std / np.sqrt(engram_n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), engram_n - 1))

    return p_value < alpha
```

**Outlier Detection**:
```python
def detect_outliers_iqr(samples: List[float]) -> Tuple[List[int], float, float]:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Returns:
        (outlier_indices, lower_bound, upper_bound)
    """
    if len(samples) < 4:
        return ([], float('-inf'), float('inf'))

    q1 = np.percentile(samples, 25)
    q3 = np.percentile(samples, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = [i for i, x in enumerate(samples) if x < lower or x > upper]
    return (outliers, lower, upper)
```

**Historical Trend Analysis**:
```python
def analyze_trend(historical_results: List[Tuple[datetime, float]],
                   metric_name: str) -> Dict[str, Any]:
    """
    Analyze historical trend for a metric using linear regression.

    Args:
        historical_results: List of (timestamp, metric_value) tuples
        metric_name: Name of metric for logging

    Returns:
        {
            "slope": float,  # Change per day
            "r_squared": float,  # Goodness of fit (0-1)
            "trend": str,  # "IMPROVING", "DEGRADING", "STABLE"
            "projection_30d": float  # Projected value in 30 days
        }
    """
    from scipy import stats

    if len(historical_results) < 2:
        return {"trend": "INSUFFICIENT_DATA"}

    # Convert to days since first measurement
    times = [(t - historical_results[0][0]).total_seconds() / 86400
             for t, _ in historical_results]
    values = [v for _, v in historical_results]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(times, values)
    r_squared = r_value ** 2

    # Classify trend
    if abs(slope) < 0.01:  # Less than 1% change per day
        trend = "STABLE"
    elif slope < 0:
        trend = "IMPROVING"
    else:
        trend = "DEGRADING"

    # Project 30 days forward
    projection = intercept + slope * (times[-1] + 30)

    return {
        "slope": slope,
        "r_squared": r_squared,
        "trend": trend,
        "projection_30d": projection,
        "p_value": p_value,
        "std_err": std_err
    }
```

### 3. Enhanced Report Generation

**Output File**: `tmp/competitive_benchmarks/<timestamp>_report.md`

**Report Structure**:

1. **Executive Summary** (Actionable Insights):
   ```markdown
   # Competitive Performance Report

   **Generated**: 2025-11-08 14:30:00 UTC
   **Engram Commit**: a1b2c3d (milestone-17/dual-memory-architecture)
   **System**: M1 Max (10-core, 32GB RAM), macOS 14.6

   ## Overall Positioning

   Engram outperforms Neo4j by 46% on graph traversal (P99: 15.12ms vs 27.96ms).
   Engram lags Qdrant by 12% on ANN search (P99: 26.4ms vs 23.5ms).
   Hybrid workload demonstrates unique competitive advantage (no direct baseline).

   **Key Insights**:
   - ✓ Graph traversal is production-ready and best-in-class
   - ⚠ ANN search needs optimization (target: <20ms P99)
   - ✓ Hybrid workload shows strong unified API performance
   ```

2. **Detailed Comparison Table**:
   ```markdown
   ## Detailed Results

   | Scenario | Metric | Engram | Competitor | Delta | Status | Significance |
   |----------|--------|--------|------------|-------|--------|--------------|
   | Neo4j Traversal | P99 Latency | 15.12ms | 27.96ms (Neo4j) | -45.99% | ✓ Better | p < 0.01 |
   | Neo4j Traversal | Throughput | 1,024 QPS | 280 QPS (Neo4j) | +265.71% | ✓ Better | p < 0.001 |
   | Qdrant ANN | P99 Latency | 26.4ms | 23.5ms (Qdrant) | +12.34% | ⚠ Worse | p < 0.05 |
   | Qdrant ANN | Recall Rate | 99.2% | 99.5% (Qdrant) | -0.30% | ≈ Comparable | p > 0.05 |
   | Hybrid Workload | P99 Latency | 18.7ms | N/A | N/A | ⓘ No Baseline | N/A |
   ```

   **Legend**:
   - ✓ Better: >10% improvement over competitor
   - ≈ Comparable: Within ±10% of competitor
   - ⚠ Worse: >10% regression vs competitor
   - ⓘ No Baseline: No competitor offers this workload

3. **ASCII Visualization** (Optional but Recommended):
   ```markdown
   ## Performance Comparison (P99 Latency)

   Neo4j Graph Traversal:
     Neo4j:  ████████████████████████████ 27.96ms
     Engram: ███████████████ 15.12ms (-46% ✓)

   Qdrant ANN Search:
     Qdrant: ███████████████████████ 23.5ms
     Engram: ██████████████████████████ 26.4ms (+12% ⚠)

   Hybrid Production Workload:
     Engram: ██████████████████ 18.7ms (unique capability)
   ```

4. **Optimization Priorities** (Data-Driven):
   ```markdown
   ## Optimization Priorities

   Based on regression magnitude and competitive gaps, prioritize:

   1. **ANN Search P99 Latency** (Qdrant scenario)
      - Current: 26.4ms, Target: <20ms (Qdrant: 23.5ms)
      - Gap: +12.34% slower than best-in-class
      - Impact: High (core vector search use case)
      - Suggested optimizations:
        - Profile HNSW index traversal (likely cache misses)
        - Consider SIMD for distance calculations
        - Review memory layout for better cache locality

   2. **Milvus Large-Scale ANN** (10M vectors)
      - Current: 142ms, Target: <100ms (Milvus: 708ms)
      - Status: Already 5x faster, but target <100ms for production
      - Impact: Medium (scalability differentiator)
   ```

5. **Historical Trends** (if `--compare-to` provided):
   ```markdown
   ## Performance Trends

   Comparing to previous baseline (2025-08-08):

   | Metric | Q2 2025 | Q3 2025 | Change | Trend |
   |--------|---------|---------|--------|-------|
   | Neo4j P99 | 18.3ms | 15.12ms | -17.38% ↓ | IMPROVING |
   | Qdrant P99 | 24.1ms | 26.4ms | +9.54% ↑ | DEGRADING |
   | Throughput | 920 QPS | 1,024 QPS | +11.30% ↑ | IMPROVING |

   **Analysis**:
   - Graph traversal improved significantly (likely M17 dual-memory architecture)
   - ANN search regressed slightly (investigate if M17 changes impacted HNSW)
   - Overall throughput trending upward (2.3% per month on average)
   ```

6. **Metadata and Reproducibility**:
   ```markdown
   ## Measurement Metadata

   **Environment**:
   - OS: macOS 14.6 (Darwin 23.6.0)
   - CPU: Apple M1 Max (10-core, 3.2GHz)
   - RAM: 32GB LPDDR5
   - Disk: NVMe SSD (2TB)

   **Software Versions**:
   - Engram Core: 0.1.0 (commit: a1b2c3d4)
   - Loadtest: 0.1.0
   - Rust: 1.83.0

   **Benchmark Parameters**:
   - Duration: 60s per scenario
   - Cooldown: 30s between scenarios
   - Seed: Deterministic (42-45)
   - Timestamp: 2025-11-08_14-30-00

   **Reproducibility**:
   ```bash
   git checkout a1b2c3d4
   cargo build --release
   ./scripts/competitive_benchmark_suite.sh
   ```
   ```

### 4. Historical Tracking and Comparison

**Usage**:
```bash
# Generate report with historical comparison
python3 scripts/generate_competitive_report.py \
    --input tmp/competitive_benchmarks/2025-11-08_14-30-00 \
    --compare-to tmp/competitive_benchmarks/2025-08-08_10-00-00 \
    --output tmp/competitive_benchmarks/2025-11-08_14-30-00_report.md
```

**Historical Comparison Logic**:
- Load both current and previous benchmark results
- Calculate delta for each metric (current vs previous)
- Perform trend analysis if 3+ historical points available
- Detect regressions relative to previous quarter
- Generate time-series visualization (ASCII sparklines)

**Trend Detection**:
```python
def classify_trend_change(current_delta: float, historical_delta: float,
                           metric_name: str) -> str:
    """
    Classify whether trend is accelerating, decelerating, or stable.

    Args:
        current_delta: Delta from competitor in current run (%)
        historical_delta: Delta from competitor in previous run (%)
        metric_name: Name of metric

    Returns:
        "ACCELERATING", "DECELERATING", "STABLE"
    """
    # For latency (lower is better):
    #   current=-45%, historical=-30% -> ACCELERATING (getting better faster)
    #   current=-45%, historical=-50% -> DECELERATING (improvement slowing)
    #
    # For throughput (higher is better): invert logic

    change = current_delta - historical_delta

    if abs(change) < 5:
        return "STABLE"

    # Latency logic (negative delta is good)
    if "latency" in metric_name.lower():
        return "ACCELERATING" if change < 0 else "DECELERATING"

    # Throughput logic (positive delta is good)
    return "ACCELERATING" if change > 0 else "DECELERATING"
```

### 5. Comprehensive Error Handling

**Error Severity Levels**:
```python
class ErrorSeverity(Enum):
    """Error severity classification for report generation."""
    FATAL = "FATAL"          # Cannot proceed (missing baseline file)
    ERROR = "ERROR"          # Major issue (scenario failed to run)
    WARNING = "WARNING"      # Minor issue (malformed field, using default)
    INFO = "INFO"            # Informational (missing optional metadata)

@dataclass
class ReportError:
    """Structured error for report generation."""
    severity: ErrorSeverity
    phase: str  # "PARSING", "VALIDATION", "ANALYSIS", "GENERATION"
    message: str
    context: Dict[str, Any]  # Additional context for debugging
    timestamp: datetime = field(default_factory=datetime.now)
```

**Error Handling Strategy**:
1. **Collect, Don't Crash**: Accumulate all errors and warnings, then decide
2. **Context Preservation**: Always include file path, line number, field name
3. **Graceful Degradation**: Generate partial report even with missing data
4. **Clear Exit Codes**:
   - 0: Success (report generated without errors)
   - 1: Partial success (warnings present, report generated)
   - 2: Failure (fatal error, no report generated)
   - 3: Validation failure (data integrity issues)

**Error Logging**:
```python
import logging
from typing import List

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("competitive_report")

class ErrorCollector:
    """Collect errors during report generation."""

    def __init__(self):
        self.errors: List[ReportError] = []

    def add(self, severity: ErrorSeverity, phase: str, message: str, **context):
        """Add error with structured context."""
        error = ReportError(
            severity=severity,
            phase=phase,
            message=message,
            context=context
        )
        self.errors.append(error)

        # Log immediately
        if severity == ErrorSeverity.FATAL:
            logger.error(f"[{phase}] {message}", extra=context)
        elif severity == ErrorSeverity.ERROR:
            logger.error(f"[{phase}] {message}", extra=context)
        elif severity == ErrorSeverity.WARNING:
            logger.warning(f"[{phase}] {message}", extra=context)
        else:
            logger.info(f"[{phase}] {message}", extra=context)

    def has_fatal(self) -> bool:
        """Check if any fatal errors occurred."""
        return any(e.severity == ErrorSeverity.FATAL for e in self.errors)

    def get_summary(self) -> Dict[str, int]:
        """Get count by severity."""
        from collections import Counter
        return Counter(e.severity for e in self.errors)
```

### 6. Code Quality and Testing

**Type Hints**: Full type annotations for all functions
```python
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats

@dataclass
class BenchmarkResult:
    """Parsed result from loadtest output."""
    scenario_name: str
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]
    throughput_qps: Optional[float]
    error_rate_pct: Optional[float]
    total_operations: Optional[int]
    duration_seconds: Optional[int]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompetitorBaseline:
    """Parsed competitor baseline from documentation."""
    system_name: str
    workload_type: str
    p99_latency_ms: Optional[float]
    throughput_qps: Optional[float]
    recall_rate_pct: Optional[float]
    dataset_size: str
    source_url: str

@dataclass
class ComparisonResult:
    """Result of comparing Engram vs competitor."""
    scenario_name: str
    competitor_name: str
    engram_p99: Optional[float]
    competitor_p99: Optional[float]
    latency_delta_pct: Optional[float]
    engram_qps: Optional[float]
    competitor_qps: Optional[float]
    throughput_delta_pct: Optional[float]
    status: str  # "BETTER", "COMPARABLE", "WORSE", "NO_BASELINE"
    statistical_significance: Optional[bool]
```

**Docstrings**: Google-style docstrings for all public functions
```python
def parse_loadtest_results(result_file: Path) -> Optional[BenchmarkResult]:
    """
    Parse loadtest output file to extract performance metrics.

    Args:
        result_file: Path to loadtest output file (*.txt)

    Returns:
        BenchmarkResult if parsing succeeds, None otherwise

    Raises:
        FileNotFoundError: If result_file does not exist
        PermissionError: If result_file is not readable

    Example:
        >>> result = parse_loadtest_results(Path("tmp/2025-11-08_qdrant.txt"))
        >>> print(f"P99: {result.p99_latency_ms}ms")
        P99: 26.4ms
    """
    # Implementation...
```

**Unit Testing**:
```bash
# Create test fixtures
mkdir -p tmp/test_competitive_report/fixtures

# Test: Parse valid loadtest output
python3 -m pytest tests/test_competitive_report.py::test_parse_valid_loadtest -v

# Test: Handle malformed JSON gracefully
python3 -m pytest tests/test_competitive_report.py::test_parse_malformed_json -v

# Test: Calculate delta with division by zero
python3 -m pytest tests/test_competitive_report.py::test_delta_division_by_zero -v

# Test: Statistical significance calculation
python3 -m pytest tests/test_competitive_report.py::test_statistical_significance -v

# Test: ASCII chart generation
python3 -m pytest tests/test_competitive_report.py::test_ascii_chart_rendering -v

# Full test suite
python3 -m pytest tests/test_competitive_report.py --cov=scripts.generate_competitive_report
```

**Linting and Formatting**:
```bash
# Type checking with mypy
mypy scripts/generate_competitive_report.py --strict

# Linting with ruff
ruff check scripts/generate_competitive_report.py

# Auto-formatting with black
black scripts/generate_competitive_report.py --line-length 100

# Import sorting with isort
isort scripts/generate_competitive_report.py
```

## File Paths

```
scripts/generate_competitive_report.py       # Main script
tests/test_competitive_report.py             # Unit tests
tmp/test_competitive_report/fixtures/        # Test fixtures
tmp/competitive_benchmarks/<timestamp>_report.md  # Generated report
```

## Enhanced Acceptance Criteria

1. **Functionality**:
   - Script parses all 4 competitive scenario results correctly (100% success rate)
   - Markdown report is human-readable and ready for quarterly review
   - Baseline comparison calculations verified manually for all scenarios
   - Historical comparison works correctly with 2+ data points
   - Statistical significance testing produces correct p-values (validated against scipy)

2. **Performance**:
   - Report generation completes in <10 seconds for 4 scenarios
   - Memory usage <500MB during parsing and analysis
   - No performance degradation with 10+ historical data points

3. **Robustness**:
   - Handles missing baseline data gracefully (warns, doesn't crash)
   - Handles missing scenario results gracefully (marks as NOT_RUN)
   - Handles malformed JSON (skips with warning)
   - Handles division by zero (returns None, logs warning)
   - Handles negative latencies (flags as measurement error)
   - Handles infinite/NaN values (sanitizes to None)

4. **Code Quality**:
   - Zero warnings from `ruff check` (strict mode)
   - Zero errors from `mypy --strict`
   - 100% type annotation coverage
   - 90%+ test coverage (measured with pytest-cov)
   - All public functions have Google-style docstrings
   - Code formatted with black (line length 100)

5. **Observability**:
   - Clear logging at INFO level for progress
   - Detailed logging at DEBUG level for debugging
   - Structured error messages with file/line context
   - Exit codes clearly documented and tested

## Testing Approach

### Phase 1: Syntax and Type Validation

```bash
# Validate Python syntax
python3 -m py_compile scripts/generate_competitive_report.py

# Type checking (strict mode)
mypy scripts/generate_competitive_report.py --strict --show-error-codes

# Linting (zero warnings allowed)
ruff check scripts/generate_competitive_report.py --select ALL

# Auto-formatting check
black scripts/generate_competitive_report.py --check --line-length 100
```

**Expected Output**: Zero errors, zero warnings.

### Phase 2: Unit Testing with Fixtures

```bash
# Create test fixtures
mkdir -p tmp/test_competitive_report/fixtures/2025-11-08_12-00-00

# Mock loadtest results (valid)
cat > tmp/test_competitive_report/fixtures/2025-11-08_12-00-00/qdrant_ann_1m_768d.txt << 'EOF'
Scenario: qdrant_ann_1m_768d
Duration: 60s
Total operations: 52,000
Error rate: 0.02%

P50 latency: 18.2ms
P95 latency: 24.1ms
P99 latency: 26.4ms
Throughput: 867 ops/sec
EOF

# Mock loadtest results (malformed - missing P99)
cat > tmp/test_competitive_report/fixtures/2025-11-08_12-00-00/neo4j_traversal_100k.txt << 'EOF'
Scenario: neo4j_traversal_100k
Duration: 60s
Total operations: 61,440

P50 latency: 10.2ms
P95 latency: 13.8ms
Throughput: 1,024 ops/sec
EOF

# Mock metadata
cat > tmp/test_competitive_report/fixtures/2025-11-08_12-00-00/metadata.txt << 'EOF'
Git commit: a1b2c3d4e5f6
Branch: milestone-17/dual-memory-architecture
OS: macOS 14.6
CPU: M1 Max (10-core)
RAM: 32GB
Engram version: 0.1.0
Timestamp: 2025-11-08T12:00:00Z
EOF

# Run unit tests
python3 -m pytest tests/test_competitive_report.py -v --cov=scripts.generate_competitive_report

# Expected output:
# - test_parse_valid_loadtest: PASS
# - test_parse_malformed_loadtest: PASS (handles missing P99 gracefully)
# - test_parse_metadata: PASS
# - test_calculate_delta_normal: PASS
# - test_calculate_delta_division_by_zero: PASS
# - test_statistical_significance: PASS
# - test_ascii_chart_rendering: PASS
# Coverage: 90%+
```

### Phase 3: Integration Testing with Real Data

```bash
# Run full competitive benchmark suite (generates real data)
./scripts/competitive_benchmark_suite.sh

# Extract timestamp
TIMESTAMP=$(ls -t tmp/competitive_benchmarks/*_metadata.txt | head -1 | xargs basename | cut -d_ -f1-3)

# Generate report
python3 scripts/generate_competitive_report.py \
    --input tmp/competitive_benchmarks/$TIMESTAMP \
    --output tmp/competitive_benchmarks/${TIMESTAMP}_report.md \
    --verbose

# Verify report structure
grep -E "Executive Summary|Detailed Results|Optimization Priorities|Historical Trends|Metadata" \
    tmp/competitive_benchmarks/${TIMESTAMP}_report.md

# Expected: All sections present
```

### Phase 4: Error Handling Validation

```bash
# Test: Missing baseline file (should be FATAL)
mv docs/reference/competitive_baselines.md docs/reference/competitive_baselines.md.bak
python3 scripts/generate_competitive_report.py --input tmp/competitive_benchmarks/$TIMESTAMP
EXIT_CODE=$?
mv docs/reference/competitive_baselines.md.bak docs/reference/competitive_baselines.md

if [ $EXIT_CODE -eq 2 ]; then
    echo "✓ Correctly exits with fatal error when baseline missing"
else
    echo "✗ Failed to handle missing baseline correctly"
fi

# Test: Malformed loadtest output (should WARN but proceed)
echo "GARBAGE DATA" > tmp/competitive_benchmarks/${TIMESTAMP}_qdrant_ann_1m_768d.txt
python3 scripts/generate_competitive_report.py \
    --input tmp/competitive_benchmarks/$TIMESTAMP \
    --output tmp/test_report.md 2>&1 | grep -q "WARNING"

if [ $? -eq 0 ]; then
    echo "✓ Correctly warns on malformed data"
else
    echo "✗ Failed to warn on malformed data"
fi

# Test: Division by zero in delta calculation
# (Manually inject competitor baseline with zero throughput)
# Expected: Delta should be None, not crash

# Test: Negative latency (measurement error)
# (Manually inject negative P99 value)
# Expected: Validation error, value sanitized to None
```

### Phase 5: Historical Comparison Testing

```bash
# Create second baseline run (simulate quarterly measurement)
sleep 5  # Ensure different timestamp
./scripts/competitive_benchmark_suite.sh
TIMESTAMP_NEW=$(ls -t tmp/competitive_benchmarks/*_metadata.txt | head -1 | xargs basename | cut -d_ -f1-3)

# Generate report with historical comparison
python3 scripts/generate_competitive_report.py \
    --input tmp/competitive_benchmarks/$TIMESTAMP_NEW \
    --compare-to tmp/competitive_benchmarks/$TIMESTAMP \
    --output tmp/competitive_benchmarks/${TIMESTAMP_NEW}_report.md

# Verify trend analysis section exists
grep -q "Performance Trends" tmp/competitive_benchmarks/${TIMESTAMP_NEW}_report.md

if [ $? -eq 0 ]; then
    echo "✓ Historical comparison section generated"
else
    echo "✗ Missing historical comparison section"
fi
```

### Phase 6: Performance Benchmarking

```bash
# Benchmark report generation time
time python3 scripts/generate_competitive_report.py \
    --input tmp/competitive_benchmarks/$TIMESTAMP \
    --output tmp/test_report.md

# Expected: <10 seconds

# Measure memory usage
/usr/bin/time -l python3 scripts/generate_competitive_report.py \
    --input tmp/competitive_benchmarks/$TIMESTAMP \
    --output tmp/test_report.md 2>&1 | grep "maximum resident set size"

# Expected: <500MB RSS
```

## Integration Points

- **Consumes output from**: Task 003 benchmark suite runner
- **Reads baseline data from**: Task 002 competitive baseline documentation
- **Referenced by**: Task 005 quarterly review workflow integration
- **Used by**: Performance engineering team for quarterly reports

## Implementation Guidance

### Script Structure

```python
#!/usr/bin/env python3
"""
Competitive performance report generator.

Parses benchmark results from loadtest suite, compares against competitor baselines,
and generates actionable markdown reports with statistical analysis.

Usage:
    python3 scripts/generate_competitive_report.py \\
        --input tmp/competitive_benchmarks/2025-11-08_14-30-00 \\
        --output tmp/competitive_benchmarks/2025-11-08_14-30-00_report.md \\
        --compare-to tmp/competitive_benchmarks/2025-08-08_10-00-00 \\
        --verbose
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("competitive_report")


# Data classes (BenchmarkResult, CompetitorBaseline, etc.)
# ...


# Parsing functions
def parse_loadtest_results(result_file: Path) -> Optional[BenchmarkResult]:
    """Parse loadtest output file to extract performance metrics."""
    # ...

def parse_metadata(metadata_file: Path) -> Dict[str, Any]:
    """Parse benchmark metadata file."""
    # ...

def parse_baseline_documentation(baseline_file: Path) -> Dict[str, CompetitorBaseline]:
    """Parse markdown table from competitive baseline documentation."""
    # ...


# Analysis functions
def calculate_delta(engram_value: float, competitor_value: float,
                     metric_name: str) -> Optional[float]:
    """Calculate percentage delta with error handling."""
    # ...

def calculate_confidence_interval(mean: float, std: float, n: int,
                                   confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for performance metrics."""
    # ...

def is_statistically_significant(engram_mean: float, engram_std: float, engram_n: int,
                                  competitor_mean: float, alpha: float = 0.05) -> bool:
    """Test if difference from competitor is statistically significant."""
    # ...

def analyze_trend(historical_results: List[Tuple[datetime, float]],
                   metric_name: str) -> Dict[str, Any]:
    """Analyze historical trend using linear regression."""
    # ...


# Report generation functions
def generate_executive_summary(comparisons: List[ComparisonResult],
                                 metadata: Dict[str, Any]) -> str:
    """Generate executive summary section."""
    # ...

def generate_comparison_table(comparisons: List[ComparisonResult]) -> str:
    """Generate detailed comparison table."""
    # ...

def generate_ascii_chart(comparisons: List[ComparisonResult]) -> str:
    """Generate ASCII bar chart visualization."""
    # ...

def generate_optimization_priorities(comparisons: List[ComparisonResult]) -> str:
    """Generate prioritized optimization list."""
    # ...

def generate_historical_trends(current_results: Dict[str, BenchmarkResult],
                                 historical_results: Dict[str, BenchmarkResult]) -> str:
    """Generate historical trend analysis section."""
    # ...

def generate_metadata_section(metadata: Dict[str, Any]) -> str:
    """Generate metadata and reproducibility section."""
    # ...


# Main orchestration
def main():
    parser = argparse.ArgumentParser(
        description="Generate competitive performance comparison report"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with benchmark results (e.g., tmp/competitive_benchmarks/2025-11-08_14-30-00)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output markdown report file"
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Previous benchmark directory for historical comparison (optional)"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("docs/reference/competitive_baselines.md"),
        help="Competitive baseline documentation file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Error collector
    errors = ErrorCollector()

    # Phase 1: Parse inputs
    logger.info("Parsing benchmark results...")
    # ...

    # Phase 2: Load baselines
    logger.info("Loading competitor baselines...")
    # ...

    # Phase 3: Perform comparisons
    logger.info("Comparing Engram vs competitors...")
    # ...

    # Phase 4: Generate report
    logger.info("Generating markdown report...")
    # ...

    # Phase 5: Write output
    logger.info(f"Writing report to {args.output}")
    # ...

    # Check for errors
    if errors.has_fatal():
        logger.error("Report generation failed due to fatal errors")
        sys.exit(2)

    summary = errors.get_summary()
    if summary.get(ErrorSeverity.WARNING, 0) > 0:
        logger.warning(f"Report generated with {summary[ErrorSeverity.WARNING]} warnings")
        sys.exit(1)

    logger.info("Report generated successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

### ASCII Chart Rendering

```python
def render_ascii_bar_chart(engram_value: float, competitor_value: float,
                            engram_label: str, competitor_label: str,
                            max_width: int = 50) -> str:
    """
    Render ASCII bar chart comparing two values.

    Args:
        engram_value: Engram metric value
        competitor_value: Competitor metric value
        engram_label: Label for Engram bar
        competitor_label: Label for competitor bar
        max_width: Maximum bar width in characters

    Returns:
        Multi-line ASCII bar chart string

    Example:
        >>> chart = render_ascii_bar_chart(15.12, 27.96, "Engram", "Neo4j")
        >>> print(chart)
        Neo4j:  ████████████████████████████ 27.96ms
        Engram: ███████████████ 15.12ms (-46% ✓)
    """
    # Find max value for scaling
    max_value = max(engram_value, competitor_value)

    # Calculate bar lengths
    competitor_len = int((competitor_value / max_value) * max_width)
    engram_len = int((engram_value / max_value) * max_width)

    # Calculate delta
    delta = ((engram_value - competitor_value) / competitor_value) * 100

    # Format delta with status indicator
    if delta < -10:
        delta_str = f"({delta:+.0f}% ✓)"
    elif delta > 10:
        delta_str = f"({delta:+.0f}% ⚠)"
    else:
        delta_str = f"({delta:+.0f}%)"

    # Render bars
    competitor_bar = f"{competitor_label}: {'█' * competitor_len} {competitor_value:.2f}ms"
    engram_bar = f"{engram_label}: {'█' * engram_len} {engram_value:.2f}ms {delta_str}"

    return f"{competitor_bar}\n{engram_bar}"
```

### Markdown Table Generation

```python
def generate_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Generate markdown table with proper alignment.

    Args:
        headers: Column headers
        rows: Table rows (each row is list of cell values)

    Returns:
        Formatted markdown table string
    """
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Generate header row
    header = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"

    # Generate separator row
    separator = "| " + " | ".join("-" * w for w in widths) + " |"

    # Generate data rows
    data_rows = []
    for row in rows:
        formatted = "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        data_rows.append(formatted)

    return "\n".join([header, separator] + data_rows)
```

## Dependencies

```bash
# Python 3.11+
# Install dependencies:
pip3 install numpy scipy

# Or use requirements.txt:
cat > requirements.txt << 'EOF'
numpy>=1.24.0
scipy>=1.10.0
EOF

pip3 install -r requirements.txt
```

## Success Criteria Summary

Task is complete when:

1. Script parses all 4 competitive scenarios correctly (validated with unit tests)
2. Markdown report is human-readable and actionable (reviewed by team)
3. Baseline comparison calculations are accurate (spot-checked manually)
4. Statistical significance testing is correct (validated against scipy reference)
5. Historical comparison works with 2+ data points (integration tested)
6. Error handling is comprehensive (all edge cases tested)
7. Code quality passes all checks: mypy strict, ruff, black, 90%+ coverage
8. Report generation completes in <10 seconds (benchmarked)
9. Script handles missing/malformed data gracefully (error injection tested)
10. Documentation is complete (Google-style docstrings, usage examples)

## References

- Welch's t-test: https://en.wikipedia.org/wiki/Welch%27s_t-test
- Cohen's d effect size: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
- Benjamini-Hochberg procedure: https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure
- Existing implementation: `scripts/analyze_benchmarks.py` (for statistical testing patterns)
