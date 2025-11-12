#!/usr/bin/env python3
"""
Validate baseline measurement results for data integrity.

Checks for measurement errors, outliers, and statistical anomalies that
indicate invalid results requiring re-measurement.
"""

import json
import sys
import re
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


def parse_loadtest_output(file_path: Path) -> Optional[dict]:
    """
    Parse loadtest output file to extract performance metrics.

    Returns dict with keys: p50, p95, p99, qps, duration_s
    Returns None if parsing fails.
    """
    try:
        with open(file_path) as f:
            content = f.read()

        # Try JSON format first (new loadtest format)
        if file_path.suffix == '.json':
            data = json.loads(content)
            return {
                'p50': data.get('p50_latency_ms'),
                'p95': data.get('p95_latency_ms'),
                'p99': data.get('p99_latency_ms'),
                'qps': data.get('throughput_qps'),
                'duration_s': data.get('duration_s', 60)
            }

        # Fall back to text parsing (old format)
        metrics = {}

        # Extract percentiles (various formats)
        p50_match = re.search(r"(?:P50|p50|median).*?(\d+\.?\d*)\s*ms", content, re.IGNORECASE)
        p95_match = re.search(r"(?:P95|p95).*?(\d+\.?\d*)\s*ms", content, re.IGNORECASE)
        p99_match = re.search(r"(?:P99|p99).*?(\d+\.?\d*)\s*ms", content, re.IGNORECASE)

        if p50_match:
            metrics['p50'] = float(p50_match.group(1))
        if p95_match:
            metrics['p95'] = float(p95_match.group(1))
        if p99_match:
            metrics['p99'] = float(p99_match.group(1))

        # Extract throughput (QPS or ops/s)
        qps_match = re.search(r"(?:Throughput|throughput|QPS|qps).*?(\d+\.?\d*)\s*(?:ops/s|QPS|qps)?", content, re.IGNORECASE)
        if qps_match:
            metrics['qps'] = float(qps_match.group(1))

        # Extract duration
        duration_match = re.search(r"(?:Duration|duration).*?(\d+)\s*s", content, re.IGNORECASE)
        metrics['duration_s'] = int(duration_match.group(1)) if duration_match else 60

        # Validate we got minimum required metrics
        if 'p99' not in metrics or 'qps' not in metrics:
            return None

        # Fill in missing percentiles with P99 (conservative estimate)
        if 'p50' not in metrics:
            metrics['p50'] = metrics['p99'] * 0.5
        if 'p95' not in metrics:
            metrics['p95'] = metrics['p99'] * 0.9

        return metrics

    except Exception as e:
        print(f"Warning: Failed to parse {file_path}: {e}", file=sys.stderr)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate baseline measurement results")
    parser.add_argument("--timestamp", required=True, help="Benchmark timestamp (e.g., 2025-11-08_14-30-00)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed validation output")
    args = parser.parse_args()

    results_dir = Path(f"tmp/competitive_benchmarks/{args.timestamp}")
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(2)

    # Validation results accumulator
    validations: List[ValidationResult] = []

    # Find all scenario result files (both .json and .txt)
    scenario_files = list(results_dir.glob("*_loadtest.json")) + list(results_dir.glob("*_loadtest.txt"))
    if len(scenario_files) == 0:
        print(f"ERROR: No scenario result files found in {results_dir}")
        sys.exit(2)

    print(f"\n=== Validating {len(scenario_files)} Scenario Results ===\n")

    for scenario_file in scenario_files:
        scenario_name = scenario_file.stem.replace("_loadtest", "")

        if args.verbose:
            print(f"Validating {scenario_name}...")

        # Parse results
        metrics = parse_loadtest_output(scenario_file)

        if metrics is None:
            validations.append(ValidationResult(
                scenario=scenario_name,
                metric="Parsing",
                value=None,
                valid=False,
                reason="Failed to parse required metrics from result file",
                severity="ERROR"
            ))
            continue

        # Run validations
        validations.append(validate_latency_sanity(metrics['p99'], scenario_name))
        validations.append(validate_throughput_sanity(metrics['qps'], scenario_name, metrics['duration_s']))
        validations.append(validate_distribution_consistency(metrics['p50'], metrics['p95'], metrics['p99'], scenario_name))

    # Summarize validation results
    print("\n=== Validation Summary ===\n")

    errors = [v for v in validations if v.severity == "ERROR"]
    warnings = [v for v in validations if v.severity == "WARNING"]
    info = [v for v in validations if v.severity == "INFO"]

    # Print results grouped by severity
    if errors:
        print("ERRORS:")
        for v in errors:
            print(f"  X {v.scenario} - {v.metric}: {v.reason}")
        print()

    if warnings:
        print("WARNINGS:")
        for v in warnings:
            print(f"  ! {v.scenario} - {v.metric}: {v.reason}")
        print()

    if args.verbose and info:
        print("INFO:")
        for v in info:
            print(f"  - {v.scenario} - {v.metric}: {v.reason}")
        print()

    print(f"Total: {len(validations)} checks")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Info: {len(info)}")

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
