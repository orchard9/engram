#!/usr/bin/env python3
"""
Benchmark regression detection using statistical hypothesis testing.

Implements Welch's t-test, Cohen's d effect size, and Benjamini-Hochberg correction
for multiple testing. Identifies performance regressions with statistical rigor.

Usage:
    python3 scripts/analyze_benchmarks.py \\
        --baseline target/criterion/baseline \\
        --current target/criterion/current \\
        --output results/regression_analysis.json \\
        --alpha 0.05
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy import stats


@dataclass
class BenchmarkResult:
    """Statistical summary of benchmark results."""
    name: str
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    samples: List[float]

    def coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (CV = std/mean)."""
        if self.mean == 0:
            return float('inf')
        return self.std / self.mean


@dataclass
class RegressionAnalysis:
    """Result of regression detection analysis."""
    name: str
    severity: str  # CRITICAL, WARNING, NOMINAL, IMPROVEMENT
    percent_change: float
    p_value: float
    effect_size: float
    baseline_mean: float
    current_mean: float
    statistical_power: float
    confidence_interval_95: Tuple[float, float]


def parse_criterion_results(criterion_dir: Path) -> Dict[str, BenchmarkResult]:
    """
    Parse Criterion benchmark results from directory structure.

    Criterion stores results as JSON in subdirectories named by benchmark.
    """
    results = {}

    # Find all benchmark.json files
    for benchmark_json in criterion_dir.rglob("benchmark.json"):
        benchmark_name = benchmark_json.parent.parent.name

        try:
            with open(benchmark_json, 'r') as f:
                data = json.load(f)

            # Extract samples from Criterion format
            if 'typical' in data and 'estimate' in data['typical']:
                mean = data['typical']['estimate']

                # Criterion stores times in nanoseconds
                samples = data.get('samples', [mean] * 100)

                # Calculate percentiles
                samples_array = np.array(samples)
                p50 = np.percentile(samples_array, 50)
                p95 = np.percentile(samples_array, 95)
                p99 = np.percentile(samples_array, 99)
                std = np.std(samples_array)

                results[benchmark_name] = BenchmarkResult(
                    name=benchmark_name,
                    mean=mean,
                    std=std,
                    p50=p50,
                    p95=p95,
                    p99=p99,
                    samples=samples
                )
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not parse {benchmark_json}: {e}", file=sys.stderr)
            continue

    return results


def welch_t_test(baseline: BenchmarkResult, current: BenchmarkResult) -> Tuple[float, float]:
    """
    Perform Welch's t-test (unequal variances) for regression detection.

    Returns (t_statistic, p_value) for one-tailed test (current > baseline).
    Null hypothesis: no performance difference from baseline.
    Alternative: performance degraded (current slower than baseline).
    """
    baseline_samples = np.array(baseline.samples)
    current_samples = np.array(current.samples)

    # Welch's t-test (unequal variances)
    t_stat, p_value_two_tailed = stats.ttest_ind(
        current_samples,
        baseline_samples,
        equal_var=False
    )

    # Convert to one-tailed: testing if current is slower
    if t_stat > 0:  # current > baseline (regression)
        p_value = p_value_two_tailed / 2
    else:  # current < baseline (improvement)
        p_value = 1 - (p_value_two_tailed / 2)

    return t_stat, p_value


def cohens_d(baseline: BenchmarkResult, current: BenchmarkResult) -> float:
    """
    Calculate Cohen's d effect size.

    Measures standardized difference between two means.
    - Small effect: |d| = 0.2
    - Medium effect: |d| = 0.5
    - Large effect: |d| = 0.8
    """
    baseline_samples = np.array(baseline.samples)
    current_samples = np.array(current.samples)

    pooled_std = np.sqrt(
        (baseline_samples.var() + current_samples.var()) / 2
    )

    if pooled_std == 0:
        return 0.0

    return (current.mean - baseline.mean) / pooled_std


def calculate_power(baseline: BenchmarkResult, current: BenchmarkResult, alpha: float) -> float:
    """
    Calculate statistical power (1 - beta) of the test.

    Power is the probability of detecting a true effect.
    Target power: >0.80 for robust detection.
    """
    effect_size = abs(cohens_d(baseline, current))
    n = len(current.samples)

    if n < 2:
        return 0.0

    # Calculate noncentrality parameter
    noncentrality = effect_size * np.sqrt(n / 2)

    # Degrees of freedom (Welch-Satterthwaite approximation)
    s1_sq = baseline.std ** 2
    s2_sq = current.std ** 2
    n1 = len(baseline.samples)
    n2 = len(current.samples)

    if n1 < 2 or n2 < 2:
        return 0.0

    df_numerator = (s1_sq / n1 + s2_sq / n2) ** 2
    df_denominator = (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)

    if df_denominator == 0:
        df = 2 * n - 2
    else:
        df = df_numerator / df_denominator

    # Critical value for one-tailed test
    critical_value = stats.t.ppf(1 - alpha, df)

    # Power calculation using noncentral t-distribution
    power = 1 - stats.nct.cdf(critical_value, df=df, nc=noncentrality)

    return float(power)


def bootstrap_confidence_interval(baseline: BenchmarkResult, current: BenchmarkResult,
                                   confidence: float = 0.95, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for difference in means.

    Uses 10,000 bootstrap resamples to estimate confidence interval
    for non-normal distributions.
    """
    baseline_samples = np.array(baseline.samples)
    current_samples = np.array(current.samples)

    differences = []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    for _ in range(n_bootstrap):
        baseline_resample = rng.choice(baseline_samples, size=len(baseline_samples), replace=True)
        current_resample = rng.choice(current_samples, size=len(current_samples), replace=True)

        diff = current_resample.mean() - baseline_resample.mean()
        differences.append(diff)

    differences = np.array(differences)
    alpha = 1 - confidence
    lower = np.percentile(differences, alpha / 2 * 100)
    upper = np.percentile(differences, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))


def detect_regression(baseline: BenchmarkResult, current: BenchmarkResult,
                       alpha: float = 0.05) -> RegressionAnalysis:
    """
    Detect performance regression using statistical hypothesis testing.

    Regression Criteria:
    - CRITICAL: P99 latency increased by >20% OR throughput decreased by >20% (p < 0.01)
    - WARNING: P99 latency increased by >10% OR throughput decreased by >10% (p < 0.05)
    - NOMINAL: Changes within 5% are attributed to measurement noise
    - IMPROVEMENT: Statistically significant performance improvement
    """
    t_stat, p_value = welch_t_test(baseline, current)
    effect_size = cohens_d(baseline, current)
    power = calculate_power(baseline, current, alpha)
    ci_95 = bootstrap_confidence_interval(baseline, current)

    percent_change = ((current.mean - baseline.mean) / baseline.mean) * 100

    # Determine severity
    if percent_change > 20 and p_value < 0.01:
        severity = "CRITICAL"
    elif percent_change > 10 and p_value < alpha:
        severity = "WARNING"
    elif percent_change < -5 and p_value < alpha:
        severity = "IMPROVEMENT"
    else:
        severity = "NOMINAL"

    return RegressionAnalysis(
        name=current.name,
        severity=severity,
        percent_change=percent_change,
        p_value=p_value,
        effect_size=effect_size,
        baseline_mean=baseline.mean,
        current_mean=current.mean,
        statistical_power=power,
        confidence_interval_95=ci_95
    )


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Benjamini-Hochberg procedure for multiple testing correction.

    Controls the False Discovery Rate (FDR) at level alpha.
    Returns boolean list indicating which tests pass correction.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]

    # Apply BH procedure
    significant = np.zeros(m, dtype=bool)

    for i in range(m - 1, -1, -1):
        threshold = (i + 1) / m * alpha
        if sorted_p_values[i] <= threshold:
            significant[sorted_indices[:i + 1]] = True
            break

    return significant.tolist()


def analyze_all_benchmarks(baseline_dir: Path, current_dir: Path,
                            alpha: float = 0.05) -> Dict[str, Any]:
    """
    Analyze all benchmarks and detect regressions.

    Returns comprehensive report with statistical analysis.
    """
    baseline_results = parse_criterion_results(baseline_dir)
    current_results = parse_criterion_results(current_dir)

    # Find common benchmarks
    common_benchmarks = set(baseline_results.keys()) & set(current_results.keys())

    if not common_benchmarks:
        print("Warning: No common benchmarks found between baseline and current", file=sys.stderr)
        return {
            "error": "No common benchmarks found",
            "baseline_benchmarks": list(baseline_results.keys()),
            "current_benchmarks": list(current_results.keys())
        }

    # Perform regression analysis on all common benchmarks
    analyses = []
    p_values = []

    for name in sorted(common_benchmarks):
        baseline = baseline_results[name]
        current = current_results[name]

        analysis = detect_regression(baseline, current, alpha)
        analyses.append(analysis)
        p_values.append(analysis.p_value)

    # Apply multiple testing correction
    corrected = benjamini_hochberg_correction(p_values, alpha)

    # Update severity based on correction
    for i, analysis in enumerate(analyses):
        if not corrected[i] and analysis.severity in ["CRITICAL", "WARNING"]:
            # Regression not significant after correction
            analysis.severity = "NOMINAL"

    # Categorize results
    critical_regressions = [a for a in analyses if a.severity == "CRITICAL"]
    warning_regressions = [a for a in analyses if a.severity == "WARNING"]
    improvements = [a for a in analyses if a.severity == "IMPROVEMENT"]
    nominal = [a for a in analyses if a.severity == "NOMINAL"]

    # Generate summary
    summary = {
        "total_benchmarks": len(common_benchmarks),
        "critical_regressions": len(critical_regressions),
        "warning_regressions": len(warning_regressions),
        "improvements": len(improvements),
        "nominal_changes": len(nominal),
        "alpha": alpha,
        "multiple_testing_correction": "Benjamini-Hochberg"
    }

    # Detailed results
    results = {
        "summary": summary,
        "critical": [asdict(a) for a in critical_regressions],
        "warnings": [asdict(a) for a in warning_regressions],
        "improvements": [asdict(a) for a in improvements],
        "all_analyses": [asdict(a) for a in analyses]
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results for performance regressions"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline Criterion results directory"
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current Criterion results directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for analysis results"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for hypothesis tests (default: 0.05)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.baseline.exists():
        print(f"Error: Baseline directory not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)

    if not args.current.exists():
        print(f"Error: Current directory not found: {args.current}", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    print(f"Analyzing benchmarks...")
    print(f"  Baseline: {args.baseline}")
    print(f"  Current:  {args.current}")
    print(f"  Alpha:    {args.alpha}")

    results = analyze_all_benchmarks(args.baseline, args.current, args.alpha)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Print summary
    if "summary" in results:
        summary = results["summary"]
        print(f"\nSummary:")
        print(f"  Total benchmarks:      {summary['total_benchmarks']}")
        print(f"  Critical regressions:  {summary['critical_regressions']}")
        print(f"  Warning regressions:   {summary['warning_regressions']}")
        print(f"  Improvements:          {summary['improvements']}")
        print(f"  Nominal changes:       {summary['nominal_changes']}")

        # Exit with error if regressions found
        if summary['critical_regressions'] > 0:
            print("\n❌ CRITICAL regressions detected!", file=sys.stderr)
            sys.exit(2)
        elif summary['warning_regressions'] > 0:
            print("\n⚠️  WARNING regressions detected!", file=sys.stderr)
            sys.exit(1)
        else:
            print("\n✓ No significant regressions detected")
            sys.exit(0)
    else:
        print("\n❌ Analysis failed", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
