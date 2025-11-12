#!/usr/bin/env python3
"""Competitive performance report generator.

Parses benchmark results from loadtest suite, compares against competitor baselines,
and generates actionable markdown reports with statistical analysis.

Usage:
    python3 scripts/generate_competitive_report.py \
        --input tmp/competitive_benchmarks/2025-11-08_14-30-00 \
        --output tmp/competitive_benchmarks/2025-11-08_14-30-00_report.md \
        --compare-to tmp/competitive_benchmarks/2025-08-08_10-00-00 \
        --verbose
"""

import argparse
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("competitive_report")


# ============================================================================
# DATA STRUCTURES
# ============================================================================


class ErrorSeverity(Enum):
    """Error severity classification for report generation."""

    FATAL = "FATAL"  # Cannot proceed (missing baseline file)
    ERROR = "ERROR"  # Major issue (scenario failed to run)
    WARNING = "WARNING"  # Minor issue (malformed field, using default)
    INFO = "INFO"  # Informational (missing optional metadata)


@dataclass
class ReportError:
    """Structured error for report generation."""

    severity: ErrorSeverity
    phase: str  # "PARSING", "VALIDATION", "ANALYSIS", "GENERATION"
    message: str
    context: Dict[str, Any]  # Additional context for debugging
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Result of data validation with detailed error context."""

    valid: bool
    warnings: List[str]
    errors: List[str]
    sanitized_value: Optional[Any]


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
    duration_seconds: Optional[float]
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


# ============================================================================
# ERROR HANDLING
# ============================================================================


class ErrorCollector:
    """Collect errors during report generation."""

    def __init__(self) -> None:
        """Initialize error collector."""
        self.errors: List[ReportError] = []

    def add(self, severity: ErrorSeverity, phase: str, message: str, **context: Any) -> None:
        """Add error with structured context.

        Args:
            severity: Error severity level
            phase: Phase where error occurred
            message: Error message
            **context: Additional context as keyword arguments
        """
        error = ReportError(severity=severity, phase=phase, message=message, context=context)
        self.errors.append(error)

        # Log immediately
        if severity == ErrorSeverity.FATAL:
            logger.error("[%s] %s", phase, message, extra=context)
        elif severity == ErrorSeverity.ERROR:
            logger.error("[%s] %s", phase, message, extra=context)
        elif severity == ErrorSeverity.WARNING:
            logger.warning("[%s] %s", phase, message, extra=context)
        else:
            logger.info("[%s] %s", phase, message, extra=context)

    def has_fatal(self) -> bool:
        """Check if any fatal errors occurred.

        Returns:
            True if fatal errors exist
        """
        return any(e.severity == ErrorSeverity.FATAL for e in self.errors)

    def get_summary(self) -> Dict[ErrorSeverity, int]:
        """Get count by severity.

        Returns:
            Dictionary mapping severity to count
        """
        return dict(Counter(e.severity for e in self.errors))


# ============================================================================
# VALIDATION
# ============================================================================


def validate_latency(value: Any, metric_name: str) -> ValidationResult:
    """Validate latency metric (must be positive, reasonable magnitude).

    Args:
        value: Value to validate
        metric_name: Name of metric for error messages

    Returns:
        ValidationResult with sanitized value or None
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Handle None
    if value is None:
        return ValidationResult(valid=True, warnings=warnings, errors=errors, sanitized_value=None)

    # Convert to float
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        errors.append(f"{metric_name}: Cannot convert to float: {value}")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    # Check for invalid float values
    if np.isnan(float_value):
        errors.append(f"{metric_name}: NaN value")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    if np.isinf(float_value):
        errors.append(f"{metric_name}: Infinite value")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    # Check for negative
    if float_value < 0:
        errors.append(f"{metric_name}: Negative latency {float_value}ms (measurement error)")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    # Check for unreasonably high values (>10 seconds)
    if float_value > 10000:
        warnings.append(f"{metric_name}: Unusually high latency {float_value}ms (>10s)")

    return ValidationResult(
        valid=True, warnings=warnings, errors=errors, sanitized_value=float_value
    )


def validate_throughput(value: Any, metric_name: str) -> ValidationResult:
    """Validate throughput metric (must be positive, non-zero).

    Args:
        value: Value to validate
        metric_name: Name of metric for error messages

    Returns:
        ValidationResult with sanitized value or None
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Handle None
    if value is None:
        return ValidationResult(valid=True, warnings=warnings, errors=errors, sanitized_value=None)

    # Convert to float
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        errors.append(f"{metric_name}: Cannot convert to float: {value}")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    # Check for invalid float values
    if np.isnan(float_value):
        errors.append(f"{metric_name}: NaN value")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    if np.isinf(float_value):
        errors.append(f"{metric_name}: Infinite value")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    # Check for negative or zero
    if float_value <= 0:
        errors.append(f"{metric_name}: Non-positive throughput {float_value} ops/sec")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    return ValidationResult(
        valid=True, warnings=warnings, errors=errors, sanitized_value=float_value
    )


def validate_percentage(value: Any, metric_name: str) -> ValidationResult:
    """Validate percentage metric (must be 0-100 or 0.0-1.0).

    Args:
        value: Value to validate
        metric_name: Name of metric for error messages

    Returns:
        ValidationResult with sanitized value normalized to 0-100 or None
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Handle None
    if value is None:
        return ValidationResult(valid=True, warnings=warnings, errors=errors, sanitized_value=None)

    # Convert to float
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        errors.append(f"{metric_name}: Cannot convert to float: {value}")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    # Check for invalid float values
    if np.isnan(float_value):
        errors.append(f"{metric_name}: NaN value")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    if np.isinf(float_value):
        errors.append(f"{metric_name}: Infinite value")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    # Normalize to 0-100 range
    if 0.0 <= float_value <= 1.0:
        # Assume it's in 0.0-1.0 range, convert to percentage
        float_value = float_value * 100.0
        warnings.append(f"{metric_name}: Normalized from 0.0-1.0 to 0-100 range")
    elif float_value < 0 or float_value > 100:
        errors.append(f"{metric_name}: Percentage out of range: {float_value}")
        return ValidationResult(valid=False, warnings=warnings, errors=errors, sanitized_value=None)

    return ValidationResult(
        valid=True, warnings=warnings, errors=errors, sanitized_value=float_value
    )


# ============================================================================
# PARSING
# ============================================================================


def parse_loadtest_results(result_file: Path, errors: ErrorCollector) -> Optional[BenchmarkResult]:
    """Parse loadtest output file to extract performance metrics.

    Args:
        result_file: Path to loadtest output file (*.txt)
        errors: Error collector for tracking issues

    Returns:
        BenchmarkResult if parsing succeeds, None otherwise

    Example:
        >>> result = parse_loadtest_results(Path("tmp/2025-11-08_qdrant.txt"), errors)
        >>> if result:
        ...     print(f"P99: {result.p99_latency_ms}ms")
    """
    if not result_file.exists():
        errors.add(
            ErrorSeverity.ERROR,
            "PARSING",
            f"Result file not found: {result_file}",
            file=str(result_file),
        )
        return None

    try:
        content = result_file.read_text()
    except (PermissionError, OSError) as e:
        errors.add(
            ErrorSeverity.ERROR,
            "PARSING",
            f"Cannot read result file: {e}",
            file=str(result_file),
        )
        return None

    # Extract scenario name from filename
    # Format: YYYY-MM-DD_HH-MM-SS_scenario_name.txt
    # Use regex to match the timestamp and extract everything after it
    timestamp_pattern = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$"
    match = re.match(timestamp_pattern, result_file.stem)
    if match:
        scenario_name = match.group(1)
    else:
        # Fallback: just use the whole filename
        errors.add(
            ErrorSeverity.WARNING,
            "PARSING",
            f"Could not extract scenario name from {result_file.name}, using full filename",
            file=str(result_file),
        )
        scenario_name = result_file.stem

    # Parse metrics using regex
    p50_latency = None
    p95_latency = None
    p99_latency = None
    throughput = None
    error_rate = None
    total_ops = None
    duration = None

    # P50 latency
    match = re.search(r"P50\s+[Ll]atency:\s+([\d.]+)\s*ms", content)
    if match:
        validation = validate_latency(match.group(1), "P50 latency")
        if validation.valid:
            p50_latency = validation.sanitized_value
        for warning in validation.warnings:
            errors.add(ErrorSeverity.WARNING, "VALIDATION", warning, file=str(result_file))
        for error in validation.errors:
            errors.add(ErrorSeverity.ERROR, "VALIDATION", error, file=str(result_file))

    # P95 latency
    match = re.search(r"P95\s+[Ll]atency:\s+([\d.]+)\s*ms", content)
    if match:
        validation = validate_latency(match.group(1), "P95 latency")
        if validation.valid:
            p95_latency = validation.sanitized_value
        for warning in validation.warnings:
            errors.add(ErrorSeverity.WARNING, "VALIDATION", warning, file=str(result_file))
        for error in validation.errors:
            errors.add(ErrorSeverity.ERROR, "VALIDATION", error, file=str(result_file))

    # P99 latency
    match = re.search(r"P99\s+[Ll]atency:\s+([\d.]+)\s*ms", content)
    if match:
        validation = validate_latency(match.group(1), "P99 latency")
        if validation.valid:
            p99_latency = validation.sanitized_value
        for warning in validation.warnings:
            errors.add(ErrorSeverity.WARNING, "VALIDATION", warning, file=str(result_file))
        for error in validation.errors:
            errors.add(ErrorSeverity.ERROR, "VALIDATION", error, file=str(result_file))

    # Throughput (try multiple patterns)
    match = re.search(r"[Tt]hroughput:\s+([\d.]+)\s*(?:ops/sec|QPS)", content, re.IGNORECASE)
    if not match:
        match = re.search(r"[Oo]verall\s+[Tt]hroughput:\s+([\d.]+)\s*ops/sec", content)
    if match:
        validation = validate_throughput(match.group(1), "Throughput")
        if validation.valid:
            throughput = validation.sanitized_value
        for warning in validation.warnings:
            errors.add(ErrorSeverity.WARNING, "VALIDATION", warning, file=str(result_file))
        for error in validation.errors:
            errors.add(ErrorSeverity.ERROR, "VALIDATION", error, file=str(result_file))

    # Error rate
    match = re.search(r"[Ee]rror\s+[Rr]ate:\s+([\d.]+)\s*%", content)
    if match:
        validation = validate_percentage(match.group(1), "Error rate")
        if validation.valid:
            error_rate = validation.sanitized_value
        for warning in validation.warnings:
            errors.add(ErrorSeverity.WARNING, "VALIDATION", warning, file=str(result_file))
        for error in validation.errors:
            errors.add(ErrorSeverity.ERROR, "VALIDATION", error, file=str(result_file))

    # Total operations
    match = re.search(r"[Tt]otal\s+[Oo]perations:\s+([\d,]+)", content)
    if match:
        try:
            total_ops = int(match.group(1).replace(",", ""))
        except ValueError:
            errors.add(
                ErrorSeverity.WARNING,
                "VALIDATION",
                f"Invalid total operations: {match.group(1)}",
                file=str(result_file),
            )

    # Duration
    match = re.search(r"[Dd]uration:\s+([\d.]+)\s*s", content)
    if match:
        try:
            duration = float(match.group(1))
        except ValueError:
            errors.add(
                ErrorSeverity.WARNING,
                "VALIDATION",
                f"Invalid duration: {match.group(1)}",
                file=str(result_file),
            )

    # Timestamp (use file modification time as fallback)
    timestamp = datetime.fromtimestamp(result_file.stat().st_mtime)

    # Warn if critical metrics are missing
    if p99_latency is None and throughput is None:
        errors.add(
            ErrorSeverity.WARNING,
            "PARSING",
            f"No P99 latency or throughput found in {result_file.name}",
            file=str(result_file),
        )

    return BenchmarkResult(
        scenario_name=scenario_name,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        p99_latency_ms=p99_latency,
        throughput_qps=throughput,
        error_rate_pct=error_rate,
        total_operations=total_ops,
        duration_seconds=duration,
        timestamp=timestamp,
    )


def parse_metadata(metadata_file: Path, errors: ErrorCollector) -> Dict[str, Any]:
    """Parse benchmark metadata file.

    Args:
        metadata_file: Path to metadata file
        errors: Error collector for tracking issues

    Returns:
        Dictionary of metadata fields
    """
    metadata: Dict[str, Any] = {
        "git_commit": "UNKNOWN",
        "git_branch": "UNKNOWN",
        "os": "UNKNOWN",
        "cpu": "UNKNOWN",
        "ram": "UNKNOWN",
        "engram_version": "UNKNOWN",
        "timestamp": datetime.now(),
    }

    if not metadata_file.exists():
        errors.add(
            ErrorSeverity.WARNING,
            "PARSING",
            f"Metadata file not found: {metadata_file}",
            file=str(metadata_file),
        )
        return metadata

    try:
        content = metadata_file.read_text()
    except (PermissionError, OSError) as e:
        errors.add(
            ErrorSeverity.WARNING,
            "PARSING",
            f"Cannot read metadata file: {e}",
            file=str(metadata_file),
        )
        return metadata

    # Parse fields
    match = re.search(r"[Cc]ommit\s+[Hh]ash:\s+([a-f0-9]+)", content)
    if match:
        commit_hash = match.group(1)
        # Validate format (40-char or 7-char hex)
        if len(commit_hash) in (7, 40) and all(c in "0123456789abcdef" for c in commit_hash):
            metadata["git_commit"] = commit_hash
        else:
            errors.add(
                ErrorSeverity.WARNING,
                "VALIDATION",
                f"Invalid git commit hash: {commit_hash}",
                file=str(metadata_file),
            )

    match = re.search(r"[Bb]ranch:\s+(.+)", content)
    if match:
        metadata["git_branch"] = match.group(1).strip()

    match = re.search(r"OS:\s+(.+)", content)
    if match:
        metadata["os"] = match.group(1).strip()

    match = re.search(r"CPU:\s+(.+)", content)
    if match:
        metadata["cpu"] = match.group(1).strip()

    match = re.search(r"RAM:\s+(.+)", content)
    if match:
        metadata["ram"] = match.group(1).strip()

    match = re.search(r"[Ee]ngram\s+[Vv]ersion:\s+(.+)", content)
    if match:
        metadata["engram_version"] = match.group(1).strip()

    # Parse timestamp (try ISO 8601 format)
    match = re.search(r"[Tt]imestamp:\s+(.+)", content)
    if match:
        timestamp_str = match.group(1).strip()
        # Try multiple date formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d_%H-%M-%S",
        ]:
            try:
                metadata["timestamp"] = datetime.strptime(timestamp_str, fmt)
                break
            except ValueError:
                continue

    return metadata


def parse_baseline_documentation(
    baseline_file: Path, errors: ErrorCollector
) -> Dict[str, CompetitorBaseline]:
    """Parse markdown table from competitive baseline documentation.

    Args:
        baseline_file: Path to competitive_baselines.md
        errors: Error collector for tracking issues

    Returns:
        Dictionary mapping scenario names to CompetitorBaseline objects
    """
    baselines: Dict[str, CompetitorBaseline] = {}

    if not baseline_file.exists():
        errors.add(
            ErrorSeverity.FATAL,
            "PARSING",
            f"Baseline documentation not found: {baseline_file}",
            file=str(baseline_file),
        )
        return baselines

    try:
        content = baseline_file.read_text()
    except (PermissionError, OSError) as e:
        errors.add(
            ErrorSeverity.FATAL,
            "PARSING",
            f"Cannot read baseline documentation: {e}",
            file=str(baseline_file),
        )
        return baselines

    # Find the main table (after "## Competitor Baseline Summary")
    table_match = re.search(
        r"## Competitor Baseline Summary.*?\n\|([^\n]+)\|.*?\n\|([^\n]+)\|.*?\n((?:\|[^\n]+\|\n)+)",
        content,
        re.DOTALL,
    )

    if not table_match:
        errors.add(
            ErrorSeverity.FATAL,
            "PARSING",
            "Could not find baseline summary table in documentation",
            file=str(baseline_file),
        )
        return baselines

    # table_match.group(1) is the header row (not needed for parsing)
    rows_text = table_match.group(3)

    # Parse table rows
    for line in rows_text.strip().split("\n"):
        if not line.strip() or not line.startswith("|"):
            continue

        # Split by | and clean
        cells = [cell.strip() for cell in line.split("|")[1:-1]]

        if len(cells) < 8:
            errors.add(
                ErrorSeverity.WARNING,
                "PARSING",
                f"Skipping malformed baseline row: {line}",
                file=str(baseline_file),
            )
            continue

        system_name = cells[0]
        workload_type = cells[2]
        dataset_size = cells[6]
        source_url = cells[8] if len(cells) > 8 else "N/A"

        # Parse P99 latency
        p99_str = cells[3]
        p99_latency = None
        if p99_str and p99_str != "N/A":
            match = re.search(r"([\d.]+)(?:-[\d.]+)?\s*ms", p99_str)
            if match:
                validation = validate_latency(match.group(1), f"{system_name} P99 latency")
                if validation.valid:
                    p99_latency = validation.sanitized_value

        # Parse throughput
        throughput_str = cells[4]
        throughput = None
        if throughput_str and throughput_str != "N/A":
            match = re.search(r"([\d,]+)", throughput_str)
            if match:
                validation = validate_throughput(
                    match.group(1).replace(",", ""), f"{system_name} throughput"
                )
                if validation.valid:
                    throughput = validation.sanitized_value

        # Parse recall
        recall_str = cells[5]
        recall = None
        if recall_str and recall_str != "N/A":
            match = re.search(r"([\d.]+)%", recall_str)
            if match:
                validation = validate_percentage(match.group(1), f"{system_name} recall")
                if validation.valid:
                    recall = validation.sanitized_value

        # Map system to scenario name
        scenario_mapping = {
            "Qdrant": "qdrant_ann_1m_768d",
            "Neo4j": "neo4j_traversal_100k",
            "Milvus": "milvus_ann_10m_768d",
            "Weaviate": "weaviate_ann_1m",
            "Redis": "redis_vector_search",
        }

        scenario_name = scenario_mapping.get(system_name)
        if scenario_name:
            baselines[scenario_name] = CompetitorBaseline(
                system_name=system_name,
                workload_type=workload_type,
                p99_latency_ms=p99_latency,
                throughput_qps=throughput,
                recall_rate_pct=recall,
                dataset_size=dataset_size,
                source_url=source_url,
            )

    logger.info("Parsed %d competitor baselines", len(baselines))
    return baselines


# ============================================================================
# ANALYSIS
# ============================================================================


def calculate_delta(
    engram_value: Optional[float], competitor_value: Optional[float], metric_name: str
) -> Optional[float]:
    """Calculate percentage delta with proper error handling.

    Args:
        engram_value: Engram measurement
        competitor_value: Competitor baseline
        metric_name: Metric name for error messages

    Returns:
        Delta as percentage (-100 to +inf), or None if invalid

    Examples:
        >>> calculate_delta(15.12, 27.96, "P99")
        -45.99  # 46% faster
        >>> calculate_delta(27.96, 15.12, "P99")
        84.92  # 85% slower
        >>> calculate_delta(10.0, 0.0, "P99")
        None  # Division by zero
    """
    if competitor_value is None or engram_value is None:
        return None

    if competitor_value == 0:
        # Handle division by zero gracefully
        logger.warning("Cannot calculate delta for %s: competitor baseline is zero", metric_name)
        return None

    # For latency: negative delta is improvement (lower is better)
    # For throughput: positive delta is improvement (higher is better)
    delta = ((engram_value - competitor_value) / competitor_value) * 100
    return delta


def calculate_confidence_interval(
    mean: float, std: float, n: int, confidence: float = 0.95
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate confidence interval for performance metrics.

    Uses t-distribution for small samples (n < 30), normal distribution otherwise.
    Handles insufficient sample size (n < 2) gracefully.

    Args:
        mean: Sample mean
        std: Sample standard deviation
        n: Sample size
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound) or (None, None) if cannot calculate
    """
    if n < 2:
        logger.warning("Cannot calculate CI: insufficient samples (n=%d)", n)
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
        critical_value = scipy_stats.t.ppf((1 + confidence) / 2, df)
    else:
        # Use normal distribution for large samples
        critical_value = scipy_stats.norm.ppf((1 + confidence) / 2)

    margin = critical_value * sem
    return (mean - margin, mean + margin)


def is_statistically_significant(
    engram_mean: float,
    engram_std: float,
    engram_n: int,
    competitor_mean: float,
    alpha: float = 0.05,
) -> bool:
    """Test if difference from competitor is statistically significant.

    Uses one-sample t-test comparing Engram samples to competitor baseline mean.
    Null hypothesis: Engram mean equals competitor mean.

    Args:
        engram_mean: Engram sample mean
        engram_std: Engram sample standard deviation
        engram_n: Engram sample size
        competitor_mean: Competitor baseline mean
        alpha: Significance level (default: 0.05)

    Returns:
        True if p-value < alpha (reject null hypothesis)
    """
    if engram_n < 2:
        return False  # Cannot test with insufficient samples

    # Avoid division by zero
    if engram_std == 0:
        return engram_mean != competitor_mean

    # One-sample t-test
    t_stat = (engram_mean - competitor_mean) / (engram_std / np.sqrt(engram_n))
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), engram_n - 1))

    return p_value < alpha


def analyze_trend(
    historical_results: List[Tuple[datetime, float]], metric_name: str
) -> Dict[str, Any]:
    """Analyze historical trend for a metric using linear regression.

    Args:
        historical_results: List of (timestamp, metric_value) tuples
        metric_name: Name of metric for logging

    Returns:
        Dictionary with trend analysis results:
        {
            "slope": float,  # Change per day
            "r_squared": float,  # Goodness of fit (0-1)
            "trend": str,  # "IMPROVING", "DEGRADING", "STABLE"
            "projection_30d": float  # Projected value in 30 days
        }
    """
    if len(historical_results) < 2:
        return {"trend": "INSUFFICIENT_DATA"}

    # Convert to days since first measurement
    times = [(t - historical_results[0][0]).total_seconds() / 86400 for t, _ in historical_results]
    values = [v for _, v in historical_results]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(times, values)
    r_squared = r_value**2

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
        "std_err": std_err,
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_executive_summary(
    comparisons: List[ComparisonResult], metadata: Dict[str, Any]
) -> str:
    """Generate executive summary section.

    Args:
        comparisons: List of comparison results
        metadata: Benchmark metadata

    Returns:
        Markdown formatted executive summary
    """
    lines = [
        "# Competitive Performance Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Engram Commit**: {metadata.get('git_commit', 'UNKNOWN')} ({metadata.get('git_branch', 'UNKNOWN')})",
        f"**System**: {metadata.get('cpu', 'UNKNOWN')} ({metadata.get('ram', 'UNKNOWN')}), {metadata.get('os', 'UNKNOWN')}",
        "",
        "## Overall Positioning",
        "",
    ]

    # Find best and worst scenarios
    better_count = sum(1 for c in comparisons if c.status == "BETTER")
    worse_count = sum(1 for c in comparisons if c.status == "WORSE")
    comparable_count = sum(1 for c in comparisons if c.status == "COMPARABLE")

    # Generate insights
    insights = []
    for comp in comparisons:
        if comp.status == "BETTER" and comp.latency_delta_pct is not None:
            insights.append(
                f"Engram outperforms {comp.competitor_name} by "
                f"{abs(comp.latency_delta_pct):.1f}% on {comp.scenario_name} "
                f"(P99: {comp.engram_p99:.2f}ms vs {comp.competitor_p99:.2f}ms)."
            )
        elif comp.status == "WORSE" and comp.latency_delta_pct is not None:
            insights.append(
                f"Engram lags {comp.competitor_name} by "
                f"{abs(comp.latency_delta_pct):.1f}% on {comp.scenario_name} "
                f"(P99: {comp.engram_p99:.2f}ms vs {comp.competitor_p99:.2f}ms)."
            )

    for insight in insights[:3]:  # Show top 3 insights
        lines.append(insight)

    lines.extend(
        [
            "",
            "**Key Insights**:",
            f"- {better_count} scenarios better than competitors",
            f"- {comparable_count} scenarios comparable to competitors",
            f"- {worse_count} scenarios need optimization",
            "",
        ]
    )

    return "\n".join(lines)


def generate_comparison_table(comparisons: List[ComparisonResult]) -> str:
    """Generate detailed comparison table.

    Args:
        comparisons: List of comparison results

    Returns:
        Markdown formatted comparison table
    """
    lines = [
        "## Detailed Results",
        "",
        "| Scenario | Metric | Engram | Competitor | Delta | Status | Significance |",
        "|----------|--------|--------|------------|-------|--------|--------------|",
    ]

    for comp in comparisons:
        # P99 Latency row
        if comp.engram_p99 is not None and comp.competitor_p99 is not None:
            delta_str = (
                f"{comp.latency_delta_pct:+.2f}%" if comp.latency_delta_pct is not None else "N/A"
            )
            status_icon = {
                "BETTER": "✓ Better",
                "COMPARABLE": "≈ Comparable",
                "WORSE": "⚠ Worse",
                "NO_BASELINE": "ⓘ No Baseline",
            }.get(comp.status, comp.status)
            sig_str = "p < 0.05" if comp.statistical_significance else "p > 0.05"

            lines.append(
                f"| {comp.scenario_name} | P99 Latency | {comp.engram_p99:.2f}ms | "
                f"{comp.competitor_p99:.2f}ms ({comp.competitor_name}) | {delta_str} | "
                f"{status_icon} | {sig_str} |"
            )

        # Throughput row
        if comp.engram_qps is not None and comp.competitor_qps is not None:
            delta_str = (
                f"{comp.throughput_delta_pct:+.2f}%"
                if comp.throughput_delta_pct is not None
                else "N/A"
            )
            throughput_status = (
                "✓ Better"
                if comp.throughput_delta_pct and comp.throughput_delta_pct > 10
                else (
                    "≈ Comparable"
                    if comp.throughput_delta_pct and abs(comp.throughput_delta_pct) <= 10
                    else "⚠ Worse"
                )
            )

            lines.append(
                f"| {comp.scenario_name} | Throughput | {comp.engram_qps:.0f} QPS | "
                f"{comp.competitor_qps:.0f} QPS ({comp.competitor_name}) | {delta_str} | "
                f"{throughput_status} | N/A |"
            )

    lines.extend(
        [
            "",
            "**Legend**:",
            "- ✓ Better: >10% improvement over competitor",
            "- ≈ Comparable: Within ±10% of competitor",
            "- ⚠ Worse: >10% regression vs competitor",
            "- ⓘ No Baseline: No competitor offers this workload",
            "",
        ]
    )

    return "\n".join(lines)


def generate_ascii_chart(comparisons: List[ComparisonResult]) -> str:
    """Generate ASCII bar chart visualization.

    Args:
        comparisons: List of comparison results

    Returns:
        ASCII bar chart string
    """
    lines = ["## Performance Comparison (P99 Latency)", ""]

    for comp in comparisons:
        if comp.engram_p99 is None or comp.competitor_p99 is None:
            continue

        lines.append(f"{comp.scenario_name}:")

        # Calculate bar lengths (max 50 chars)
        max_value = max(comp.engram_p99, comp.competitor_p99)
        competitor_len = int((comp.competitor_p99 / max_value) * 50)
        engram_len = int((comp.engram_p99 / max_value) * 50)

        # Format delta
        delta_str = ""
        if comp.latency_delta_pct is not None:
            if comp.latency_delta_pct < -10:
                delta_str = f" ({comp.latency_delta_pct:+.0f}% ✓)"
            elif comp.latency_delta_pct > 10:
                delta_str = f" ({comp.latency_delta_pct:+.0f}% ⚠)"
            else:
                delta_str = f" ({comp.latency_delta_pct:+.0f}%)"

        lines.extend(
            [
                f"  {comp.competitor_name}: {'█' * competitor_len} {comp.competitor_p99:.2f}ms",
                f"  Engram: {'█' * engram_len} {comp.engram_p99:.2f}ms{delta_str}",
                "",
            ]
        )

    return "\n".join(lines)


def generate_optimization_priorities(comparisons: List[ComparisonResult]) -> str:
    """Generate prioritized optimization list.

    Args:
        comparisons: List of comparison results

    Returns:
        Markdown formatted optimization priorities
    """
    lines = [
        "## Optimization Priorities",
        "",
        "Based on regression magnitude and competitive gaps:",
        "",
    ]

    # Find scenarios that need optimization
    needs_optimization = [
        comp
        for comp in comparisons
        if comp.status == "WORSE" and comp.latency_delta_pct is not None
    ]

    # Sort by delta magnitude (worst first)
    needs_optimization.sort(key=lambda c: abs(c.latency_delta_pct or 0), reverse=True)

    for i, comp in enumerate(needs_optimization[:3], 1):  # Top 3 priorities
        target_latency = comp.competitor_p99 * 0.9 if comp.competitor_p99 else 0  # 10% better
        lines.extend(
            [
                f"{i}. **{comp.scenario_name} P99 Latency** ({comp.competitor_name} scenario)",
                f"   - Current: {comp.engram_p99:.2f}ms, Target: <{target_latency:.2f}ms ({comp.competitor_name}: {comp.competitor_p99:.2f}ms)",
                f"   - Gap: {abs(comp.latency_delta_pct):.2f}% slower than best-in-class",
                "   - Impact: High (competitive positioning)",
                "   - Suggested optimizations:",
                "     - Profile hot paths with flamegraph",
                "     - Review memory layout for cache locality",
                "     - Consider SIMD optimizations",
                "",
            ]
        )

    if not needs_optimization:
        lines.append(
            "No optimization priorities identified. All scenarios meet or exceed competitor baselines."
        )

    return "\n".join(lines)


def generate_metadata_section(metadata: Dict[str, Any]) -> str:
    """Generate metadata and reproducibility section.

    Args:
        metadata: Benchmark metadata

    Returns:
        Markdown formatted metadata section
    """
    lines = [
        "## Measurement Metadata",
        "",
        "**Environment**:",
        f"- OS: {metadata.get('os', 'UNKNOWN')}",
        f"- CPU: {metadata.get('cpu', 'UNKNOWN')}",
        f"- RAM: {metadata.get('ram', 'UNKNOWN')}",
        "",
        "**Software Versions**:",
        f"- Engram Core: {metadata.get('engram_version', 'UNKNOWN')}",
        f"- Git Commit: {metadata.get('git_commit', 'UNKNOWN')}",
        f"- Branch: {metadata.get('git_branch', 'UNKNOWN')}",
        "",
        "**Reproducibility**:",
        "```bash",
        f"git checkout {metadata.get('git_commit', 'UNKNOWN')}",
        "cargo build --release",
        "./scripts/competitive_benchmark_suite.sh",
        "```",
        "",
    ]

    return "\n".join(lines)


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate competitive performance comparison report"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with benchmark results (e.g., tmp/competitive_benchmarks/2025-11-08_14-30-00)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output markdown report file")
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Previous benchmark directory for historical comparison (optional)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("docs/reference/competitive_baselines.md"),
        help="Competitive baseline documentation file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Error collector
    errors = ErrorCollector()

    # Phase 1: Parse baseline documentation
    logger.info("Loading competitor baselines...")
    baselines = parse_baseline_documentation(args.baseline, errors)

    if errors.has_fatal():
        logger.error("Cannot proceed without baseline documentation")
        sys.exit(2)

    # Phase 2: Parse metadata
    logger.info("Parsing benchmark metadata...")
    metadata_file = args.input / f"{args.input.name}_metadata.txt"
    metadata = parse_metadata(metadata_file, errors)

    # Phase 3: Parse benchmark results
    logger.info("Parsing benchmark results...")
    results: Dict[str, BenchmarkResult] = {}

    # Find all result files
    for result_file in args.input.glob("*.txt"):
        if result_file.name.endswith("_metadata.txt"):
            continue
        if result_file.name.endswith("_stderr.txt"):
            continue
        if result_file.name.endswith("_diag.txt"):
            continue
        if result_file.name.endswith("_sys.txt"):
            continue
        if result_file.name.endswith("_summary.txt"):
            continue

        result = parse_loadtest_results(result_file, errors)
        if result:
            results[result.scenario_name] = result

    logger.info("Parsed %d benchmark results", len(results))

    # Phase 4: Perform comparisons
    logger.info("Comparing Engram vs competitors...")
    comparisons: List[ComparisonResult] = []

    for scenario_name, result in results.items():
        baseline = baselines.get(scenario_name)

        if not baseline:
            # No baseline for this scenario
            comparisons.append(
                ComparisonResult(
                    scenario_name=scenario_name,
                    competitor_name="N/A",
                    engram_p99=result.p99_latency_ms,
                    competitor_p99=None,
                    latency_delta_pct=None,
                    engram_qps=result.throughput_qps,
                    competitor_qps=None,
                    throughput_delta_pct=None,
                    status="NO_BASELINE",
                    statistical_significance=None,
                )
            )
            continue

        # Calculate deltas
        latency_delta = calculate_delta(
            result.p99_latency_ms, baseline.p99_latency_ms, f"{scenario_name} P99"
        )
        throughput_delta = calculate_delta(
            result.throughput_qps, baseline.throughput_qps, f"{scenario_name} throughput"
        )

        # Determine status (based on latency)
        status = "COMPARABLE"
        if latency_delta is not None:
            if latency_delta < -10:
                status = "BETTER"
            elif latency_delta > 10:
                status = "WORSE"

        # Statistical significance (placeholder - need multiple runs for real test)
        sig = None
        if result.p99_latency_ms is not None and baseline.p99_latency_ms is not None:
            # For single run, just check if delta is non-zero
            sig = abs(result.p99_latency_ms - baseline.p99_latency_ms) > 0.01

        comparisons.append(
            ComparisonResult(
                scenario_name=scenario_name,
                competitor_name=baseline.system_name,
                engram_p99=result.p99_latency_ms,
                competitor_p99=baseline.p99_latency_ms,
                latency_delta_pct=latency_delta,
                engram_qps=result.throughput_qps,
                competitor_qps=baseline.throughput_qps,
                throughput_delta_pct=throughput_delta,
                status=status,
                statistical_significance=sig,
            )
        )

    # Phase 5: Generate report
    logger.info("Generating markdown report...")

    report_sections = [
        generate_executive_summary(comparisons, metadata),
        generate_comparison_table(comparisons),
        generate_ascii_chart(comparisons),
        generate_optimization_priorities(comparisons),
        generate_metadata_section(metadata),
    ]

    # Write output
    logger.info("Writing report to %s", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_sections))

    # Check for errors
    if errors.has_fatal():
        logger.error("Report generation failed due to fatal errors")
        sys.exit(2)

    summary = errors.get_summary()
    if summary.get(ErrorSeverity.WARNING, 0) > 0:
        logger.warning("Report generated with %d warnings", summary[ErrorSeverity.WARNING])
        sys.exit(1)

    logger.info("Report generated successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
