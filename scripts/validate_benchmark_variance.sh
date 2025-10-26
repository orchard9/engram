#!/usr/bin/env bash
#
# Validate that benchmark variance is less than 5% across multiple runs
#
# Usage: ./scripts/validate_benchmark_variance.sh [benchmark] [metric_pattern]
# Example: ./scripts/validate_benchmark_variance.sh baseline_performance "vector_similarity"
#
# This script runs the specified benchmark multiple times and computes
# the coefficient of variation for each metric to ensure statistical stability.

set -euo pipefail

RUNS=${RUNS:-10}
BENCHMARK=${1:-"baseline_performance"}
METRIC_PATTERN=${2:-"vector_similarity"}
VARIANCE_THRESHOLD=5.0  # 5% coefficient of variation

echo "====================================="
echo "  Benchmark Variance Validation"
echo "====================================="
echo "Benchmark: $BENCHMARK"
echo "Metric pattern: $METRIC_PATTERN"
echo "Runs: $RUNS"
echo "Variance threshold: ${VARIANCE_THRESHOLD}%"
echo

# Create tmp directory
mkdir -p tmp/variance_validation

echo "Running benchmark $RUNS times..."
for i in $(seq 1 $RUNS); do
    echo -n "  Run $i/$RUNS..."
    cargo bench --bench "$BENCHMARK" -- "$METRIC_PATTERN" \
        --noplot \
        --save-baseline "variance_run_$i" 2>&1 \
        > "tmp/variance_validation/run_${i}.log"
    echo " done"
done

echo
echo "Analyzing results..."

# Extract mean times from each run and compute variance
python3 << 'EOF'
import re
import statistics
import sys
import glob

# Parse all run logs
run_times = {}
for log_file in sorted(glob.glob('tmp/variance_validation/run_*.log')):
    with open(log_file) as f:
        content = f.read()

    # Find all "time: [lower mean upper]" lines
    # Format: "time:   [59.585 µs 60.037 µs 60.559 µs]"
    for match in re.finditer(r'time:\s+\[([0-9.]+)\s+([µnm]s)\s+([0-9.]+)\s+([µnm]s)\s+([0-9.]+)\s+([µnm]s)\]', content):
        # Extract middle value (mean estimate)
        mean_value = float(match.group(3))
        unit = match.group(4)

        # Convert to nanoseconds for consistency
        if unit == 'µs':
            mean_value *= 1000
        elif unit == 'ms':
            mean_value *= 1_000_000

        # Find the benchmark name (appears in the line before "time:")
        # Look backward in content to find the benchmark name
        prefix = content[max(0, match.start()-300):match.start()]
        # Extract last non-whitespace word before "time:"
        bench_name_match = re.search(r'(\S+)\s*$', prefix)
        if bench_name_match:
            bench_name = bench_name_match.group(1)
        else:
            # Fallback: look for "Benchmarking" line
            bench_line_match = re.search(r'Benchmarking\s+(\S+)', prefix)
            if bench_line_match:
                bench_name = bench_line_match.group(1)
            else:
                bench_name = "unknown"

        if bench_name not in run_times:
            run_times[bench_name] = []
        run_times[bench_name].append(mean_value)

# Analyze variance for each benchmark
print("\nVariance Analysis:")
print("=" * 90)
print(f"{'Benchmark':<55} {'Mean':<12} {'StdDev':<8} {'CV%':<8} {'Status'}")
print("=" * 90)

all_pass = True
total_benchmarks = 0
pass_count = 0

for bench_name, times in sorted(run_times.items()):
    if len(times) < 2:
        continue

    total_benchmarks += 1
    mean = statistics.mean(times)
    stdev = statistics.stdev(times)
    cv = (stdev / mean) * 100  # Coefficient of variation (%)

    # Format time nicely
    if mean < 1000:
        mean_str = f"{mean:.2f} ns"
    elif mean < 1_000_000:
        mean_str = f"{mean/1000:.2f} µs"
    else:
        mean_str = f"{mean/1_000_000:.2f} ms"

    status = "PASS" if cv < 5.0 else "FAIL"
    if cv < 5.0:
        pass_count += 1
    else:
        all_pass = False

    print(f"{bench_name:<55} {mean_str:>12}  {cv:>6.2f}%  {status}")

print("=" * 90)
print(f"\nSummary: {pass_count}/{total_benchmarks} benchmarks passed variance check")

if all_pass:
    print("\n✓ All benchmarks have variance < 5%")
    sys.exit(0)
else:
    print(f"\n✗ {total_benchmarks - pass_count} benchmark(s) exceed 5% variance threshold")
    print("\nRecommendations:")
    print("  1. Close background applications")
    print("  2. Disable CPU frequency scaling")
    print("  3. Run benchmarks multiple times and take median")
    print("  4. Increase sample size in benchmark configuration")
    sys.exit(1)
EOF
