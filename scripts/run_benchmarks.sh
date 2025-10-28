#!/usr/bin/env bash
# Run comprehensive benchmark suite with statistical validation
#
# Usage: ./scripts/run_benchmarks.sh [--baseline baseline_name] [--output results.json]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Benchmark configuration (per task spec requirements)
WARMUP_SECS=3
MEASUREMENT_SECS=10
SAMPLE_SIZE=50
CONFIDENCE_LEVEL=0.95
MAX_VARIANCE=0.05  # 5% coefficient of variation threshold

# Statistical validation using Welch's t-test
# Null hypothesis: no performance difference from baseline
# Alternative: performance degraded (one-tailed test)
ALPHA=0.05  # Significance level

# Parse arguments
BASELINE_NAME=""
OUTPUT_FILE=""
SAVE_BASELINE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            BASELINE_NAME="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --save-baseline)
            SAVE_BASELINE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --baseline NAME      Compare against baseline NAME"
            echo "  --output FILE        Save results to FILE (JSON format)"
            echo "  --save-baseline NAME Save current results as baseline NAME"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

echo "================================"
echo "Engram Benchmark Suite"
echo "================================"
echo ""
echo "Configuration:"
echo "  Warmup time:      ${WARMUP_SECS}s"
echo "  Measurement time: ${MEASUREMENT_SECS}s"
echo "  Sample size:      ${SAMPLE_SIZE}"
echo "  Confidence level: ${CONFIDENCE_LEVEL}"
echo "  Significance:     α = ${ALPHA}"
echo ""

cd "$PROJECT_ROOT"

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required for statistical analysis"
    exit 1
fi

# Check for required Python packages
if ! python3 -c "import scipy" 2>/dev/null; then
    echo "Error: scipy is required. Install with: pip3 install scipy numpy"
    exit 1
fi

# Build benchmarks in release mode
echo "Building benchmarks..."
cargo build --release --benches

# Run comprehensive benchmark suite
echo ""
echo "Running benchmarks..."
echo "This may take 10-15 minutes depending on hardware."
echo ""

CRITERION_ARGS="--verbose"

if [ -n "$BASELINE_NAME" ]; then
    echo "Comparing against baseline: $BASELINE_NAME"
    CRITERION_ARGS="$CRITERION_ARGS --baseline $BASELINE_NAME"
fi

if [ -n "$SAVE_BASELINE" ]; then
    echo "Saving results as baseline: $SAVE_BASELINE"
    CRITERION_ARGS="$CRITERION_ARGS --save-baseline $SAVE_BASELINE"
fi

# Run all benchmarks
cargo bench --bench comprehensive -- $CRITERION_ARGS

# Also run deterministic overhead benchmark
cargo bench --bench deterministic_overhead -- $CRITERION_ARGS

echo ""
echo "Benchmark execution complete!"
echo ""

# Perform regression analysis if baseline specified
if [ -n "$BASELINE_NAME" ]; then
    echo "Performing regression analysis..."

    ANALYSIS_OUTPUT="${OUTPUT_FILE:-target/benchmark_analysis.json}"

    python3 "$SCRIPT_DIR/analyze_benchmarks.py" \
        --baseline "target/criterion/$BASELINE_NAME" \
        --current "target/criterion" \
        --output "$ANALYSIS_OUTPUT" \
        --alpha "$ALPHA"

    ANALYSIS_EXIT=$?

    if [ $ANALYSIS_EXIT -eq 0 ]; then
        echo ""
        echo "✓ No regressions detected"
        echo "  Analysis saved to: $ANALYSIS_OUTPUT"
    elif [ $ANALYSIS_EXIT -eq 1 ]; then
        echo ""
        echo "⚠️  WARNING: Performance regressions detected"
        echo "  See analysis: $ANALYSIS_OUTPUT"
        exit 1
    elif [ $ANALYSIS_EXIT -eq 2 ]; then
        echo ""
        echo "❌ CRITICAL: Severe performance regressions detected"
        echo "  See analysis: $ANALYSIS_OUTPUT"
        exit 2
    else
        echo ""
        echo "❌ Analysis failed"
        exit 3
    fi
fi

# Generate HTML report
echo ""
echo "Benchmark results available at:"
echo "  target/criterion/report/index.html"
echo ""
echo "To view the report:"
echo "  open target/criterion/report/index.html"
echo ""
