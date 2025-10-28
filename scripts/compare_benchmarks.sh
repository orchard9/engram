#!/usr/bin/env bash
# Compare two benchmark baselines for regression detection
#
# Usage: ./scripts/compare_benchmarks.sh BASELINE CURRENT [--output results.json]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ALPHA=0.05

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 BASELINE CURRENT [--output FILE]"
    echo ""
    echo "Compare benchmark results between two baselines."
    echo ""
    echo "Arguments:"
    echo "  BASELINE  Name of baseline benchmark"
    echo "  CURRENT   Name of current benchmark to compare"
    echo ""
    echo "Options:"
    echo "  --output FILE  Save comparison results to FILE (JSON)"
    echo ""
    echo "Example:"
    echo "  $0 v0.1.0 main --output regression_report.json"
    exit 1
fi

BASELINE="$1"
CURRENT="$2"
shift 2

OUTPUT_FILE="target/benchmark_comparison.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo "================================"
echo "Benchmark Comparison"
echo "================================"
echo ""
echo "Baseline: $BASELINE"
echo "Current:  $CURRENT"
echo "Alpha:    $ALPHA"
echo ""

# Check that both baselines exist
BASELINE_DIR="target/criterion/$BASELINE"
CURRENT_DIR="target/criterion/$CURRENT"

if [ ! -d "$BASELINE_DIR" ]; then
    echo "Error: Baseline directory not found: $BASELINE_DIR"
    echo ""
    echo "Available baselines:"
    ls -1 target/criterion/ | grep -v "report" | grep -v "\.json" || echo "  (none)"
    exit 1
fi

if [ ! -d "$CURRENT_DIR" ]; then
    echo "Error: Current directory not found: $CURRENT_DIR"
    echo ""
    echo "Available baselines:"
    ls -1 target/criterion/ | grep -v "report" | grep -v "\.json" || echo "  (none)"
    exit 1
fi

# Run statistical analysis
echo "Analyzing performance differences..."
python3 "$SCRIPT_DIR/analyze_benchmarks.py" \
    --baseline "$BASELINE_DIR" \
    --current "$CURRENT_DIR" \
    --output "$OUTPUT_FILE" \
    --alpha "$ALPHA"

ANALYSIS_EXIT=$?

echo ""
echo "Comparison complete!"
echo "Results saved to: $OUTPUT_FILE"

exit $ANALYSIS_EXIT
