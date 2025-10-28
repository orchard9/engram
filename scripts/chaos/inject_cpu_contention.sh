#!/usr/bin/env bash
# Inject CPU contention for chaos engineering testing
#
# Usage: ./inject_cpu_contention.sh [NUM_WORKERS] [DURATION_SECS]

set -euo pipefail

NUM_WORKERS="${1:-4}"
DURATION_SECS="${2:-300}"

echo "================================"
echo "Chaos: CPU Contention Injection"
echo "================================"
echo ""
echo "Workers:  ${NUM_WORKERS}"
echo "Duration: ${DURATION_SECS}s"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Platform: Linux"

    if command -v stress-ng &> /dev/null; then
        echo "Using stress-ng..."
        stress-ng --cpu "$NUM_WORKERS" --timeout "${DURATION_SECS}s" &
        STRESS_PID=$!
        echo "✓ CPU contention injected (PID: $STRESS_PID)"
    else
        echo "stress-ng not found. Using fallback CPU burner..."

        # Fallback: spawn CPU-burning processes
        for i in $(seq 1 "$NUM_WORKERS"); do
            dd if=/dev/zero of=/dev/null bs=1M &
        done

        echo "✓ CPU contention injected (${NUM_WORKERS} workers)"
        sleep "$DURATION_SECS"
        killall dd 2>/dev/null || true
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Platform: macOS"
    echo "Using CPU burner..."

    # Spawn CPU-burning processes
    for i in $(seq 1 "$NUM_WORKERS"); do
        yes > /dev/null &
    done

    STRESS_PID=$!
    echo "✓ CPU contention injected (${NUM_WORKERS} workers)"

    # Wait and cleanup
    sleep "$DURATION_SECS"
    killall yes 2>/dev/null || true

else
    echo "Error: Unsupported operating system: $OSTYPE"
    exit 1
fi

echo ""
echo "To monitor CPU usage:"
echo "  top"
echo ""
echo "CPU contention will release after ${DURATION_SECS}s"
echo ""
