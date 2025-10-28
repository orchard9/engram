#!/usr/bin/env bash
# Inject memory pressure for chaos engineering testing
#
# Usage: ./inject_memory_pressure.sh [MEMORY_MB] [DURATION_SECS]

set -euo pipefail

MEMORY_MB="${1:-1024}"
DURATION_SECS="${2:-300}"

echo "================================"
echo "Chaos: Memory Pressure Injection"
echo "================================"
echo ""
echo "Allocate: ${MEMORY_MB}MB"
echo "Duration: ${DURATION_SECS}s"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Platform: Linux"

    if command -v stress-ng &> /dev/null; then
        echo "Using stress-ng..."
        stress-ng --vm 1 --vm-bytes "${MEMORY_MB}M" --timeout "${DURATION_SECS}s" &
        STRESS_PID=$!
        echo "✓ Memory pressure injected (PID: $STRESS_PID)"
    else
        echo "stress-ng not found. Using fallback memory allocator..."

        # Fallback: Python memory allocator
        python3 -c "
import time
import sys

memory_mb = int(sys.argv[1])
duration = int(sys.argv[2])

print(f'Allocating {memory_mb}MB of memory...')
# Allocate memory (list of bytes)
data = bytearray(memory_mb * 1024 * 1024)

# Fill with data to ensure physical allocation
for i in range(0, len(data), 4096):
    data[i] = 0xFF

print(f'Memory allocated. Holding for {duration}s...')
time.sleep(duration)
print('Releasing memory...')
" "$MEMORY_MB" "$DURATION_SECS" &

        STRESS_PID=$!
        echo "✓ Memory pressure injected (PID: $STRESS_PID)"
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Platform: macOS"
    echo "Using Python memory allocator..."

    python3 -c "
import time
import sys

memory_mb = int(sys.argv[1])
duration = int(sys.argv[2])

print(f'Allocating {memory_mb}MB of memory...')
data = bytearray(memory_mb * 1024 * 1024)

# Fill with data to ensure physical allocation
for i in range(0, len(data), 4096):
    data[i] = 0xFF

print(f'Memory allocated. Holding for {duration}s...')
time.sleep(duration)
print('Releasing memory...')
" "$MEMORY_MB" "$DURATION_SECS" &

    STRESS_PID=$!
    echo "✓ Memory pressure injected (PID: $STRESS_PID)"

else
    echo "Error: Unsupported operating system: $OSTYPE"
    exit 1
fi

echo ""
echo "To monitor memory usage:"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  watch -n 1 free -h"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  watch -n 1 vm_stat"
fi

echo ""
echo "To stop early:"
echo "  kill $STRESS_PID"
echo ""
echo "Memory pressure will automatically release after ${DURATION_SECS}s"
echo ""
