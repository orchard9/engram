#!/usr/bin/env bash
# Inject disk I/O slowdown for chaos engineering testing
#
# Usage: ./inject_disk_slow.sh [BANDWIDTH_MBPS]

set -euo pipefail

BANDWIDTH_MBPS="${1:-10}"

echo "================================"
echo "Chaos: Disk I/O Throttling"
echo "================================"
echo ""
echo "Bandwidth limit: ${BANDWIDTH_MBPS} MB/s"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Platform: Linux"
    echo ""

    # Check for ionice
    if ! command -v ionice &> /dev/null; then
        echo "Error: 'ionice' command not found"
        exit 1
    fi

    echo "To throttle a specific process, use:"
    echo "  ionice -c 3 -p PID"
    echo ""
    echo "Where PID is the Engram process ID"
    echo ""
    echo "Class 3 = Idle (lowest priority)"
    echo ""

    # Optionally use cgroups for more precise control
    if [ -d /sys/fs/cgroup/blkio ]; then
        echo "For more precise control, you can use cgroups:"
        echo "  # Create cgroup"
        echo "  sudo cgcreate -g blkio:/engram_throttle"
        echo "  # Set read/write limits (bytes per second)"
        echo "  echo $((${BANDWIDTH_MBPS} * 1024 * 1024)) | sudo tee /sys/fs/cgroup/blkio/engram_throttle/blkio.throttle.read_bps_device"
        echo "  # Move process to cgroup"
        echo "  echo PID | sudo tee /sys/fs/cgroup/blkio/engram_throttle/tasks"
        echo ""
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Platform: macOS"
    echo ""
    echo "macOS does not have built-in I/O throttling like Linux."
    echo ""
    echo "To simulate slow disk:"
    echo "1. Use a RAM disk with synthetic delays"
    echo "2. Or run I/O-intensive background tasks:"
    echo "   dd if=/dev/zero of=/tmp/disk_load bs=1m count=10000 &"
    echo ""

else
    echo "Error: Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "⚠️  This is a manual configuration step"
echo ""
