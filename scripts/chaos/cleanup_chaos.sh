#!/usr/bin/env bash
# Cleanup all chaos engineering injections
#
# Usage: ./cleanup_chaos.sh

set -euo pipefail

echo "================================"
echo "Chaos Cleanup"
echo "================================"
echo ""

CLEANED=0

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Platform: Linux"
    echo ""

    # Remove tc rules
    echo "Cleaning up traffic control rules..."
    for IFACE in eth0 lo; do
        if sudo tc qdisc del dev "$IFACE" root 2>/dev/null; then
            echo "  ✓ Removed rules from $IFACE"
            CLEANED=1
        fi
    done

    # Kill stress processes
    if killall -9 stress-ng 2>/dev/null; then
        echo "  ✓ Killed stress-ng processes"
        CLEANED=1
    fi

    if killall -9 dd 2>/dev/null; then
        echo "  ✓ Killed dd processes"
        CLEANED=1
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Platform: macOS"
    echo ""

    # Remove pf rules
    echo "Cleaning up packet filter rules..."
    if sudo pfctl -a engram_chaos -F all 2>/dev/null; then
        echo "  ✓ Flushed pf anchor rules"
        CLEANED=1
    fi

    # Flush dummynet pipes
    if sudo dnctl -q flush 2>/dev/null; then
        echo "  ✓ Flushed dummynet pipes"
        CLEANED=1
    fi

    # Disable pf (only if no other rules)
    # sudo pfctl -d 2>/dev/null || true

    # Kill stress processes
    if killall yes 2>/dev/null; then
        echo "  ✓ Killed CPU burner processes"
        CLEANED=1
    fi

    # Cleanup temp files
    rm -f /tmp/engram_chaos_*.pf
fi

# Kill Python stress processes
if killall -9 python3 2>/dev/null | grep -q "memory"; then
    echo "  ✓ Killed Python memory stress"
    CLEANED=1
fi

echo ""
if [ "$CLEANED" -eq 1 ]; then
    echo "✓ Chaos injections cleaned up"
else
    echo "No active chaos injections found"
fi

echo ""
