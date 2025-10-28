#!/usr/bin/env bash
# Inject network latency for chaos engineering testing
#
# Usage: ./inject_network_latency.sh [LATENCY_MS] [INTERFACE]

set -euo pipefail

LATENCY_MS="${1:-100}"
INTERFACE="${2:-lo0}"

echo "================================"
echo "Chaos: Network Latency Injection"
echo "================================"
echo ""
echo "Latency:   ${LATENCY_MS}ms"
echo "Interface: ${INTERFACE}"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux using tc (traffic control)
    echo "Platform: Linux"
    echo ""

    # Check if tc is available
    if ! command -v tc &> /dev/null; then
        echo "Error: 'tc' command not found. Install with: sudo apt-get install iproute2"
        exit 1
    fi

    # Add latency
    echo "Adding ${LATENCY_MS}ms latency to ${INTERFACE}..."
    sudo tc qdisc add dev "$INTERFACE" root netem delay "${LATENCY_MS}ms"

    echo ""
    echo "✓ Latency injected successfully"
    echo ""
    echo "To remove:"
    echo "  sudo tc qdisc del dev $INTERFACE root"
    echo ""

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS using dnctl and pfctl
    echo "Platform: macOS"
    echo ""

    # Create dummynet pipe with latency
    echo "Creating dummynet pipe with ${LATENCY_MS}ms delay..."

    PIPE_NUM=1
    sudo dnctl pipe "$PIPE_NUM" config delay "${LATENCY_MS}ms"

    # Create pf rules
    PF_RULES="/tmp/engram_chaos_latency.pf"
    cat > "$PF_RULES" <<EOF
dummynet-anchor "engram_chaos"
anchor "engram_chaos"
EOF

    # Create anchor rules
    PF_ANCHOR="/tmp/engram_chaos_anchor.pf"
    cat > "$PF_ANCHOR" <<EOF
dummynet in proto tcp from any to any pipe $PIPE_NUM
dummynet out proto tcp from any to any pipe $PIPE_NUM
EOF

    echo "Loading pf rules..."
    sudo pfctl -f "$PF_RULES"
    sudo pfctl -a engram_chaos -f "$PF_ANCHOR"
    sudo pfctl -e 2>/dev/null || true

    echo ""
    echo "✓ Latency injected successfully"
    echo ""
    echo "To remove:"
    echo "  sudo pfctl -a engram_chaos -F all"
    echo "  sudo dnctl -q flush"
    echo "  sudo pfctl -d"
    echo ""
    echo "Or run: ./scripts/chaos/cleanup_chaos.sh"
    echo ""

else
    echo "Error: Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "Testing latency (ping localhost):"
ping -c 3 localhost || true

echo ""
echo "⚠️  Remember to clean up after testing!"
echo ""
