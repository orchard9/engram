#!/usr/bin/env bash
# Inject packet loss for chaos engineering testing
#
# Usage: ./inject_packet_loss.sh [LOSS_PERCENT] [INTERFACE]

set -euo pipefail

LOSS_PERCENT="${1:-5}"
INTERFACE="${2:-lo0}"

echo "================================"
echo "Chaos: Packet Loss Injection"
echo "================================"
echo ""
echo "Loss rate: ${LOSS_PERCENT}%"
echo "Interface: ${INTERFACE}"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux using tc (traffic control)
    echo "Platform: Linux"
    echo ""

    if ! command -v tc &> /dev/null; then
        echo "Error: 'tc' command not found. Install with: sudo apt-get install iproute2"
        exit 1
    fi

    echo "Adding ${LOSS_PERCENT}% packet loss to ${INTERFACE}..."
    sudo tc qdisc add dev "$INTERFACE" root netem loss "${LOSS_PERCENT}%"

    echo ""
    echo "✓ Packet loss injected successfully"
    echo ""
    echo "To remove:"
    echo "  sudo tc qdisc del dev $INTERFACE root"
    echo ""

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS using dnctl and pfctl
    echo "Platform: macOS"
    echo ""

    PIPE_NUM=2
    echo "Creating dummynet pipe with ${LOSS_PERCENT}% loss..."
    sudo dnctl pipe "$PIPE_NUM" config plr "${LOSS_PERCENT}"

    # Create pf rules
    PF_RULES="/tmp/engram_chaos_loss.pf"
    cat > "$PF_RULES" <<EOF
dummynet-anchor "engram_chaos"
anchor "engram_chaos"
EOF

    PF_ANCHOR="/tmp/engram_chaos_loss_anchor.pf"
    cat > "$PF_ANCHOR" <<EOF
dummynet in proto tcp from any to any pipe $PIPE_NUM
dummynet out proto tcp from any to any pipe $PIPE_NUM
EOF

    echo "Loading pf rules..."
    sudo pfctl -f "$PF_RULES"
    sudo pfctl -a engram_chaos -f "$PF_ANCHOR"
    sudo pfctl -e 2>/dev/null || true

    echo ""
    echo "✓ Packet loss injected successfully"
    echo ""
    echo "To remove:"
    echo "  sudo pfctl -a engram_chaos -F all"
    echo "  sudo dnctl -q flush"
    echo "  sudo pfctl -d"
    echo ""

else
    echo "Error: Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "⚠️  Remember to clean up after testing!"
echo ""
