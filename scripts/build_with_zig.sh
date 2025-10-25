#!/usr/bin/env bash
# Build Engram with Zig performance kernels enabled
#
# This script simplifies building with the zig-kernels feature flag.
# It verifies Zig is installed and builds with appropriate optimizations.
#
# Usage:
#   ./scripts/build_with_zig.sh           # Debug build
#   ./scripts/build_with_zig.sh release   # Release build

set -euo pipefail

# Change to project root
cd "$(dirname "$0")/.."

# Check if Zig is installed
if ! command -v zig &> /dev/null; then
    echo "ERROR: Zig compiler not found"
    echo ""
    echo "Install Zig 0.13.0 from: https://ziglang.org/download/"
    echo ""
    echo "Or build without Zig kernels:"
    echo "  cargo build"
    exit 1
fi

# Display Zig version
ZIG_VERSION=$(zig version)
echo "Found Zig version: $ZIG_VERSION"

# Warn if not using recommended version
if [[ "$ZIG_VERSION" != "0.13.0" ]]; then
    echo "WARNING: Recommended Zig version is 0.13.0, found $ZIG_VERSION"
    echo "ABI compatibility may be affected."
    echo ""
fi

# Determine build profile
PROFILE="${1:-debug}"

case "$PROFILE" in
    debug)
        echo "Building Engram with Zig kernels (debug mode)..."
        cargo build --features zig-kernels
        ;;
    release)
        echo "Building Engram with Zig kernels (release mode)..."
        cargo build --release --features zig-kernels
        ;;
    *)
        echo "ERROR: Unknown profile '$PROFILE'"
        echo "Usage: $0 [debug|release]"
        exit 1
        ;;
esac

echo ""
echo "Build completed successfully!"
echo ""
echo "To verify Zig kernels are linked:"
echo "  nm target/$PROFILE/engram-cli | grep engram_vector_similarity"
echo ""
echo "To run with Zig kernels:"
echo "  cargo run --features zig-kernels"
