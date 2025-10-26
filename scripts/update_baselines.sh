#!/usr/bin/env bash
# Update performance baselines for regression testing
#
# This script runs benchmarks in UPDATE_BASELINES mode to record new
# baseline performance measurements. Use this after:
# 1. Intentional performance improvements
# 2. API changes that affect benchmark behavior
# 3. Platform or compiler upgrades
#
# The updated baselines should be reviewed and committed to version control.
#
# Usage:
#   ./scripts/update_baselines.sh

set -euo pipefail

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================"
echo "Update Performance Baselines"
echo "================================================"
echo ""

BASELINES_FILE="engram-core/benches/regression/baselines.json"

# Backup existing baselines if they exist
if [ -f "$BASELINES_FILE" ]; then
    BACKUP_FILE="${BASELINES_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}Backing up existing baselines to:${NC}"
    echo "  $BACKUP_FILE"
    cp "$BASELINES_FILE" "$BACKUP_FILE"
    echo ""
fi

# Display platform information
echo "Platform: $(uname -s) $(uname -m)"
echo "Date: $(date)"
echo ""

# Warn about system conditions
echo -e "${YELLOW}IMPORTANT: For accurate baselines:${NC}"
echo "  - Close unnecessary applications"
echo "  - Disable CPU frequency scaling if possible"
echo "  - Ensure system is not under load"
echo "  - Run on the same hardware as CI/production"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Clean build to ensure consistent results
echo "Cleaning previous build artifacts..."
cargo clean --package engram-core
echo ""

# Build with release optimizations
echo "Building with release optimizations..."
cargo build --release --package engram-core
echo ""

# Run benchmarks in update mode
echo "Running benchmarks to establish new baselines..."
echo "This will take several minutes..."
echo ""

UPDATE_BASELINES=1 cargo bench --package engram-core --bench regression -- --noplot

echo ""
echo "================================================"
echo -e "${GREEN}✓ Baselines updated successfully${NC}"
echo "================================================"
echo ""

# Show diff if baselines existed before
if [ -f "${BASELINES_FILE}.backup."* ]; then
    echo -e "${BLUE}Changes to baselines:${NC}"
    # Find most recent backup
    LATEST_BACKUP=$(ls -t "${BASELINES_FILE}.backup."* 2>/dev/null | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        git diff --no-index "$LATEST_BACKUP" "$BASELINES_FILE" || true
    fi
    echo ""
fi

echo "Next steps:"
echo "  1. Review the changes:"
echo "     cat $BASELINES_FILE"
echo ""
echo "  2. Verify baselines are reasonable for your platform"
echo ""
echo "  3. Commit the updated baselines:"
echo "     git add $BASELINES_FILE"
echo "     git commit -m 'chore: update performance baselines for $(uname -m)'"
echo ""
echo "  4. Run regression check to verify:"
echo "     ./scripts/benchmark_regression.sh"
echo ""
