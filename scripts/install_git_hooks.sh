#!/usr/bin/env bash
# Install git hooks for Engram development
#
# This script installs production-grade git hooks that enforce code quality
# and competitive benchmark freshness requirements.

set -euo pipefail
IFS=$'\n\t'

# Configuration
# shellcheck disable=SC2155
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2155
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly HOOKS_DIR="$PROJECT_ROOT/.git/hooks"
readonly TEMPLATE_FILE="$SCRIPT_DIR/pre-commit.template"

# Colors for output
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# Error handling
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Verify we're in a git repository
if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
    error_exit "Not a git repository. Please run from within the Engram project."
fi

# Verify template file exists
if [[ ! -f "$TEMPLATE_FILE" ]]; then
    error_exit "Template file not found: $TEMPLATE_FILE"
fi

echo "Installing Engram git hooks..."

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Install pre-commit hook
HOOK_FILE="$HOOKS_DIR/pre-commit"

# Backup existing hook if present (not a symlink)
if [[ -f "$HOOK_FILE" && ! -L "$HOOK_FILE" ]]; then
    backup_file="${HOOK_FILE}.backup.$(date +%s)"
    echo -e "${YELLOW}  Backing up existing pre-commit hook to $(basename "$backup_file")${NC}"
    mv "$HOOK_FILE" "$backup_file"
fi

# Copy template to hooks directory
cp "$TEMPLATE_FILE" "$HOOK_FILE"
chmod +x "$HOOK_FILE"

echo -e "${GREEN}Installed pre-commit hook: $HOOK_FILE${NC}"
echo ""
echo "The pre-commit hook will:"
echo "  - Enforce quarterly competitive benchmarking (every 90 days)"
echo "  - Block commits if benchmarks are stale"
echo "  - Warn when benchmarks expire in <14 days"
echo ""
echo "To bypass the hook (not recommended):"
echo "  SKIP_BENCHMARK_CHECK=1 git commit -m \"message\""
echo ""
echo "To run benchmarks:"
echo "  ./scripts/competitive_benchmark_suite.sh"
echo ""
echo "Installation complete."
