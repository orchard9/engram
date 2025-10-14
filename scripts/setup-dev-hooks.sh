#!/bin/bash
# Developer environment setup - installs git hooks for code quality
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "🔧 Setting up development environment for Engram"
echo ""

# Install pre-commit hook
echo "📋 Installing pre-commit hook..."
cp "$SCRIPT_DIR/git-hooks/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed"
echo ""
echo "The hook will run 'make quality' before each commit to ensure:"
echo "  • Code formatting (cargo fmt)"
echo "  • Linting (cargo clippy)"
echo "  • Tests pass (cargo test)"
echo "  • Documentation builds"
echo ""
echo "To bypass the hook (not recommended): git commit --no-verify"
echo ""
echo "✨ Development environment ready!"
