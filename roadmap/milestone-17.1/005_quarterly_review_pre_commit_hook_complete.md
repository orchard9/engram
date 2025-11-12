# Task 005: Quarterly Review Pre-Commit Hook

**Status**: Pending
**Complexity**: Simple
**Dependencies**: Task 003 (requires benchmark results structure)
**Estimated Effort**: 2 hours

## Objective

Create a git pre-commit hook that enforces quarterly competitive benchmarking by preventing commits if competitive benchmarks haven't been run within the last 90 days (one quarter). This ensures the team maintains up-to-date competitive positioning data.

## Specifications

### 1. Pre-Commit Hook Script

**Create**: `.git/hooks/pre-commit` (or template for installation)

**Behavior**:
- Check `tmp/competitive_benchmarks/` for most recent metadata file
- Extract timestamp from filename (format: `YYYY-MM-DD_HH-MM-SS`)
- Calculate days since last benchmark
- If >90 days: Block commit with clear error message
- If ≤90 days: Allow commit
- If no benchmarks found: Block with setup instructions

**Edge Cases**:
- Empty benchmark directory: Treat as no benchmarks (block with instructions)
- Malformed timestamp: Skip file, continue checking others
- Missing tmp directory: Block with setup instructions
- CI/CD environments: Allow bypass with `SKIP_BENCHMARK_CHECK=1` env var

### 2. Hook Implementation

```bash
#!/bin/bash
# Pre-commit hook: Enforce quarterly competitive benchmarking
#
# This hook prevents commits if competitive benchmarks haven't been run
# within the last 90 days (one quarter).
#
# Bypass: SKIP_BENCHMARK_CHECK=1 git commit -m "message"

set -e

# Configuration
readonly BENCHMARK_DIR="tmp/competitive_benchmarks"
readonly MAX_AGE_DAYS=90
readonly BYPASS_VAR="SKIP_BENCHMARK_CHECK"

# Colors for output
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly GREEN='\033[0;32m'
readonly NC='\033[0m' # No Color

# Check for bypass
if [[ -n "${!BYPASS_VAR}" ]]; then
    echo -e "${YELLOW}Bypassing benchmark freshness check (SKIP_BENCHMARK_CHECK=1)${NC}"
    exit 0
fi

# Check if benchmark directory exists
if [[ ! -d "$BENCHMARK_DIR" ]]; then
    echo -e "${RED}ERROR: Competitive benchmark directory not found${NC}"
    echo ""
    echo "Competitive benchmarks must be run quarterly (every 90 days)."
    echo ""
    echo "To create initial benchmark:"
    echo "  1. Build release binary: cargo build --release"
    echo "  2. Run benchmark suite: ./scripts/competitive_benchmark_suite.sh"
    echo "  3. Retry your commit"
    echo ""
    echo "To bypass this check (not recommended):"
    echo "  SKIP_BENCHMARK_CHECK=1 git commit -m \"your message\""
    exit 1
fi

# Find most recent metadata file
latest_metadata=$(find "$BENCHMARK_DIR" -name "*_metadata.txt" -type f 2>/dev/null | sort -r | head -1)

if [[ -z "$latest_metadata" ]]; then
    echo -e "${RED}ERROR: No competitive benchmarks found${NC}"
    echo ""
    echo "Competitive benchmarks must be run quarterly (every 90 days)."
    echo ""
    echo "To run benchmarks:"
    echo "  ./scripts/competitive_benchmark_suite.sh"
    echo ""
    echo "Duration: ~10 minutes"
    echo ""
    echo "To bypass this check (not recommended):"
    echo "  SKIP_BENCHMARK_CHECK=1 git commit -m \"your message\""
    exit 1
fi

# Extract timestamp from filename (format: YYYY-MM-DD_HH-MM-SS_metadata.txt)
filename=$(basename "$latest_metadata")
timestamp_str="${filename%%_metadata.txt}"

# Parse date (handle both macOS and Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS date command
    benchmark_epoch=$(date -j -f "%Y-%m-%d_%H-%M-%S" "$timestamp_str" "+%s" 2>/dev/null)
else
    # Linux date command
    benchmark_date="${timestamp_str//_/ }"
    benchmark_epoch=$(date -d "$benchmark_date" "+%s" 2>/dev/null)
fi

if [[ -z "$benchmark_epoch" ]]; then
    echo -e "${YELLOW}WARNING: Could not parse benchmark timestamp: $timestamp_str${NC}"
    echo "Allowing commit, but please verify benchmark data integrity."
    exit 0
fi

# Calculate age in days
current_epoch=$(date "+%s")
age_seconds=$((current_epoch - benchmark_epoch))
age_days=$((age_seconds / 86400))

# Format dates for display
if [[ "$OSTYPE" == "darwin"* ]]; then
    benchmark_date_display=$(date -j -f "%s" "$benchmark_epoch" "+%Y-%m-%d %H:%M:%S" 2>/dev/null)
    expiry_epoch=$((benchmark_epoch + MAX_AGE_DAYS * 86400))
    expiry_date_display=$(date -j -f "%s" "$expiry_epoch" "+%Y-%m-%d" 2>/dev/null)
else
    benchmark_date_display=$(date -d "@$benchmark_epoch" "+%Y-%m-%d %H:%M:%S" 2>/dev/null)
    expiry_epoch=$((benchmark_epoch + MAX_AGE_DAYS * 86400))
    expiry_date_display=$(date -d "@$expiry_epoch" "+%Y-%m-%d" 2>/dev/null)
fi

# Check if benchmarks are stale
if (( age_days > MAX_AGE_DAYS )); then
    days_overdue=$((age_days - MAX_AGE_DAYS))

    echo -e "${RED}ERROR: Competitive benchmarks are stale${NC}"
    echo ""
    echo "  Last benchmark: $benchmark_date_display ($age_days days ago)"
    echo "  Maximum age: $MAX_AGE_DAYS days"
    echo "  Days overdue: $days_overdue"
    echo ""
    echo "Competitive benchmarks must be run quarterly to ensure up-to-date"
    echo "competitive positioning data."
    echo ""
    echo "To update benchmarks:"
    echo "  ./scripts/competitive_benchmark_suite.sh"
    echo ""
    echo "Duration: ~10 minutes"
    echo ""
    echo "To bypass this check (not recommended):"
    echo "  SKIP_BENCHMARK_CHECK=1 git commit -m \"your message\""
    exit 1
fi

# Success: benchmarks are fresh
days_remaining=$((MAX_AGE_DAYS - age_days))

echo -e "${GREEN}✓ Competitive benchmarks are up to date${NC}"
echo "  Last benchmark: $benchmark_date_display ($age_days days ago)"
echo "  Next benchmark due: $expiry_date_display ($days_remaining days remaining)"

# Warning if approaching expiry (within 14 days)
if (( days_remaining <= 14 )); then
    echo ""
    echo -e "${YELLOW}⚠  Benchmarks expire in $days_remaining days${NC}"
    echo "  Schedule quarterly review: ./scripts/competitive_benchmark_suite.sh"
fi

exit 0
```

### 3. Installation Script

**Create**: `scripts/install_git_hooks.sh`

```bash
#!/bin/bash
# Install git hooks for Engram development

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "Installing Engram git hooks..."

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Install pre-commit hook
HOOK_FILE="$HOOKS_DIR/pre-commit"

if [[ -f "$HOOK_FILE" && ! -L "$HOOK_FILE" ]]; then
    echo "  Backing up existing pre-commit hook..."
    mv "$HOOK_FILE" "${HOOK_FILE}.backup.$(date +%s)"
fi

cat > "$HOOK_FILE" << 'HOOK_CONTENT'
#!/bin/bash
# Pre-commit hook: Enforce quarterly competitive benchmarking
#
# This hook prevents commits if competitive benchmarks haven't been run
# within the last 90 days (one quarter).
#
# Bypass: SKIP_BENCHMARK_CHECK=1 git commit -m "message"

set -e

# [Full hook content from above]
HOOK_CONTENT

chmod +x "$HOOK_FILE"

echo "✓ Installed pre-commit hook: $HOOK_FILE"
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
```

### 4. Documentation

**Add to** `docs/reference/competitive_baselines.md`:

```markdown
## Automated Enforcement: Pre-Commit Hook

To ensure competitive benchmarks stay current, a git pre-commit hook enforces quarterly measurements. This hook prevents commits if benchmarks haven't been run within the last 90 days.

**Installation**:
```bash
./scripts/install_git_hooks.sh
```

**Behavior**:
- Checks timestamp of most recent benchmark in `tmp/competitive_benchmarks/`
- Blocks commit if >90 days old
- Warns if expiring within 14 days
- Clear error messages with instructions to run benchmarks

**Bypass** (use sparingly):
```bash
SKIP_BENCHMARK_CHECK=1 git commit -m "hotfix: critical bug"
```

**Recommended bypass scenarios**:
- Hotfixes that must deploy immediately
- CI/CD environments (set `SKIP_BENCHMARK_CHECK=1` in CI)
- Local development branches (not merging to main)

**Not recommended**:
- Regular development work
- Feature branches
- Release branches
```

### 5. CI/CD Integration

**Add to** `.github/workflows/ci.yml` or equivalent:

```yaml
env:
  # Bypass benchmark check in CI (benchmarks run separately in scheduled job)
  SKIP_BENCHMARK_CHECK: 1
```

**Create scheduled benchmark workflow** (optional):

```yaml
name: Quarterly Competitive Benchmarks

on:
  schedule:
    # Run on first Monday of January, April, July, October at 2 AM UTC
    - cron: '0 2 1-7 1,4,7,10 1'
  workflow_dispatch:  # Allow manual trigger

jobs:
  competitive_benchmarks:
    runs-on: ubuntu-latest-4-cores
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Build release binary
        run: cargo build --release

      - name: Run competitive benchmark suite
        run: ./scripts/competitive_benchmark_suite.sh

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: competitive-benchmarks
          path: tmp/competitive_benchmarks/
          retention-days: 365  # Keep quarterly results for a year
```

## Acceptance Criteria

1. **Functionality**:
   - Hook correctly calculates days since last benchmark
   - Hook blocks commits when >90 days stale
   - Hook allows commits when ≤90 days fresh
   - Hook handles missing benchmark directory gracefully
   - Hook handles malformed timestamps gracefully
   - Bypass via `SKIP_BENCHMARK_CHECK=1` works correctly

2. **User Experience**:
   - Error messages are clear and actionable
   - Instructions provided to run benchmarks
   - Warning displays when benchmarks expire soon (<14 days)
   - Success message confirms fresh benchmarks

3. **Cross-platform**:
   - Works on macOS (BSD date)
   - Works on Linux (GNU date)
   - Installation script works on both platforms

4. **Integration**:
   - Installation script backs up existing hooks
   - Hook doesn't interfere with other git hooks
   - CI/CD can bypass with environment variable
   - Hook is idempotent (safe to re-install)

5. **Code Quality**:
   - Shell script passes `shellcheck` with zero warnings
   - Installation script is idempotent
   - Hook script has clear comments

## Testing Approach

### Test 1: Fresh Benchmarks (Allow Commit)

```bash
# Setup: Run recent benchmark
./scripts/competitive_benchmark_suite.sh

# Install hook
./scripts/install_git_hooks.sh

# Attempt commit
echo "test" > test.txt
git add test.txt
git commit -m "test: fresh benchmarks"

# Expected: Commit succeeds with success message
```

### Test 2: Stale Benchmarks (Block Commit)

```bash
# Simulate old benchmark (modify timestamp)
latest=$(find tmp/competitive_benchmarks -name "*_metadata.txt" | sort -r | head -1)
# Rename to 100 days ago
old_date=$(date -d "100 days ago" "+%Y-%m-%d_%H-%M-%S")
mv "$latest" "tmp/competitive_benchmarks/${old_date}_metadata.txt"

# Attempt commit
echo "test2" > test.txt
git add test.txt
git commit -m "test: stale benchmarks"

# Expected: Commit blocked with error message
```

### Test 3: No Benchmarks (Block Commit)

```bash
# Remove all benchmarks
rm -rf tmp/competitive_benchmarks/*

# Attempt commit
echo "test3" > test.txt
git add test.txt
git commit -m "test: no benchmarks"

# Expected: Commit blocked with setup instructions
```

### Test 4: Bypass (Allow Commit)

```bash
# With stale benchmarks, use bypass
SKIP_BENCHMARK_CHECK=1 git commit -m "hotfix: emergency fix"

# Expected: Commit succeeds with bypass message
```

### Test 5: Expiry Warning (Allow with Warning)

```bash
# Simulate benchmark 80 days old (10 days remaining)
latest=$(find tmp/competitive_benchmarks -name "*_metadata.txt" | sort -r | head -1)
old_date=$(date -d "80 days ago" "+%Y-%m-%d_%H-%M-%S")
mv "$latest" "tmp/competitive_benchmarks/${old_date}_metadata.txt"

# Attempt commit
git commit -m "test: expiring soon"

# Expected: Commit succeeds with warning about expiry in 10 days
```

### Test 6: Cross-Platform Date Parsing

```bash
# Test on macOS
uname -a | grep -q Darwin && ./scripts/install_git_hooks.sh && git commit --allow-empty -m "test"

# Test on Linux
uname -a | grep -q Linux && ./scripts/install_git_hooks.sh && git commit --allow-empty -m "test"

# Expected: Both succeed with correct date parsing
```

## File Paths

```
.git/hooks/pre-commit                   # Git pre-commit hook
scripts/install_git_hooks.sh            # Hook installation script
docs/reference/competitive_baselines.md # Documentation (update)
```

## Implementation Notes

1. **Date Parsing**: Use separate logic for macOS (BSD date) and Linux (GNU date)
2. **Bypass Variable**: Check `$SKIP_BENCHMARK_CHECK` not `${SKIP_BENCHMARK_CHECK}` to handle unset variable
3. **Exit Codes**: Return 1 to block commit, 0 to allow
4. **Colors**: Disable if not a TTY (`-t 1`)
5. **Idempotency**: Installation script should be safe to run multiple times

## Success Criteria

Task is complete when:

1. Pre-commit hook correctly enforces 90-day freshness requirement
2. Installation script successfully installs hook on macOS and Linux
3. All 6 test scenarios pass
4. Hook passes `shellcheck` with zero warnings
5. Documentation clearly explains bypass scenarios
6. Hook doesn't interfere with other git operations

## References

- Git hooks documentation: https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
- BSD date vs GNU date: https://www.unix.com/man-page/osx/1/date/
- Bash date arithmetic: https://www.gnu.org/software/bash/manual/html_node/Shell-Arithmetic.html
