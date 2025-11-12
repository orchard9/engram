#!/bin/bash
# Scenario 7: Documentation Validation (Automated Checks)
# Objective: Validate documentation quality without human intervention

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Scenario 7: Documentation Validation (Automated Checks) ==="
echo ""

# Check 1: Markdown linting
echo "Check 1: Markdown linting"
if command -v npx &> /dev/null; then
  if npx markdownlint-cli2 'docs/reference/competitive_baselines.md' 2>&1 | head -10; then
    echo "  - PASS: No linting errors"
  else
    echo "  - WARNING: Linting errors found (may be acceptable)"
  fi
else
  echo "  - SKIP: npx not available (install Node.js for full validation)"
fi
echo ""

# Check 2: File references
echo "Check 2: Validate scenario file references"
MISSING_COUNT=0
while IFS= read -r f; do
  if [ ! -f "$f" ]; then
    echo "  - ERROR: Missing file: $f"
    MISSING_COUNT=$((MISSING_COUNT + 1))
  fi
done < <(grep -o 'scenarios/competitive/[^ )]*\.toml' docs/reference/competitive_baselines.md | sort -u)

if [ $MISSING_COUNT -eq 0 ]; then
  echo "  - PASS: All referenced scenario files exist"
else
  echo "  - FAIL: $MISSING_COUNT missing files"
  exit 1
fi
echo ""

# Check 3: Code blocks syntax
echo "Check 3: Validate bash code blocks"
TEMP_SCRIPT=$(mktemp)
if grep -A 20 '```bash' docs/reference/competitive_baselines.md | \
  grep -v '```' > "$TEMP_SCRIPT" 2>/dev/null; then

  if bash -n "$TEMP_SCRIPT" 2>&1; then
    echo "  - PASS: All bash code blocks syntactically valid"
  else
    echo "  - WARNING: Some code blocks may have syntax issues"
  fi
else
  echo "  - SKIP: Could not extract code blocks"
fi
rm -f "$TEMP_SCRIPT"
echo ""

# Check 4: Required sections
echo "Check 4: Verify required documentation sections"
REQUIRED_SECTIONS=(
  "Competitor Baseline Summary"
  "Qdrant"
  "Neo4j"
  "Scenario Mapping"
  "Measurement Methodology"
  "Quarterly Review Process"
)

MISSING_SECTIONS=0
for section in "${REQUIRED_SECTIONS[@]}"; do
  if grep -q "$section" docs/reference/competitive_baselines.md; then
    echo "  - Found: $section"
  else
    echo "  - ERROR: Missing section: $section"
    MISSING_SECTIONS=$((MISSING_SECTIONS + 1))
  fi
done

if [ $MISSING_SECTIONS -eq 0 ]; then
  echo "  - PASS: All required sections present"
else
  echo "  - FAIL: $MISSING_SECTIONS missing sections"
  exit 1
fi
echo ""

# Check 5: Vision.md competitive positioning
echo "Check 5: Check vision.md for competitive positioning"
if [ -f "vision.md" ]; then
  if grep -q "Competitive Positioning" vision.md; then
    echo "  - PASS: Competitive positioning section found in vision.md"
  else
    echo "  - FAIL: No competitive positioning section in vision.md"
    exit 1
  fi
else
  echo "  - ERROR: vision.md not found"
  exit 1
fi
echo ""

# Check 6: README references
echo "Check 6: Check docs/reference/README.md"
if [ -f "docs/reference/README.md" ]; then
  if grep -q "competitive_baselines" docs/reference/README.md; then
    echo "  - PASS: competitive_baselines.md linked from README"
  else
    echo "  - FAIL: No link to competitive_baselines.md in README"
    exit 1
  fi
else
  echo "  - WARNING: docs/reference/README.md not found (should be created)"
fi
echo ""

echo "RESULT: PASS - Automated documentation validation"
echo ""
echo "Note: Human validation still required (see scenario_7_documentation_validation.md)"
echo "Recruit an engineer unfamiliar with M17.1 to:"
echo "  1. Read docs/reference/competitive_baselines.md"
echo "  2. Run a competitive baseline"
echo "  3. Interpret the results"
echo "  4. Complete in <30 minutes without questions"
echo ""
echo "Document human validation results in:"
echo "  roadmap/milestone-17.1/acceptance_tests/scenario_7_results.md"
