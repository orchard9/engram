# Scenario 7: Documentation Completeness (Human Validation)

**Status**: Manual validation required
**Objective**: Verify another engineer can use the framework without assistance

## Test Protocol

1. **Recruit Test Subject**: Engineer unfamiliar with M17.1 competitive framework
2. **Provide Materials**: Only `docs/reference/competitive_baselines.md` and main `README.md`
3. **Task**: "Run a competitive baseline and interpret the results"
4. **Observation**: Document success/failure, questions asked, time taken

## Success Criteria

- [ ] Engineer completes task in <30 minutes
- [ ] Zero clarifying questions about how to run tools
- [ ] Correctly interprets P99 latency and trend data
- [ ] Identifies which scenarios Engram wins/loses vs competitors
- [ ] No errors encountered during execution

## Documentation Validation Checklist

Automated checks (run before human testing):

### Link Validation
```bash
npx markdown-link-check docs/reference/competitive_baselines.md
# Expected: 0 broken links
```

### Code Block Syntax
```bash
grep -A 20 '```bash' docs/reference/competitive_baselines.md | bash -n
# Expected: No syntax errors
```

### File References
```bash
grep -o 'scenarios/competitive/[^ ]*\.toml' docs/reference/competitive_baselines.md | \
  while read f; do [ -f "$f" ] || echo "Missing: $f"; done
# Expected: No output (all files exist)
```

### Markdown Linting
```bash
npx markdownlint-cli2 'docs/reference/competitive_baselines.md'
# Expected: 0 warnings
```

## Manual Validation Steps

### Prerequisites Check
- [ ] All commands are copy-pasteable
- [ ] Prerequisites explicitly stated (tools, specs)
- [ ] Expected outputs shown with examples
- [ ] Troubleshooting section covers common errors
- [ ] Links to source code/config files working
- [ ] Terminology defined (P99, QPS, recall)
- [ ] Comparisons clearly explained

### Execution Observation

**Observer Notes Template**:

```
Test Date: ____________
Test Subject: ____________ (GitHub handle if applicable)
Experience Level: [ ] Junior [ ] Mid-level [ ] Senior
Familiar with benchmarking: [ ] Yes [ ] No

Execution Timeline:
- Start time: _______
- First question (if any): _______
  Question: _________________________________
- First command executed: _______
- First error (if any): _______
  Error: _________________________________
- Completion time: _______
Total duration: _______ minutes

Questions Asked:
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

Errors Encountered:
1. _______________________________________________
2. _______________________________________________

Result Interpretation:
- Correctly identified P99 latencies: [ ] Yes [ ] No
- Understood competitive positioning: [ ] Yes [ ] No
- Could identify optimization priorities: [ ] Yes [ ] No

Overall Assessment:
[ ] PASS - Completed independently in <30min
[ ] PARTIAL - Completed but needed clarification
[ ] FAIL - Could not complete without significant help

Recommendations for Documentation Improvements:
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
```

## Post-Test Actions

If test fails:
1. Document specific pain points in test notes
2. Create issues for documentation improvements
3. Update `docs/reference/competitive_baselines.md` with clarifications
4. Re-test with another engineer after fixes

If test passes:
1. Archive test notes in `roadmap/milestone-17.1/acceptance_tests/scenario_7_results.md`
2. Mark Scenario 7 as PASS in completion checklist
3. Use feedback for future documentation improvements

## Automated Validation (Run Automatically)

This automated script validates documentation quality without human intervention:

```bash
#!/bin/bash
# Automated documentation validation checks

set -euo pipefail

echo "=== Scenario 7: Documentation Validation (Automated Checks) ==="
echo ""

# Check 1: Markdown linting
echo "Check 1: Markdown linting"
if command -v npx &> /dev/null; then
  npx markdownlint-cli2 'docs/reference/competitive_baselines.md' 2>&1 | head -10
  if [ $? -eq 0 ]; then
    echo "  - PASS: No linting errors"
  else
    echo "  - FAIL: Linting errors found"
    exit 1
  fi
else
  echo "  - SKIP: npx not available"
fi
echo ""

# Check 2: File references
echo "Check 2: Validate scenario file references"
MISSING_FILES=0
grep -o 'scenarios/competitive/[^ )]*\.toml' docs/reference/competitive_baselines.md | sort -u | while read f; do
  if [ ! -f "$f" ]; then
    echo "  - ERROR: Missing file: $f"
    MISSING_FILES=$((MISSING_FILES + 1))
  fi
done

if [ $MISSING_FILES -eq 0 ]; then
  echo "  - PASS: All referenced scenario files exist"
else
  echo "  - FAIL: $MISSING_FILES missing files"
  exit 1
fi
echo ""

# Check 3: Code blocks syntax
echo "Check 3: Validate bash code blocks"
TEMP_SCRIPT=$(mktemp)
grep -A 20 '```bash' docs/reference/competitive_baselines.md | \
  grep -v '```' > "$TEMP_SCRIPT"
if bash -n "$TEMP_SCRIPT" 2>&1; then
  echo "  - PASS: All bash code blocks syntactically valid"
else
  echo "  - FAIL: Syntax errors in code blocks"
  rm -f "$TEMP_SCRIPT"
  exit 1
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

for section in "${REQUIRED_SECTIONS[@]}"; do
  if grep -q "$section" docs/reference/competitive_baselines.md; then
    echo "  - Found: $section"
  else
    echo "  - ERROR: Missing section: $section"
    exit 1
  fi
done
echo "  - PASS: All required sections present"
echo ""

echo "RESULT: PASS - Automated documentation validation"
echo ""
echo "Note: Human validation still required (see scenario_7_documentation_validation.md)"
echo "Recruit an engineer unfamiliar with M17.1 to follow the docs and provide feedback"
```

Save this script as `scenario_7_documentation_validation_automated.sh` and run it as part of acceptance testing.

## Completion

Mark Scenario 7 complete when:
- [ ] Automated validation passes
- [ ] Human validation conducted with documented results
- [ ] Any identified issues addressed or documented as known limitations
- [ ] Test notes archived for future reference
