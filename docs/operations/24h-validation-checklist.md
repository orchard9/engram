# 24-Hour Validation Checklist

Quick reference for running and validating the 24-hour production soak test.

## Before Starting

- [ ] Latest code is committed and clean working directory
- [ ] All tests passing: `cargo test --workspace --lib`
- [ ] At least 20GB free disk space
- [ ] No other Engram processes running
- [ ] Release binary builds: `cargo build --release --bin consolidation_soak`

## Running the Test

### Full 24-Hour Test (Production Validation)
```bash
./scripts/validate_24h_production.sh
```

### Quick 1-Hour Test (Development/Testing)
```bash
./scripts/validate_24h_production.sh --quick-test
```

## Critical Criteria (MUST PASS)

- [ ] **Zero crashes** - No panics/segfaults in logs
- [ ] **Zero memory leaks** - RSS growth <25% over test duration
- [ ] **Zero data corruption** - All data recoverable post-test
- [ ] **Consolidation cadence** - ≥95% of runs within 60s±2s
- [ ] **Tests passing** - Full test suite passes pre and post

## High Priority Criteria (≥80% SHOULD PASS)

- [ ] **Spreading P99 latency** <50ms
- [ ] **Vector query P99 latency** <2ms
- [ ] **Throughput sustained** at 1K obs/sec baseline
- [ ] **CPU usage** <80% sustained
- [ ] **Disk growth** <10GB over 24 hours
- [ ] **Thread count stable** - No thread leaks
- [ ] **File descriptors stable** - No fd leaks
- [ ] **Backpressure working** - Queue depth managed
- [ ] **Metrics recording** - All metrics collected
- [ ] **Error handling** - Graceful recovery from failures

## What to Monitor

### Every Hour (Manual Checks)
- [ ] Check RSS in `metrics.csv` - Should be growing <1% per hour
- [ ] Check consolidation.log - No errors or panics
- [ ] Check process still running: `ps aux | grep consolidation_soak`
- [ ] Check disk usage: `du -sh /tmp/engram-24h`

### At 12 Hours
- [ ] Review intermediate results
- [ ] Decide if continuing or aborting
- [ ] Check memory trend - Linear or stabilizing?

### At 24 Hours
- [ ] Wait for script to complete
- [ ] Review `VALIDATION_REPORT.md` in output directory
- [ ] Check all criteria in validation.log

## Output Files

All files written to `/tmp/engram-24h/`:

- `VALIDATION_REPORT.md` - **Final report (read this first)**
- `validation.log` - Detailed test log
- `metrics.csv` - Memory/CPU samples
- `baseline.csv` - Initial baseline metrics
- `consolidation.log` - Consolidation soak output
- `consolidation/snapshots.jsonl` - Consolidation snapshots
- `consolidation/metrics.jsonl` - Metrics snapshots
- `tests_pre.log` - Pre-test suite results
- `tests_post.log` - Post-test suite results
- `bench_during_soak.log` - Benchmark results

## Interpreting Results

### PASS ✅
- All 5 critical criteria met
- At least 8/10 high criteria met (80%)
- **Action**: Proceed to production deployment

### CONDITIONAL PASS ⚠️
- All 5 critical criteria met
- At least 6/10 high criteria met (60%)
- **Action**: Deploy with extra monitoring and documented mitigations

### FAIL ❌
- Any critical criterion failed
- **Action**: Fix issues and re-run validation. DO NOT deploy.

## Common Issues & Fixes

### Issue: Memory growing >1% per hour
**Investigate**: Check for unbounded collections, missing cleanup
**Fix**: Profile with valgrind, fix leaks, re-run test

### Issue: Consolidation cadence irregular
**Investigate**: Check scheduler logs, CPU contention
**Fix**: Adjust scheduler config, check system load

### Issue: Process crashes
**Investigate**: Check panic backtrace in logs
**Fix**: Fix panic, add defensive checks, re-run test

### Issue: Tests fail post-soak
**Investigate**: State corruption, resource exhaustion
**Fix**: Ensure proper cleanup, check for side effects

## Next Steps After PASS

1. **Document baselines**
   ```bash
   cp /tmp/engram-24h/VALIDATION_REPORT.md docs/operations/production-baselines.md
   ```

2. **Set Grafana alerts** based on observed thresholds:
   - RSS alert at 150% of ending value
   - Consolidation delay alert at 5s
   - Spreading P99 alert at 100ms

3. **Create runbooks** for any issues encountered during test

4. **Update capacity planning** with actual resource usage

5. **Schedule production deployment**

## Troubleshooting

### Script hangs or takes too long
- Check if consolidation_soak process is running
- Check CPU usage - may be under heavy load
- Check disk I/O - may be I/O bound

### Can't find output files
- Check OUTPUT_DIR environment variable
- Default is /tmp/engram-24h
- Ensure you have write permissions

### Python analysis fails
- Requires Python 3
- Check snapshots.jsonl file exists
- Check file is valid JSON (one object per line)

## Quick Validation (Development)

For quick iteration during development:

```bash
# 1-hour test (fastest)
./scripts/validate_24h_production.sh --quick-test

# Check just the report
cat /tmp/engram-24h/VALIDATION_REPORT.md

# Re-run specific phase
cargo run --release --bin consolidation_soak -- \
    --duration-secs 3600 \
    --output-dir /tmp/test-consolidation
```

---

**See also**:
- Full validation plan: `docs/operations/24-hour-validation-plan.md`
- Validation script: `scripts/validate_24h_production.sh`
- Consolidation soak source: `engram-cli/src/bin/consolidation_soak.rs`
