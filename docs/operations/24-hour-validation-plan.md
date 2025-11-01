# 24-Hour Single-Node Validation Plan

**Purpose**: Validate production readiness for single-node Engram deployment through comprehensive 24-hour soak testing.

**Duration**: 24 hours continuous operation
**Target**: Single-node deployment validation before M14 (distributed architecture)

---

## Test Coverage Checklist

### 1. Memory Stability ✓
- [ ] No memory leaks detected over 24 hours
- [ ] RSS stays within 25% of starting value
- [ ] No unbounded growth in any memory pools
- [ ] Graceful handling of memory pressure
- [ ] Arena allocations properly freed

**How**: Monitor RSS every minute, plot growth rate, verify < 1% per hour

### 2. Consolidation System ✓
- [ ] Scheduler runs on perfect 60s cadence (±2s tolerance)
- [ ] 1,440 successful consolidation runs (60/hour × 24 hours)
- [ ] Zero scheduler crashes or panics
- [ ] Latency remains <10ms P99 throughout
- [ ] Pattern detection accuracy stable
- [ ] No consolidation re-entry loops

**How**: Parse consolidation logs, verify timestamps, count successes

### 3. Spreading Activation Performance ✓
- [ ] P50 latency <5ms throughout 24 hours
- [ ] P95 latency <15ms throughout 24 hours
- [ ] P99 latency <50ms throughout 24 hours
- [ ] No timeouts after increasing base timeout to 60s
- [ ] Thread pool remains responsive
- [ ] No deadlocks in parallel spreading

**How**: Sample activation spreading every 5 minutes, track latency distribution

### 4. HNSW Index Scaling ✓
- [ ] Query time remains <2ms for vector search
- [ ] Index build time scales linearly with additions
- [ ] No index corruption after 24 hours
- [ ] Concurrent queries don't degrade performance >20%
- [ ] Space-partitioned HNSW maintains isolation

**How**: Run vector queries every minute, measure P50/P95/P99

### 5. Observation Throughput ✓
- [ ] Sustained 10K observations/sec for 1 hour burst
- [ ] Sustained 1K observations/sec baseline
- [ ] Queue never fills (backpressure working)
- [ ] Zero dropped observations
- [ ] Latency stable under load

**How**: Inject observations at varying rates, monitor queue depth

### 6. Persistence & WAL ✓
- [ ] WAL writes sustain throughout 24 hours
- [ ] No WAL corruption
- [ ] Recovery from mid-test crash succeeds
- [ ] WAL compaction triggers appropriately
- [ ] Disk space usage predictable

**How**: Monitor WAL size, trigger deliberate crash at hour 12, verify recovery

### 7. Multi-Tenancy Isolation ✓
- [ ] Memory space metrics isolated correctly
- [ ] No cross-space contamination
- [ ] Label preservation in all metrics
- [ ] Space creation/deletion stable
- [ ] Concurrent space access performs well

**How**: Create 10 memory spaces, verify isolation in metrics and data

### 8. Cognitive Patterns ✓
- [ ] Priming effects remain consistent
- [ ] Interference detection stable
- [ ] Reconsolidation triggers appropriately
- [ ] No metric overflow or NaN values
- [ ] DRM false memory rate within 60%±10%

**How**: Run cognitive pattern benchmarks every 2 hours

### 9. Resource Limits ✓
- [ ] CPU usage <80% sustained
- [ ] Thread count stable (no thread leaks)
- [ ] File descriptor count stable
- [ ] Network connections properly closed
- [ ] Graceful degradation under resource pressure

**How**: Monitor system resources every minute

### 10. Error Handling ✓
- [ ] All errors logged with context
- [ ] No unhandled panics
- [ ] Graceful recovery from transient failures
- [ ] Evidence chains maintained during errors
- [ ] Metrics continue during degraded mode

**How**: Inject errors (disk full, network issues), verify recovery

---

## Acceptance Criteria

### CRITICAL (Must Pass)
1. **Zero crashes** during 24-hour period
2. **Zero memory leaks** (RSS growth <1% per hour)
3. **Zero data corruption** (all data recoverable)
4. **Consolidation cadence** within 60s±2s for 95% of runs
5. **All tests passing** at start and end of soak test

### HIGH (Should Pass)
6. **P99 latency** <50ms for spreading activation
7. **P99 latency** <2ms for vector queries
8. **Throughput** sustains 1K obs/sec baseline
9. **CPU usage** <80% sustained
10. **Disk growth** <10GB over 24 hours

### MEDIUM (Nice to Have)
11. **P95 latency** <15ms for spreading activation
12. **Throughput** sustains 10K obs/sec burst for 1 hour
13. **Memory** RSS <2GB throughout
14. **Thread count** <200 throughout
15. **Zero warnings** in logs

---

## Test Phases

### Phase 1: Baseline (Hours 0-2)
- Start system with minimal load
- Establish baseline metrics
- Verify all systems operational
- **Inject**: 100 obs/sec, 10 queries/sec

### Phase 2: Steady State (Hours 2-8)
- Sustained moderate load
- Monitor for degradation
- Verify stable performance
- **Inject**: 1,000 obs/sec, 100 queries/sec

### Phase 3: Burst Load (Hours 8-9)
- High throughput stress test
- Verify backpressure works
- Check queue depth limits
- **Inject**: 10,000 obs/sec, 1,000 queries/sec

### Phase 4: Recovery (Hours 9-10)
- Return to baseline load
- Verify system recovers
- Check for lingering effects
- **Inject**: 100 obs/sec, 10 queries/sec

### Phase 5: Crash Injection (Hour 12)
- Deliberate process kill
- Verify WAL recovery
- Confirm no data loss
- **Inject**: SIGKILL process, restart, validate

### Phase 6: Multi-Tenant Load (Hours 12-18)
- Create 10 memory spaces
- Run concurrent workloads
- Verify isolation
- **Inject**: 100 obs/sec per space, 10 queries/sec per space

### Phase 7: Resource Pressure (Hours 18-20)
- Artificially limit memory
- Verify graceful degradation
- Check error handling
- **Inject**: Memory pressure via ulimit

### Phase 8: Cognitive Validation (Hours 20-22)
- Run full cognitive pattern suite
- Verify psychology accuracy
- Check pattern stability
- **Inject**: DRM paradigm, priming tests, interference tests

### Phase 9: Burn-in (Hours 22-24)
- Final sustained load
- Verify no accumulated issues
- Prepare for final validation
- **Inject**: 1,000 obs/sec, 100 queries/sec

### Phase 10: Final Validation (Hour 24)
- Run full test suite
- Compare against baseline
- Generate final report
- **Verify**: All acceptance criteria met

---

## Metrics Collection

### Every Minute
- RSS memory usage
- CPU usage
- Thread count
- File descriptor count
- Queue depth (observation queue)
- Active spreading operations

### Every 5 Minutes
- Spreading activation latency (P50/P95/P99)
- Vector query latency (P50/P95/P99)
- Consolidation run count
- Pattern count
- Episode count per space

### Every Hour
- Full benchmark suite
- Cognitive pattern validation
- Memory snapshot
- WAL size and compaction status
- Disk usage
- Network stats

### Continuous
- Consolidation timestamps (verify cadence)
- Error/warning logs
- Evidence chain integrity
- Metrics streaming
- Backpressure events

---

## Data Collection

### Log Files
- `/tmp/engram-24h/system.log` - Main system log
- `/tmp/engram-24h/consolidation.log` - Consolidation events
- `/tmp/engram-24h/metrics.jsonl` - Metrics snapshots
- `/tmp/engram-24h/errors.log` - All errors/warnings
- `/tmp/engram-24h/cognitive.log` - Cognitive pattern results

### Metric Files
- `/tmp/engram-24h/memory_rss.csv` - RSS over time
- `/tmp/engram-24h/latency_spreading.csv` - Spreading latency
- `/tmp/engram-24h/latency_vector.csv` - Vector query latency
- `/tmp/engram-24h/throughput.csv` - Observations/sec
- `/tmp/engram-24h/consolidation_cadence.csv` - Timestamp deltas

### Snapshots
- `/tmp/engram-24h/snapshot_hour_00.json` - Baseline
- `/tmp/engram-24h/snapshot_hour_12.json` - Pre-crash
- `/tmp/engram-24h/snapshot_hour_12_post.json` - Post-recovery
- `/tmp/engram-24h/snapshot_hour_24.json` - Final

---

## Success Definition

**PASS**: All CRITICAL criteria met + ≥80% of HIGH criteria met
**CONDITIONAL PASS**: All CRITICAL met + ≥60% of HIGH criteria + issues documented
**FAIL**: Any CRITICAL criterion fails

### Post-Test Analysis Required
1. Plot all time-series metrics
2. Identify any anomalies or trends
3. Compare against documented baselines
4. Document any warnings or errors
5. Generate final validation report

---

## Report Format

```markdown
# 24-Hour Validation Report

**Date**: YYYY-MM-DD
**Duration**: 24:00:00
**Status**: PASS | CONDITIONAL PASS | FAIL

## Critical Criteria (5/5)
✅ Zero crashes
✅ Zero memory leaks
✅ Zero data corruption
✅ Consolidation cadence stable
✅ All tests passing

## High Criteria (8/10)
✅ Spreading P99 <50ms
✅ Vector P99 <2ms
⚠️ Throughput 800/sec (target 1K/sec)
...

## Issues Encountered
1. Minor latency spike at hour 8 during burst phase (recovered)
2. ...

## Baseline Metrics
- RSS: Start 450MB, End 520MB, Growth 0.3%/hour ✅
- Consolidation: 1,438/1,440 runs successful (99.86%) ✅
- Spreading P99: 42ms average ✅
...

## Recommendation
APPROVED for production deployment with monitoring thresholds:
- RSS alert at 1.5GB
- Consolidation alert if >5s delay
- Spreading alert if P99 >100ms
```

---

## Next Steps After Passing

1. **Document baselines** in `docs/operations/production-baselines.md`
2. **Set alert thresholds** in Grafana based on observed metrics
3. **Create runbooks** for any issues encountered during soak test
4. **Update capacity estimates** based on actual resource usage
5. **Schedule production deployment** with validated configuration

## Next Steps If Failing

1. **Root cause analysis** for each failed criterion
2. **Create follow-up tasks** in roadmap
3. **Re-run focused tests** for failed areas
4. **Fix issues** and re-validate
5. **Do NOT proceed to production** until PASS achieved
