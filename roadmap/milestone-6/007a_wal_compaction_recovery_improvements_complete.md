# Task 007a: WAL Compaction and Recovery Improvements

## Status
IN_PROGRESS

## Priority
P1 (Production Readiness - Before Task 007)

## Effort Estimate
4 hours

## Dependencies
- Task 006 (Consolidation Metrics & Observability)

## Origin
UAT Issue 1 - Found during User Acceptance Testing of Engram v0.1.0

## Objective
Implement automatic WAL compaction after recovery to eliminate corrupt entry accumulation and improve observability of WAL health.

## Problem Statement
Server logs 3 corrupted WAL entries on every restart:
```
WAL DESERIALIZATION ERROR: InvalidTagEncoding(51)
WAL DESERIALIZATION ERROR: InvalidTagEncoding(154)
WAL DESERIALIZATION ERROR: InvalidTagEncoding(102)
```

**Root Cause**: WAL files accumulate corrupt entries indefinitely with no compaction mechanism. Recovery gracefully skips corrupt entries but never cleans them up.

## Technical Approach

### Component 1: Add WAL Metrics
**File**: `engram-core/src/metrics/mod.rs`

Add 5 new metrics:
- `WAL_RECOVERY_SUCCESSES_TOTAL`: Episodes successfully recovered
- `WAL_RECOVERY_FAILURES_TOTAL`: WAL entries that failed deserialization
- `WAL_RECOVERY_DURATION_SECONDS`: Time taken for recovery
- `WAL_COMPACTION_RUNS_TOTAL`: Compaction operations performed
- `WAL_COMPACTION_BYTES_RECLAIMED`: Bytes saved by compaction

### Component 2: WAL Compaction Method
**File**: `engram-core/src/storage/wal.rs`

Implement `compact_from_memory()` method:
- Creates new temporary WAL file
- Writes all valid episodes from memory
- Atomic rename (crash-safe commit point)
- Removes old WAL files
- Returns `CompactionStats` for observability

**Safety Guarantees**:
- Temporary file prevents corruption of existing WAL
- Atomic rename ensures crash safety
- Old WAL only removed after successful rename
- Full fsync before rename

### Component 3: Update Recovery
**File**: `engram-core/src/store.rs` (lines 883-974)

Update `recover_from_wal()`:
- Remove `eprintln!` on line 940
- Track recovery success/failure counts
- Update metrics for each recovery attempt
- Log recovery summary with corruption rate
- Automatically compact WAL after recovery
- Handle compaction failures gracefully (non-fatal)

## Implementation Checklist
- [ ] Add WAL metrics to `engram-core/src/metrics/mod.rs`
- [ ] Implement `compact_from_memory()` in `engram-core/src/storage/wal.rs`
- [ ] Add `CompactionStats` struct
- [ ] Update `recover_from_wal()` in `engram-core/src/store.rs`
- [ ] Remove `eprintln!` on line 940
- [ ] Add recovery summary logging
- [ ] Create test file `engram-core/tests/wal_compaction_tests.rs`
- [ ] Test: WAL compaction removes corrupt entries
- [ ] Test: WAL compaction is crash-safe
- [ ] Test: WAL recovery metrics accuracy
- [ ] Update `docs/metrics-schema-changelog.md` with schema v1.3.0
- [ ] Run `make quality` - ensure zero warnings
- [ ] Manual test: Verify corrupt WAL cleaned up on restart

## Acceptance Criteria
- [ ] WAL compaction runs automatically after recovery
- [ ] Corrupt entries removed from WAL on every restart
- [ ] No `eprintln!` statements (proper logging only)
- [ ] All 5 WAL metrics exposed and accurate
- [ ] Recovery summary logged with corruption rate
- [ ] Compaction failures are non-fatal
- [ ] Zero clippy warnings
- [ ] All tests passing

## Testing Approach
1. **Unit Tests** (wal_compaction_tests.rs):
   - Test compaction removes corrupt entries
   - Test crash safety (temp file handling)
   - Test metrics accuracy

2. **Manual Testing**:
   - Start server with corrupt WAL
   - Verify compaction runs
   - Verify old WAL removed
   - Restart server
   - Verify no corrupt entry warnings

## Expected Impact
- Eliminates log pollution on every restart
- Enables WAL corruption rate monitoring
- Prevents unbounded WAL file growth
- Improves production observability

## Technical Design Reference
Complete implementation details in `/tmp/uat_issues_technical_design.md` (Issue 1)

## Notes
- Following PostgreSQL/SQLite WAL compaction model
- Atomic operations ensure crash safety
- Non-breaking, additive change
- Compaction failure doesn't affect recovery (old WAL preserved)
