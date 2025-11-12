# M17 Performance Testing Bug Report

Date: 2025-11-09
Context: Milestone 17 baseline performance test failures
Goal: 0% error rate baseline

## Summary

Performance test showing 23.9% overall error rate (3,475/14,555 operations) with systematic failures across all operation types.

## Bug Breakdown

### Bug #1: PatternCompletion 100% Failure Rate (686/686)
**Location**: `tools/loadtest/src/main.rs:327-358`
**Status**: CRITICAL
**Root Cause**: Unknown - requires investigation

**Details**:
- PatternCompletion operations fail 100% of the time
- Loadtest sends JSON with `partial_episode` containing:
  - `known_fields`: HashMap
  - `partial_embedding`: Vec<Option<f32>> (768 dimensions)
  - `cue_strength`: 0.7
- API schema (engram-cli/src/handlers/complete.rs:43-59) expects this format
- Need to capture actual HTTP error responses to diagnose

**Next Steps**:
1. Add debug logging to loadtest to capture HTTP error bodies
2. Test PatternCompletion API endpoint directly with curl
3. Compare loadtest JSON with known-working examples

### Bug #2: Store 20.5% Failure Rate (1,497/7,304)
**Location**: `tools/loadtest/src/main.rs:277-293`
**Status**: CRITICAL
**Root Cause**: Unknown

**Details**:
- Consistent ~20% error rate suggests systematic issue
- Loadtest sends POST /api/v1/memories with:
  ```json
  {
    "content": "Load test memory {uuid}",
    "confidence": 0.7-1.0,
    "embedding": [768 floats]
  }
  ```
- Manual curl test with 1-dimensional embedding succeeded (!)
- Suggests API may not validate embedding size for Store
- 20% rate suggests timeout/rate-limiting issue rather than validation

**Next Steps**:
1. Check for connection pool exhaustion (30s timeout, 1000 ops/sec rate)
2. Review Engram server logs for Store operation errors
3. Test with lower request rate to eliminate rate-limiting

### Bug #3: Recall 19.7% Failure Rate (1,156/5,882)
**Location**: `tools/loadtest/src/main.rs:294-310`
**Status**: CRITICAL
**Root Cause**: Embedding validation or URL encoding issue

**Details**:
- Uses GET with URL-encoded embedding query param
- API validates embedding must be exactly 768 dimensions
- Format: `GET /api/v1/memories/recall?embedding=[...]&threshold=...&max_results=...&space=...`
- Manual test with 1-dimensional embedding returned: "Embedding must be exactly 768 dimensions, got 1"
- Workload generator uses `embedding_distribution.generate(rng, 768)` - should produce correct size
- URL encoding of 768-float JSON array may hit length limits or parsing issues

**Hypothesis**: URL length limits or JSON parsing errors in query param

**Next Steps**:
1. Test recall with full 768-dimensional embedding via curl
2. Check if URL-encoded JSON array exceeds limits
3. Consider switching Recall to POST with JSON body

###Bug #4: Search 19.9% Failure Rate (136/683)
**Location**: `tools/loadtest/src/main.rs:311-326`
**Status**: CRITICAL
**Root Cause**: Unknown

**Details**:
- Uses GET /api/v1/memories/search with text query
- Sends synthetic query: `format!("test query {}", uuid)`
- Simplest operation - just text search
- 20% error rate matching Store/Recall suggests common root cause
- May be timeout/connection issue rather than validation

**Next Steps**:
1. Test search endpoint directly
2. Check if text query format has restrictions
3. Review for same timeout/rate-limiting as Store

## Common Patterns

All bugs share ~20% error rate (except PatternCompletion at 100%), suggesting:

1. **Timeout/Connection Issues**:
   - 30s timeout per request
   - 1000 ops/sec target rate
   - May exhaust connection pool or hit server limits

2. **Rate Limiting**:
   - Server may reject requests under high load
   - Need to check server-side logs

3. **Test Harness Bugs**:
   - Loadtest may have error handling bugs
   - Could be marking successful operations as failed

## Testing Methodology Issues

### Issue #1: No Error Logging
- Loadtest doesn't log actual HTTP error messages
- Only marks operations as failed via `result.is_ok()`
- Need to add error body capture

### Issue #2: No Server Logs Review
- Haven't checked Engram server logs for errors
- Server may be logging actual failure reasons

### Issue #3: No Direct API Testing
- Haven't verified each endpoint works with realistic data
- Should test each operation type independently with curl

## Fixed Bugs (For Reference)

### ✓ Wikipedia XML Namespace Mismatch
**Files**: `tools/wikipedia_ingest/ingest.py:71`, `debug_parse.py:10`
**Fix**: Changed namespace from `{http://www.mediawiki.org/xml/export-0.10/}` to `{http://www.mediawiki.org/xml/export-0.11/}`
**Result**: Successfully ingested 99/100 Wikipedia articles

### ✓ Divide-by-Zero in Comparison Script
**File**: `scripts/compare_m17_performance.sh:71-81`
**Fix**: Added `calc_pct_change()` function with zero-check
**Result**: Script handles zero baseline values correctly

### ✓ PatternCompletion API Mismatch in Replay.rs
**File**: `tools/loadtest/src/replay.rs:119-145`
**Fix**: Wrapped partial data in `partial_episode` object with correct schema
**Note**: This fix doesn't help because M17 tests use `run` command (main.rs), not `replay` command

## Recommended Next Actions

1. **Add debug logging to loadtest**:
   ```rust
   if let Err(e) = result {
       tracing::error!("Operation failed: {:?} - {}", operation, e);
   }
   ```

2. **Review Engram server logs** from test run:
   ```bash
   grep ERROR tmp/m17_performance/001_before_*_loadtest.log
   ```

3. **Test each endpoint directly** with realistic payloads

4. **Run at lower rate** (100 ops/sec) to eliminate timeout issues

5. **Check connection pool** configuration in loadtest

## Current Baseline State

**File**: `tmp/m17_performance/001_before_20251108_225318.json`

- Total operations: 14,555
- Total errors: 3,475 (23.9%)
- P99 latency: 0.65ms ✓
- Throughput: 242 ops/sec (target: 800)

**This baseline is INVALID** due to high error rate. Cannot use for regression tracking until bugs are fixed.
