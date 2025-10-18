# Task 030: Fix Graceful Shutdown Timeout

**Status**: In Progress
**Priority**: High (Production-Critical)
**Milestone**: 0 (Infrastructure)

## Problem

The Engram server fails to shut down gracefully within the timeout window, requiring forced termination:

```
[WARN] ⏰ Graceful shutdown timeout, forcing stop
[INFO] Server stopped with TERM signal
```

This can cause:
- Data loss if writes are in-progress
- Resource leaks (file handles, sockets)
- Inconsistent state on restart
- Poor production behavior in orchestrated deployments

## Root Cause Analysis

**Suspected causes:**
1. Background tasks not checking cancellation tokens
2. Long-running operations (consolidation, HNSW indexing) blocking shutdown
3. Missing or improper `tokio::select!` on shutdown channel
4. Cleanup handlers exceeding timeout window (likely 5 seconds)

**Files to investigate:**
- `engram-cli/src/cli/server.rs` - shutdown signal handling
- `engram-cli/src/server.rs` - server lifecycle
- Any background task spawning (consolidation, metrics, monitoring)
- HNSW index persistence if blocking on flush

## Requirements

1. Server MUST shut down gracefully within 5 seconds under normal conditions
2. All background tasks MUST respond to shutdown signals
3. Critical state (memories, indexes) MUST be flushed before exit
4. Test coverage for graceful shutdown with active workload
5. Logging to show shutdown progress for debugging

## Implementation Plan

### Phase 1: Investigation
- [ ] Map all spawned background tasks
- [ ] Identify blocking operations
- [ ] Trace shutdown signal flow
- [ ] Measure time spent in each cleanup phase

### Phase 2: Fix Implementation
- [ ] Add shutdown channel to all background tasks
- [ ] Wrap long operations in `tokio::select!` with shutdown branch
- [ ] Add timeout guards on cleanup operations
- [ ] Implement prioritized shutdown (critical ops first)
- [ ] Add progress logging

### Phase 3: Testing
- [ ] Unit test: shutdown with idle server
- [ ] Unit test: shutdown with active memory operations
- [ ] Unit test: shutdown with background consolidation running
- [ ] Integration test: shutdown completes within 5s
- [ ] Test: verify no data loss on shutdown

### Phase 4: Verification
- [ ] Run `make quality` - zero warnings
- [ ] Manual testing: `./target/debug/engram start` then `stop`
- [ ] Verify clean shutdown in logs
- [ ] Run diagnostics script

## Acceptance Criteria

- [ ] `./target/debug/engram stop` completes within 5 seconds
- [ ] No timeout warnings in logs
- [ ] All tests pass
- [ ] Zero clippy warnings
- [ ] Diagnostics show clean shutdown
- [ ] No data loss when stopping during operations

## Technical Approach

Use Tokio's structured concurrency pattern:

```rust
// Background task pattern
tokio::spawn(async move {
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                // Cleanup and exit
                break;
            }
            result = long_operation() => {
                // Process result
            }
        }
    }
});

// Shutdown sequence
async fn shutdown_gracefully(shutdown_tx: watch::Sender<()>, timeout: Duration) {
    shutdown_tx.send(()).ok(); // Signal all tasks

    tokio::time::timeout(timeout, async {
        // Wait for critical flushes
        flush_memory_store().await?;
        flush_hnsw_index().await?;
    }).await
}
```

## Dependencies

None - standalone infrastructure fix

## Notes

This is production-critical. A cognitive memory system MUST persist state correctly on shutdown to prevent memory corruption or loss.
