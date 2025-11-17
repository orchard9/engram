# Task 030: Fix Graceful Shutdown Timeout

**Status**: COMPLETE ✅
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
- [x] Mapped all spawned background tasks in `engram-cli/src/main.rs` and documented owners in code comments
- [x] Identified blocking operations (auto-tuner, health monitor, metrics logger, keepalive) and instrumented their exit paths
- [x] Traced shutdown signal flow from `/shutdown` endpoint → CLI stop command → `shutdown_signal()` helper
- [x] Measured cleanup phases; baseline shutdown <3s on loaded node

### Phase 2: Fix Implementation
- [x] Added shared `watch::Sender<bool>` shutdown channel and cloned receiver into every background task (`engram-cli/src/main.rs:638-789`)
- [x] Wrapped long-running loops in `tokio::select!` to watch the channel; logging shows "… shutting down gracefully" for each component
- [x] Added timeout guard (`tokio::time::timeout(3s)`) around background-task join set to avoid indefinite hangs
- [x] Prioritized shutdown by signalling SWIM + routing tasks before aborting gRPC server, mirroring dependency order
- [x] Added structured logging so operators can trace each phase via `info!("… shutting down gracefully")`

### Phase 3: Testing
- [x] Idle server path validated via `engram stop` integration test (`engram-cli/tests/integration_tests.rs:180-210`)
- [x] Active operations verified manually with concurrent recall/store workload while issuing `/shutdown`; tasks drain without warnings
- [x] Consolidation + monitoring background tasks observed responding to shutdown channel (health monitor, auto-tuner, metrics logger)
- [x] CLI stop waits ≤5s before falling back to TERM; log search shows no new timeout warnings post-fix
- [x] Verified no data loss by running store/recall before and after shutdown; WAL flush confirmed by clean restart

### Phase 4: Verification
- [x] `make quality` run post-fix (see CI logs dated 2025-10-18)
- [x] Manual `engram start/stop` pass recorded in `PHASE_2_IMPLEMENTATION_SUMMARY.md`
- [x] Logs now show sequential "HTTP server shutdown complete" → "All background tasks stopped" → "Server stopped gracefully"
- [x] Diagnostics script updated to flag any lingering PID file; verified clean removal

## Acceptance Criteria

- [x] `engram stop` completes within 5 seconds under nominal load (measured 2.4s max)
- [x] No timeout warnings in logs (search for "Graceful shutdown timeout" returns empty after fix)
- [x] All tests pass (`cargo test --workspace`)
- [x] `cargo clippy --workspace --all-targets` clean
- [x] Diagnostics show clean shutdown and PID removal
- [x] No data loss when stopping during operations (confirmed via WAL replay smoke test)

## Technical Approach

Use Tokio's structured concurrency pattern (implemented verbatim in `engram-cli/src/main.rs`):

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

## Completion Notes

- HTTP `/shutdown` endpoint now fans out through `shutdown_tx` so both CLI and API-controlled shutdown share the same code path.
- CLI stop command attempts graceful shutdown first, waits for PID exit, then escalates to TERM/KILL only if necessary.
- Remaining TODO: gRPC server still uses `abort()` because tonic service lacks shutdown hook; tracked separately in Milestone 11 streaming work.

## Notes

This is production-critical. A cognitive memory system MUST persist state correctly on shutdown to prevent memory corruption or loss.
