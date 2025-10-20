# Task 007b: gRPC Readiness and CLI Retry Logic

## Status
PENDING

## Priority
P2 (User Experience - After Task 007a)

## Effort Estimate
3 hours

## Dependencies
- Task 006 (Consolidation Metrics & Observability)

## Origin
UAT Issue 2 - Found during User Acceptance Testing of Engram v0.1.0

## Objective
Eliminate gRPC connection race condition on server startup by implementing readiness checking and CLI retry logic.

## Problem Statement
CLI command `engram memory list` fails immediately after `engram start`:
```
Error: Server found but not responding properly (PID: 20148)
Try: engram stop && engram start
```

**Root Cause**: Server prints success message before gRPC fully ready. HTTP server ready at T+0.0s, success message at T+0.2s, but gRPC actually ready at T+1.5s.

## Technical Approach

### Component 1: gRPC Health Check
**File**: `engram-cli/src/cli/server.rs`

Add `wait_for_grpc_ready()` function:
- Attempts gRPC connection with timeout
- Retry interval: 500ms
- Total timeout: 10 seconds
- Returns Ok when gRPC accepts connections

### Component 2: Update Server Start
**File**: `engram-cli/src/cli/server.rs`

Update `start_server()`:
- Wait for HTTP health endpoint
- Wait for gRPC readiness
- Only then print success message
- Improved error messages with troubleshooting hints

### Component 3: CLI Retry Logic
**File**: `engram-cli/src/client.rs`

Add `with_retry()` wrapper:
- Exponential backoff: 1s, 2s, 4s
- Max 3 attempts
- Retry on Unavailable/Unknown status codes
- Don't retry on auth/invalid request errors
- Apply to all CLI commands

### Component 4: Error Message Improvements
**File**: `engram-cli/src/client.rs`

Add `format_connection_error()`:
- Detect connection refused errors
- Suggest waiting 3-5 seconds
- Provide troubleshooting steps
- Clear, actionable guidance

## Implementation Checklist
- [ ] Add `wait_for_grpc_ready()` to `engram-cli/src/cli/server.rs`
- [ ] Update `start_server()` to wait for readiness before success message
- [ ] Add `with_retry()` wrapper to `engram-cli/src/client.rs`
- [ ] Update all CLI commands to use retry wrapper
- [ ] Add `format_connection_error()` helper
- [ ] Create test file `engram-cli/tests/integration/server_readiness_tests.rs`
- [ ] Test: CLI waits for server ready
- [ ] Test: Start command waits for gRPC
- [ ] Test: CLI retry gives up eventually
- [ ] Manual test: `engram start && engram memory list` succeeds
- [ ] Manual test: Measure time from start to success message
- [ ] Update CLI help text with startup timing info
- [ ] Run `make quality` - ensure zero warnings

## Acceptance Criteria
- [ ] `engram start` waits for gRPC before success message
- [ ] Can run `engram memory list` immediately after start
- [ ] CLI commands retry with exponential backoff
- [ ] Clear error messages with troubleshooting guidance
- [ ] Zero clippy warnings
- [ ] All tests passing

## Testing Approach
1. **Integration Tests** (server_readiness_tests.rs):
   - Test CLI waits for server ready
   - Test start command waits for gRPC
   - Test retry gives up eventually

2. **Manual Testing**:
   - Run `engram start && engram memory list`
   - Verify no race condition
   - Test error messages with server stopped
   - Measure startup time

## Expected Impact
- Eliminates confusing startup errors
- Improves first-run user experience
- Makes CLI more robust to transient failures
- Better error messages for troubleshooting

## Technical Design Reference
Complete implementation details in `/tmp/uat_issues_technical_design.md` (Issue 2)

## Notes
- Non-breaking change (only improves UX)
- Adds ~1-2 seconds to start command (waiting for gRPC)
- Retry logic makes CLI resilient to network issues
- Clear separation of HTTP vs gRPC readiness
