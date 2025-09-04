# Create infallible store() operation with graceful degradation

## Status: PENDING

## Description
Implement store() operation that never returns errors, instead degrading gracefully under pressure while maintaining system stability.

## Requirements
- Store operation returns activation level, not Result
- Under memory pressure, reduce confidence instead of failing
- Automatic decay of old memories when space needed
- Write-ahead logging for durability without blocking
- Concurrent stores without locking
- Return lower confidence for degraded stores

## Acceptance Criteria
- [ ] store() has signature: fn store(&self, episode: Episode) -> Activation
- [ ] Never panics regardless of system state
- [ ] Handles OOM by evicting low-activation memories
- [ ] Concurrent stores don't block each other
- [ ] Degraded stores indicated by returned activation level

## Dependencies
- Task 006 (Memory types)

## Notes
- Use lock-free data structures for hot tier
- Consider write buffer with background flushing
- Activation level indicates store quality
- May need backpressure mechanism