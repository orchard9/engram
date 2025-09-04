# Create engram status showing memory count, nodes, consolidation state

## Status: PENDING

## Description
Implement status command that provides clear, actionable information about the current state of the Engram database.

## Requirements
- Show total memory count and size
- Display cluster node information
- Report consolidation progress
- Show current activation statistics
- Display resource usage (memory, CPU)
- Health status with specific indicators

## Acceptance Criteria
- [ ] `engram status` shows: "HEALTHY: 42K memories, 3 nodes, consolidating"
- [ ] JSON output with --json flag
- [ ] Real-time updates with --watch flag
- [ ] Clear indication of any problems
- [ ] Sub-second response time

## Dependencies
- Task 010 (engram start)

## Notes
- Use tabular output for human readable
- Consider colored output for health states
- Show recent recall performance metrics
- Include uptime and last consolidation time