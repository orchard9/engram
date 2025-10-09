# Create engram status showing memory count, nodes, consolidation state

## Status: COMPLETE

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

### Cognitive Design Principles
- Status output should be scannable at a glance (working memory constraints)
- Hierarchical information organization for progressive detail discovery
- Use visual indicators (✓, ⚠, ✗) for pre-attentive processing
- Group related metrics to support chunking (7±2 rule)

### Implementation Strategy
- Use tabular output for human readable
- Consider colored output for health states (green/yellow/red)
- Show recent recall performance metrics
- Include uptime and last consolidation time
- Tree structure for hierarchical display:
  ```
  Engram Status: HEALTHY ✓
  ├─ Memory: 42K episodes (3.2GB)
  ├─ Cluster: 3 nodes active
  └─ Performance: 12ms avg recall
  ```

### Research Integration
- Pre-attentive visual processing (<200ms) for colored status indicators
- Working memory constraints require grouped, chunked information
- Hierarchical display leverages spatial-hierarchical cognition
- See content/0_developer_experience_foundation/011_cli_startup_cognitive_ergonomics_research.md for status communication patterns