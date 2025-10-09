# Implement engram stop with graceful shutdown and state preservation

## Status: COMPLETE

## Description
Create stop command that gracefully shuts down Engram server while preserving state for next startup.

## Requirements
- Graceful shutdown with configurable timeout
- Flush pending writes before termination
- Save activation states and statistics
- Clean up temporary files and locks
- Report what was preserved
- Support forced shutdown with --force flag

## Acceptance Criteria
- [ ] `engram stop` gracefully shuts down server
- [ ] Pending operations complete or timeout gracefully
- [ ] State saved to disk for next startup
- [ ] Clear message: "Preserved 42K memories, 3.2GB indexed"
- [ ] --force flag for immediate termination

## Dependencies
- Task 010 (engram start)

## Notes

### Cognitive Design Principles
- Shutdown should provide closure and confidence about data preservation
- Progress indication during shutdown prevents uncertainty about hang vs processing
- Summary of what was preserved builds trust in data durability
- Clear distinction between graceful and forced shutdown consequences

### Implementation Strategy
- Use SIGTERM for graceful, SIGKILL for forced
- Save statistics for next startup
- Consider checkpoint file for crash recovery
- Maximum 30 second graceful shutdown
- Show progress: "Flushing writes... Saving state... Preserved X memories"

### Research Integration
- Closure psychology: explicit confirmation of preservation reduces anxiety
- Progressive disclosure during shutdown maintains engagement without panic
- See content/0_developer_experience_foundation/011_cli_startup_cognitive_ergonomics_research.md for shutdown as inverse of startup patterns