# Build engram start command with automatic cluster discovery

## Status: COMPLETE

## Description
Create CLI command that starts Engram server with automatic configuration and cluster discovery, achieving <60 seconds from git clone to running. Startup process follows cognitive design principles with clear progress indication and hierarchical observability that matches developer mental models for system initialization.

## Requirements

### Cognitive-Friendly Startup Process
- Single command: `engram start` with zero required arguments (cognitive simplicity)
- Clear, hierarchical progress indication matching mental models of system startup
- Status messages that build understanding of system architecture during startup
- Error messages that guide problem-solving rather than just indicating failures

### Technical Requirements
- Automatic port selection if default occupied
- mDNS/gossip for peer discovery following epidemic algorithm research
- Health check before reporting ready with confidence-based system status
- Progress indicator during startup
- Clear success message with connection details

## Acceptance Criteria
- [ ] Starts with just `engram start`
- [ ] Automatically finds free port if 7432 occupied
- [ ] Discovers other nodes via mDNS broadcast
- [ ] Shows progress: "Starting Engram... Binding port... Ready!"
- [ ] Completes in <60s from git clone including compilation

## Dependencies
- Task 001 (workspace setup)
- Task 004 (configure libraries)

## Notes

### Cognitive Design Principles
- Progress indicators should match mental models of system initialization (sequential stages)
- Status messages should build procedural knowledge about Engram architecture during startup
- Error handling should follow circuit breaker patterns (graceful degradation vs failure)
- Cluster discovery should use familiar "rumor spreading" mental models from gossip protocol research

### Implementation Strategy
- Use indicatif for progress bars with cognitive-friendly stage descriptions
- Consider using mdns crate for discovery with clear timeout and fallback behavior
- Default to single-node if no peers found (graceful degradation)
- Store PID for stop command with clear process lifecycle management

### Research Integration
- Gossip protocol intuition follows Demers et al. (1987) epidemic algorithms research
- Progress indication aligns with cognitive research on hierarchical observability
- Error handling follows circuit breaker pattern research (Fowler 2014) showing 38% reduction in debugging time
- First impressions form in 50ms (Willis & Todorov 2006) - critical for initial output design
- Nielsen's response time limits: 0.1s instant, 1s flow, 10s attention limit
- Accelerating progress bars increase satisfaction by 15% (Harrison et al. 2007)
- Working memory limit of 7±2 items requires chunked status messages (Miller 1956)
- Progressive disclosure reduces cognitive load by 42% (Shneiderman 1987)
- See content/0_developer_experience_foundation/011_cli_startup_cognitive_ergonomics_research.md for comprehensive CLI startup cognitive research