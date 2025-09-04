# Build engram start command with automatic cluster discovery

## Status: PENDING

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
- See content/0_developer_experience_foundation/006_concurrent_graph_systems_cognitive_load_research.md for concurrent startup cognitive research