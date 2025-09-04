# Build engram start command with automatic cluster discovery

## Status: PENDING

## Description
Create CLI command that starts Engram server with automatic configuration and cluster discovery, achieving <60 seconds from git clone to running.

## Requirements
- Single command: `engram start` with zero required arguments
- Automatic port selection if default occupied
- mDNS/gossip for peer discovery
- Health check before reporting ready
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
- Use indicatif for progress bars
- Consider using mdns crate for discovery
- Default to single-node if no peers found
- Store PID for stop command