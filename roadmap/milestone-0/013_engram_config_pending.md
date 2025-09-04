# Add engram config for runtime configuration management

## Status: PENDING

## Description
Create configuration management system allowing runtime parameter adjustment without restart.

## Requirements
- View current configuration values
- Modify runtime parameters on the fly
- Validate configuration changes before applying
- Persist configuration changes
- Support configuration profiles
- Import/export configuration

## Acceptance Criteria
- [ ] `engram config get <key>` retrieves values
- [ ] `engram config set <key> <value>` updates configuration
- [ ] Changes take effect without restart where possible
- [ ] Invalid configurations rejected with helpful message
- [ ] Configuration persisted across restarts

## Dependencies
- Task 010 (engram start)

## Notes
- Use TOML for configuration format
- Hot-reload for applicable settings
- Some settings may require restart (document which)
- Support environment variable overrides