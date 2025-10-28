# Documentation Changelog

## Unreleased

### Added

- **Multi-Tenant Memory Spaces** (Milestone 7): Complete isolation infrastructure for multi-tenant deployments
  - `MemorySpaceRegistry` for thread-safe space management using DashMap
  - Per-space persistence with isolated WAL/tier storage directories
  - `X-Memory-Space` HTTP header for space routing (header → query → body precedence)
  - CLI commands: `space list`, `space create`, `status --space <name>`
  - Per-space health metrics endpoint `/api/v1/system/health` with spaces array
  - Automatic space discovery and WAL recovery on startup
  - 100% backward compatibility via `default` space fallback

- Comprehensive migration guide at `docs/operations/memory-space-migration.md`

- Memory Spaces section added to README with usage examples

- Multi-tenancy documentation in API reference and usage guide

### Changed

- Health endpoint response structure: changed from `{"memory": {...}}` to `{"spaces": [{...}]}`

- Status CLI command now displays per-space metrics in formatted table

- gRPC message schema: added `memory_space_id` field to request messages

### Known Issues

- HTTP routing gap: X-Memory-Space header extracted but not fully wired to operations (Task 004 follow-up pending)

- Streaming API: Space extraction added but full isolation incomplete (Task 005c follow-up pending)

- Health endpoint: Response format mismatch in validation tests (Task 006b follow-up pending)

### Previous Features

- Marked the spreading activation API as **beta**. The CLI now manages a `spreading_api_beta` feature flag stored in `~/.config/engram/config.toml` with defaults from `engram-cli/config/default.toml`.

- Added Diátaxis-aligned documentation for spreading activation (tutorial, how-to guides, explanation, and reference), along with deterministic visualization tooling.
