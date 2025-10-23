# 001: Memory Space Registry & Lifecycle — _complete_

## Outcome
Multi-tenant control plane for Engram is now in place. We implemented a validated `MemorySpaceId`, the `MemorySpaceRegistry`, and space-scoped stores, then wired the registry through CLI startup, APIs, and gRPC. Operators and tests can create/list spaces end-to-end.

### Key Deliverables
- **ID & Errors**: Added `MemorySpaceId` newtype with validation plus `MemorySpaceError` (`engram-core/src/types.rs`, `engram-core/src/registry/error.rs`).
- **Registry Module**: Implemented `MemorySpaceRegistry`, `SpaceHandle`, directory helpers, and summaries (`engram-core/src/registry/{memory_space.rs,mod.rs}`) with unit coverage (`engram-core/tests/memory_space_registry.rs`).
- **CLI Integration**: Startup now provisions the registry, bootstraps configured spaces, and threads registry/default-space through `ApiState` and `MemoryService` (`engram-cli/src/main.rs`, `src/api.rs`, `src/grpc.rs`).
- **Config & Commands**: Extended CLI config defaults (`engram-cli/src/config.rs`, `config/default.toml`) and added `engram space list/create` commands plus helper module (`engram-cli/src/cli/{commands.rs,mod.rs,space.rs}`) with integration test (`engram-cli/tests/space_commands.rs`).
- **Control-Plane APIs**: REST exposes `/api/v1/spaces` list/create endpoints, `/health/alive` liveness, and tests now provision per-space registries with keepalive subscribers (`engram-cli/src/api.rs`, updated suites in `engram-cli/tests/`).

## Tests
- `cargo test -p engram-core --test memory_space_registry -- memory_space`
- `cargo test -p engram-cli http_api_tests::test_remember_memory_success -- --nocapture`
- `cargo test -p engram-cli space_commands::test_space_list_and_create_commands -- --nocapture`
- `cargo test --workspace`

## Follow-ups
- Downstream milestones (Tasks 002–008) depend on the registry; coordinate their updates with the new APIs.
- Document operational playbooks under Milestone 7 docs once remaining tasks land.
