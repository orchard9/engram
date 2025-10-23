# Memory Space Registry — Research Notes

## Current Architecture
- `engram-core::MemoryStore` constructor (`engram-core/src/store.rs:405`) creates a single in-memory graph and shared state without any namespace concept.
- The CLI boots one `MemoryStore` and shares it across all routes (`engram-cli/src/main.rs:174`), implying a global tenant.
- Persistence defaults to one WAL buffer per process (`engram-core/src/store.rs:520`), so multi-tenant isolation must be layered above the store creation boundary.

## Registry Requirements
- Need a `MemorySpaceId` newtype to prevent passing raw strings throughout the codebase and to centralize validation.
- Registry must provision per-space stores lazily while guaranteeing a deterministic default space for backward compatibility.
- Concurrency control: registry operations will run in async contexts (HTTP/gRPC) and synchronous CLI commands, so choose primitives compatible with both (e.g., `DashMap`, `parking_lot::RwLock`, or `tokio::sync::RwLock`).
- Lifecycle operations should surface rich errors rather than `Option`, matching Engram’s ergonomic error style (`CLAUDE.md` guidance on educating users).

## CLI & Config Considerations
- CLI currently reads `engram-cli/config/default.toml` for feature flags only; registry introduces `default_memory_space` and optional bootstrap list.
- New subcommands (`engram space list/create`) must align with existing CLI patterns in `engram-cli/src/cli/memory.rs` (clap-based subcommands).

## Persistence Hooks
- Registry needs to hand out per-space persistence roots before Task 003 partitions storage.
- Directory layout likely rooted in an operator-configurable base path derived from existing `resolve_data_directory()` helper (`engram-cli/src/main.rs:183`).

## Testing Strategy Inputs
- Stress concurrent `create/get` calls using `tokio::spawn` or `rayon` to model server usage.
- CLI integration tests can reuse existing harnesses in `engram-cli/tests/` to verify command outputs.
