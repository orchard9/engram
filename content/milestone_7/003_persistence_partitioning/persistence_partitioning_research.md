# Persistence Partitioning Research

## Current Architecture
- `MemoryStore` owns a single `wal_buffer`, `wal_writer`, and tier migration worker that assume one global namespace (`engram-core/src/store.rs`).
- WAL files are emitted into a shared directory with a fixed naming scheme and no tenant awareness (`engram-core/src/storage/wal.rs`).
- Tiered storage (hot/warm/cold) and compaction workers are initialized without space identifiers (`engram-core/src/storage/tiers.rs`).
- CLI bootstrap only provisions a single `MemoryStore` instance and binds persistence once at startup (`engram-cli/src/main.rs`).

## Requirements for Memory Spaces
- Deterministic per-space directory layout under the configured data root.
- Independent WAL writers/readers to avoid shared file handles.
- Separate tier migration threads to keep IO pressure bounded per space.
- Registry lifecycle hooks to create, recover, and tear down persistence for each space.

## Design Considerations
- Registry must lazily instantiate persistence handles to limit idle resource usage.
- Directory names should sanitize `MemorySpaceId` and disallow traversal.
- Compaction needs to respect per-space capacity limits and metrics.
- Recovery workflow should enumerate discovered spaces and surface partial failures.

## Validation Targets
- Concurrent writes across spaces produce isolated WAL segments.
- Crash recovery replays only the intended space’s log.
- Deleting a space removes persistence artifacts without touching siblings.
- Benchmarks quantify overhead from per-space workers.
