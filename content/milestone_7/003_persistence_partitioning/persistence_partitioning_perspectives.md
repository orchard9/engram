# Persistence Partitioning Perspectives

## Systems-Architecture View
- Introduce a per-space persistence supervisor that manages WAL writers, tier workers, and compaction queues behind lightweight handles (`engram-core/src/store.rs`).
- Normalize directory layout to `<data_root>/<space>/` to simplify tooling and backups while adding guardrails against path traversal (`engram-core/src/storage/wal.rs`).
- Isolate IO metrics per space so operators can derive pressure budgets before throttling or migrating tenants (`engram-core/src/storage/tiers.rs`).

## Rust-Graph-Engine View
- Feed `MemorySpaceId` into `MemoryStore` persistence hooks so eviction, deduplication, and recovery flows remain type-safe (`engram-core/src/store.rs`).
- Ensure crossbeam queues and DashMaps are scoped per space to avoid contention and guarantee lock-free operations stay local (`engram-core/src/store.rs`).
- Provide an abstraction for batched HNSW updates that can be sharded by space without duplicating code paths (`engram-core/src/store.rs`).

## Memory-Systems View
- Treat each space as an independent memory consolidation pipeline; persistence isolation prevents one agent’s “sleep cycle” from disturbing another’s (`engram-core/src/store.rs`).
- Allow configurable tier capacities per space so high-recall agents can tune forgetting curves without starving peers (`milestones.md`).
- Preserve consolidation audit trails per space to maintain explainability of semantic schema evolution (`engram-core/src/consolidation`).

## Operations & SRE View
- Enable hot path observability: log WAL lag, compaction backlog, and tier utilization tagged with `memory_space_id` (`engram-core/src/metrics`).
- Provide tooling to snapshot or migrate a single space without downtime for others (`engram-cli/src/docs.rs`).
- Document runbooks for recovering a failed space while others continue serving traffic (`docs/operations/operations.md`).
