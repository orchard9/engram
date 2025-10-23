# Memory Space Support – Research Notes

## Current State Analysis
- `MemoryStore` exposes a single global namespace that mixes all hot memories, WAL buffers, and activation pipelines, so tenant isolation is currently impossible.[source: roadmap/milestone-7/000_milestone_overview_pending.md]
- HTTP, gRPC, and CLI entry points never accept a tenant identifier; every request resolves to the singleton store created at startup.[source: roadmap/milestone-7/000_milestone_overview_pending.md]
- Persistence layers (WAL, tier migration) emit to one directory tree, so recovery and compaction operations inherently co-mingle data.[source: roadmap/milestone-7/000_milestone_overview_pending.md]

## Target Capabilities
- Introduce `MemorySpaceId` and registry to own lifecycle (create/list/delete) and provide handles to space-specific stores.[source: roadmap/milestone-7/001_memory_space_registry_pending.md]
- Enforce space awareness across engine internals—deduplication, spreading activation, event streaming—so cross-space access is structurally impossible.[source: roadmap/milestone-7/002_engine_isolation_pending.md]
- Partition persistence by space with deterministic directory layout and independent workers for WAL and tier migration.[source: roadmap/milestone-7/003_persistence_partitioning_pending.md]
- Elevate `memory_space_id` to first-class input on REST, CLI, and gRPC calls with backwards-compatible defaults for single-space users.[source: roadmap/milestone-7/004_api_cli_surface_pending.md][source: roadmap/milestone-7/005_grpc_proto_multi_tenant_pending.md]
- Surface per-space metrics, health, and diagnostics so operators can monitor isolation and capacity.[source: roadmap/milestone-7/006_metrics_observability_pending.md]
- Establish validation suite and docs to confirm isolation and teach migrations.[source: roadmap/milestone-7/007_multi_tenant_validation_pending.md][source: roadmap/milestone-7/008_docs_migration_pending.md]

## External Benchmarks & Practices
- Multi-tenant databases typically combine registry + per-tenant schema or database; the plan mirrors registry + per-space store approach used in systems like PostgreSQL schemas and Elasticsearch indices. (Industry prior art)
- Observability patterns require per-tenant labels to avoid cardinality explosion—hints from milestone tasks to keep label sets controlled.[source: roadmap/milestone-7/006_metrics_observability_pending.md]

## Risks & Open Questions
- Registry bootstrap needs atomic creation to avoid races when multiple agents request the same space simultaneously.[source: roadmap/milestone-7/001_memory_space_registry_pending.md]
- Introducing space awareness into hot-path activation may affect latency; benchmarking is part of the validation task.[source: roadmap/milestone-7/002_engine_isolation_pending.md][source: roadmap/milestone-7/007_multi_tenant_validation_pending.md]
- Client migration must balance strict enforcement with compatibility; warnings and docs are required before removing defaults.[source: roadmap/milestone-7/004_api_cli_surface_pending.md][source: roadmap/milestone-7/005_grpc_proto_multi_tenant_pending.md]
