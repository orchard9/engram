# Memory Space Support – Perspectives

## Systems Architecture Perspective
Designing a registry that brokers `MemorySpaceId` → `MemoryStore` mappings centralizes lifecycle and persistence provisioning decisions.[source: roadmap/milestone-7/001_memory_space_registry_pending.md] The control plane must initialize per-space WAL writers and tier workers, ensuring IO isolation from the first write.[source: roadmap/milestone-7/003_persistence_partitioning_pending.md]

## Graph Engine Perspective
Space enforcement burrows into the activation pipeline: deduplicator, eviction queue, and spreading engine all need explicit space context so activation waves never cross boundaries.[source: roadmap/milestone-7/002_engine_isolation_pending.md] Embedding this in compile-time signatures guarantees multi-tenant safety before runtime checks.

## API & Developer Experience Perspective
Surfaces must elevate `memory_space_id` to required inputs across REST, CLI, and gRPC while preserving default behavior for single-space adopters, delivering helpful guidance when callers omit the field.[source: roadmap/milestone-7/004_api_cli_surface_pending.md][source: roadmap/milestone-7/005_grpc_proto_multi_tenant_pending.md] CLI ergonomics hinge on sensible defaults and introspection commands.

## Observability & Operations Perspective
Per-space metrics and health output allow operators to spot hot tenants, capacity pressure, or skewed consolidation while keeping dashboards readable.[source: roadmap/milestone-7/006_metrics_observability_pending.md] Diagnostics scripts and docs must translate raw telemetry into actionable steps during incidents.[source: roadmap/milestone-7/008_docs_migration_pending.md]

## Quality & Validation Perspective
Isolation guarantees depend on exhaustive testing: concurrent stress across spaces, persistence crash-recovery, and fuzzing around registry races confirm the architecture holds under pressure.[source: roadmap/milestone-7/007_multi_tenant_validation_pending.md]
