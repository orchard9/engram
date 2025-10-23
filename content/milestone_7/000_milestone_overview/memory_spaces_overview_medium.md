# Milestone 7: Giving Every Agent a Private Memory Space

Engram’s seventh milestone rewires the cognitive database around explicit memory spaces so multiple agents can coexist without blending their recollections. Today, the system spins up a single `MemoryStore`, accepts requests without tenant hints, and writes everything into one persistence tree.[source: roadmap/milestone-7/000_milestone_overview_pending.md] That simplicity works for demos but collapses in multi-agent deployments. This milestone turns isolation into a first-class capability.

## Step 1: Install a Control Plane
We begin with a registry that mints `MemorySpaceId` values, provisions associated stores, and guards lifecycle operations whether they come from operators or auto-scaling workflows.[source: roadmap/milestone-7/001_memory_space_registry_pending.md] Registry lookups must be lock-free on the hot path yet atomic when establishing a new space. It also seeds persistence directories, so every other subsystem immediately inherits the right boundaries.

## Step 2: Thread Space Into the Engine
Once the registry exists, the core engine must refuse to operate without a space context. Store, recall, consolidation, and spreading activation all accept a space-scoped handle, ensuring a recall in “alpha” cannot see “beta” memories even if a bug tries to sneak through.[source: roadmap/milestone-7/002_engine_isolation_pending.md] We attach the same context to event broadcasts so downstream consumers—SSE feeds, monitoring—know which tenant generated an activation.

## Step 3: Partition Persistence
Multi-tenancy is meaningless if WAL records and tiered data still mingle on disk. Partitioning defines a deterministic directory structure (`<data_root>/<memory_space_id>/…`), spins up space-specific WAL writers, and ensures recovery or compaction runs never cross folders.[source: roadmap/milestone-7/003_persistence_partitioning_pending.md] The registry owns these lifecycles, so deleting a space can eventually clean up its artifacts safely.

## Step 4: Evolve the Surfaces
With internals safeguarded, we raise `memory_space_id` to the UI level: REST routes, CLI commands, and configuration must declare a target space.[source: roadmap/milestone-7/004_api_cli_surface_pending.md] gRPC contracts follow suit, adding optional fields with a default for legacy clients.[source: roadmap/milestone-7/005_grpc_proto_multi_tenant_pending.md] We pair strict validation with clear error messages, helping developers adopt the new parameter without surprises.

## Step 5: Observe and Validate
Operators need per-space metrics, health, and diagnostics to keep deployments healthy.[source: roadmap/milestone-7/006_metrics_observability_pending.md] Meanwhile, the validation suite attacks every seam—concurrent writes, crash recovery, fuzzing—to confirm isolation holds even under pathological load.[source: roadmap/milestone-7/007_multi_tenant_validation_pending.md] The milestone wraps with documentation and migration notes that turn the plan into an actionable upgrade guide.[source: roadmap/milestone-7/008_docs_migration_pending.md]

By the end, Engram behaves like a true multi-tenant cognitive database: each agent owns a private memory corridor, operations remain fast, and observability tooling keeps administrators confident in the separation.
