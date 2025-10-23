1/ Engram Milestone 7 is all about **memory spaces**—true multi-tenant isolation so every agent keeps its own recollections.[source: roadmap/milestone-7/000_milestone_overview_pending.md]

2/ We’re adding a registry that mints `MemorySpaceId` values, provisions stores, and keeps lifecycle operations sane even under concurrent load.[source: roadmap/milestone-7/001_memory_space_registry_pending.md]

3/ The engine soon refuses to run without a space handle. Store, recall, spreading activation—everything stays within its lane.[source: roadmap/milestone-7/002_engine_isolation_pending.md]

4/ Persistence splits per space: deterministic directories, WAL writers, tier workers, and recovery paths that never touch the wrong tenant.
[source: roadmap/milestone-7/003_persistence_partitioning_pending.md]

5/ REST, CLI, and gRPC surfaces adopt `memory_space_id` with friendly defaults for existing users plus migration docs to ease the switch.[source: roadmap/milestone-7/004_api_cli_surface_pending.md][source: roadmap/milestone-7/005_grpc_proto_multi_tenant_pending.md]

6/ Observability grows per-space metrics and health views, while a new validation suite hammers the system to prove isolation.[source: roadmap/milestone-7/006_metrics_observability_pending.md][source: roadmap/milestone-7/007_multi_tenant_validation_pending.md]

7/ Final stop: documentation that shows how to upgrade, troubleshoot, and operate Engram in a multi-agent world.[source: roadmap/milestone-7/008_docs_migration_pending.md]
