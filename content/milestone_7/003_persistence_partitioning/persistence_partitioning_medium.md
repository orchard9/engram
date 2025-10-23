# Building Tenant-Isolated Persistence for Engram

Memory spaces turn Engram from a single-tenant research system into infrastructure that can host many autonomous agents. Achieving that vision demands more than tagging requests—we have to partition persistence so every tenant’s memories live, recover, and compact independently.

## Why Partitioning Matters
Today `MemoryStore` wires a single WAL buffer, tier migration worker, and compaction queue for the entire process (`engram-core/src/store.rs`). When the CLI boots, it mounts one data directory and all memories flow into the same log (`engram-cli/src/main.rs`). That works for demos, but the moment we host two agents the guarantees break: a WAL replay could resurrect another agent’s memories, compaction might evict the wrong tier, and IO pressure becomes impossible to attribute (`engram-core/src/storage/wal.rs`).

## Laying Out Per-Space Storage
The first step is normalizing directory structure. We will derive a sanitized `MemorySpaceId` and map it to `<data_root>/<space>/wal`, `<data_root>/<space>/hot`, and friends, keeping parity with the existing tier modules (`engram-core/src/storage/tiers.rs`). The registry becomes the fabric—during `create`, it ensures directories exist and returns handles; during `delete`, it tears them down safely. Enumeration flows the other direction: on startup, the registry scans known directories, recovers each space, and reports partial failures without blocking others.

## Dedicated WAL and Tier Workers
Once the layout is deterministic, we can spin up per-space WAL writers and readers. Each writer keeps its own sequence numbers, fsync cadence, and health metrics. Tier migration follows the same pattern: worker threads pull from space-specific queues, so a noisy tenant cannot starve consolidation elsewhere. To keep resource use in check we’ll lazily instantiate workers and shut them down when a space becomes idle.

## Threading Space IDs Through the Engine
Persistence hooks currently hide inside `MemoryStore`. We’ll thread a lightweight context that carries the `MemorySpaceId` through store, eviction, and recovery operations. That context selects the correct WAL handle, indexes the proper tier, and adds the space tag to metrics. Downstream components—HNSW updates, consolidation jobs—inherit the same pattern, ensuring workloads stay compartmentalized end-to-end.

## Operating the New System
Partitioning only helps if operators can understand it. Every metric—WAL lag, compaction backlog, tier utilization—will be labeled by space so dashboards reveal hotspots instantly. The CLI gets commands to list, create, and inspect spaces, plus diagnostics that read from per-space directories. Runbooks can now capture precise steps for migrating or restoring a single tenant while the rest of the cluster keeps running.

## What’s Next
With persistence isolated, follow-on tasks can expose the feature to APIs, update gRPC clients, and expand the validation suite. By the end of Milestone 7, multi-tenant Engram won’t just be an idea in docs—it will be a dependable storage substrate that keeps every agent’s memories truly their own.
