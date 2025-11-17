# Task 009: Distributed Query Execution (Scatter-Gather)

**Status**: Pending
**Estimated Duration**: 3-4 days
**Dependencies**: Task 004 (Space Assignment), Task 005 (Replication), Task 006 (Routing), Task 007/008 (state convergence)
**Owner**: TBD

## Objective

Enable Engram’s recall/spreading APIs to execute across partitions transparently. This requires a query planner, scatter/gather executor, partial-result aggregation with confidence penalties, and timeout handling that downgraded confidence instead of failing outright.

## Current Implementation Snapshot

- Queries run only on the local node; there is no awareness of remote partitions.
- The router can proxy writes to primaries but reads still hit the local store.
- There is no query planner or scatter/gather executor.

## Technical Specification

### Query Types to Support

1. `recall` (space-specific) – can stay local if the space is on this node, otherwise route to the primary.
2. `spread` (activation across graph) – may touch multiple spaces/partitions; scatter to all relevant nodes.
3. `complete` (pattern completion) – generally local.
4. `imagine` / cross-space queries – multi-partition.

### Planner

Add `engram-core/src/query/distributed/planner.rs` with a `QueryPlanner` that:
- Uses `SpaceAssignmentManager` to map `MemorySpaceId`s to nodes.
- Consults `PartitionDetector` to avoid routing to unreachable nodes.
- Determines execution strategy: `SinglePartition`, `MultiPartition`, or `Broadcast`.
- Produces a `QueryPlan { targets, expected_partitions, timeout, strategy }`.

### Executor

Add `engram-cli/src/query/executor.rs` (or extend existing executor) with:
- `execute_plan(plan, request)` that spawns async tasks per target node.
- Uses `Router` to reuse connection pooling + retries.
- Collects `PartialResult`s (memory hits, confidences, latencies).
- Aggregates into a final `QueryResult` with adjusted confidence: `final_confidence = base * (responding_partitions / expected_partitions)`.
- If some partitions timeout, include them in response metadata (`missing_partitions`) and log warnings.

### Partial Result Aggregation

`PartialResult` should include:
- `space_id`, `node_id`
- `matches` (memories/patterns)
- `confidence_adjustment`
- `latency`

Aggregation logic:
- Merge matches by `memory_id`, keeping highest confidence or combining as appropriate.
- Penalize final confidence proportionally to missing partitions.
- Provide optional strict consistency modes (future: QUORUM/ALL).

### Timeouts & Retries

- Use router retry/backoff for per-node RPCs.
- Overall query deadline (configurable, e.g., 2s) enforced at coordinator.
- On timeout, mark partition as missing and reduce confidence; do not fail the entire query unless `strict_consistency` is requested.

### Metrics & Observability

Add metrics:
- `engram_query_scatter_count{strategy}`
- `engram_query_partial_timeouts_total`
- `engram_query_latency_seconds{strategy}`
- `engram_query_missing_partitions_total`

Expose in `/cluster/health` and CLI stats.

### Configuration

Extend `engram-cli/config/default.toml` with:
- `query.scatter_timeout_ms`
- `query.partial_confidence_penalty`
- `query.max_fanout`

### Testing

1. **Unit tests** for planner strategies: ensure targeted queries hit one node, spread queries include all relevant nodes.
2. **Integration tests** with multiple nodes to verify aggregator merges partial results and handles timeouts.
3. **Failure injection**: drop one partition’s response to ensure confidence penalty is applied.
4. **Performance test**: measure added latency to ensure <2x slowdown for typical multi-partition queries.

## Acceptance Criteria

1. Query planner determines target nodes per query type, avoiding unreachable nodes.
2. Scatter/gather executor fans out requests, retries failures, and aggregates results.
3. Partial responses lead to reduced confidence rather than hard failures.
4. Metrics/CLI show scatter/gather activity, missing partitions, and latency.
5. Tests cover planner decisions, aggregation, and timeout handling.
