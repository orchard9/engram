# Task 008: Conflict Resolution for Divergent Consolidations

**Status**: Pending
**Estimated Duration**: 4-5 days
**Dependencies**: Task 007 (Consolidation Gossip), Task 005 (Replication lag metrics)
**Owner**: TBD

## Objective

Define and implement deterministic merge rules for semantic patterns that diverge across nodes. When Task 007’s gossip detects differences, this module decides whether to merge, keep separate, or flag conflicts based on pattern similarity, vector clocks, and biological heuristics. The result must be convergent (CRDT-like), preserving confidence calibration and citations.

## Current Implementation Snapshot

- Semantic patterns are stored per node with no distributed metadata (no vector clocks or merge history).
- Gossip doesn’t run yet (Task 007), so no conflicts are detected in code.
- No merge/resolution APIs exist in the CLI or consolidation engine.

## Technical Specification

### Metadata Requirements

Extend `SemanticPattern` (or wrap it in `DistributedPattern`) with:
- `vector_clock: VectorClock` (node_id → logical timestamp)
- `origin_nodes: HashSet<String>` for auditing
- `reconsolidation_window: Timestamp` to enforce biological timing (optional)
- `schema_labels` to bias merges (optional)

Vector clocks are incremented whenever a node updates a pattern and merged via `max()` during exchanges.

### Conflict Detection

When gossip requests deltas, compare local vs remote entries:
1. If `vector_clock` indicates happens-before (`self < other`), fast-forward to the newer version.
2. If concurrent (`||`), run semantic resolution.
3. If pattern IDs differ but similarity > threshold (e.g., 0.85 cosine), treat as same conceptual pattern (Type 2 conflict).
4. If episodes overlap but embeddings diverge, treat as Type 1 conflict (same pattern different support).

### Merge Strategies

| Conflict Type | Detection | Resolution |
| --- | --- | --- |
| Type 1: same core episodes, minor divergence | high intersection of `episode_ids` and cosine similarity > 0.8 | Merge embeddings via confidence-weighted average, union `episode_ids`, `confidence = max`, merge citations, merge vector clocks |
| Type 2: independent patterns with similar semantics | cosine similarity between embeddings > threshold and overlapping schema labels | Optionally merge if both confidence > min threshold; otherwise keep both but reduce confidence to reflect ambiguity |
| Type 3: concurrent updates to same pattern | `vector_clock` indicates concurrency | Choose semantic merge function: combine citations, average confidence (biased toward higher citation count), merge vector clocks |
| Type 4: same episode assigned to multiple patterns | `episode_ids` overlap but overall similarity low | Allow multi-membership but decay confidence to reflect competition; store metadata so future consolidation can resolve |

All merges must be associative/commutative/idempotent to guarantee convergence.

### Implementation Plan

1. Add `engram-core/src/consolidation/conflict.rs` with:
   - `VectorClock` type + serialization.
   - `ConflictResolver` with methods per conflict type.
   - Similarity helpers (cosine, overlap ratios).
2. Extend consolidation storage to persist vector clocks and conflict metadata in WAL so merges survive restarts.
3. Integrate with Task 007’s gossip flow: when the gossip delta RPC receives remote patterns, pass them through `ConflictResolver::merge(local, remote)` before storing.
4. Update metrics to track conflicts resolved vs unresolved (`engram_consolidation_conflicts_total{type}`) and confidence adjustments.
5. For unresolved conflicts (e.g., low similarity multi-membership), surface them via CLI/monitoring so operators know patterns need review.

### Observability

- Add per-pattern audit logs when merges occur (log `pattern_id`, participating nodes, resulting confidence).
- `/cluster/consolidation` should include conflict counts and last merge timestamp.
- Metrics for merge latency and number of concurrent conflicts.

### Testing Strategy

1. **Unit tests** for `VectorClock` operations and conflict detection thresholds.
2. **Merge tests** covering each conflict type with synthetic patterns (embedding arrays, episode sets, confidence values).
3. **Integration tests** with two mocked nodes performing concurrent consolidations; after running the resolver, states should converge.
4. **Regression tests** to ensure merges are associative/commutative by applying them in different orders.

## Acceptance Criteria

1. Patterns carry vector clocks and conflict metadata, persisted across restarts.
2. `ConflictResolver` merges divergent patterns deterministically; concurrent updates converge regardless of merge order.
3. Gossip (Task 007) routes all remote patterns through the resolver before applying.
4. Conflicts that cannot be reconciled automatically are surfaced via metrics/logs/CLI.
5. Tests cover all conflict types and verify convergence.
