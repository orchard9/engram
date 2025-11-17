# Task 007: Gossip Protocol for Consolidation State

**Status**: Pending
**Estimated Duration**: 4 days
**Dependencies**: Task 001 (SWIM membership), Task 004 (Space Assignment), Task 005 (Replication for lag), Task 006 (Routing)
**Owner**: TBD

## Objective

Build an anti-entropy pipeline so semantic consolidation state (patterns, citations, versions) converges across nodes even when consolidation runs happen independently. Use Merkle-tree fingerprints to detect divergence, a gossip transport piggybacked on SWIM, and a merge/resolution path that respects vector clocks/conflict rules.

## Current Implementation Snapshot

- Consolidation runs are local only; results never leave the node unless an operator manually copies data.
- No Merkle tree, version tracking across nodes, or gossip hooks exist in the codebase.
- SWIM gossip currently carries only membership updates; we need to add consolidation payloads.
- Conflict resolution (vector clocks, merging patterns) is not implemented—Task 008 is supposed to define these rules, so we must align with that spec.

## Technical Specification

### Data to Synchronize

For each `SemanticPattern`:
- `pattern_id`
- `embedding` (or hash/fingerprint to avoid shipping full vector unless needed)
- `confidence`, `citations`, `updated_at`
- `version` (vector clock) to detect concurrent edits

Additionally, track `consolidation_generation` per node to know the latest update cycle.

### Merkle Tree Fingerprinting

- Implement a Merkle tree builder under `engram-core/src/consolidation/merkle.rs` (or reuse existing caching structures).
- Tree parameters:
  - Depth: 12 (4096 leaves) to cover up to ~100K patterns.
  - Leaf hash = SHA-256(pattern_id || confidence || updated_at || version hash). Embedding can be hashed to 32 bytes to avoid large payloads.
  - Internal hash = SHA-256(left || right).
- Maintain a mapping from leaf index → `pattern_id`s for delta extraction.
- Update tree incrementally whenever consolidation writes a pattern; only recompute affected path.

### Gossip Protocol

- Extend SWIM’s `SwimMessage::Gossip` payload to include consolidation Merkle roots.
- Add a `ConsolidationGossip` struct containing:
  - `space_id`
  - `root_hash`
  - `generation`
  - `node_id`
- When two nodes compare roots and detect mismatch, they should request a range sync:
  1. Exchange Merkle subtree hashes until the differing leaves are found.
  2. Fetch the actual patterns for those leaves via a new RPC (`FetchConsolidationDelta`).
- Rate-limit gossip rounds (e.g., once per minute) to avoid overwhelming SWIM.

### Delta Transfer & Merge

- Add a gRPC/HTTP RPC in `engram-cli` (`/consolidation/delta`) that accepts `pattern_id` lists and returns serialized `SemanticPattern`s.
- When receiving remote patterns:
  - Use vector clocks (Task 008 spec) to decide whether to accept, merge, or flag conflicts.
  - Update local state and rebuild affected Merkle path.
  - Record origin node for auditing.
- If a conflict is detected (concurrent updates), hand off to Task 008’s resolution rules (e.g., highest confidence wins, merge citations, etc.).

### Integration with Task 005/006

- Replication lag metrics help decide whether a replica should become a gossip peer (avoid gossiping with replicas that are far behind).
- Routing layer should expose an endpoint to request deltas from any node; ensure circuit breaker/retry logic handles transient failures.

### Metrics & Observability

- Add Prometheus metrics:
  - `engram_consolidation_gossip_rounds_total`
  - `engram_consolidation_deltas_transferred_bytes`
  - `engram_consolidation_conflicts_total`
  - `engram_consolidation_merkle_compare_seconds`
- Expose `/cluster/consolidation` in the CLI with state: root hash, generation, pending conflicts.

### Testing Strategy

1. **Unit tests** for Merkle tree update/compare logic.
2. **Integration tests** spinning up multiple nodes (using `tokio::test`) to verify gossip converges after independent consolidations.
3. **Conflict tests** to ensure concurrent edits are detected and handed to Task 008’s resolution logic.
4. **Performance tests** to confirm gossip bandwidth stays within targets (<5KB/s per node).

## Acceptance Criteria

1. Each node maintains a Merkle fingerprint of its consolidation state, updated after every consolidation run.
2. SWIM gossip exchanges root hashes and triggers delta sync when roots differ.
3. Delta transfers fetch only changed patterns and merge them respecting vector clocks/conflict rules.
4. Metrics and CLI endpoints expose gossip progress, conflicts, and root hashes.
5. Tests demonstrate convergence across multiple nodes even after partitions.
