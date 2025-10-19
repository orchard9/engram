# Next Steps – Consolidation Follow-Through

## 1. Integrate Background Scheduler Snapshots
- **Objective**: Serve `/api/v1/consolidations*` data from the asynchronous scheduler instead of on-demand replay.
- **Relevant Roadmap**: `roadmap/milestone-6/002_pattern_detection_engine_pending.md`, `roadmap/milestone-6/003_semantic_memory_extraction_pending.md`.
- **Actions**:
  - Wire `ConsolidationScheduler` to emit snapshots into a durable cache (e.g., `Arc<RwLock<Option<ConsolidationSnapshot>>>`).
  - Update `MemoryStore::consolidation_snapshot` to read from that cache, falling back to eager generation only during scheduler warm-up.
  - Ensure scheduler triggers pattern detection semantics (Task 002) prior to snapshot publication and records stats for Task 003 acceptance criteria.
  - Add regression tests validating that the REST and SSE layers reflect scheduler-generated data and do not block on fresh replay.

## 2. Persisted Metrics & Alerts for Task 006 Completion *(in progress per roadmap update)*
- **Objective**: Expand consolidation observability to satisfy Milestone 6 Task 006 deliverables.
- **Relevant Roadmap**: `roadmap/milestone-6/006_consolidation_metrics_observability_pending.md`.
- **Actions**:
  - Export consolidation run metrics (`total_replays`, `successful_consolidations`, novelty rates) through the existing metrics registry (Prometheus/SSE) with stable names.
  - Record belief update events—confidence deltas, citation churn—into a persistent log (e.g., WAL segment or structured JSONL) for alerting and audits.
  - Define alert thresholds (e.g., failed_consolidations spike, freshness_hours exceeding SLA) and document operator responses.
  - Update docs/OpenAPI to cover new metrics endpoints or dashboards, and add integration smoke tests to guard against regressions.
