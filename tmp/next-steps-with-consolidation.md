# Next Steps – Consolidation Follow-Through

## 1. Integrate Background Scheduler Snapshots
- **Objective**: Serve `/api/v1/consolidations*` data from the asynchronous scheduler instead of on-demand replay.
- **Relevant Roadmap**: `roadmap/milestone-6/002_pattern_detection_engine_in_progress.md`, `roadmap/milestone-6/003_semantic_memory_extraction_in_progress.md`.
- **Actions**:
  - Wire `ConsolidationScheduler` to publish `ConsolidationSnapshot` instances into a durable cache (`Arc<RwLock<Option<CachedSnapshot>>>`).
  - Update `MemoryStore::consolidation_snapshot` to prefer the cached snapshot, falling back to eager generation only when no scheduler output is available.
  - Ensure scheduler-driven runs execute pattern detection semantics (Task 002) before publishing snapshots and surface the semantic extraction metrics (Task 003).
  - Extend REST + SSE handlers to route through the cached snapshot so consolidation responses never block on on-demand recomputation.
  - Add regression coverage proving that recall/consolidation endpoints emit scheduler-backed data and respect warm-up fallback rules.
  - Formalize scheduler health contract (cadence, failover, SLA) and surface it in operations docs and dashboards.

## 2. Persisted Metrics & Alerts for Task 006 Completion *(roadmap status: in progress)*
- **Objective**: Expand consolidation observability to satisfy Milestone 6 Task 006 deliverables.
- **Relevant Roadmap**: `roadmap/milestone-6/006_consolidation_metrics_observability_in_progress.md`.
- **Actions**:
  - Export consolidation run counters (`total_runs`, `successful_consolidations`, failure counts, freshness deltas) via the `MetricsRegistry` with stable names consumed by Prometheus/SSE.
  - Persist belief update events—confidence deltas, citation churn—into an append-only alert log with rotation so operators can raise tickets post-incident.
  - Define alert thresholds (e.g., consecutive failed runs, novelty stagnation, stale snapshots) and document remediation playbooks alongside the roadmap.
  - Refresh OpenAPI + operations docs to describe the new metrics and alert workflows, and land integration smoke tests guarding the persistence path.
  - Baseline long-window smoke/regression runs (≥1h) capturing metric snapshots and belief logs for reference dashboards *(30s sample captured via `consolidation-soak` in `docs/assets/consolidation/baseline/`; rerun for full hour before sign-off).* 

## 3. Future-proof Consolidation Architecture
- **Objective**: Reduce tech debt by consolidating scheduling, caching, and observability inside a dedicated consolidation module boundary.
- **Relevant Roadmap**: feeds follow-on RFC under Milestone 6/7; see `docs/architecture/consolidation_boundary.md`.
- **Actions**:
  - Extract cache + metrics façade behind a `ConsolidationService` trait so `MemoryStore` depends on an interface.
  - Flesh out RFC for distributed scheduler support and remote cache backend options (Redis, S3 snapshots) — see updated `docs/architecture/rfc_consolidation_service.md`.
  - Plan integration/load fixtures that replay high-ingest workloads to validate cache coherency before flipping the module boundary (scaffold added in `engram-core/tests/consolidation_load_tests.rs`).
