# Task 006: Consolidation Metrics Observability

## Status
IN_PROGRESS

## Priority
P0 (Critical Path)

## Effort Estimate
1 days

## Dependencies
- Task 005

## Objective
See MILESTONE_5_6_ROADMAP.md for complete specification.

## Technical Approach
Complete technical specification available in MILESTONE_5_6_ROADMAP.md (Milestone 6, Task 006).
Refer to main roadmap document for detailed implementation plan, code examples, integration points, and acceptance criteria.

## Current Progress
- `/api/v1/consolidations` and `/api/v1/consolidations/{id}` now expose semantic belief snapshots with citation trails, schema confidence, and freshness metrics.
- Consolidation SSE stream emits `belief`, `progress`, and keepalive events with novelty thresholds and consolidation run statistics.
- Temporal provenance (`observed_at`, `stored_at`, `last_access`) is available in write and recall responses to support drift monitoring.
- Scheduler-backed consolidation snapshots update metrics gauges/counters (`engram_consolidation_*`) and persist belief-delta logs to `data/consolidation/alerts/` for alerting.
- `consolidation-soak` harness + baseline artifacts (docs/assets/consolidation/baseline) provide repeatable telemetry for dashboard validation.

## Remaining Work
- Wire the new consolidation metrics into dashboards/alerting rules and document operator response playbooks (use `docs/operations/consolidation_dashboard.md`).
- Expand integration coverage around the persisted belief-update log and scheduler gauges to guard against regressions (promote `consolidation_soak` + future load tests into CI once stable).
- Run long-window soak tests to baseline freshness/novelty thresholds and capture reference dashboards for docs (replace baseline artifacts with 1h capture before release).

## Acceptance Criteria
See MILESTONE_5_6_ROADMAP.md for detailed acceptance criteria.

## Testing Approach
See MILESTONE_5_6_ROADMAP.md for comprehensive testing strategy.

## Notes
This task file provides summary information. Complete implementation-ready specifications are in MILESTONE_5_6_ROADMAP.md.
