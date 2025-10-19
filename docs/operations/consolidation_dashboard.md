# Consolidation Dashboard Checklist

> Status: Placeholder documenting required follow-up once scheduler metrics are wired into the observability stack.

## Dashboard Goals

- Visualize scheduler cadence (`engram_consolidation_runs_total` vs failures) and freshness gauge trends.
- Surface belief update deltas (confidence/citation churn) with links to the persisted alert log.
- Highlight SLA breaches defined in the health contract (see `consolidation_observability.md`).

## Required Widgets

1. **Run Cadence**: counter diff per 5-minute window with alert overlay when failures spike.
2. **Freshness Heatmap**: shows `engram_consolidation_freshness_seconds` percentile bands.
3. **Novelty Trend**: tracks `engram_consolidation_novelty_gauge` over time with stagnation threshold lines.
4. **Belief Update Feed**: tail of `data/consolidation/alerts/*.jsonl` with filters for confidence/citation deltas.
5. **Failover Indicator**: boolean showing whether API fell back to on-demand snapshots in the last hour.

### Baseline Artifacts

- Reference soak output lives in `docs/assets/consolidation/baseline/` (captured with `consolidation-soak`).
- Use these JSONL files to prototype panels and validate schema mappings before live integration.
- Replace the checked-in sample with a fresh 1h capture prior to shipping dashboards.

## Follow-Up Tasks

- Implement exporters for the dashboard target (Grafana, Looker, etc.).
- Capture baseline screenshots after running the planned 1h soak test.
- Document remediation runbook steps directly in dashboard annotations.

> Track progress under Milestone 6 Task 006 and update this file once visualizations are live.
