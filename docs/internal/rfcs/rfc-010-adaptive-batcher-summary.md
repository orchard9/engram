# RFC 010 Review Prep – Adaptive Batcher

**Meeting target:** 2025-10-16 @ 17:00 UTC (30 min)

## Purpose

- Walk reviewers through AdaptiveBatcher design (RFC 010) before implementation.

- Validate EWMA parameter defaults and guardrail plan.

- Confirm telemetry footprint and rollout steps across CLI/API/docs.

## Key Links

- Full RFC: `docs/rfcs/rfc-010-adaptive-batcher.md`

- Metrics artifacts: `docs/assets/metrics/` (`sample_metrics.json`, `2025-10-12-longrun/`)

- Bench plan: `engram-core/benches/README.md`

- Task tracker: `tmp/010—rewiring-todo.md`

## Agenda (30 min)

1. **Context recap (5 min):** why adaptive batching now, resolved prerequisites (pool/cache/metrics).

2. **Design walkthrough (10 min):** topology probe, EWMA loop, guardrails, telemetry surface.

3. **Validation plan (8 min):** deterministic tests, soak metrics, perf benchmarking timeline.

4. **Open questions (5 min):** NUMA resets, GPU hand-off expectations, instrumentation gaps.

5. **Next steps (2 min):** reviewer actions, implementation window, follow-up tasks.

## Reviewer Checklist

- **Systems Architecture (Aditi):** Validate lock-free interactions, scheduler integration, NUMA handling.

- **Verification/Test (Rowan):** Assess test coverage plan (unit/property/integration, soak).

- **Product/Roadmap (Noah):** Confirm rollout flag, operator messaging, alignment with Task 012.

- **Tech Writing (Mira):** Ensure metrics/runbook updates, CLI help references, On-call guidance.

## Pre-reading

- RFC sections: Summary, Design Overview, Metrics & Telemetry, Failure Modes, Rollout Plan.

- Task 010 status snapshot + tech debt log for outstanding dependencies.

- `docs/operations/metrics_streaming.md` for updated operator guidance.

## Prep Tasks

- [ ] Send calendar invite with this summary and RFC link.

- [ ] Attach latest metrics snapshots (`docs/assets/metrics`).

- [ ] Confirm hardware reservation ticket once ops approves (perf-rig-02).

- [ ] Gather benchmark delta expectations (baseline vs adaptive) for discussion.
