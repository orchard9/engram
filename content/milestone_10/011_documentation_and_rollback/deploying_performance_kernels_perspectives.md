# Documentation and Rollback - Perspectives

## Operations Perspective

Operators need three things:
1. How to deploy (step-by-step procedures)
2. How to monitor (what metrics indicate problems)
3. How to rollback (emergency procedures)

Documentation that skips any of these is incomplete.

For Zig kernels:
- Deploy: Build with --features zig-kernels, configure arena size
- Monitor: Track arena overflows, kernel execution time, error rates
- Rollback: Rebuild without feature flag, restart service

All three documented clearly.

## Systems Architecture Perspective

Rollback speed matters more than deployment speed.

Deployments are planned. You have time to test and validate.

Rollbacks are emergencies. You need to restore service in minutes.

For Zig kernels, rollback is deliberately simple: rebuild Rust-only binary. No database migrations, no state cleanup, no complex procedures.

RTO: 5 minutes from incident to service restored.
