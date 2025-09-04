# Build health check system with clear status reporting

## Status: PENDING

## Description
Implement comprehensive health checking system that provides clear, actionable status information for operations.

## Requirements
- HTTP health endpoint at /health
- Readiness vs liveness distinction
- Component-level health status
- Degraded state detection
- Health history tracking
- Alerting thresholds

## Acceptance Criteria
- [ ] GET /health returns 200 OK when healthy
- [ ] Detailed JSON with component statuses
- [ ] Readiness check for load balancers
- [ ] Clear status: GREEN/YELLOW/RED
- [ ] Sub-100ms response time

## Dependencies
- Task 010 (engram start)

## Notes
- Follow Kubernetes health check patterns
- Include recent error rate in health
- Check storage tier availability
- Monitor activation spreading performance