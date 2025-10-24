# Task 006: Scaling Strategies & Capacity Planning — pending

**Priority:** P1 (High)
**Estimated Effort:** 2 days
**Dependencies:** Tasks 001 (Deployment), 003 (Monitoring)

## Objective

Define vertical and horizontal scaling procedures with capacity planning guidance. Enable operators to predict resource needs, scale proactively, and optimize infrastructure costs while meeting SLAs.

## Key Deliverables

- `/scripts/estimate_capacity.sh` - Capacity planning calculator
- `/docs/operations/scaling.md` - Complete scaling guide
- `/docs/operations/capacity-planning.md` - Capacity planning worksheet
- `/docs/howto/scale-vertically.md` - Step-by-step vertical scaling
- `/docs/reference/resource-requirements.md` - Resource specifications

## Technical Specifications

**Vertical Scaling:**
- CPU: 1 core → 2 cores → 4 cores → 8 cores (2x throughput per doubling up to 32 cores)
- Memory: 2GB → 4GB → 8GB → 16GB (proportional to active memory set)
- Storage: 20GB → 50GB → 100GB → 500GB (1.5x data size for overhead)

**Capacity Planning Formula:**
```
Required CPU = (target_ops_per_sec / 1000) cores
Required RAM = (active_nodes * 1KB) + (hot_tier_size)
Required Disk = (total_nodes * 2KB * 1.5)  # 1.5x for WAL/indices
```

**Scaling Triggers:**
- CPU >70% sustained → Add CPU cores
- Memory >80% → Increase RAM or reduce hot tier
- Disk >80% → Expand volume or archive old data
- P99 latency >50ms → Add CPU or optimize queries

## Acceptance Criteria

- [ ] Capacity calculator predicts resource needs within 15% accuracy
- [ ] Vertical scaling procedures tested for CPU/memory/storage
- [ ] Scaling thresholds defined for all key metrics
- [ ] Cost optimization reduces spend by >20% without SLA impact
- [ ] External operator successfully scales deployment following docs

## Follow-Up Tasks

- Milestone 14: Horizontal scaling across distributed nodes
- Task 004: Reference performance baselines in capacity planning
