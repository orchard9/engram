# Example 08: Multi-Tenant Operations

**Learning Goal**: Implement secure multi-tenant memory isolation with per-tenant spaces and resource limits.

**Difficulty**: Advanced
**Time**: 20 minutes
**Prerequisites**: Completed Example 06 (Authentication)

## Cognitive Concept

Multi-tenant isolation in Engram:
- Each tenant gets a separate memory space (cognitive "namespace")
- No cross-tenant memory leakage via spreading activation
- Per-space resource limits prevent noisy neighbors
- Shared knowledge spaces for common facts

## What You'll Learn

- Create per-tenant memory spaces
- Scope JWT tokens to specific spaces
- Implement noisy neighbor protection
- Share common knowledge across tenants
- Monitor per-tenant resource usage

## Architecture Patterns

### Pattern 1: Strong Isolation (SaaS)
```
Tenant A → memory_space_a (isolated)
Tenant B → memory_space_b (isolated)
Tenant C → memory_space_c (isolated)
```

### Pattern 2: Shared + Private (Hybrid)
```
Tenant A → [memory_space_a_private, shared_knowledge]
Tenant B → [memory_space_b_private, shared_knowledge]
Common facts in shared_knowledge, private data isolated
```

### Pattern 3: Hierarchical (Enterprise)
```
Organization → org_shared
  ↳ Team A → team_a_space
  ↳ Team B → team_b_space
```

## Code Examples

See language-specific implementations in this directory:

- `rust.rs` - Rust with space management
- `python.py` - Python with tenant middleware
- `typescript.ts` - TypeScript with space routing
- `go.go` - Go with context propagation
- `java.java` - Java with tenant context

## Security Guarantees

1. **No cross-tenant reads**: Spreading activation stops at space boundary
2. **Resource isolation**: Per-space capacity limits
3. **Audit trail**: Per-tenant operation logs
4. **Token validation**: Server enforces space claim matches request

## Resource Limits

Configure per-space limits:

```toml
[memory_spaces.limits]
hot_capacity_per_space = 100_000
warm_capacity_per_space = 1_000_000
cold_capacity_per_space = 10_000_000
max_spreading_depth = 5
rate_limit_per_space = 100  # ops/sec
```

## Monitoring Per-Tenant

```promql
# Memory usage by tenant
engram_storage_memory_bytes{space="tenant_acme"}

# Operation rate by tenant
rate(engram_operations_total{space="tenant_acme"}[5m])

# Latency by tenant
histogram_quantile(0.99,
  engram_spreading_latency_seconds_bucket{space="tenant_acme"}
)
```

## Next Steps

- [Security Guide](/operations/security.md) - Production security
- [Capacity Planning](/operations/capacity-planning.md) - Resource allocation
