# Task 010: NUMA Cross-Socket Performance

**Status**: Pending
**Estimated Duration**: 5-6 days
**Priority**: Medium - Tier 3 hardware validation

## Objective

Validate NUMA-aware memory allocation on multi-socket systems. Measure >80% NUMA-local references and <2x latency penalty for remote access. Implement automatic thread placement on NUMA topologies.

## Hardware Requirements

**Tier 3 Systems Only** (multi-socket):
- 2+ NUMA nodes (e.g., dual-socket Xeon, EPYC)
- 128GB+ RAM distributed across sockets
- Linux with numactl installed

## Key Measurements

```rust
pub struct NumaMetrics {
    pub local_access_pct: f64,    // Target >80%
    pub remote_access_pct: f64,   // Should be <20%
    pub cross_socket_latency_ns: u64,
    pub local_latency_ns: u64,
    pub latency_penalty_ratio: f64, // Target <2.0x
}
```

## Implementation Strategy

1. **Pin threads to NUMA nodes**: Use `hwloc` for topology detection
2. **Allocate memory locally**: Use `numa_alloc_onnode()` for hot-tier
3. **Measure with perf**: `perf stat -e node-loads,node-load-misses`

## Success Criteria

- **Local Access**: >80% of memory references are NUMA-local
- **Latency Penalty**: <2x for remote access
- **Automatic Placement**: Thread pinning requires no manual configuration
- **Graceful Fallback**: Works correctly on single-socket systems

## Files

- `engram-core/src/numa/affinity.rs` (350 lines)
- `engram-core/src/numa/allocator.rs` (280 lines)
- `scripts/measure_numa_perf.sh` (120 lines)
- `scenarios/numa/cross_socket.toml`
