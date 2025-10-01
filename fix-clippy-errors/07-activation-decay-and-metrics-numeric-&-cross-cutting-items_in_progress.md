# 07 Activation Decay And Metrics – Numeric & Cross-Cutting Items

## Scope
- `engram-core/src/metrics/lockfree.rs`
- `engram-core/src/metrics/numa_aware.rs`
- `engram-core/src/metrics/hardware.rs`
- `engram-core/src/metrics/streaming.rs`
- `engram-core/src/metrics/health.rs`
- `engram-core/src/metrics/mod.rs`
- `engram-core/src/batch/collector.rs`
- `engram-core/src/decay/two_component.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/metrics/lockfree.rs` | 12 | inline always (7), cast precision loss (2), while float (1) |
| `engram-core/src/metrics/numa_aware.rs` | 10 | inline always (4), cast precision loss (4), needless pass by value (1) |
| `engram-core/src/metrics/hardware.rs` | 8 | cast precision loss (6), inline always (1), needless pass by value (1) |
| `engram-core/src/metrics/streaming.rs` | 7 | cast precision loss (3), inline always (1), needless pass by value (1) |
| `engram-core/src/metrics/health.rs` | 6 | trivially copy pass by ref (4), cast precision loss (2) |
| `engram-core/src/metrics/mod.rs` | 6 | inline always (6) |
| `engram-core/src/batch/collector.rs` | 4 | cast precision loss (4) |
| `engram-core/src/decay/two_component.rs` | 4 | cast precision loss (4) |

## Recommended approach
- Replace lossy `as` casts with checked conversions or widen data types as appropriate.
- Exploit `mul_add` / fused ops or precompute invariants to silence floating-point efficiency lints.
- Drop `#[inline(always)]` unless perf data justifies forcing inlining.
- Pass heavy data (embeddings, counters) by reference to avoid copies and appease Clippy.
- Audit float comparisons/conversions; use tolerances or intermediate integer types where needed.
