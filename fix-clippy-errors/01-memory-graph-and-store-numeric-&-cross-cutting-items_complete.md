# 01 Memory Graph And Store – Numeric & Cross-Cutting Items

## Scope
- `engram-core/src/store.rs`
- `engram-core/src/memory_graph/backends/infallible.rs`
- `engram-core/src/memory_graph/backends/hashmap.rs`
- `engram-core/src/memory.rs`
- `engram-core/src/memory_graph/tests.rs`
- `engram-core/src/memory_graph/traits.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/store.rs` | 17 | e0277 (5), needless pass by value (4), trivially copy pass by ref (4) |
| `engram-core/src/memory_graph/backends/infallible.rs` | 6 | cast precision loss (4), significant drop tightening (2) |
| `engram-core/src/memory_graph/backends/hashmap.rs` | 5 | significant drop tightening (5) |
| `engram-core/src/memory.rs` | 4 | large types passed by value (4) |
| `engram-core/src/memory_graph/tests.rs` | 3 | e0277 (3) |
| `engram-core/src/memory_graph/traits.rs` | 1 | suboptimal flops (1) |

## Recommended approach
- Replace lossy `as` casts with checked conversions or widen data types as appropriate.
- Exploit `mul_add` / fused ops or precompute invariants to silence floating-point efficiency lints.
- Pass heavy data (embeddings, counters) by reference to avoid copies and appease Clippy.
- Audit float comparisons/conversions; use tolerances or intermediate integer types where needed.
