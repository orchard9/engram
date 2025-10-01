# 05 Query And Cue Pipeline – Numeric & Cross-Cutting Items

## Scope
- `engram-core/src/query/integration.rs`
- `engram-core/src/cue/handlers.rs`
- `engram-core/src/query/evidence.rs`
- `engram-core/src/query/mod.rs`
- `engram-core/src/index/confidence_metrics.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/query/integration.rs` | 10 | assigning clones (4), cast sign loss (2), trivially copy pass by ref (1) |
| `engram-core/src/cue/handlers.rs` | 3 | cast precision loss (3) |
| `engram-core/src/query/evidence.rs` | 3 | cast precision loss (2), needless pass by value (1) |
| `engram-core/src/query/mod.rs` | 2 | cast precision loss (2) |
| `engram-core/src/index/confidence_metrics.rs` | 1 | cast possible truncation (1) |

## Recommended approach
- Replace lossy `as` casts with checked conversions or widen data types as appropriate.
- Pass heavy data (embeddings, counters) by reference to avoid copies and appease Clippy.
