# 06 Diagnostics And Error Reporting – Numeric & Cross-Cutting Items

## Scope
- `engram-core/src/differential/reporting.rs`
- `engram-core/src/error_testing.rs`
- `engram-core/src/error_review.rs`
- `engram-core/src/lib.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/differential/reporting.rs` | 8 | cast possible truncation (3), cast precision loss (3), needless pass by value (1) |
| `engram-core/src/error_testing.rs` | 8 | cast precision loss (4), cast possible truncation (2), cast sign loss (2) |
| `engram-core/src/error_review.rs` | 2 | cast precision loss (2) |
| `engram-core/src/lib.rs` | 1 | cast possible truncation (1) |

## Recommended approach
- Replace lossy `as` casts with checked conversions or widen data types as appropriate.
- Pass heavy data (embeddings, counters) by reference to avoid copies and appease Clippy.
