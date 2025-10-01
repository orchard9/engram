# 02 Storage Addressing And Indexing – Numeric & Cross-Cutting Items

## Scope
- `engram-core/src/storage/content_addressing.rs`
- `engram-core/src/storage/access_tracking.rs`
- `engram-core/src/storage/confidence.rs`
- `engram-core/src/storage/deduplication.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/storage/content_addressing.rs` | 17 | cast precision loss (7), cast sign loss (4), cast possible truncation (3) |
| `engram-core/src/storage/access_tracking.rs` | 16 | cast precision loss (11), suboptimal flops (2), cast possible truncation (1) |
| `engram-core/src/storage/confidence.rs` | 3 | cast precision loss (2), suboptimal flops (1) |
| `engram-core/src/storage/deduplication.rs` | 3 | cast precision loss (2), assigning clones (1) |

## Recommended approach
- Replace lossy `as` casts with checked conversions or widen data types as appropriate.
- Exploit `mul_add` / fused ops or precompute invariants to silence floating-point efficiency lints.
