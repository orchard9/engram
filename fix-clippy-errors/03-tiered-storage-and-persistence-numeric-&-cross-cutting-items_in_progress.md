# 03 Tiered Storage And Persistence – Numeric & Cross-Cutting Items

## Scope
- `engram-core/src/storage/cold_tier.rs`
- `engram-core/src/storage/cache.rs`
- `engram-core/src/storage/tiers.rs`
- `engram-core/src/storage/hot_tier.rs`
- `engram-core/src/storage/mapped.rs`
- `engram-core/src/storage/warm_tier.rs`
- `engram-core/src/storage/wal.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/storage/cold_tier.rs` | 21 | cast precision loss (9), cast possible truncation (4), significant drop tightening (3) |
| `engram-core/src/storage/cache.rs` | 11 | cast possible truncation (5), cast precision loss (2), cast ptr alignment (2) |
| `engram-core/src/storage/tiers.rs` | 11 | cast precision loss (5), cast possible truncation (3), cast sign loss (3) |
| `engram-core/src/storage/hot_tier.rs` | 9 | cast precision loss (6), cast possible truncation (2), cast sign loss (1) |
| `engram-core/src/storage/mapped.rs` | 5 | cast possible truncation (5) |
| `engram-core/src/storage/warm_tier.rs` | 3 | cast precision loss (1), cast possible truncation (1), cast sign loss (1) |
| `engram-core/src/storage/wal.rs` | 1 | significant drop tightening (1) |

## Recommended approach
- Replace lossy `as` casts with checked conversions or widen data types as appropriate.
- Pass heavy data (embeddings, counters) by reference to avoid copies and appease Clippy.
