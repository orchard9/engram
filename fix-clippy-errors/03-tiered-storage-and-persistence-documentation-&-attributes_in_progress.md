# 03 Tiered Storage And Persistence – Documentation & Attributes

## Scope
- `engram-core/src/storage/wal.rs`
- `engram-core/src/storage/cache.rs`
- `engram-core/src/storage/mapped.rs`
- `engram-core/src/storage/hot_tier.rs`
- `engram-core/src/storage/cold_tier.rs`
- `engram-core/src/storage/tiers.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/storage/wal.rs` | 12 | missing errors doc (6), must use candidate (3), doc markdown (2) |
| `engram-core/src/storage/cache.rs` | 9 | missing const for fn (8), doc markdown (1) |
| `engram-core/src/storage/mapped.rs` | 6 | missing const for fn (2), must use candidate (2), missing errors doc (2) |
| `engram-core/src/storage/hot_tier.rs` | 5 | doc markdown (3), must use candidate (2) |
| `engram-core/src/storage/cold_tier.rs` | 4 | must use candidate (2), missing const for fn (1), missing errors doc (1) |
| `engram-core/src/storage/tiers.rs` | 2 | missing const for fn (2) |

## Recommended approach
- Add the missing `# Errors` / `# Panics` sections so linted APIs describe failure modes.
- Apply `#[must_use]` or adjust return signatures where results should not be ignored.
- Promote simple helpers to `const fn` (or justify why they cannot be const).
- Fix intra-doc markdown (backticks around identifiers, valid headings).
