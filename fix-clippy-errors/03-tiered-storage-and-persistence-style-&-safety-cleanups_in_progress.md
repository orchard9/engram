# 03 Tiered Storage And Persistence – Style & Safety Cleanups

## Scope
- `engram-core/src/storage/tiers.rs`
- `engram-core/src/storage/cache.rs`
- `engram-core/src/storage/hot_tier.rs`
- `engram-core/src/storage/wal.rs`
- `engram-core/src/storage/mapped.rs`
- `engram-core/src/storage/cold_tier.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/storage/tiers.rs` | 13 | unused implementer of `std::future::Future` that must be used (4), unused self (2), unnecessary wraps (2) |
| `engram-core/src/storage/cache.rs` | 8 | unused self (4), fields `generation` and `allocator` are never read (1), needless borrow (1) |
| `engram-core/src/storage/hot_tier.rs` | 8 | option if let else (3), explicit iter loop (3), unused self (1) |
| `engram-core/src/storage/wal.rs` | 8 | use self (2), needless borrow (2), fields `current file` and `wal dir` are never read (1) |
| `engram-core/src/storage/mapped.rs` | 6 | unused self (2), ptr as ptr (2), field `numa topology` is never read (1) |
| `engram-core/src/storage/cold_tier.rs` | 5 | unused self (2), option if let else (1), too many lines (1) |

## Recommended approach
- Replace `unwrap`/`expect` with error propagation or tailored assertions.
- Remove, rename, or gate unused parameters and fields to keep APIs tight.
- Adopt idiomatic combinators (`map_or`, `ok_or`, `matches!`) in place of manual branching.
- Trim unused struct members or wire them into tests so the compiler sees their value.
- Switch to iterator adapters (e.g., `for item in collection`) for clarity and perf hints.
