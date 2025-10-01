# 02 Storage Addressing And Indexing – Style & Safety Cleanups

## Scope
- `engram-core/src/storage/access_tracking.rs`
- `engram-core/src/storage/content_addressing.rs`
- `engram-core/src/storage/recovery.rs`
- `engram-core/src/storage/deduplication.rs`
- `engram-core/src/storage/index.rs`
- `engram-core/src/storage/compact.rs`
- `engram-core/src/storage/mod.rs`
- `engram-core/src/storage/confidence.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/storage/access_tracking.rs` | 8 | map unwrap or (7), redundant closure for method calls (1) |
| `engram-core/src/storage/content_addressing.rs` | 6 | use self (2), associated functions `quantize embedding`, `compute semantic hash`, `compute lsh bucket fast`, `compute lsh bucket`, `random projection`, and `pseudo random` are never used (1), option if let else (1) |
| `engram-core/src/storage/recovery.rs` | 4 | unused self (2), unnecessary wraps (2) |
| `engram-core/src/storage/deduplication.rs` | 3 | derive partial eq without eq (1), should implement trait (1), unwrap used (1) |
| `engram-core/src/storage/index.rs` | 3 | unused self (2), unnecessary wraps (1) |
| `engram-core/src/storage/compact.rs` | 2 | unused self (1), unnecessary wraps (1) |
| `engram-core/src/storage/mod.rs` | 2 | items after statements (1), manual clamp (1) |
| `engram-core/src/storage/confidence.rs` | 1 | struct field names (1) |

## Recommended approach
- Replace `unwrap`/`expect` with error propagation or tailored assertions.
- Remove, rename, or gate unused parameters and fields to keep APIs tight.
- Adopt idiomatic combinators (`map_or`, `ok_or`, `matches!`) in place of manual branching.
- Trim unused struct members or wire them into tests so the compiler sees their value.
- Switch to iterator adapters (e.g., `for item in collection`) for clarity and perf hints.
