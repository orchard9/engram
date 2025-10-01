# 02 Storage Addressing And Indexing – Documentation & Attributes

## Scope
- `engram-core/src/storage/confidence.rs`
- `engram-core/src/storage/content_addressing.rs`
- `engram-core/src/storage/access_tracking.rs`
- `engram-core/src/storage/index.rs`
- `engram-core/src/storage/compact.rs`
- `engram-core/src/storage/deduplication.rs`
- `engram-core/src/storage/recovery.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/storage/confidence.rs` | 12 | must use candidate (7), missing const for fn (4), return self not must use (1) |
| `engram-core/src/storage/content_addressing.rs` | 12 | must use candidate (10), missing const for fn (2) |
| `engram-core/src/storage/access_tracking.rs` | 8 | must use candidate (8) |
| `engram-core/src/storage/index.rs` | 7 | missing const for fn (3), must use candidate (2), new without default (1) |
| `engram-core/src/storage/compact.rs` | 5 | missing const for fn (2), new without default (1), must use candidate (1) |
| `engram-core/src/storage/deduplication.rs` | 5 | must use candidate (2), missing const for fn (2), missing panics doc (1) |
| `engram-core/src/storage/recovery.rs` | 2 | missing const for fn (2) |

## Recommended approach
- Add the missing `# Errors` / `# Panics` sections so linted APIs describe failure modes.
- Apply `#[must_use]` or adjust return signatures where results should not be ignored.
- Promote simple helpers to `const fn` (or justify why they cannot be const).
- Provide `Default` impls or rename constructors that require parameters.
