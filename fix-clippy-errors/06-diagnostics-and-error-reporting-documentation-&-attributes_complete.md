# 06 Diagnostics And Error Reporting – Documentation & Attributes

## Scope
- `engram-core/src/error/recovery.rs`
- `engram-core/src/lib.rs`
- `engram-core/src/differential/reporting.rs`
- `engram-core/src/error/cognitive.rs`
- `engram-core/src/error_review.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/error/recovery.rs` | 4 | return self not must use (4) |
| `engram-core/src/lib.rs` | 2 | unused import: `anyhow` (1), missing errors doc (1) |
| `engram-core/src/differential/reporting.rs` | 1 | missing errors doc (1) |
| `engram-core/src/error/cognitive.rs` | 1 | missing const for fn (1) |
| `engram-core/src/error_review.rs` | 1 | missing errors doc (1) |

## Recommended approach
- Add the missing `# Errors` / `# Panics` sections so linted APIs describe failure modes.
- Apply `#[must_use]` or adjust return signatures where results should not be ignored.
- Promote simple helpers to `const fn` (or justify why they cannot be const).
