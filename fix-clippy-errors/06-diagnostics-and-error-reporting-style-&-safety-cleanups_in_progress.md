# 06 Diagnostics And Error Reporting – Style & Safety Cleanups

## Scope
- `engram-core/src/differential/reporting.rs`
- `engram-core/src/error_review.rs`
- `engram-core/src/error_testing.rs`
- `engram-core/src/error/cognitive.rs`
- `engram-core/src/differential/mod.rs`
- `engram-core/src/error/recovery.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/differential/reporting.rs` | 22 | format push string (19), unwrap used (2), too many lines (1) |
| `engram-core/src/error_review.rs` | 15 | format push string (12), unused self (2), unnecessary wraps (1) |
| `engram-core/src/error_testing.rs` | 15 | unused self (12), struct excessive bools (3) |
| `engram-core/src/error/cognitive.rs` | 5 | panic (5) |
| `engram-core/src/differential/mod.rs` | 1 | option if let else (1) |
| `engram-core/src/error/recovery.rs` | 1 | cognitive complexity (1) |

## Recommended approach
- Replace `unwrap`/`expect` with error propagation or tailored assertions.
- Return structured errors instead of panicking in `Result`-returning flows.
- Remove, rename, or gate unused parameters and fields to keep APIs tight.
- Adopt idiomatic combinators (`map_or`, `ok_or`, `matches!`) in place of manual branching.
