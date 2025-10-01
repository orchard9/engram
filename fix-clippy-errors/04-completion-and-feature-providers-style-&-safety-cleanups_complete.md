# 04 Completion And Feature Providers – Style & Safety Cleanups

## Scope
- `engram-core/src/completion/context.rs`
- `engram-core/src/completion/hippocampal.rs`
- `engram-core/src/completion/consolidation.rs`
- `engram-core/src/completion/reconstruction.rs`
- `engram-core/src/features/mod.rs`
- `engram-core/src/features/monitoring.rs`
- `engram-core/src/completion/confidence.rs`
- `engram-core/src/completion/hypothesis.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/completion/context.rs` | 12 | unused self (6), unwrap used (3), option if let else (2) |
| `engram-core/src/completion/hippocampal.rs` | 11 | unused self (4), option if let else (3), expect used (2) |
| `engram-core/src/completion/consolidation.rs` | 10 | unused self (4), fields `episodes`, `timestamp`, `ripple frequency`, and `ripple duration` are never read (1), field `failed consolidations` is never read (1) |
| `engram-core/src/completion/reconstruction.rs` | 3 | map unwrap or (2), fields `config` and `pattern cache` are never read (1) |
| `engram-core/src/features/mod.rs` | 2 | panic (1), write with newline (1) |
| `engram-core/src/features/monitoring.rs` | 2 | field `config` is never read (1), field `name` is never read (1) |
| `engram-core/src/completion/confidence.rs` | 1 | field `source` is never read (1) |
| `engram-core/src/completion/hypothesis.rs` | 1 | fields `activation` and `source` are never read (1) |

## Recommended approach
- Replace `unwrap`/`expect` with error propagation or tailored assertions.
- Return structured errors instead of panicking in `Result`-returning flows.
- Remove, rename, or gate unused parameters and fields to keep APIs tight.
- Adopt idiomatic combinators (`map_or`, `ok_or`, `matches!`) in place of manual branching.
- Trim unused struct members or wire them into tests so the compiler sees their value.
