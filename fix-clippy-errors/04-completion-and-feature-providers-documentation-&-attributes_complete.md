# 04 Completion And Feature Providers – Documentation & Attributes

## Scope
- `engram-core/src/features/decay.rs`
- `engram-core/src/features/index.rs`
- `engram-core/src/features/completion.rs`
- `engram-core/src/features/mod.rs`
- `engram-core/src/features/monitoring.rs`
- `engram-core/src/completion/context.rs`
- `engram-core/src/features/null_impls.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/features/decay.rs` | 9 | missing const for fn (5), must use candidate (2), missing errors doc (1) |
| `engram-core/src/features/index.rs` | 9 | missing errors doc (4), must use candidate (2), missing const for fn (2) |
| `engram-core/src/features/completion.rs` | 8 | missing errors doc (3), must use candidate (2), missing const for fn (2) |
| `engram-core/src/features/mod.rs` | 8 | doc markdown (6), double must use (2) |
| `engram-core/src/features/monitoring.rs` | 5 | must use candidate (2), missing errors doc (1), new without default (1) |
| `engram-core/src/completion/context.rs` | 2 | missing panics doc (2) |
| `engram-core/src/features/null_impls.rs` | 1 | doc markdown (1) |

## Recommended approach
- Add the missing `# Errors` / `# Panics` sections so linted APIs describe failure modes.
- Apply `#[must_use]` or adjust return signatures where results should not be ignored.
- Promote simple helpers to `const fn` (or justify why they cannot be const).
- Fix intra-doc markdown (backticks around identifiers, valid headings).
- Provide `Default` impls or rename constructors that require parameters.
