# 05 Query And Cue Pipeline – Documentation & Attributes

## Scope
- `engram-core/src/query/verification.rs`
- `engram-core/src/index/mod.rs`
- `engram-core/src/query/evidence.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/query/verification.rs` | 3 | must use candidate (3) |
| `engram-core/src/index/mod.rs` | 1 | missing errors doc (1) |
| `engram-core/src/query/evidence.rs` | 1 | missing errors doc (1) |

## Recommended approach
- Add the missing `# Errors` / `# Panics` sections so linted APIs describe failure modes.
- Apply `#[must_use]` or adjust return signatures where results should not be ignored.
